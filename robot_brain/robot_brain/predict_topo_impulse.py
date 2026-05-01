import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import matplotlib.pyplot as plt

# ================= 1. 配置中心 =================
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = "/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_big.pt"
    MODEL_PATH = "/home/brandon/brandon/hyrd_robot/lifelong_data/models/topo_best_model_big.pth"
    SAVE_DIR = "prediction_plots_final_v10"
    
    # 根据诊断报告：Ratio = 92.71，我们将物理项放大以补偿单位丢失
    PHYSICS_COMPENSATION = 92.71 
    
    SEQ_LEN = 50
    BATCH_SIZE = 256 
    PLOT_START_IDX = 1000  
    PLOT_LEN = 500       

# ================= 2. 核心模型 (物理增强版) =================
class TopoImpulseNet(nn.Module):
    def __init__(self, scalers, input_dim=15):
        super().__init__()
        self.register_buffer('jac_mean', scalers['jacobian_mean'])
        self.register_buffer('jac_std', scalers['jacobian_std'])
        self.register_buffer('cmd_mean', scalers['dq_cmd_mean'])
        self.register_buffer('cmd_std', scalers['dq_cmd_std'])

        # 网络层定义 (需严格对应训练时的结构)
        self.stage1 = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 64)
        ) for _ in range(5)])
        
        self.stage2_gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        
        self.stage3_heads = nn.ModuleList([nn.Linear(64 + 128 + 6, 3) for _ in range(5)])
        self.stage3_gates = nn.ModuleList([nn.Sequential(
            nn.Linear(64 + 128 + 6, 32), nn.Tanh(), nn.Linear(32, 3), nn.Sigmoid()
        ) for _ in range(5)])

    def compute_nominal_delta(self, jac_norm, dq_cmd_norm):
        B, N, _ = jac_norm.shape
        # 反归一化
        jac_real = jac_norm * self.jac_std + self.jac_mean
        dq_real = dq_cmd_norm * self.cmd_std + self.cmd_mean
        
        # 物理计算：J * dq
        J_matrix = jac_real.view(B, N, 3, 2)
        dq_vec = dq_real.unsqueeze(-1)
        
        # ⚠️ 这里应用补偿系数，修正单位失配导致的量级萎缩
        nominal_delta = torch.matmul(J_matrix, dq_vec).squeeze(-1)
        return nominal_delta * Config.PHYSICS_COMPENSATION

    def forward(self, x_inputs, jac_norm, cmd_norm):
        # 1. 物理基准 (已补偿)
        A = self.compute_nominal_delta(jac_norm, cmd_norm)
        
        # 2. 网络特征提取
        feats = torch.stack([m(x_inputs[:, i]) for i, m in enumerate(self.stage1)], dim=1)
        gru_out, _ = self.stage2_gru(feats)
        
        outputs, alphas = [], []
        for i in range(5):
            h_in = torch.cat([feats[:, i], gru_out[:, i], jac_norm[:, i]], dim=1)
            B = self.stage3_heads[i](h_in)
            alpha = self.stage3_gates[i](h_in)
            
            # Hybrid = A + alpha * B
            final_pred = A[:, i] + alpha * B
            outputs.append(final_pred)
            alphas.append(alpha)
            
        return A, torch.stack(outputs, dim=1), torch.stack(alphas, dim=1)

# ================= 3. 推理与可视化 =================
def run_final_prediction():
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    data = torch.load(Config.DATA_PATH)
    scalers = data['scalers']
    
    # 准备测试数据 (简化版，直接从 tensor 取)
    tensors = data['data']
    s, e = Config.PLOT_START_IDX, Config.PLOT_START_IDX + Config.PLOT_LEN
    
    def norm(t, k): return (t - scalers[f'{k}_mean']) / scalers[f'{k}_std']
    
    # 加载张量并移动到设备
    q = norm(tensors['q_curr'][s:e], 'q_curr').to(Config.DEVICE).float()
    dq = norm(tensors['dq_hist'][s:e], 'dq_hist').to(Config.DEVICE).float()
    pose = norm(tensors['pose_loc'][s:e], 'pose_loc').to(Config.DEVICE).float()
    cmd = norm(tensors['dq_cmd'][s:e], 'dq_cmd').to(Config.DEVICE).float()
    jac = norm(tensors['jacobian'][s:e], 'jacobian').to(Config.DEVICE).float()
    gt_delta = tensors['tgt_delta'][s:e].to(Config.DEVICE).float()

    input_vec = torch.cat([q, dq, pose, cmd, jac], dim=2)

    # 加载模型
    model = TopoImpulseNet(scalers).to(Config.DEVICE)
    ckpt = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    model.eval()

    with torch.no_grad():
        nom, hyb, alp = model(input_vec, jac, cmd)

    # 绘图逻辑：针对 Tip 节点 (Node 5) 的 X 和 Y 坐标
    time_steps = np.arange(Config.PLOT_LEN)
    node_idx = 4 # Tip
    
    for dim_idx, dim_name in enumerate(['DX', 'DY']):
        plt.figure(figsize=(14, 7))
        plt.plot(time_steps, gt_delta[:, node_idx, dim_idx].cpu(), 'k-', label='GT (Vision)', linewidth=1.5)
        plt.plot(time_steps, nom[:, node_idx, dim_idx].cpu(), 'b--', label='Physics (Nominal)', alpha=0.8)
        plt.plot(time_steps, hyb[:, node_idx, dim_idx].cpu(), 'r-', label='Hybrid Prediction', alpha=0.7)
        
        plt.title(f"Tip Node - {dim_name} Step Size Analysis (Physics Corrected)")
        plt.xlabel("Frames")
        plt.ylabel("Step Size (mm)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.5)
        
        save_path = f"{Config.SAVE_DIR}/tip_{dim_name.lower()}_corrected.png"
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"🖼️ Saved plot to: {save_path}")

    print(f"✅ Final analysis complete. Ratio {Config.PHYSICS_COMPENSATION}x applied.")

if __name__ == "__main__":
    run_final_prediction()