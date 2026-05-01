import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import argparse
import time
import random
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt  # 新增绘图库

# ==========================================
# 🔧 全局配置中心 (Config V9 - Debug Mode)
# ==========================================
class Config:
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "/home/brandon/brandon/hyrd_robot/lifelong_data/models"
    DATA_PATH = "/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_big.pt"
    PLOT_DIR = os.path.join(SAVE_DIR, "plots") # 新增绘图保存路径
    
    NUM_WORKERS = 8          
    PIN_MEMORY = True         
    
    EPOCHS = 3000             
    BATCH_SIZE = 2048
    SEQ_LEN = 50
    
    LR = 1e-5     
    WARMUP_EPOCHS = 30        
    LR_PATIENCE = 10          
    LR_FACTOR = 0.5           
    MIN_LR = 1e-7  
    
    WEIGHT_DECAY = 1e-3
    DROPOUT = 0.25             
    
    LAMBDA_LOCAL = 10.0
    LAMBDA_SHAPE = 1.0
    
    VAL_RATIO = 0.15
    TEST_RATIO = 0.05
    PATIENCE = 50             
    DELTA = 1e-6

os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(Config.PLOT_DIR, exist_ok=True)

# ================= 1. 工具类 =================
class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

    def load_state(self, state_dict):
        self.best_score = state_dict['best_score']
        self.counter = state_dict['counter']
        self.early_stop = state_dict['early_stop']

    def state_dict(self):
        return {
            'best_score': self.best_score,
            'counter': self.counter,
            'early_stop': self.early_stop
        }

# ================= 2. 数学层 =================
class DifferentiableFK(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, current_local_pose, pred_local_delta):
        batch_size = current_local_pose.shape[0]
        next_state = current_local_pose + pred_local_delta
        x, y, theta = next_state[:,:,0], next_state[:,:,1], next_state[:,:,2]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        zeros, ones = torch.zeros_like(x), torch.ones_like(x)
        r0 = torch.stack([cos_t, -sin_t, x], dim=2)
        r1 = torch.stack([sin_t, cos_t, y], dim=2)
        r2 = torch.stack([zeros, zeros, ones], dim=2)
        T_local = torch.stack([r0, r1, r2], dim=2)
        T_curr = torch.eye(3, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        global_points_list = []
        for i in range(5):
            T_curr = torch.bmm(T_curr, T_local[:, i, :, :])
            global_points_list.append(T_curr[:, :2, 2])
        return torch.stack(global_points_list, dim=1)

# ================= 3. 网络架构 (修改版: 支持中间变量输出) =================
class TopoImpulseNet(nn.Module):
    def __init__(self, scalers, input_dim=18, feature_dim=64, gru_dim=64, dropout=0.1):
        super().__init__()
        
        # 注册 Buffer
        self.register_buffer('jac_mean', scalers['jacobian_mean'])
        self.register_buffer('jac_std', scalers['jacobian_std'])
        self.register_buffer('cmd_mean', scalers['dq_cmd_mean'])
        self.register_buffer('cmd_std', scalers['dq_cmd_std'])

        self.num_nodes = 5
        self.stage1_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(self.num_nodes)
        ])
        
        # 注意: GRU 是在节点维度(spatial)上运行，不是时间维度
        self.stage2_gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=gru_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.gru_dropout = nn.Dropout(dropout)
        
        head_input_dim = feature_dim + (gru_dim * 2) + 6 
        
        self.stage3_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 3) 
            ) for _ in range(self.num_nodes)
        ])
        
        self.stage3_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 3), 
                nn.Sigmoid()
            ) for _ in range(self.num_nodes)
        ])

    def compute_nominal_delta(self, jac_norm, dq_cmd_norm):
        """
        计算物理基准项 A (Physics Base)
        公式: A = Unnormalized(J) * Unnormalized(dq)
        """
        B, N, _ = jac_norm.shape
        
        # 1. 反归一化 (Recover Physics Data)
        jac_real = jac_norm * self.jac_std + self.jac_mean
        dq_real = dq_cmd_norm * self.cmd_std + self.cmd_mean
        
        # 2. 重塑矩阵 [Batch, 5, 6] -> [Batch, 5, 3, 2]
        J_matrix = jac_real.view(B, N, 3, 2)
        
        # 3. 显式物理计算: dx = J * dq
        dq_vec = dq_real.unsqueeze(-1)
        nominal_delta = torch.matmul(J_matrix, dq_vec).squeeze(-1) # [B, N, 3]
        
        return nominal_delta

    def forward(self, x_inputs, jacobian_norm, dq_cmd_norm, return_internals=False):
        # --- Stage 1 & 2 (Feature Extraction) ---
        local_features = []
        for i in range(self.num_nodes):
            feat = self.stage1_experts[i](x_inputs[:, i, :]) 
            local_features.append(feat)
        local_features_tensor = torch.stack(local_features, dim=1) 
        gru_out, _ = self.stage2_gru(local_features_tensor) 
        gru_out = self.gru_dropout(gru_out)
        
        # --- Stage 3 (Physics + Residual) ---
        nominal_delta = self.compute_nominal_delta(jacobian_norm, dq_cmd_norm)
        
        outputs = []
        alphas_list = []
        residuals_list = []
        
        for i in range(self.num_nodes):
            head_in = torch.cat([local_features[i], gru_out[:, i, :], jacobian_norm[:, i, :]], dim=1) 
            
            # 计算残差 B
            raw_residual = self.stage3_heads[i](head_in)
            
            # 计算门控 Alpha
            alpha = self.stage3_gates[i](head_in)
            
            # 融合: Output = A + alpha * B
            final_pred = nominal_delta[:, i, :] + alpha * raw_residual
            
            outputs.append(final_pred)
            
            if return_internals:
                alphas_list.append(alpha)
                residuals_list.append(raw_residual)
            
        final_stack = torch.stack(outputs, dim=1)
        
        if return_internals:
            return final_stack, nominal_delta, torch.stack(alphas_list, dim=1), torch.stack(residuals_list, dim=1)
        else:
            return final_stack

# ================= 4. 数据集 =================
class RoboSeqDataset(Dataset):
    def __init__(self, pt_path, seq_len=50):
        data = torch.load(pt_path)
        self.tensors = data['data']
        self.scalers = data['scalers']
        self.seq_len = seq_len
        self.total_frames = self.tensors['q_curr'].shape[0]
        self.num_seqs = self.total_frames // seq_len
        
    def __len__(self): return self.num_seqs
    def _norm(self, tensor, key):
        mean = self.scalers[f'{key}_mean']
        std = self.scalers[f'{key}_std']
        return (tensor - mean) / std
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        q_feat = self._norm(self.tensors['q_curr'][start:end], 'q_curr')
        dq_feat = self._norm(self.tensors['dq_hist'][start:end], 'dq_hist')
        pose_feat = self.tensors['pose_loc'][start:end]
        pose_norm = self._norm(pose_feat, 'pose_loc')
        cmd_feat = self._norm(self.tensors['dq_cmd'][start:end], 'dq_cmd')
        jac_feat = self.tensors['jacobian'][start:end]
        jac_norm = self._norm(jac_feat, 'jacobian')
        tgt_delta = self.tensors['tgt_delta'][start:end]
        tgt_global = self.tensors['tgt_global'][start:end]
        return {
            'q': q_feat.float(), 'dq': dq_feat.float(),
            'pose': pose_norm.float(), 'pose_raw': pose_feat.float(),
            'cmd': cmd_feat.float(), 'jac': jac_norm.float(),
            'tgt_delta': tgt_delta.float(), 'tgt_global': tgt_global.float()
        }

# ================= 5. 可视化模块 (新功能) =================
def visualize_physics_alignment(model, dataset, device, epoch, save_path):
    """
    随机抽取一段连续 100 帧的数据，绘制物理基准 vs 真实值 vs 神经网络输出
    用于验证 Scale Mismatch 问题
    """
    model.eval()
    
    # 1. 随机找个起点，取连续两段 (50+50 = 100帧)
    idx = random.randint(0, len(dataset) - 2)
    
    # 获取两段数据
    seq1 = dataset[idx]
    seq2 = dataset[idx+1]
    
    # 拼接成 100 长度的序列
    # Dictionary of tensors
    combined = {}
    for k in seq1.keys():
        combined[k] = torch.cat([seq1[k], seq2[k]], dim=0) # [100, 5, D]
        
    # 2. 构造 Batch
    # 由于模型中的 GRU 是 spatial (node) 维度的，时间上没有 hidden state 依赖
    # 我们可以直接把 Time 维度作为 Batch 维度输入，一次性推理 100 帧
    
    with torch.no_grad():
        # Move to device: [100, 5, D]
        q_t = combined['q'].to(device)
        dq_t = combined['dq'].to(device)
        pose_t = combined['pose'].to(device)
        cmd_t = combined['cmd'].to(device)
        jac_t = combined['jac'].to(device)
        tgt_delta = combined['tgt_delta'].to(device)
        
        input_vec = torch.cat([q_t, dq_t, pose_t, cmd_t, jac_t], dim=2)
        
        # 开启 return_internals
        final_pred, nom_delta, alphas, residuals = model(input_vec, jac_t, cmd_t, return_internals=True)
        
    # 3. 提取 Node 4 (Tip) 的数据进行绘图
    # Shape: [100, 3] (dx, dy, dtheta)
    node_idx = 4 
    
    gt_np = tgt_delta[:, node_idx, :].cpu().numpy()
    nom_np = nom_delta[:, node_idx, :].cpu().numpy()
    pred_np = final_pred[:, node_idx, :].cpu().numpy()
    alpha_np = alphas[:, node_idx, :].cpu().numpy()
    res_np = residuals[:, node_idx, :].cpu().numpy()
    
    t_axis = np.arange(100)
    
    # 4. 绘图
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    labels = ['Delta X (mm)', 'Delta Y (mm)', 'Delta Theta (rad)']
    
    for i in range(3): # x, y, theta
        # Row 1: 物理基准 vs 真实值 vs 最终预测
        ax = axes[0, i]
        ax.plot(t_axis, gt_np[:, i], 'k-', linewidth=2, label='Ground Truth', alpha=0.6)
        ax.plot(t_axis, nom_np[:, i], 'r--', linewidth=1.5, label='Physics (Nominal)')
        ax.plot(t_axis, pred_np[:, i], 'g:', linewidth=2, label='Net Prediction')
        ax.set_title(f"{labels[i]} - Physics Alignment")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
        
        # Row 2: Alpha 门控值
        ax = axes[1, i]
        ax.plot(t_axis, alpha_np[:, i], 'b-', label='Gate Alpha')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"Alpha Gate (Confidence) - {['X','Y','Th'][i]}")
        ax.grid(True, alpha=0.3)
        
        # Row 3: Residual 修正量
        ax = axes[2, i]
        ax.plot(t_axis, res_np[:, i], 'm-', label='Raw Residual (B)')
        # 叠加物理误差用于对比
        phy_err = gt_np[:, i] - nom_np[:, i]
        ax.plot(t_axis, phy_err, 'k--', alpha=0.3, label='Ideally Needed Correction')
        ax.set_title(f"Residual Correction - {['X','Y','Th'][i]}")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"debug_epoch_{epoch:03d}.png"))
    plt.close()

# ================= 6. 常规函数 =================
def split_continuous_blocks(total_seqs, val_ratio, test_ratio, seed):
    if seed is None: seed = int(time.time())
    random.seed(seed)
    n_val = int(total_seqs * val_ratio)
    n_test = int(total_seqs * test_ratio)
    val_start = random.randint(0, total_seqs - n_val)
    val_indices = list(range(val_start, val_start + n_val))
    possible_starts = []
    if val_start - n_test >= 0: possible_starts.extend(range(0, val_start - n_test + 1))
    if total_seqs - n_test >= val_start + n_val: possible_starts.extend(range(val_start + n_val, total_seqs - n_test + 1))
    test_start = random.choice(possible_starts)
    test_indices = list(range(test_start, test_start + n_test))
    reserved = set(val_indices + test_indices)
    train_indices = [i for i in range(total_seqs) if i not in reserved]
    return train_indices, val_indices, test_indices

def run_epoch(model, dataloader, math_layer, device, optimizer=None, is_train=True):
    if is_train: model.train()
    else: model.eval()
    
    total_loss = 0      
    total_mse = 0       
    total_node_errors = np.zeros(5)
    max_error_mm_epoch = 0.0
    
    pbar = tqdm(dataloader, desc="Step", leave=False, dynamic_ncols=True)
    
    for batch in pbar:
        loss_seq = 0
        mse_seq = 0
        batch_node_errors = np.zeros(5)
        batch_max_error = 0.0
        
        with torch.set_grad_enabled(is_train):
            for t in range(Config.SEQ_LEN): 
                q_t = batch['q'][:, t].to(device)
                dq_t = batch['dq'][:, t].to(device)
                pose_t = batch['pose'][:, t].to(device)
                cmd_t = batch['cmd'][:, t].to(device) 
                jac_t = batch['jac'][:, t].to(device) 
                pose_raw_t = batch['pose_raw'][:, t].to(device)
                gt_delta = batch['tgt_delta'][:, t].to(device)
                gt_global = batch['tgt_global'][:, t].to(device)
                
                input_vec = torch.cat([q_t, dq_t, pose_t, cmd_t, jac_t], dim=2)
                
                # 正常训练不需要 internal
                pred_delta = model(input_vec, jac_t, cmd_t, return_internals=False)
                
                pred_global = math_layer(pose_raw_t, pred_delta)
                
                l_local = nn.HuberLoss(delta=1.0)(pred_delta, gt_delta)
                l_shape = nn.HuberLoss(delta=1.0)(pred_global, gt_global)
                loss_step = l_shape + Config.LAMBDA_LOCAL * l_local
                loss_seq += loss_step
                
                mse_step = nn.MSELoss()(pred_global, gt_global)
                mse_seq += mse_step
                
                with torch.no_grad():
                    dist = torch.norm(pred_global - gt_global, dim=2) 
                    batch_node_errors += dist.mean(dim=0).cpu().numpy() 
                    current_max = dist.max().item()
                    if current_max > batch_max_error: 
                        batch_max_error = current_max
            
            loss_final = loss_seq / Config.SEQ_LEN
            if is_train:
                optimizer.zero_grad()
                loss_final.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss_final.item()
            total_mse += (mse_seq / Config.SEQ_LEN).item()
            total_node_errors += (batch_node_errors / Config.SEQ_LEN)
            if batch_max_error > max_error_mm_epoch: 
                max_error_mm_epoch = batch_max_error
            
            pbar.set_postfix({'MSE': f"{(mse_seq/Config.SEQ_LEN).item():.2f}"})
            
    return {
        'loss': total_loss / len(dataloader),
        'mse': total_mse / len(dataloader),
        'node_errs': total_node_errors / len(dataloader),
        'max_err': max_error_mm_epoch
    }

def save_full_state(path, epoch, model, optimizer, scheduler, early_stopping):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'es_state_dict': early_stopping.state_dict(),
    }
    torch.save(state, path)

# ================= 7. 训练主循环 =================
def train(args):    
    print(f"🚀 Topo-Impulse V9 (Physics Diagnostic) | Device: {Config.DEVICE}")
    print(f"📦 BS={Config.BATCH_SIZE} | Workers={Config.NUM_WORKERS} | Target LR={Config.LR:.2e}")
    
    torch.backends.cudnn.benchmark = True

    full_dataset = RoboSeqDataset(os.path.expanduser(Config.DATA_PATH), seq_len=Config.SEQ_LEN)
    train_idx, val_idx, test_idx = split_continuous_blocks(
        len(full_dataset), Config.VAL_RATIO, Config.TEST_RATIO, Config.SEED
    )
    
    # 保存 Validation Dataset 引用用于绘图
    val_dataset = Subset(full_dataset, val_idx)
    
    kwargs = {
        'num_workers': Config.NUM_WORKERS, 
        'pin_memory': Config.PIN_MEMORY,
        'persistent_workers': True
    }
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=Config.BATCH_SIZE, shuffle=False, **kwargs)
    
    model = TopoImpulseNet(
        scalers=full_dataset.scalers,
        input_dim=15, 
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    math_layer = DifferentiableFK().to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=Config.LR_FACTOR, patience=Config.LR_PATIENCE,
        min_lr=Config.MIN_LR
    )
    
    best_model_path = os.path.join(Config.SAVE_DIR, 'topo_best_model_big.pth')
    resume_path = os.path.join(Config.SAVE_DIR, 'topo_latest_checkpoint_big.pth')
    
    early_stopping = EarlyStopping(patience=Config.PATIENCE, delta=Config.DELTA, path=best_model_path)

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': [], 'val_max_err': [], 'lr': []}
    log_path = os.path.join(Config.SAVE_DIR, 'training_log.json')
    
    start_epoch = 0
    if args.resume:
        load_path = args.resume if args.resume != 'latest' else resume_path
        if os.path.exists(load_path):
            print(f"🔄 Resuming from: {load_path}")
            checkpoint = torch.load(load_path, map_location=Config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint: plateau_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'es_state_dict' in checkpoint: early_stopping.load_state(checkpoint['es_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
    epoch_pbar = tqdm(range(start_epoch, Config.EPOCHS), desc="Progress", unit="ep", initial=start_epoch, total=Config.EPOCHS)
    
    for epoch in epoch_pbar:
        if epoch < Config.WARMUP_EPOCHS:
            warmup_lr = Config.LR * (epoch + 1) / Config.WARMUP_EPOCHS
            for param_group in optimizer.param_groups: param_group['lr'] = warmup_lr
            lr_status = f"Warmup {warmup_lr:.2e}"
        else:
            lr_status = f"LR {optimizer.param_groups[0]['lr']:.2e}"
            
        train_metrics = run_epoch(model, train_loader, math_layer, Config.DEVICE, optimizer, is_train=True)
        val_metrics = run_epoch(model, val_loader, math_layer, Config.DEVICE, is_train=False)
        
        # --- 🔍 关键 Debug 步骤：绘制物理对齐图 ---
        # 每个 Epoch 结束后画一次图，看看 Nominal Delta 到底有没有活着
        visualize_physics_alignment(model, val_dataset, Config.DEVICE, epoch+1, Config.PLOT_DIR)
        
        if epoch >= Config.WARMUP_EPOCHS:
            plateau_scheduler.step(val_metrics['mse'])
        early_stopping(val_metrics['mse'], model)

        train_rmse = train_metrics['mse'] ** 0.5
        val_rmse = val_metrics['mse'] ** 0.5
        node_err_str = "/".join([f"{e:.2f}" for e in val_metrics['node_errs']])

        tqdm.write(
            f"\n[Ep {epoch+1}] {lr_status} | "
            f"T_RMSE: {train_rmse:.4f} | V_RMSE: {val_rmse:.4f} | "
            f"Max: {val_metrics['max_err']:.1f}mm | Nodes: {node_err_str} | ES: {early_stopping.counter}"
        )

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_mse'].append(train_metrics['mse'])
        history['val_mse'].append(val_metrics['mse'])
        history['val_max_err'].append(val_metrics['max_err'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        import json
        with open(log_path, 'w') as f: json.dump(history, f, indent=4)
        save_full_state(resume_path, epoch, model, optimizer, plateau_scheduler, early_stopping)
        
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered!")
            break

    print("\n🏁 Final Test...")
    if os.path.exists(best_model_path): model.load_state_dict(torch.load(best_model_path))
    test_metrics = run_epoch(model, test_loader, math_layer, Config.DEVICE, is_train=False)
    print(f"🧪 Test RMSE: {test_metrics['mse']**0.5:.4f} | Max Err: {test_metrics['max_err']:.1f}mm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=Config.DATA_PATH)
    parser.add_argument('--resume', nargs='?', const='latest', help="Path to checkpoint")
    args = parser.parse_args()
    
    if args.resume == 'interrupted': args.resume = os.path.join(Config.SAVE_DIR, 'topo_interrupted_big.pth')
    elif args.resume == 'latest': args.resume = os.path.join(Config.SAVE_DIR, 'topo_latest_checkpoint_big.pth')
        
    Config.DATA_PATH = os.path.expanduser(args.data_path)
    train(args)