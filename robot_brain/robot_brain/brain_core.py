import torch
import torch.nn as nn
import numpy as np
import os
from numba import jit

# ==============================================================================
# ⚙️ 1. 物理引擎与几何参数 (必须严格匹配训练代码)
# ==============================================================================
C_LIST = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL = 22.0
H0_VAL = 52.0
M_VAL = H0_VAL - 2 * N_VAL

@jit(nopython=True, cache=True)
def get_local_pose_from_q(q_l, q_r, c_val, n_val, m_val):
    delta_q = q_l - q_r; sum_q = q_l + q_r
    theta = delta_q / c_val 
    L_c = m_val + sum_q / 2.0
    if abs(theta) < 1e-6: return np.array([0.0, 2*n_val + L_c, 0.0])
    rho = L_c / theta
    c = np.cos(-theta); s = np.sin(-theta)
    lx = rho * (1.0 - np.cos(theta)); ly = rho * np.sin(theta)
    return np.array([-s * n_val + lx, c * n_val + ly + n_val, -theta])

@jit(nopython=True, cache=True)
def compute_local_jacobian(q_pair, c_val, n_val, m_val):
    eps = 1e-4 
    J = np.zeros((3, 2), dtype=np.float64)
    curr = get_local_pose_from_q(q_pair[0], q_pair[1], c_val, n_val, m_val)
    p_l = get_local_pose_from_q(q_pair[0] + eps, q_pair[1], c_val, n_val, m_val)
    m_l = get_local_pose_from_q(q_pair[0] - eps, q_pair[1], c_val, n_val, m_val)
    J[:, 0] = (p_l - m_l) / (2 * eps)
    p_r = get_local_pose_from_q(q_pair[0], q_pair[1] + eps, c_val, n_val, m_val)
    m_r = get_local_pose_from_q(q_pair[0], q_pair[1] - eps, c_val, n_val, m_val)
    J[:, 1] = (p_r - m_r) / (2 * eps)
    return J.flatten() # Returns 6 elements

# ==============================================================================
# 🧠 2. 网络架构定义 (完全复用 train_topo_impulse_big.py)
# ==============================================================================
class DifferentiableFK(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, current_local_pose, pred_local_delta):
        # 注意：训练时 current_local_pose 是 pose_raw，物理值
        # pred_local_delta 也是物理值
        batch_size = current_local_pose.shape[0]
        next_state = current_local_pose + pred_local_delta
        x, y, theta = next_state[:,:,0], next_state[:,:,1], next_state[:,:,2]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        zeros, ones = torch.zeros_like(x), torch.ones_like(x)
        r0 = torch.stack([cos_t, -sin_t, x], dim=2)
        r1 = torch.stack([sin_t, cos_t, y], dim=2)
        r2 = torch.stack([zeros, zeros, ones], dim=2)
        T_local = torch.stack([r0, r1, r2], dim=2)
        
        # Base frame is Identity
        T_curr = torch.eye(3, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        global_points_list = []
        for i in range(5):
            T_curr = torch.bmm(T_curr, T_local[:, i, :, :])
            global_points_list.append(T_curr[:, :2, 2]) # Extract (x, y) global
        return torch.stack(global_points_list, dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.gelu = nn.GELU()
    def forward(self, x): return self.gelu(x + self.net(x))

class TopoImpulseNet(nn.Module):
    def __init__(self, scalers, input_dim=15, feature_dim=128, gru_dim=128, dropout=0.0):
        super().__init__()
        # 注册 Buffer 以便 forward 内部使用
        self.register_buffer('jac_mean', scalers['jacobian_mean'])
        self.register_buffer('jac_std', scalers['jacobian_std'])
        self.register_buffer('cmd_mean', scalers['dq_cmd_mean'])
        self.register_buffer('cmd_std', scalers['dq_cmd_std'])
        
        self.num_nodes = 5
        self.stage1_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, feature_dim), nn.LayerNorm(feature_dim),
                nn.GELU(), nn.Dropout(dropout),
                ResidualBlock(feature_dim, dropout), ResidualBlock(feature_dim, dropout) 
            ) for _ in range(self.num_nodes)
        ])
        self.stage2_gru = nn.GRU(feature_dim, gru_dim, 2, batch_first=True, bidirectional=True, dropout=dropout)
        self.gru_dropout = nn.Dropout(dropout)
        
        head_input_dim = feature_dim + (gru_dim * 2) + 6 
        self.net_expert_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, 128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3) 
            ) for _ in range(self.num_nodes)
        ])
        self.confidence_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, 64), nn.Tanh(), nn.Linear(64, 3), nn.Sigmoid() 
            ) for _ in range(self.num_nodes)
        ])

    def compute_nominal_delta(self, jac_norm, dq_cmd_norm):
        B, N, _ = jac_norm.shape
        # 必须还原到物理值进行计算
        jac_real = jac_norm * self.jac_std + self.jac_mean
        dq_real = dq_cmd_norm * self.cmd_std + self.cmd_mean
        J_matrix = jac_real.view(B, N, 3, 2)
        dq_vec = dq_real.unsqueeze(-1)
        nominal_delta = torch.matmul(J_matrix, dq_vec).squeeze(-1)
        return nominal_delta

    def forward(self, x_inputs, jacobian_norm, dq_cmd_norm):
        local_features = [self.stage1_experts[i](x_inputs[:, i, :]) for i in range(self.num_nodes)]
        local_features_tensor = torch.stack(local_features, dim=1) 
        gru_out, _ = self.stage2_gru(local_features_tensor)
        gru_out = self.gru_dropout(gru_out)
        nominal_delta = self.compute_nominal_delta(jacobian_norm, dq_cmd_norm)
        
        outputs, betas = [], []
        for i in range(self.num_nodes):
            feature = torch.cat([local_features[i], gru_out[:, i, :], jacobian_norm[:, i, :]], dim=1)
            net_prediction = self.net_expert_head[i](feature) 
            beta = self.confidence_gate[i](feature)
            final_pred = beta * nominal_delta[:, i, :] + (1.0 - beta) * net_prediction
            outputs.append(final_pred)
            betas.append(beta)
            
        return torch.stack(outputs, dim=1), torch.stack(betas, dim=1)

# ==============================================================================
# 🚀 3. 推理接口 (修正版：解决 cuDNN RNN Backward 问题)
# ==============================================================================
class NeuralBrain:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        self.MOT_INDICES_MAP = [[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]
        
        # 1. 加载 Scalers
        data_path = "/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_smooth_strided.pt"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ 数据集文件未找到: {data_path}")

        print(f"📊 Loading Scalers from {data_path}...")
        dataset_content = torch.load(data_path, map_location=self.device)
        self.scalers = dataset_content['scalers']
        
        # 2. 加载权重
        print(f"🧠 Loading TopoImpulse Weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("💡 Detected Full Checkpoint format.")
        else:
            state_dict = checkpoint
            print("💡 Detected Pure State Dict format.")

        # 3. 初始化模型
        # 注意：dropout=0.0 确保 train 模式下没有随机性
        self.model = TopoImpulseNet(self.scalers, input_dim=15, feature_dim=128, gru_dim=128, dropout=0.0).to(self.device)
        
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"❌ 权重加载失败! 请检查 input_dim。\n详细错误: {e}")
            raise e

        # 🔥 [关键修改] 必须使用 train() 模式！
        # 原因：eval() 模式下，PyTorch 的 cuDNN RNN 后端不会保存中间状态，
        # 导致无法对输入(dq_cmd)进行求导(Autograd)。
        # 因为我们已经设置了 dropout=0.0 且冻结了参数，train() 模式是安全的。
        self.model.train() 
        
        self.diff_fk = DifferentiableFK().to(self.device)
        
        # 4. 冻结所有权重 (Double Safe)
        for param in self.model.parameters():
            param.requires_grad = False
            
        print("✅ Brain Ready with Autograd Support (Train Mode Forced).")

    def _norm(self, tensor, key):
        return (tensor - self.scalers[f'{key}_mean']) / self.scalers[f'{key}_std']

    def _denorm(self, tensor, key):
        return tensor * self.scalers[f'{key}_std'] + self.scalers[f'{key}_mean']

    def get_shape_jacobian(self, q_curr, dq_hist, pose_loc_phys):
        """
        全状态雅可比计算 (10x10) - 优化版
        不再使用循环 10 次 autograd，而是使用 functional.jacobian
        """
        # --- 1. 数据组装 (与原代码一致) ---
        q_nodes, dq_nodes, pose_nodes, jac_nodes = [], [], [], []
        
        for i in range(5):
            idx = self.MOT_INDICES_MAP[i]
            q_pair = q_curr[idx]
            dq_pair = dq_hist[idx]
            j_flat = compute_local_jacobian(q_pair, C_LIST[i], N_VAL, M_VAL)
            q_nodes.append(q_pair); dq_nodes.append(dq_pair); 
            pose_nodes.append(pose_loc_phys[i]); jac_nodes.append(j_flat)

        # 转换为 Tensor，注意不需要 requires_grad，因为它们是常量条件
        t_q = torch.tensor(np.array([q_nodes]), dtype=torch.float32, device=self.device)
        t_dq = torch.tensor(np.array([dq_nodes]), dtype=torch.float32, device=self.device)
        t_pose_phys = torch.tensor(np.array([pose_nodes]), dtype=torch.float32, device=self.device)
        t_jac = torch.tensor(np.array([jac_nodes]), dtype=torch.float32, device=self.device)

        n_q = self._norm(t_q, 'q_curr')
        n_dq = self._norm(t_dq, 'dq_hist')
        n_pose = self._norm(t_pose_phys, 'pose_loc')
        n_jac = self._norm(t_jac, 'jacobian')

        # 初始指令为 0 (线性化点)
        t_cmd_norm_init = torch.zeros((1, 5, 2), dtype=torch.float32, device=self.device)

        # --- 2. 定义闭包函数 (Forward Logic) ---
        # 这个函数必须包含从 cmd -> x_inputs -> model -> diff_fk -> output 的完整链路
        # 这样 autograd 才能追踪到 cmd 的梯度
        def forward_func(cmd_norm_arg):
            # 关键：必须在函数内部重新拼接 x_inputs，
            # 否则计算图无法建立 cmd_norm_arg 与输入的联系
            x_in = torch.cat([n_q, n_dq, n_pose, cmd_norm_arg, n_jac], dim=2)
            
            # Model Forward
            pred_delta_norm, _ = self.model(x_in, n_jac, cmd_norm_arg)
            
            # Denorm & FK
            pred_delta_phys = self._denorm(pred_delta_norm, 'tgt_delta')
            global_shape = self.diff_fk(t_pose_phys, pred_delta_phys)
            
            # 返回扁平化的 [10] 向量
            return global_shape.view(-1)

        # --- 3. 一次性计算 Jacobian ---
        # 输入形状: (1, 5, 2) -> 总参数量 10
        # 输出形状: (10)
        # 结果 J_raw 形状: (10, 1, 5, 2) —— (Output_dim, Input_dims...)
        J_raw = torch.autograd.functional.jacobian(forward_func, t_cmd_norm_init)
        
        # --- 4. 后处理与 Scaling ---
        # Reshape to (10, 10)
        # 行: 10个输出坐标 (x0, y0, ... x4, y4)
        # 列: 10个输入指令 (cmd0_L, cmd0_R, ... cmd4_L, cmd4_R)
        J_matrix = J_raw.view(10, 10)
        
        # 应用 Scaling (Chain Rule: dOut/dPhy = dOut/dNorm * 1/std)
        cmd_std = self.scalers['dq_cmd_std']
        scale_vec = 1.0 / (cmd_std + 1e-6) # [2]
        
        # 构建完整的 Scaling 矩阵 (10,) -> (1, 10)
        # 输入指令排列是 (N0_L, N0_R, N1_L, N1_R...)，对应的 std 也是重复 5 次
        full_scale = scale_vec.repeat(5).to(self.device).view(1, 10)
        
        # 广播乘法: 每一列 j 都乘以对应的 scale[j]
        J_final = J_matrix * full_scale
        
        # 获取 Beta 值 (只需要做一次正向传播，或者直接取最近一次的值)
        # 为了性能，这里可以复用计算，或者简单地再跑一次无梯度的 forward
        with torch.no_grad():
            x_in_static = torch.cat([n_q, n_dq, n_pose, t_cmd_norm_init, n_jac], dim=2)
            _, betas = self.model(x_in_static, n_jac, t_cmd_norm_init)
            beta_val = betas.mean().item()

        return J_final.cpu().numpy(), beta_val
    
    def get_physics_jacobian(self, q_curr, dq_hist, pose_loc_phys):
        """🔥 纯物理雅可比 (Analytical CCM + FK Chain Rule) 🔥"""
        # 数据准备 (同上)
        q_nodes, pose_nodes, jac_nodes = [], [], []
        for i in range(5):
            idx = self.MOT_INDICES_MAP[i]
            q_pair = q_curr[idx]
            j_flat = compute_local_jacobian(q_pair, C_LIST[i], N_VAL, M_VAL)
            q_nodes.append(q_pair)
            pose_nodes.append(pose_loc_phys[i])
            jac_nodes.append(j_flat) # [6] -> [dx_L, dx_R, dy_L, dy_R, dth_L, dth_R]

        # 转换为 Tensor
        t_pose_phys = torch.tensor(np.array([pose_nodes]), dtype=torch.float32, device=self.device) # [1, 5, 3]
        t_jac = torch.tensor(np.array([jac_nodes]), dtype=torch.float32, device=self.device) # [1, 5, 6]

        # 指令输入 (Normalized -> Denormalized)
        t_cmd_norm_init = torch.zeros((1, 5, 2), dtype=torch.float32, device=self.device)
        
        # 核心闭包: 输入 norm_cmd -> 还原为 phys_cmd -> 乘以 J_local -> 传入 FK -> 输出 Global
        def forward_func_phys(cmd_norm_arg):
            # 1. Denorm cmd (dq)
            cmd_std = self.scalers['dq_cmd_std']
            cmd_mean = self.scalers['dq_cmd_mean']
            dq_phys = cmd_norm_arg * cmd_std + cmd_mean # [1, 5, 2]
            
            # 2. Compute local delta (Linear: delta = J_local * dq)
            # t_jac: [1, 5, 6] -> Reshape to [1, 5, 3, 2]
            J_local = t_jac.view(1, 5, 3, 2)
            dq_col = dq_phys.unsqueeze(-1) # [1, 5, 2, 1]
            
            # Matrix Mult: (3x2) * (2x1) -> (3x1)
            delta_local_phys = torch.matmul(J_local, dq_col).squeeze(-1) # [1, 5, 3]
            
            # 3. Apply FK (Chain Rule propagation)
            # current_pose + delta -> global
            global_shape = self.diff_fk(t_pose_phys, delta_local_phys)
            
            return global_shape.view(-1)
        
        # 计算雅可比
        J_raw = torch.autograd.functional.jacobian(forward_func_phys, t_cmd_norm_init)
        J_matrix = J_raw.view(10, 10)
        
        # 这里的 J_matrix 已经是 dGlobal / dNormCmd
        # 我们需要 dGlobal / dPhysCmd = J_matrix * (1/std)
        # 实际上 autograd 已经追踪了 Denorm 过程，所以 J_raw 已经是 dG/dNorm。
        # 等等，如果我在 forward_func_phys 里面做了 denorm，那么 dG/dNorm 就已经包含了 *std。
        # 不对，chain rule: y = f(x*std + mu) -> dy/dx = f'(...) * std.
        # 所以 J_raw = J_phys * std.
        # 因此 J_phys = J_raw / std.
        
        cmd_std = self.scalers['dq_cmd_std']
        scale_vec = 1.0 / (cmd_std + 1e-6) 
        full_scale = scale_vec.repeat(5).to(self.device).view(1, 10)
        
        J_final = J_matrix * full_scale
        
        return J_final.cpu().numpy(), 0.0 # Beta is 0 for physics only