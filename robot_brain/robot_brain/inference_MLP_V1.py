#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import os
import time
from numba import jit

# ==============================================================================
# ⚙️ 全局配置 (Robot Constants)
# ==============================================================================
C_LIST_CONFIG = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL = 22.0
H0_VAL = 52.0
M_VAL = H0_VAL - 2 * N_VAL

# ==============================================================================
# ⚡ 1. Numba 物理核心 (极致优化版)
# ==============================================================================
@jit(nopython=True, cache=True)
def get_section_transform_numba(q_l, q_r, c_val, n, m):
    """ 计算单节变换矩阵 T (3x3) """
    if c_val < 1e-5: c_val = 100.0 
    
    delta_q = q_l - q_r; sum_q = q_l + q_r
    theta = delta_q / c_val
    L_c = m + sum_q / 2.0
    
    if abs(theta) < 1e-6:
        lx, ly, c, s = 0.0, L_c, 1.0, 0.0
    else:
        rho = L_c / theta
        c = np.cos(-theta); s = np.sin(-theta)
        lx = rho * (1.0 - np.cos(theta)); ly = rho * np.sin(theta)
    
    # T矩阵构造 (符合之前的修正逻辑)
    T = np.eye(3, dtype=np.float64)
    T[0,0]=c; T[0,1]=-s; T[0,2]=lx - s*n
    T[1,0]=s; T[1,1]=c;  T[1,2]=ly + c*n + n
    return T

@jit(nopython=True, cache=True)
def get_xy_state_fast(q_vec, c_list, n_val, m_val):
    """ FK: Base -> Tip (返回 10D 坐标) """
    state = np.zeros(10, dtype=np.float64)
    T_curr = np.eye(3, dtype=np.float64)
    num_sections = len(c_list)
    for i in range(num_sections):
        idx = (num_sections - 1 - i) * 2
        T_sec = get_section_transform_numba(q_vec[idx], q_vec[idx+1], c_list[i], n_val, m_val)
        T_curr = T_curr @ T_sec 
        state[i*2] = T_curr[0, 2]; state[i*2+1] = T_curr[1, 2]
    return state

@jit(nopython=True, cache=True)
def compute_jacobian_xy_single(q_curr, c_list, n_val, m_val):
    """ 
    🔥 [核心接口] 物理雅可比
    方法: 数值差分
    耗时: ~0.05 ms
    """
    n_dof = 10; eps = 1e-4; J = np.zeros((10, 10), dtype=np.float64)
    base = get_xy_state_fast(q_curr, c_list, n_val, m_val)
    for k in range(n_dof):
        q_p = q_curr.copy(); q_p[k] += eps
        diff = (get_xy_state_fast(q_p, c_list, n_val, m_val) - base) / eps
        for m in range(10): J[m, k] = diff[m]
    return J

# ==============================================================================
# 🧠 2. 神经网络架构 (保持不变)
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
        )
    def forward(self, x): return x + self.block(x)

class PhysicsGatedGNK(nn.Module):
    def __init__(self, state_dim=20, action_dim=10, output_dim=10, hidden_dim=512):
        super().__init__()
        self.register_buffer('target_mean', torch.zeros(output_dim)) 
        self.register_buffer('target_std', torch.ones(output_dim))
        self.residual_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.SiLU(),
            ResidualBlock(hidden_dim), ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim), ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, state_norm, action_norm, action_phys, j_phys):
        dx_phys_raw = torch.bmm(j_phys, action_phys.unsqueeze(-1)).squeeze(-1)
        dx_phys_norm = (dx_phys_raw - self.target_mean) / (self.target_std + 1e-7)
        net_in = torch.cat([state_norm, action_norm], dim=1)
        dx_residual_norm = self.residual_net(net_in)
        alpha = self.gate_net(state_norm)
        return dx_phys_norm + (1.0 - alpha) * dx_residual_norm

# ==============================================================================
# 🚀 3. JacobianCore (数学计算核)
# ==============================================================================
class JacobianCore:
    def __init__(self, model_path_override=None):
        """ 初始化并自动预热 """
        self.device = torch.device("cpu") # CPU 推理对小 Batch 延迟最低
        torch.set_num_threads(4) 
        
        # 1. 预热 Numba (关键!)
        self._warmup_numba()

        # 2. 加载模型
        if model_path_override: path = model_path_override
        else: path = os.path.expanduser("~/brandon/hyrd_robot/lifelong_data/models/gnk_offline_big.pth")
            
        if not os.path.exists(path):
            print(f"⚠️ Warning: Model not found at {path}. Running in PHYSICS-ONLY mode.")
            self.model = None
            return
            
        print(f"🧠 [Core] Loading model from: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            scalers = checkpoint['scalers']
            self.state_mean = scalers['state_mean'].to(self.device); self.state_std = scalers['state_std'].to(self.device)
            self.act_mean = scalers['action_mean'].to(self.device); self.act_std = scalers['action_std'].to(self.device)
            self.tgt_mean = scalers['target_mean'].to(self.device); self.tgt_std = scalers['target_std'].to(self.device)
            
            self.model = PhysicsGatedGNK(state_dim=20, action_dim=10, output_dim=10).to(self.device)
            state_dict = checkpoint['model_state_dict']; state_dict['target_mean'] = self.tgt_mean; state_dict['target_std'] = self.tgt_std
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # 3. 预热 NN (关键!)
            self._warmup_nn()
            print("✅ [Core] All Systems Ready & Optimized.")
            
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            self.model = None

    def _warmup_numba(self):
        print("🔥 [Core] Warming up Numba Physics Engine...")
        dummy_q = np.zeros(10, dtype=np.float64)
        # 运行一次以触发 JIT 编译
        get_xy_state_fast(dummy_q, C_LIST_CONFIG, N_VAL, M_VAL)
        compute_jacobian_xy_single(dummy_q, C_LIST_CONFIG, N_VAL, M_VAL)

    def _warmup_nn(self):
        print("🔥 [Core] Warming up Neural Network (Batch Mode)...")
        # 模拟一个 Batch=11 的推理 (Hybrid J 计算场景)
        dummy_s = torch.zeros(11, 20).to(self.device)
        dummy_a = torch.zeros(11, 10).to(self.device)
        dummy_j = torch.zeros(11, 10, 10).to(self.device)
        with torch.no_grad():
            self.model(dummy_s, dummy_a, dummy_a, dummy_j)

    # --------------------------------------------------------------------------
    # 接口 1: 获取物理雅可比 (最快)
    # --------------------------------------------------------------------------
    def get_physics_jacobian(self, q_curr):
        """
        返回: 10x10 Numpy Matrix (Physics J)
        耗时: < 0.1 ms
        """
        return compute_jacobian_xy_single(q_curr.astype(np.float64), C_LIST_CONFIG, N_VAL, M_VAL)

    # --------------------------------------------------------------------------
    # 接口 2: 获取融合雅可比 (含 NN 修正)
    # --------------------------------------------------------------------------
    def get_hybrid_jacobian(self, q_curr, xy_curr):
        """
        返回: 10x10 Numpy Matrix (Hybrid J)
        耗时: ~1.5 ms
        原理: 使用 Batch=11 的并行推理一次性算出所有维度的偏导
        """
        # 如果模型没加载，回退到物理 J
        if self.model is None:
            return self.get_physics_jacobian(q_curr)
            
        # 1. 计算基准物理 J
        q_dbl = q_curr.astype(np.float64)
        j_phys_np = compute_jacobian_xy_single(q_dbl, C_LIST_CONFIG, N_VAL, M_VAL)
        
        # 2. 构造 Batch 数据 (11组: 1基准 + 10扰动)
        # 这样只用过一次网络就能算出整个 J
        eps = 1e-4
        batch_size = 11
        
        # 复制状态: (11, 20)
        state_batch = np.tile(np.concatenate([q_curr, xy_curr]), (batch_size, 1)).astype(np.float32)
        # 复制物理J: (11, 10, 10)
        j_phys_batch = np.tile(j_phys_np, (batch_size, 1, 1)).astype(np.float32)
        
        # 构造动作扰动: 第0个全0，第1-10个分别在对应维度加 eps
        dq_batch = np.zeros((batch_size, 10), dtype=np.float32)
        idx = np.arange(10)
        dq_batch[idx + 1, idx] = eps
        
        # 3. 神经网络并行推理
        with torch.no_grad():
            s_t = torch.from_numpy(state_batch)
            a_t = torch.from_numpy(dq_batch)
            j_t = torch.from_numpy(j_phys_batch)
            
            pred_norm = self.model(
                (s_t - self.state_mean)/self.state_std,
                (a_t - self.act_mean)/self.act_std,
                a_t, j_t
            )
            # 反归一化
            dx_pred = (pred_norm * self.tgt_std + self.tgt_mean).numpy()

        # 4. 差分计算 J_hybrid
        # J_col_i = (f(q + eps*e_i) - f(q)) / eps
        # 利用 Numpy 广播直接计算
        J_hybrid = ((dx_pred[1:] - dx_pred[0]) / eps).T 
        
        return J_hybrid

# ==============================================================================
# 🧪 性能基准测试 (只测速度)
# ==============================================================================
if __name__ == "__main__":
    core = JacobianCore()
    
    q_test = np.ones(10) * 10.0
    xy_test = get_xy_state_fast(q_test, C_LIST_CONFIG, N_VAL, M_VAL)
    
    print("\n" + "="*40)
    print("🚀 Speed Benchmark")
    print("="*40)
    
    # 1. 测物理 J
    t0 = time.time()
    loops = 1000
    for _ in range(loops):
        J_p = core.get_physics_jacobian(q_test)
    avg_p = (time.time() - t0) / loops * 1000
    print(f"⚡ Physics J Time : {avg_p:.4f} ms")
    
    # 2. 测融合 J
    t0 = time.time()
    loops = 100
    for _ in range(loops):
        J_h = core.get_hybrid_jacobian(q_test, xy_test)
    avg_h = (time.time() - t0) / loops * 1000
    print(f"🧠 Hybrid J Time  : {avg_h:.4f} ms")
    
    print("-" * 40)
    if avg_h < 2.0:
        print("✅ Status: SUPER FAST (Ready for 500Hz Control)")
    else:
        print("⚠️ Status: Acceptable")