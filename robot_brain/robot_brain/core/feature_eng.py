# File: robot_brain/core/feature_eng.py
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from numba import jit, prange

# ================= ⚙️ 核心配置 =================
# 连续体机器人几何参数
C_LIST = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL = 22.0
H0_VAL = 52.0
M_VAL = H0_VAL - 2 * N_VAL

# ================= ⚡ Numba 加速核 (XY-Only Version) =================

@jit(nopython=True, cache=True)
def get_section_transform_fast(q_l, q_r, c_val, n_val, m_val):
    """ 单节变换矩阵计算 (保持不变) """
    delta_q = q_l - q_r
    sum_q = q_l + q_r
    theta = delta_q / c_val
    L_c = m_val + sum_q / 2.0
    
    if abs(theta) < 1e-6:
        # 近似直线
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 2*n_val + L_c],
            [0.0, 0.0, 1.0]
        ])
    else:
        rho = L_c / theta
        c = np.cos(-theta)
        s = np.sin(-theta)
        lx = rho * (1.0 - np.cos(theta))
        ly = rho * np.sin(theta)
        
        T02 = -s * n_val + lx
        T12 = c * n_val + ly + n_val
        
        return np.array([
            [c,  -s,  T02],
            [s,   c,  T12],
            [0.0, 0.0, 1.0]
        ])

@jit(nopython=True, cache=True)
def get_xy_state_fast(q_vec, c_list, n_val, m_val):
    """ 
    [修改] 仅计算 XY 坐标
    输出: vector dim 10 [x1, y1, x2, y2, ..., x5, y5]
    """
    state = np.zeros(10, dtype=np.float64)
    T_curr = np.eye(3, dtype=np.float64)
    
    num_sections = len(c_list)
    
    for i in range(num_sections):
        # 倒序索引 (Base -> Tip)
        idx_base = (num_sections - 1 - i) * 2
        q_l = q_vec[idx_base]
        q_r = q_vec[idx_base + 1]
        
        T_sec = get_section_transform_fast(q_l, q_r, c_list[i], n_val, m_val)
        T_curr = T_curr @ T_sec 
        
        # 仅保存 x, y
        base_idx = i * 2
        state[base_idx + 0] = T_curr[0, 2] # x
        state[base_idx + 1] = T_curr[1, 2] # y
        
    return state

@jit(nopython=True, parallel=True, cache=True) 
def compute_jacobian_xy_fast(q_array, c_list, n_val, m_val):
    """ 
    [修改] 并行计算 XY 雅可比
    输出: (Batch, 10, 10) -> 映射 dq (10) 到 d(xy) (10)
    """
    L = q_array.shape[0]
    n_dof = 10
    eps = 1e-4
    J_batch = np.zeros((L, 10, 10), dtype=np.float64)
    
    for t in prange(L):
        q_curr = q_array[t]
        
        for k in range(n_dof):
            # 正扰动
            q_p = q_curr.copy()
            q_p[k] += eps
            f_p = get_xy_state_fast(q_p, c_list, n_val, m_val)
            
            # 负扰动
            q_m = q_curr.copy()
            q_m[k] -= eps
            f_m = get_xy_state_fast(q_m, c_list, n_val, m_val)
            
            # 差分
            diff = (f_p - f_m) / (2 * eps)
            
            # 填充雅可比 (无需行交换，因为 get_xy_state_fast 已经按 x,y 排列)
            for row in range(10):
                J_batch[t, row, k] = diff[row]
                
    return J_batch

# ================= 3. 主逻辑类 =================

class FeatureEngineer:
    def __init__(self):
        # 预热 Numba
        print(f"🔄 [FeatureEng] Pre-heating Numba JIT (XY Mode)...")
        dummy_q = np.ones((2, 10), dtype=np.float64) * 10.0
        _ = compute_jacobian_xy_fast(dummy_q, C_LIST, N_VAL, M_VAL)
        print(f"✅ [FeatureEng] JIT Ready.")

    def process_segment_optimized(self, segment):
        """
        处理单个数据段，提取 XY 特征
        Input segment: [q(10) | x1,y1,th1 ... x5,y5,th5 (15)]
        """
        q = segment[:, :10]       
        x_raw = segment[:, 10:]   
        
        if len(segment) < 2: return None, None, None, None

        # 1. 计算物理雅可比 (XY Only)
        # 使用 t 时刻的 q 计算雅可比
        s_t_q = q[:-1].astype(np.float64) 
        j_phys_np = compute_jacobian_xy_fast(s_t_q, C_LIST, N_VAL, M_VAL)

        # 2. 提取视觉 XY 特征 (丢弃 Theta)
        # x_raw 结构: [x1, y1, th1, x2, y2, th2, ...]
        x_vis_list = []
        for i in range(5):
            col_start = i * 3
            # 取 x, y (索引 0, 1)
            curr_xy = x_raw[:, col_start : col_start+2] 
            x_vis_list.append(curr_xy)
        
        x_vis = np.concatenate(x_vis_list, axis=1) # Shape: (N, 10)

        # 3. 构建状态向量
        # State = [q (10) | xy (10)] -> Dim 20
        state = np.concatenate([q, x_vis], axis=1)

        # 4. 构建时间差分 (t -> t+1)
        s_t = state[:-1]
        s_next = state[1:]
        
        delta_q = s_next[:, :10] - s_t[:, :10]   # Action (dq)
        delta_x = s_next[:, 10:] - s_t[:, 10:]   # Target (d_xy)

        return s_t, delta_q, delta_x, j_phys_np

    def process_batch(self, clean_path, seg_path, output_pt_path):
        print(f"📂 [FeatureEng] Processing: {os.path.basename(clean_path)}")
        if not os.path.exists(clean_path):
            print("❌ Clean data not found.")
            return

        raw_data = np.load(clean_path)
        indices = np.load(seg_path)
        
        data_states, data_actions, data_targets, data_j_phys = [], [], [], []

        # 处理每个 Segment
        for start_idx, end_idx in tqdm(indices, desc="Computing XY Physics"):
            segment = raw_data[start_idx:end_idx]
            s, a, t, j = self.process_segment_optimized(segment)
            if s is not None:
                data_states.append(s)
                data_actions.append(a)
                data_targets.append(t)
                data_j_phys.append(j)

        if not data_states:
            print("❌ No valid segments processed.")
            return

        # 合并张量
        states_np = np.concatenate(data_states, axis=0)   # (N, 20)
        actions_np = np.concatenate(data_actions, axis=0) # (N, 10)
        targets_np = np.concatenate(data_targets, axis=0) # (N, 10)
        j_phys_np = np.concatenate(data_j_phys, axis=0)   # (N, 10, 10)
        
        print(f"📊 Total Samples: {states_np.shape[0]}")
        print(f"   State Dim: {states_np.shape[1]} (Expect 20)")
        print(f"   Target Dim: {targets_np.shape[1]} (Expect 10)")
        print(f"   Jacobian Shape: {j_phys_np.shape} (Expect N,10,10)")

        # 归一化计算 (StandardScaler)
        # 注意: 加 1e-6 防止除零
        state_mean = np.mean(states_np, axis=0); state_std = np.std(states_np, axis=0) + 1e-6
        action_mean = np.mean(actions_np, axis=0); action_std = np.std(actions_np, axis=0) + 1e-6
        target_mean = np.mean(targets_np, axis=0); target_std = np.std(targets_np, axis=0) + 1e-6

        states_norm = (states_np - state_mean) / state_std
        actions_norm = (actions_np - action_mean) / action_std
        targets_norm = (targets_np - target_mean) / target_std

        # 保存 .pt 文件
        os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
        torch.save({
            "inputs_state": torch.FloatTensor(states_norm),
            "inputs_action": torch.FloatTensor(actions_norm),
            "targets": torch.FloatTensor(targets_norm),
            "inputs_action_phys": torch.FloatTensor(actions_np), # Raw Action (rad or mm)
            "j_phys": torch.FloatTensor(j_phys_np),              # Jacobian (10x10)
            "scalers": {
                "target_mean": torch.FloatTensor(target_mean),
                "target_std": torch.FloatTensor(target_std),
                "action_mean": torch.FloatTensor(action_mean),
                "action_std": torch.FloatTensor(action_std),
                "state_mean": torch.FloatTensor(state_mean),
                "state_std": torch.FloatTensor(state_std)
            },
            "meta": {"type": "xy_only", "dims": "20-10-10"}
        }, output_pt_path)
        print(f"💾 Saved XY-Only Tensor to {output_pt_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', type=str, required=True, help="Clean .npy path")
    parser.add_argument('--seg', type=str, required=True, help="Segments .npy path")
    parser.add_argument('--out_pt', type=str, required=True, help="Output .pt path")
    args = parser.parse_args()
    
    fe = FeatureEngineer()
    fe.process_batch(args.clean, args.seg, args.out_pt)

if __name__ == "__main__":
    main()