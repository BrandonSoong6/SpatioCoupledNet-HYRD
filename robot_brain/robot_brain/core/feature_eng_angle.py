# File: robot_brain/core/feature_eng.py
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from numba import jit, prange

# ================= ⚙️ 核心配置 (复刻 preprocess_with_Jphysics.py) =================
# 常量定义 (转为 Numpy 数组以便 Numba 消化)
C_LIST = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL = 22.0
H0_VAL = 52.0
M_VAL = H0_VAL - 2 * N_VAL

# ================= ⚡ Numba 加速核 (完全复刻，零修改) =================

@jit(nopython=True, cache=True)
def get_section_transform_fast(q_l, q_r, c_val, n_val, m_val):
    """ 单节变换矩阵计算 """
    delta_q = q_l - q_r
    sum_q = q_l + q_r
    theta = delta_q / c_val
    L_c = m_val + sum_q / 2.0
    
    if abs(theta) < 1e-6:
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
def get_full_state_fast(q_vec, c_list, n_val, m_val):
    """ 计算单个状态向量 [x, y, cos, sin] * 5 """
    state = np.zeros(20, dtype=np.float64)
    T_curr = np.eye(3, dtype=np.float64)
    
    num_sections = len(c_list)
    
    for i in range(num_sections):
        idx_base = (num_sections - 1 - i) * 2
        q_l = q_vec[idx_base]
        q_r = q_vec[idx_base + 1]
        
        T_sec = get_section_transform_fast(q_l, q_r, c_list[i], n_val, m_val)
        T_curr = T_curr @ T_sec 
        
        base_idx = i * 4
        state[base_idx + 0] = T_curr[0, 2] 
        state[base_idx + 1] = T_curr[1, 2] 
        state[base_idx + 2] = T_curr[0, 0] 
        state[base_idx + 3] = T_curr[1, 0] 
        
    return state

@jit(nopython=True, parallel=True, cache=True) 
def compute_jacobian_batch_fast(q_array, c_list, n_val, m_val):
    """ 
    并行计算雅可比 (包含行交换逻辑)
    """
    L = q_array.shape[0]
    n_dof = 10
    eps = 1e-4
    J_batch = np.zeros((L, 20, 10), dtype=np.float64)
    
    for t in prange(L):
        q_curr = q_array[t]
        
        for k in range(n_dof):
            # 正扰动
            q_p = q_curr.copy()
            q_p[k] += eps
            f_p = get_full_state_fast(q_p, c_list, n_val, m_val)
            
            # 负扰动
            q_m = q_curr.copy()
            q_m[k] -= eps
            f_m = get_full_state_fast(q_m, c_list, n_val, m_val)
            
            # 差分
            diff = (f_p - f_m) / (2 * eps)
            
            # --- 行交换逻辑 ---
            for m in range(5):
                base_src = m * 4 # x, y, cos, sin
                base_dst = m * 4 # x, y, sin, cos
                
                J_batch[t, base_dst + 0, k] = diff[base_src + 0] 
                J_batch[t, base_dst + 1, k] = diff[base_src + 1] 
                J_batch[t, base_dst + 2, k] = diff[base_src + 3] # sin
                J_batch[t, base_dst + 3, k] = diff[base_src + 2] # cos
                
    return J_batch

# ================= 3. 主逻辑类 =================

class FeatureEngineer:
    def __init__(self):
        # 预热 Numba (第一次调用会编译)
        print(f"🔄 [FeatureEng] Pre-heating Numba JIT...")
        dummy_q = np.ones((2, 10), dtype=np.float64) * 10.0
        _ = compute_jacobian_batch_fast(dummy_q, C_LIST, N_VAL, M_VAL)
        print(f"✅ [FeatureEng] JIT Ready.")

    def process_segment_optimized(self, segment):
        q = segment[:, :10]       
        x_raw = segment[:, 10:]   
        
        if len(segment) < 2: return None, None, None, None

        # 1. 计算 J_phys
        s_t_q = q[:-1].astype(np.float64) 
        j_phys_np = compute_jacobian_batch_fast(s_t_q, C_LIST, N_VAL, M_VAL)

        # 2. 数据整理 (Feature Expansion)
        x_expanded_list = []
        for i in range(5):
            col_start = i * 3
            curr_x = x_raw[:, col_start : col_start+1]
            curr_y = x_raw[:, col_start+1 : col_start+2]
            curr_th_deg = x_raw[:, col_start+2 : col_start+3]
            curr_th_rad = np.deg2rad(curr_th_deg)
            feat = np.concatenate([curr_x, curr_y, np.sin(curr_th_rad), np.cos(curr_th_rad)], axis=1)
            x_expanded_list.append(feat)
        
        x_expanded = np.concatenate(x_expanded_list, axis=1)
        state = np.concatenate([q, x_expanded], axis=1)

        s_t = state[:-1]
        s_next = state[1:]
        delta_q = s_next[:, :10] - s_t[:, :10] 
        delta_x = s_next[:, 10:] - s_t[:, 10:] 

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
        for start_idx, end_idx in tqdm(indices, desc="Computing Physics"):
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
        states_np = np.concatenate(data_states, axis=0)
        actions_np = np.concatenate(data_actions, axis=0)
        targets_np = np.concatenate(data_targets, axis=0)
        j_phys_np = np.concatenate(data_j_phys, axis=0)
        
        print(f"📊 Total Samples: {states_np.shape[0]} | J_phys Shape: {j_phys_np.shape}")

        # 归一化计算 (StandardScaler)
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
            "inputs_action_phys": torch.FloatTensor(actions_np), # Unnormalized Action for Physics
            "j_phys": torch.FloatTensor(j_phys_np),              # Jacobian
            "scalers": {
                "target_mean": torch.FloatTensor(target_mean),
                "target_std": torch.FloatTensor(target_std),
                "action_mean": torch.FloatTensor(action_mean),
                "action_std": torch.FloatTensor(action_std),
                "state_mean": torch.FloatTensor(state_mean),     # 保存State scaler以备后用
                "state_std": torch.FloatTensor(state_std)
            }
        }, output_pt_path)
        print(f"💾 Saved Tensor to {output_pt_path}")

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