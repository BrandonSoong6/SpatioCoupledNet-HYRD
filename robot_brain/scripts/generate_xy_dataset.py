#!/usr/bin/env python3
# File: scripts/generate_xy_dataset.py
import numpy as np
import torch
import os
import glob
import argparse
from tqdm import tqdm
from numba import jit, prange

# ================= ⚙️ Core Configuration =================
# Robot physical constants (same as before)
C_LIST = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL = 22.0
H0_VAL = 52.0
M_VAL = H0_VAL - 2 * N_VAL

# ================= ⚡ Numba Kernels (Optimized for XY Only) =================

@jit(nopython=True, cache=True)
def get_section_transform_fast(q_l, q_r, c_val, n_val, m_val):
    """ Single section transformation matrix calculation (Unchanged) """
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
def get_xy_state_fast(q_vec, c_list, n_val, m_val):
    """ 
    [MODIFIED] Compute state vector with ONLY X and Y coordinates.
    Output Shape: (10,) -> [x1, y1, x2, y2, ..., x5, y5]
    """
    state = np.zeros(10, dtype=np.float64) # Reduced from 20 to 10
    T_curr = np.eye(3, dtype=np.float64)
    
    num_sections = len(c_list)
    
    for i in range(num_sections):
        idx_base = (num_sections - 1 - i) * 2
        q_l = q_vec[idx_base]
        q_r = q_vec[idx_base + 1]
        
        T_sec = get_section_transform_fast(q_l, q_r, c_list[i], n_val, m_val)
        T_curr = T_curr @ T_sec 
        
        base_idx = i * 2 # Stride is now 2 (x, y)
        state[base_idx + 0] = T_curr[0, 2] # x
        state[base_idx + 1] = T_curr[1, 2] # y
        # Sin/Cos removed
        
    return state

@jit(nopython=True, parallel=True, cache=True) 
def compute_jacobian_xy_fast(q_array, c_list, n_val, m_val):
    """ 
    [MODIFIED] Parallel Jacobian computation for XY only.
    Input: q_array (Batch, 10)
    Output: J_batch (Batch, 10, 10) -> (Batch, Output_Dim=10, Input_Dim=10)
    Row mapping: 0->x1, 1->y1, ..., 8->x5, 9->y5
    """
    L = q_array.shape[0]
    n_dof = 10
    eps = 1e-4
    J_batch = np.zeros((L, 10, 10), dtype=np.float64) # Reduced output dim to 10
    
    for t in prange(L):
        q_curr = q_array[t]
        
        for k in range(n_dof):
            # Positive perturbation
            q_p = q_curr.copy()
            q_p[k] += eps
            f_p = get_xy_state_fast(q_p, c_list, n_val, m_val)
            
            # Negative perturbation
            q_m = q_curr.copy()
            q_m[k] -= eps
            f_m = get_xy_state_fast(q_m, c_list, n_val, m_val)
            
            # Finite difference
            diff = (f_p - f_m) / (2 * eps)
            
            # Fill Jacobian (No complex row swapping needed as f_p structure matches target)
            for m in range(10):
                J_batch[t, m, k] = diff[m]
                
    return J_batch

# ================= 3. Main Logic Class =================

class XYFeatureEngineer:
    def __init__(self):
        # Pre-heat Numba
        print(f"🔄 [XY-FeatureEng] Pre-heating Numba JIT...")
        dummy_q = np.ones((2, 10), dtype=np.float64) * 10.0
        _ = compute_jacobian_xy_fast(dummy_q, C_LIST, N_VAL, M_VAL)
        print(f"✅ [XY-FeatureEng] JIT Ready.")

    def process_segment_xy(self, segment):
        """
        Process a single continuous data segment.
        """
        # 1. Extract Joint Angles (q)
        q = segment[:, :10] 
        
        # 2. Extract Raw Measurements (x_raw)
        x_raw_full = segment[:, 10:] 
        
        if len(segment) < 2: return None, None, None, None

        # 3. Compute Jacobian (Physics Prior)
        s_t_q = q[:-1].astype(np.float64) 
        j_phys_np = compute_jacobian_xy_fast(s_t_q, C_LIST, N_VAL, M_VAL)

        # 4. Filter Measurement Data (Keep only X, Y)
        xy_indices = []
        for i in range(5):
            base = i * 3
            xy_indices.append(base)     # x
            xy_indices.append(base + 1) # y
        
        x_xy_only = x_raw_full[:, xy_indices]

        # 5. Construct State Vector
        state = np.concatenate([q, x_xy_only], axis=1)

        # 6. Compute Transitions (s_t, s_next)
        s_t = state[:-1]
        s_next = state[1:]
        
        # --- 🛠️ 关键修改开始：处理角度跳变 ---
        
        # 原始差值
        raw_diff_q = s_next[:, :10] - s_t[:, :10]
        
        # 假设 q 是弧度制 (如果是角度制，请把 np.pi 换成 180.0)
        # 公式：diff = (diff + pi) % (2*pi) - pi
        # 这会将所有差值强制转换到 [-pi, +pi] 区间内，消除 2pi 的跳变
        action_delta_q = (raw_diff_q + np.pi) % (2 * np.pi) - np.pi
        
        # --- 关键修改结束 ---
        
        # Target is delta_x (Cartesian space doesn't wrap, linear subtraction is fine)
        target_delta_x = s_next[:, 10:] - s_t[:, 10:] 

        return s_t, action_delta_q, target_delta_x, j_phys_np

    def process_all_batches(self, data_root, output_pt_path):
        """
        Scans all batch directories and aggregates data into one file.
        """
        print(f"🚀 [XY-FeatureEng] Scanning data root: {data_root}")
        search_pattern = os.path.join(data_root, "**", "batch_*")
        batch_dirs = sorted(glob.glob(search_pattern, recursive=True))
        
        if not batch_dirs:
            print("❌ No batches found!")
            return

        global_states = []
        global_actions = []
        global_targets = []
        global_j_phys = []

        total_batches = 0

        for b_dir in tqdm(batch_dirs, desc="Aggregating Batches"):
            clean_path = os.path.join(b_dir, "clean_data.npy")
            seg_path = os.path.join(b_dir, "segments.npy")

            if not (os.path.exists(clean_path) and os.path.exists(seg_path)):
                continue # Skip incomplete batches

            raw_data = np.load(clean_path)
            indices = np.load(seg_path)
            
            # Process segments in this batch
            for start_idx, end_idx in indices:
                segment = raw_data[start_idx:end_idx]
                s, a, t, j = self.process_segment_xy(segment)
                if s is not None:
                    global_states.append(s)
                    global_actions.append(a)
                    global_targets.append(t)
                    global_j_phys.append(j)
            
            total_batches += 1

        if not global_states:
            print("❌ No valid data processed.")
            return

        # Concatenate Everything
        print("📦 Concatenating massive arrays...")
        states_np = np.concatenate(global_states, axis=0)
        actions_np = np.concatenate(global_actions, axis=0)
        targets_np = np.concatenate(global_targets, axis=0)
        j_phys_np = np.concatenate(global_j_phys, axis=0)
        
        print(f"✅ Data Aggregation Complete!")
        print(f"   - Total Samples: {states_np.shape[0]}")
        print(f"   - State Dim: {states_np.shape[1]} (10q + 10xy)")
        print(f"   - Target Dim: {targets_np.shape[1]} (10 delta_xy)")
        print(f"   - Jacobian Dim: {j_phys_np.shape}")

        # Compute Scalers (Global Normalization)
        print("🧮 Computing Global Scalers...")
        state_mean = np.mean(states_np, axis=0); state_std = np.std(states_np, axis=0) + 1e-6
        action_mean = np.mean(actions_np, axis=0); action_std = np.std(actions_np, axis=0) + 1e-6
        target_mean = np.mean(targets_np, axis=0); target_std = np.std(targets_np, axis=0) + 1e-6

        # Normalize
        states_norm = (states_np - state_mean) / state_std
        actions_norm = (actions_np - action_mean) / action_std
        targets_norm = (targets_np - target_mean) / target_std

        # Save Mega Dataset
        os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
        print(f"💾 Saving to {output_pt_path} ...")
        torch.save({
            "inputs_state": torch.FloatTensor(states_norm),
            "inputs_action": torch.FloatTensor(actions_norm),
            "targets": torch.FloatTensor(targets_norm),
            "inputs_action_phys": torch.FloatTensor(actions_np), # Raw actions for physics calc
            "j_phys": torch.FloatTensor(j_phys_np),
            "scalers": {
                "target_mean": torch.FloatTensor(target_mean),
                "target_std": torch.FloatTensor(target_std),
                "action_mean": torch.FloatTensor(action_mean),
                "action_std": torch.FloatTensor(action_std),
                "state_mean": torch.FloatTensor(state_mean),
                "state_std": torch.FloatTensor(state_std)
            },
            "meta": {
                "description": "XY-Only Mega Dataset",
                "xy_only": True,
                "original_batches": total_batches
            }
        }, output_pt_path)
        print("🎉 Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=os.path.expanduser("~/brandon/hyrd_robot/lifelong_data"), help="Root data directory")
    parser.add_argument('--out', type=str, default=os.path.expanduser("~/brandon/hyrd_robot/lifelong_data/mega_xy_dataset.pt"), help="Output path")
    args = parser.parse_args()
    
    fe = XYFeatureEngineer()
    fe.process_all_batches(args.root, args.out)

if __name__ == "__main__":
    main()