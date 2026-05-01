import numpy as np
import torch
import os
import glob
import argparse
from tqdm import tqdm
from numba import jit, prange

# ================= ⚙️ 机器人几何参数 =================
# C_LIST: 从 Base 到 Tip 的常曲率参数
# 注意：Vision ID1 (Base) 对应 C_LIST[0], Vision ID5 (Tip) 对应 C_LIST[4]
C_LIST = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL = 22.0
H0_VAL = 52.0
M_VAL = H0_VAL - 2 * N_VAL

# ================= ⚡ Numba 核心算法 (Local Frame) =================

@jit(nopython=True, cache=True)
def get_local_pose_from_q(q_l, q_r, c_val, n_val, m_val):
    """
    正运动学 (FK): 计算单节在局部坐标系下的末端位姿 (x, y, theta)
    Input: q_l, q_r (rad)
    Output: [x, y, theta]
    """
    delta_q = q_l - q_r
    sum_q = q_l + q_r
    theta = delta_q / c_val
    L_c = m_val + sum_q / 2.0
    
    if abs(theta) < 1e-6:
        # 近似直线
        x = 0.0
        y = 2*n_val + L_c
        return np.array([x, y, 0.0])
    else:
        rho = L_c / theta
        # 几何推导 (Local Frame: Base is 0,0,0, pointing Y+)
        # 这里的公式需要适配你的物理模型，假设标准CC模型
        # Arc geometry:
        # x = rho * (1 - cos(theta))
        # y = rho * sin(theta) + 2*n_val (加上直段偏移) -> 简化模型通常忽略直段或包含在L_c
        # 依据你之前的feature_eng，保留之前的变换逻辑，但只取最后一列
        
        c = np.cos(-theta)
        s = np.sin(-theta)
        lx = rho * (1.0 - np.cos(theta))
        ly = rho * np.sin(theta)
        
        # T02 矩阵中的平移项
        x_local = -s * n_val + lx
        y_local = c * n_val + ly + n_val
        
        return np.array([x_local, y_local, -theta]) # 注意theta方向

@jit(nopython=True, cache=True)
def compute_local_jacobian(q_pair, c_val, n_val, m_val):
    """
    数值微分法计算局部雅可比 J_local (3x2)
    Input: q_pair [q_l, q_r]
    Output: Flat Jacobian (6,) representing [[dx/dq_l, dx/dq_r], [dy/..., ...], [dth/..., ...]]
    """
    eps = 1e-4
    J = np.zeros((3, 2), dtype=np.float64)
    
    curr_pose = get_local_pose_from_q(q_pair[0], q_pair[1], c_val, n_val, m_val)
    
    # 对 q_l 扰动
    p_l = get_local_pose_from_q(q_pair[0] + eps, q_pair[1], c_val, n_val, m_val)
    m_l = get_local_pose_from_q(q_pair[0] - eps, q_pair[1], c_val, n_val, m_val)
    J[:, 0] = (p_l - m_l) / (2 * eps)
    
    # 对 q_r 扰动
    p_r = get_local_pose_from_q(q_pair[0], q_pair[1] + eps, c_val, n_val, m_val)
    m_r = get_local_pose_from_q(q_pair[0], q_pair[1] - eps, c_val, n_val, m_val)
    J[:, 1] = (p_r - m_r) / (2 * eps)
    
    return J.flatten()

@jit(nopython=True, cache=True)
def global_to_local(parent_pose, current_pose):
    """
    将 current_pose 转换到 parent_pose 的局部坐标系下
    Poses are [x, y, theta]
    """
    x_p, y_p, th_p = parent_pose
    x_c, y_c, th_c = current_pose
    
    # 相对角度
    th_local = th_c - th_p
    
    # 旋转平移变换 (逆变换)
    # R^T * (P_c - P_p)
    dx = x_c - x_p
    dy = y_c - y_p
    
    cos_t = np.cos(th_p)
    sin_t = np.sin(th_p)
    
    x_local = dx * cos_t + dy * sin_t
    y_local = -dx * sin_t + dy * cos_t
    
    return np.array([x_local, y_local, th_local])

# ================= 🧠 数据生成器类 =================

class DataGeneratorV2:
    def __init__(self):
        # 索引配置
        # Raw Data Indices (Based on data_cleaner.py)
        self.IDX_MOT_START = 0
        self.IDX_VIS_START = 10
        
        # 视觉ID (Base->Tip) 到 电机数组索引 (0-9) 的映射
        # Vis 1 (Base, Section 0) -> Motor ID 9,10 -> indices 8,9
        # Vis 2 (Section 1)       -> Motor ID 7,8  -> indices 6,7
        # ...
        # Vis 5 (Tip, Section 4)  -> Motor ID 1,2  -> indices 0,1
        self.MOT_INDICES_MAP = [
            [8, 9], # Section 0
            [6, 7], # Section 1
            [4, 5], # Section 2
            [2, 3], # Section 3
            [0, 1]  # Section 4
        ]

    def process_segment(self, segment_data):
        """
        处理单个连续片段
        Input: (T, Raw_Dim)
        Output: Dict of processed tensors for this segment
        """
        T = segment_data.shape[0]
        if T < 5: return None # 忽略太短的序列

        # 1. 提取原始数据
        raw_mot = segment_data[:, self.IDX_MOT_START : self.IDX_MOT_START+10] # (T, 10)
        raw_vis = segment_data[:, self.IDX_VIS_START : self.IDX_VIS_START+15] # (T, 15)

        # 容器
        seq_q = []         # (T, 5, 2)
        seq_vis_global = []# (T, 5, 3)
        seq_vis_local = [] # (T, 5, 3)
        seq_jacobian = []  # (T, 5, 6)

        # 2. 空间遍历 (Per Frame)
        for t in range(T):
            frame_q = []
            frame_vis_global = []
            frame_vis_local = []
            frame_jac = []
            
            # 记录上一节的Pose用于坐标变换 (初始为原点)
            prev_pose_global = np.array([0.0, 0.0, 0.0]) 

            # 遍历 5 个 Section
            for i in range(5):
                # --- A. Motor Data ---
                m_idx = self.MOT_INDICES_MAP[i]
                q_pair = raw_mot[t, m_idx] # (2,)
                frame_q.append(q_pair) # 保存原始 q (2维，不是1维，因为雅可比需要2个)
                
                # --- B. Physics Jacobian (Local) ---
                # 计算 J (6,)
                j_loc = compute_local_jacobian(q_pair, C_LIST[i], N_VAL, M_VAL)
                frame_jac.append(j_loc)

                # --- C. Vision Data (Global) ---
                v_idx_start = i * 3
                curr_pose_global = raw_vis[t, v_idx_start : v_idx_start+3] # x,y,theta
                frame_vis_global.append(curr_pose_global)

                # --- D. Vision Data (Local) ---
                # 核心：计算当前节相对于上一节的局部 Pose
                # Section 0 的 Local 就是 Global (相对于世界原点)
                # Section i (i>0) 的 Local 是相对于 Section i-1 的
                if i == 0:
                    # 对于Base节，假设它相对于世界坐标系
                    pose_local = curr_pose_global 
                    # 或者如果是相对于上一时刻... 不，拓扑网络定义Local是空间上的相对
                    # 但Vision ID1是Base，通常不动或微动。
                    # 如果Vision ID1是第一节末端，那么相对于Base(0,0,0)就是它的Global
                else:
                    pose_local = global_to_local(prev_pose_global, curr_pose_global)
                
                frame_vis_local.append(pose_local)
                
                # 更新 prev 为当前，供下一节使用
                prev_pose_global = curr_pose_global

            seq_q.append(np.stack(frame_q))          # (5, 2)
            seq_vis_global.append(np.stack(frame_vis_global)) # (5, 3)
            seq_vis_local.append(np.stack(frame_vis_local))   # (5, 3)
            seq_jacobian.append(np.stack(frame_jac))          # (5, 6)

        # 转换为 Numpy 数组 (T, 5, D)
        np_q = np.array(seq_q)          # (T, 5, 2)
        np_vis_loc = np.array(seq_vis_local)    # (T, 5, 3)
        np_vis_glob = np.array(seq_vis_global)  # (T, 5, 3)
        np_jac = np.array(seq_jacobian) # (T, 5, 6)

        # 3. 时间序列构建 (t-1, t, t+1)
        # 有效数据长度 T-2 (去掉头尾用于差分)
        # Input t 范围: [1 : T-1]
        
        # --- Inputs Construction ---
        # 1. q_t (Current State): np_q[1:-1]
        # 2. dq_history (Hysteresis): np_q[1:-1] - np_q[0:-2]
        # 3. Pose_local (Vis Base): np_vis_loc[1:-1]
        # 4. dq_cmd (Intent): np_q[2:] - np_q[1:-1] (用下一帧作为Command意图)
        # 5. Jacobian: np_jac[1:-1]
        
        # --- Targets Construction ---
        # 1. delta_local_gt: np_vis_loc[2:] - np_vis_loc[1:-1] (预测下一帧的增量)
        # 2. global_pos_gt: np_vis_glob[2:, :, :2] (预测下一帧的全局XY，用于Shape Loss)

        # 截取
        valid_q_curr = np_q[1:-1] # (N, 5, 2) -> 我们只需要 1 个 q? 你的定义是 "基础状态qt:维度1"。
        # 通常 ql, qr 近似对称，或者网络需要知道两个？
        # 系统说明书说: "基础状态qt: 维度1"。 假设取平均或只取左？建议保留2个或取平均。
        # 这里为了尽可能保留信息，我将保留2个，或者如果必须为1，则取平均 (q_l + q_r)/2
        # *修正*：为了兼容性，我输出 2，你在网络里可以用 Linear(2, 1) 融合。
        
        valid_dq_hist = np_q[1:-1] - np_q[0:-2]
        valid_pose_loc = np_vis_loc[1:-1]
        valid_dq_cmd = np_q[2:] - np_q[1:-1]
        valid_jac = np_jac[1:-1]
        
        valid_target_delta = np_vis_loc[2:] - np_vis_loc[1:-1]
        valid_target_global = np_vis_glob[2:, :, :2] # 只取 XY
        
        return {
            "q_curr": valid_q_curr,      # (N, 5, 2)
            "dq_hist": valid_dq_hist,    # (N, 5, 2)
            "pose_loc": valid_pose_loc,  # (N, 5, 3)
            "dq_cmd": valid_dq_cmd,      # (N, 5, 2)
            "jacobian": valid_jac,       # (N, 5, 6)
            "tgt_delta": valid_target_delta, # (N, 5, 3)
            "tgt_global": valid_target_global # (N, 5, 2)
        }

    def process_all_batches(self, root_dir, output_path):
        batch_dirs = sorted(glob.glob(os.path.join(root_dir, "batch_*")))
        print(f"Found {len(batch_dirs)} batches in {root_dir}")

        all_data = {
            "q_curr": [], "dq_hist": [], "pose_loc": [], 
            "dq_cmd": [], "jacobian": [], 
            "tgt_delta": [], "tgt_global": []
        }

        total_frames = 0

        for b_dir in batch_dirs:
            clean_path = os.path.join(b_dir, "clean_data.npy")
            seg_path = os.path.join(b_dir, "segments.npy")
            
            if not (os.path.exists(clean_path) and os.path.exists(seg_path)):
                continue
                
            raw_data = np.load(clean_path)
            segments = np.load(seg_path)
            
            for (start, end) in segments:
                # 提取片段
                chunk = raw_data[start:end]
                processed = self.process_segment(chunk)
                
                if processed:
                    for k, v in processed.items():
                        all_data[k].append(v)
                    total_frames += processed["q_curr"].shape[0]

        print(f"✅ Processing complete. Total frames: {total_frames}")
        
        # 合并数据
        print("🔄 Concatenating tensors...")
        final_tensors = {}
        for k in all_data.keys():
            if len(all_data[k]) > 0:
                final_tensors[k] = np.concatenate(all_data[k], axis=0).astype(np.float32)
            else:
                print(f"❌ Warning: No data for {k}")
                return

        # 计算归一化统计量 (Scalers)
        print("🧮 Computing Statistics...")
        scalers = {}
        keys_to_norm = ["q_curr", "dq_hist", "pose_loc", "dq_cmd", "jacobian", "tgt_delta"]
        
        for k in keys_to_norm:
            data = final_tensors[k]
            # 计算全局 Mean/Std (跨 Batch 和 Section 均值，或者每节单独？通常全局通用比较好)
            # Shape (N, 5, D) -> Flatten to (N*5, D) to compute mean per feature channel
            flat_data = data.reshape(-1, data.shape[-1])
            mean = np.mean(flat_data, axis=0)
            std = np.std(flat_data, axis=0) + 1e-6
            
            scalers[f"{k}_mean"] = torch.tensor(mean)
            scalers[f"{k}_std"] = torch.tensor(std)
            
            # 这里我们**不**预先归一化数据，保留原始物理数值
            # 归一化在训练时的 DataLoader 或 Network Input Layer 进行
            # 这样方便 Physics Layer 使用真实的物理量计算
        
        # 保存
        save_dict = {
            "data": {k: torch.tensor(v) for k, v in final_tensors.items()},
            "scalers": scalers,
            "meta": {
                "description": "Topo-Impulse Training Data V2",
                "coordinate_system": "Local Frames (Sequential)",
                "mapping": "Motor Reversed (Base=Idx8,9), Vision Sequential",
                "shapes": {k: v.shape for k, v in final_tensors.items()}
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(save_dict, output_path)
        print(f"💾 Saved full dataset to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=False, default='~/brandon/hyrd_robot/lifelong_data', help="Path to lifelong_data folder")
    parser.add_argument('--out', type=str, required=False, default='~/brandon/hyrd_robot/lifelong_data/mega_expert.pt', help="Output .pt file path")
    args = parser.parse_args()
    root_abs = os.path.abspath(os.path.expanduser(args.root))
    out_abs = os.path.abspath(os.path.expanduser(args.out))
    
    print(f"🔍 Searching in: {root_abs}") # 打印出来确认一下
    
    gen = DataGeneratorV2()
    gen.process_all_batches(root_abs, out_abs)

if __name__ == "__main__":
    main()