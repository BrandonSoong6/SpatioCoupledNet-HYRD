import numpy as np
import torch
import os
import glob
import argparse
from tqdm import tqdm
from numba import jit
from scipy.signal import savgol_filter  # ✨ 新增依赖

# ================= ⚙️ 机器人几何参数 =================
C_LIST = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL = 22.0
H0_VAL = 52.0
M_VAL = H0_VAL - 2 * N_VAL

# ================= ⚡ Numba 核心算法 =================

@jit(nopython=True, cache=True)
def get_local_pose_from_q(q_l, q_r, c_val, n_val, m_val):
    delta_q = q_l - q_r
    sum_q = q_l + q_r
    theta = delta_q / c_val 
    L_c = m_val + sum_q / 2.0
    
    if abs(theta) < 1e-6:
        x = 0.0
        y = 2*n_val + L_c
        return np.array([x, y, 0.0])
    else:
        rho = L_c / theta
        c = np.cos(-theta)
        s = np.sin(-theta)
        lx = rho * (1.0 - np.cos(theta))
        ly = rho * np.sin(theta)
        
        x_local = -s * n_val + lx
        y_local = c * n_val + ly + n_val
        return np.array([x_local, y_local, -theta])

@jit(nopython=True, cache=True)
def compute_local_jacobian(q_pair, c_val, n_val, m_val):
    eps = 1e-4 
    J = np.zeros((3, 2), dtype=np.float64)
    curr_pose = get_local_pose_from_q(q_pair[0], q_pair[1], c_val, n_val, m_val)
    p_l = get_local_pose_from_q(q_pair[0] + eps, q_pair[1], c_val, n_val, m_val)
    m_l = get_local_pose_from_q(q_pair[0] - eps, q_pair[1], c_val, n_val, m_val)
    J[:, 0] = (p_l - m_l) / (2 * eps)
    p_r = get_local_pose_from_q(q_pair[0], q_pair[1] + eps, c_val, n_val, m_val)
    m_r = get_local_pose_from_q(q_pair[0], q_pair[1] - eps, c_val, n_val, m_val)
    J[:, 1] = (p_r - m_r) / (2 * eps)
    return J.flatten()

@jit(nopython=True, cache=True)
def global_to_local(parent_pose, current_pose):
    x_p, y_p, th_p = parent_pose
    x_c, y_c, th_c = current_pose
    th_local = th_c - th_p
    dx = x_c - x_p
    dy = y_c - y_p
    cos_t = np.cos(th_p)
    sin_t = np.sin(th_p)
    x_local = dx * cos_t + dy * sin_t
    y_local = -dx * sin_t + dy * cos_t
    return np.array([x_local, y_local, th_local])

@jit(nopython=True, cache=True)
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

@jit(nopython=True, cache=True)
def compute_pose_delta(next_pose, curr_pose):
    dx = next_pose[0] - curr_pose[0]
    dy = next_pose[1] - curr_pose[1]
    raw_dtheta = next_pose[2] - curr_pose[2]
    dtheta = wrap_angle(raw_dtheta)
    return np.array([dx, dy, dtheta])

# ================= 🧠 数据生成器类 (V4 - Smoothed & Strided) =================

class DataGeneratorV4:
    def __init__(self, window_len=31, poly_order=3, stride=5):
        """
        Args:
            window_len: SavGol 滤波窗口长度 (建议 31)
            poly_order: SavGol 多项式阶数 (建议 3)
            stride: 跨步预测步长 (建议 5, 即 50ms)
        """
        self.IDX_MOT_START = 0
        self.IDX_VIS_START = 10
        self.MOT_INDICES_MAP = [[8, 9], [6, 7], [4, 5], [2, 3], [0, 1]]
        
        # ✨ 新增参数
        self.window_len = window_len
        self.poly_order = poly_order
        self.stride = stride
        
        # 计算滤波边缘需要切除的长度 (Head/Tail)
        self.edge_cut = self.window_len // 2

    def process_segment(self, segment_data):
        T = segment_data.shape[0]
        # 最小长度检查：
        # 必须大于 2*edge_cut (滤波边缘) + stride (跨步) + 1 (基础样本)
        min_required = 2 * self.edge_cut + self.stride + 2
        if T < min_required: 
            return None 

        # 1. 提取原始数据
        raw_mot = segment_data[:, self.IDX_MOT_START : self.IDX_MOT_START+10] 
        raw_vis_mixed = segment_data[:, self.IDX_VIS_START : self.IDX_VIS_START+15]

        raw_vis = np.copy(raw_vis_mixed)
        angle_indices = [2, 5, 8, 11, 14]
        raw_vis[:, angle_indices] = np.radians(raw_vis[:, angle_indices])

        # 2. 先计算出整段的 q 和 pose_loc (Raw)
        full_seq_q = []
        full_seq_vis_local = []
        full_seq_vis_global = []
        full_seq_jac = []
        
        prev_pose_global = np.array([0.0, 0.0, 0.0]) 

        for t in range(T):
            frame_q = []
            frame_vis_local = []
            frame_vis_global = []
            frame_jac = []
            
            for i in range(5):
                m_idx = self.MOT_INDICES_MAP[i]
                q_pair = raw_mot[t, m_idx]
                frame_q.append(q_pair)
                
                # Jacobian 计算依赖原始 q (无滤波)，保持瞬时性
                j_loc = compute_local_jacobian(q_pair, C_LIST[i], N_VAL, M_VAL)
                frame_jac.append(j_loc)

                v_idx_start = i * 3
                curr_pose_global = raw_vis[t, v_idx_start : v_idx_start+3]
                frame_vis_global.append(curr_pose_global)

                if i == 0:
                    pose_local = curr_pose_global 
                else:
                    pose_local = global_to_local(prev_pose_global, curr_pose_global)
                
                pose_local[2] = wrap_angle(pose_local[2])
                frame_vis_local.append(pose_local)
                
                prev_pose_global = curr_pose_global

            full_seq_q.append(np.stack(frame_q))
            full_seq_vis_local.append(np.stack(frame_vis_local))
            full_seq_vis_global.append(np.stack(frame_vis_global))
            full_seq_jac.append(np.stack(frame_jac))

        np_q = np.array(full_seq_q)         # [T, 5, 2]
        np_vis_loc = np.array(full_seq_vis_local) # [T, 5, 3]
        np_vis_glob = np.array(full_seq_vis_global)
        np_jac = np.array(full_seq_jac)     # [T, 5, 6]

        # 3. ✨ 核心：应用 SavGol 平滑 (In-Segment Smoothing)
        # 对 pose_loc 进行平滑 (沿时间轴 axis=0)
        # 形状保持不变 [T, 5, 3]
        smoothed_pose_loc = np.zeros_like(np_vis_loc)
        for s in range(5):
            for d in range(3):
                smoothed_pose_loc[:, s, d] = savgol_filter(
                    np_vis_loc[:, s, d], 
                    window_length=self.window_len, 
                    polyorder=self.poly_order, 
                    axis=0
                )
        
        # 4. ✨ 核心：切除边缘 & 生成 Strided 样本
        # 有效数据区间: [edge_cut, T - edge_cut]
        # 但我们还需要往后找 +stride 的目标，所以 Loop 终点要提前 stride
        
        valid_start_idx = self.edge_cut
        valid_end_idx = T - self.edge_cut - self.stride # 保证 t+stride 不越界且不在尾部噪声区
        
        if valid_end_idx <= valid_start_idx:
            return None

        # 初始化列表
        out_q_curr = []
        out_dq_hist = []
        out_pose_loc = []
        out_dq_cmd = []
        out_jac = []
        out_tgt_delta = []
        out_tgt_global = []

        # 循环采样
        for t in range(valid_start_idx, valid_end_idx):
            # t 是当前时刻
            # t_target 是预测目标时刻 (跨步)
            t_target = t + self.stride
            
            # --- Input Features ---
            # 1. q_curr: 当前关节角 (未平滑，保持真实反馈)
            out_q_curr.append(np_q[t])
            
            # 2. dq_hist: 历史速度 (t - t-1)
            # 这里可以使用平滑后的 q 也可以用原始 q，通常 history 用原始的反应快
            out_dq_hist.append(np_q[t] - np_q[t-1])
            
            # 3. pose_loc: 当前平滑后的位姿
            out_pose_loc.append(smoothed_pose_loc[t])
            
            # 4. jacobian: 当前时刻雅可比
            out_jac.append(np_jac[t])
            
            # --- Control & Labels (Strided) ---
            
            # 5. dq_cmd: 跨步期间的总电机指令
            # 这是 input，告诉网络：未来 stride 步内，电机总共动了多少
            cmd_stride = np_q[t_target] - np_q[t]
            out_dq_cmd.append(cmd_stride)
            
            # 6. tgt_delta: 跨步期间的总位移 (Label)
            # 使用平滑后的 pose 计算，去噪效果最好
            # 计算每个 segment 的 delta
            frame_delta = []
            for s in range(5):
                p_curr = smoothed_pose_loc[t, s]
                p_next = smoothed_pose_loc[t_target, s]
                frame_delta.append(compute_pose_delta(p_next, p_curr))
            out_tgt_delta.append(np.stack(frame_delta))
            
            # 7. Global Target (Optionally Strided or Next)
            # 通常用于可视化，取 target 时刻的 global
            out_tgt_global.append(np_vis_glob[t_target, :, :2])

        # 堆叠结果
        return {
            "q_curr": np.array(out_q_curr),
            "dq_hist": np.array(out_dq_hist),
            "pose_loc": np.array(out_pose_loc),
            "dq_cmd": np.array(out_dq_cmd),   # Strided Command
            "jacobian": np.array(out_jac),
            "tgt_delta": np.array(out_tgt_delta), # Strided Delta
            "tgt_global": np.array(out_tgt_global)
        }

    def process_all_batches(self, root_dir, output_path):
        search_pattern = os.path.join(root_dir, "**", "batch_*")
        batch_dirs = sorted(glob.glob(search_pattern, recursive=True))
        print(f"🚀 Found {len(batch_dirs)} batches in {root_dir}")
        print(f"🔧 Config: Window={self.window_len}, Poly={self.poly_order}, Stride={self.stride}")

        all_data = {
            "q_curr": [], "dq_hist": [], "pose_loc": [], 
            "dq_cmd": [], "jacobian": [], 
            "tgt_delta": [], "tgt_global": []
        }

        total_frames = 0
        pbar = tqdm(batch_dirs, desc="Processing")

        for b_dir in pbar:
            clean_path = os.path.join(b_dir, "clean_data.npy")
            seg_path = os.path.join(b_dir, "segments.npy")
            
            if not (os.path.exists(clean_path) and os.path.exists(seg_path)): continue
                
            raw_data = np.load(clean_path)
            segments = np.load(seg_path)
            
            for (start, end) in segments:
                chunk = raw_data[start:end]
                processed = self.process_segment(chunk)
                
                if processed:
                    for k, v in processed.items():
                        all_data[k].append(v)
                    total_frames += processed["q_curr"].shape[0]

        print(f"✅ Total valid frames: {total_frames}")
        
        if total_frames == 0:
            print("❌ Error: No valid frames!")
            return

        print("🔄 Concatenating tensors...")
        final_tensors = {}
        for k in all_data.keys():
            final_tensors[k] = np.concatenate(all_data[k], axis=0).astype(np.float32)

        print("🧮 Computing Statistics...")
        scalers = {}
        keys_to_norm = ["q_curr", "dq_hist", "pose_loc", "dq_cmd", "jacobian", "tgt_delta"]
        
        for k in keys_to_norm:
            data = final_tensors[k]
            flat_data = data.reshape(-1, data.shape[-1])
            mean = np.mean(flat_data, axis=0)
            std = np.std(flat_data, axis=0) + 1e-6
            
            scalers[f"{k}_mean"] = torch.tensor(mean)
            scalers[f"{k}_std"] = torch.tensor(std)
        
        save_dict = {
            "data": {k: torch.tensor(v) for k, v in final_tensors.items()},
            "scalers": scalers,
            "config": { # 保存配置以便后续查阅
                "window": self.window_len,
                "poly": self.poly_order,
                "stride": self.stride
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(save_dict, output_path)
        print(f"💾 Saved SMOOTHED & STRIDED dataset to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/brandon/brandon/hyrd_robot/lifelong_data')
    parser.add_argument('--out', type=str, default='/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_smooth_strided.pt')
    
    # 开放参数配置
    parser.add_argument('--window', type=int, default=31, help='SavGol Window Length')
    parser.add_argument('--poly', type=int, default=3, help='SavGol Poly Order')
    parser.add_argument('--stride', type=int, default=5, help='Prediction Horizon Stride')
    
    args = parser.parse_args()
    
    gen = DataGeneratorV4(window_len=args.window, poly_order=args.poly, stride=args.stride)
    gen.process_all_batches(args.root, args.out)

if __name__ == "__main__":
    main()