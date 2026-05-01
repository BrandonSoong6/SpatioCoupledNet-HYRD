import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
from numba import jit
import matplotlib.pyplot as plt

# ================= ⚙️ 核心参数 (与 Cleaner 一致) =================
C_LIST = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL = 22.0
H0_VAL = 52.0
M_VAL = H0_VAL - 2 * N_VAL

CLEAN_THRESHOLD_MM = 60.0    # Cleaner 的阈值
MAX_GAP_TOLERANCE = 0.05     # 时间连接阈值

# ================= ⚡ Numba 坐标变换 =================
@jit(nopython=True, cache=True)
def global_to_local(parent_pose, current_pose):
    x_p, y_p, th_p = parent_pose
    x_c, y_c, th_c = current_pose
    if np.isnan(x_c) or np.isnan(x_p): return np.array([np.nan, np.nan, np.nan])
    
    th_local = th_c - th_p
    dx = x_c - x_p
    dy = y_c - y_p
    cos_t = np.cos(th_p)
    sin_t = np.sin(th_p)
    x_local = dx * cos_t + dy * sin_t
    y_local = -dx * sin_t + dy * cos_t
    return np.array([x_local, y_local, th_local])

def calc_jumps(vis_data):
    """通用函数：计算视觉数据的帧间位移 (mm)"""
    T = vis_data.shape[0]
    if T < 2: return []
    
    seq_vis_local = []
    for t in range(T):
        frame_vis_local = []
        prev_pose = np.array([0.0, 0.0, 0.0])
        for i in range(5):
            idx = i * 3
            curr_pose = vis_data[t, idx:idx+3]
            if i == 0: pose_local = curr_pose
            else: pose_local = global_to_local(prev_pose, curr_pose)
            frame_vis_local.append(pose_local)
            prev_pose = curr_pose
        seq_vis_local.append(np.stack(frame_vis_local))
    
    np_vis = np.array(seq_vis_local) # (T, 5, 3)
    # 计算 Delta: (T-1, 5, 3)
    deltas = np_vis[1:] - np_vis[:-1]
    # 计算模长 (XY only)
    jumps = np.linalg.norm(deltas[:, :, :2], axis=2) # (T-1, 5)
    return jumps.flatten()

# ================= 🕵️‍♂️ 逻辑复刻 =================

class PipelineInspector:
    def __init__(self):
        self.stage1_jumps = [] # Raw
        self.stage2_jumps = [] # Cleaned (Pre-Spline)
        self.stage3_jumps = [] # Final (Post-Spline)

    def find_stable_islands(self, data, dist_threshold=60.0):
        """复刻 DataCleaner 的逻辑"""
        # 计算每一帧的跳变
        diffs = np.abs(np.diff(data, axis=0))
        # 只要任意维度超过阈值，或者已经是NaN
        max_jumps = np.max(diffs, axis=1)
        # 这里的逻辑是：如果跳变 < 60，则认为是连接的
        is_connected = (max_jumps < dist_threshold) & (~np.isnan(max_jumps))
        
        # 构造掩码
        # 注意：如果 i 和 i+1 连接，则保留。
        # 这里简化处理：找出被切断的地方，把切断点设为 NaN
        
        # 为了严格模拟，我们直接标记那些导致断开的点
        # 但 DataCleaner 的逻辑是提取 continuous islands
        
        cleaned_data = data.copy()
        
        # 简单粗暴模拟：如果 diff > 60，把后一帧设为 NaN (视为断开)
        # DataCleaner 实际上是提取片段，我们这里为了看分布，把断开处设为NaN即可避免计算该处Delta
        
        # 更精确的模拟：复刻 find_stable_islands 的输出
        n_frames = len(data)
        padded = np.concatenate(([False], is_connected, [False]))
        change_indices = np.where(np.diff(padded))[0]
        ranges = change_indices.reshape(-1, 2)
        
        mask = np.zeros(n_frames, dtype=bool)
        for start, end in ranges:
            # DataCleaner 中 min_len=6
            if (end + 1) - start >= 6:
                mask[start : end+1] = True
                
        cleaned_data[~mask] = np.nan
        return cleaned_data

    def process_batch(self, b_dir):
        raw_path = os.path.join(b_dir, "raw_data.npy")
        clean_path = os.path.join(b_dir, "clean_data.npy")
        seg_path = os.path.join(b_dir, "segments.npy")
        
        if not (os.path.exists(raw_path) and os.path.exists(clean_path)):
            return

        # Load Raw
        raw_data = np.load(raw_path)
        raw_vis = raw_data[:, 44:59]
        raw_times = raw_data[:, 0]
        
        # --- Stage 1: Raw Distribution ---
        # 过滤掉时间间隔过大的 (模拟实验间隙)
        jumps = calc_jumps(raw_vis)
        dt = np.diff(raw_times)
        dt_expanded = np.repeat(dt[:, np.newaxis], 5, axis=1).flatten()
        
        # 只统计 NaN 以外且 dt < 0.05 的跳变
        valid_mask = (~np.isnan(jumps)) & (dt_expanded < MAX_GAP_TOLERANCE)
        self.stage1_jumps.extend(jumps[valid_mask].tolist())
        
        # --- Stage 2: Cleaned (Pre-Spline) ---
        # 运行 Cleaner 逻辑
        cleaned_vis = self.find_stable_islands(raw_vis, CLEAN_THRESHOLD_MM)
        
        # 计算跳变
        # 注意：这里计算的是“被认为是连贯”的帧之间的跳变
        # 如果 Cleaner 工作正常，这里所有的跳变都应该 < 60mm
        jumps_2 = calc_jumps(cleaned_vis)
        
        # 同样需要时间过滤，因为 Cleaner 是分段处理的
        # 但更重要的是，我们需要看“还有没有大跳变残留”
        valid_mask_2 = (~np.isnan(jumps_2)) & (dt_expanded < MAX_GAP_TOLERANCE)
        self.stage2_jumps.extend(jumps_2[valid_mask_2].tolist())
        
        # --- Stage 3: Final (Post-Spline) ---
        final_data = np.load(clean_path)
        segments = np.load(seg_path)
        final_vis = final_data[:, 10:25] # clean_data结构不同，vis在后15位
        
        for start, end in segments:
            chunk = final_vis[start:end]
            jumps_3 = calc_jumps(chunk)
            self.stage3_jumps.extend(jumps_3.tolist())

    def plot(self):
        plt.figure(figsize=(18, 6))
        
        titles = ["Stage 1: Raw Data", "Stage 2: Post-Clean (Pre-Spline)", "Stage 3: Final (Post-Spline)"]
        datasets = [self.stage1_jumps, self.stage2_jumps, self.stage3_jumps]
        colors = ['gray', 'orange', 'blue']
        
        for i, (data, title, col) in enumerate(zip(datasets, titles, colors)):
            data = np.array(data)
            plt.subplot(1, 3, i+1)
            
            if len(data) == 0:
                plt.title(f"{title}\n(No Data)")
                continue

            # 统计
            p99 = np.percentile(data, 99)
            max_val = np.max(data)
            
            # 画图 (Log Scale)
            plt.hist(data, bins=100, color=col, edgecolor='black', log=True)
            plt.title(f"{title}\nMax: {max_val:.1f}mm | P99: {p99:.1f}mm")
            plt.xlabel("Jump (mm)")
            if i == 0: plt.ylabel("Count (Log)")
            
            # 标注 60mm 阈值
            plt.axvline(60, color='red', linestyle='--', alpha=0.5, label='60mm Threshold')
            
            # 标注 1400mm (如果有)
            if max_val > 1000:
                 plt.axvline(max_val, color='purple', linestyle=':', label=f'Max {max_val:.0f}')
            
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("pipeline_inspection.png")
        print(f"📊 Plot saved to pipeline_inspection.png")
        
        # 打印数值报告
        print("\n" + "="*60)
        print(f"{'Stage':<25} | {'Max (mm)':<10} | {'P99.9 (mm)':<10} | {'>100mm Count'}")
        print("-" * 60)
        for name, d in zip(titles, datasets):
            d = np.array(d)
            count_bad = np.sum(d > 100)
            print(f"{name:<25} | {np.max(d):<10.1f} | {np.percentile(d, 99.9):<10.1f} | {count_bad}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/brandon/brandon/hyrd_robot/lifelong_data', help="Root path")
    args = parser.parse_args()
    
    inspector = PipelineInspector()
    
    search_pattern = os.path.join(args.root, "**", "batch_*")
    batches = sorted(glob.glob(search_pattern, recursive=True))
    print(f"🔍 Scanning {len(batches)} batches...")
    
    for b in tqdm(batches):
        inspector.process_batch(b)
        
    inspector.plot()

if __name__ == "__main__":
    main()