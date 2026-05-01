# File: robot_brain/core/data_cleaner.py
import numpy as np
import os
import sys
import json
import argparse
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

class DataCleaner:
    def __init__(self):
        # ================= 配置区域 (复刻 2_clean_and_br.py) =================
        self.TARGET_FREQ = 100.0          
        self.DT_TARGET = 1.0 / self.TARGET_FREQ
        
        # [动态] Latency 现在从外部文件读取，不再硬编码
        self.dynamic_latency_s = None
        
        self.MAX_GAP_TOLERANCE = 0.05     
        self.CLEAN_THRESHOLD_MM = 60.0    
        self.MIN_ISLAND_LEN = 6           

        self.RAW_IDX_TIME = 0 
        self.RAW_IDX_Q_START = 13
        self.RAW_IDX_Q_END = 23
        self.RAW_IDX_VIS_START = 44
        self.RAW_IDX_VIS_END = 59

        self.SG_WINDOW_LEN = 21  
        self.SG_POLY_ORDER = 3   

    def load_latency(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Latency file not found: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.dynamic_latency_s = float(data['recommended_latency_s'])
            print(f"[Cleaner] Loaded Dynamic Latency: {self.dynamic_latency_s*1000:.2f} ms")

    def find_stable_islands(self, data, dist_threshold, min_len):
        """
        [Step 1] 粗筛：找出内部连续的孤岛，将剧烈跳变处的数据置为 NaN
        注意：这一步只是把数据“标记”为坏点，并没有在数组索引上把它们切开。
        """
        n_frames = len(data)
        if n_frames < min_len:
            return np.full((n_frames, data.shape[1]), np.nan), n_frames

        # 计算相邻帧差分
        diffs = np.abs(np.diff(data, axis=0))
        max_jumps = np.max(diffs, axis=1)
        
        # 判定连接性
        is_connected = max_jumps < dist_threshold
        
        padded = np.concatenate(([False], is_connected, [False]))
        change_indices = np.where(np.diff(padded))[0]
        ranges = change_indices.reshape(-1, 2)
        
        cleaned_data = np.full_like(data, np.nan)
        kept_count = 0
        
        for start, end in ranges:
            island_len = (end + 1) - start
            if island_len >= min_len:
                cleaned_data[start : end+1] = data[start : end+1]
                kept_count += island_len
                
        return cleaned_data, n_frames - kept_count

    def split_by_validity(self, times, motor, vision, max_gap):
        """
        [Step 3 核心修复] 
        根据 NaN 和 物理跳变 将数据切分成适合插值的小块 (Sub-blocks)。
        
        修复逻辑：
        除了检查时间断档 (dt > max_gap)，还必须检查空间断档 (dist > 60mm)。
        如果两个孤岛在时间上紧挨着，但在空间上差了1米，必须在这里切一刀，
        否则插值器会生成一条 1.4米的过冲曲线。
        """
        # 1. 找出所有非 NaN 的有效索引
        valid_mask = ~np.isnan(vision).any(axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 2: return []

        # 提取有效数据
        t_valid = times[valid_indices]
        v_valid = vision[valid_indices]

        # --- A. 检查时间断档 (Time Gap) ---
        dt = np.diff(t_valid)
        time_split_locs = np.where(dt > max_gap)[0] + 1
        
        # --- B. [新增] 检查空间断档 (Spatial Gap) ---
        # 必须把 (N, 15) 重塑为 (N, 5, 3) 才能计算物理距离
        v_reshaped = v_valid.reshape(-1, 5, 3)
        # 计算相邻有效帧之间的位移
        dv = v_reshaped[1:] - v_reshaped[:-1] # (N-1, 5, 3)
        d_dist = np.linalg.norm(dv, axis=2)    # (N-1, 5) -> 每个节点的位移
        max_spatial_jump = np.max(d_dist, axis=1) # (N-1,) -> 取跳得最远的那个节点
        
        # 如果物理跳变 > 阈值，强制切分
        spatial_split_locs = np.where(max_spatial_jump > self.CLEAN_THRESHOLD_MM)[0] + 1
        
        # --- C. 合并切分点 ---
        # 使用 unique 自动去重并排序
        all_split_locs = np.unique(np.concatenate([time_split_locs, spatial_split_locs]))
        
        # --- D. 执行切分 ---
        if len(all_split_locs) > 0:
            sub_idx_groups = np.split(valid_indices, all_split_locs)
        else:
            sub_idx_groups = [valid_indices]
            
        sub_blocks = []
        for grp in sub_idx_groups:
            # 忽略太短的碎片 (插值需要至少几个点)
            if len(grp) < 10: continue 
            
            block_t = times[grp]
            block_m = motor[grp]
            block_v = vision[grp]
            sub_blocks.append((block_t, block_m, block_v))
            
        return sub_blocks

    def process_block_spline(self, times, motor_data, vis_data, start_t, end_t):
        if self.dynamic_latency_s is None:
            raise ValueError("Latency not initialized!")

        grid_times = np.arange(start_t, end_t, self.DT_TARGET)
        if len(grid_times) < 5: return None

        try:
            cs_motor = CubicSpline(times, motor_data, axis=0, bc_type='natural')
            new_motor = cs_motor(grid_times)

            cs_vision = CubicSpline(times, vis_data, axis=0, bc_type='natural')
            
            # 使用动态时延
            sample_times = grid_times + self.dynamic_latency_s
            
            if sample_times[-1] > times[-1] + 0.02: return None
            if sample_times[0] < times[0] - 0.02: return None 

            new_vision = cs_vision(sample_times)

            new_motor_smooth = savgol_filter(new_motor, self.SG_WINDOW_LEN, self.SG_POLY_ORDER, axis=0)
            new_vision_smooth = savgol_filter(new_vision, self.SG_WINDOW_LEN, self.SG_POLY_ORDER, axis=0)

            return np.hstack([new_motor_smooth, new_vision_smooth])

        except Exception:
            return None

    def process_file(self, raw_path, latency_path, output_clean_path, output_seg_path):
        self.load_latency(latency_path)
        
        print(f"📂 Processing (V4 Auto-Split): {os.path.basename(raw_path)}")
        if not os.path.exists(raw_path): 
            print("❌ File not found.")
            return

        raw_data = np.load(raw_path).astype(np.float64)
        raw_times = raw_data[:, self.RAW_IDX_TIME]
        raw_motor = raw_data[:, self.RAW_IDX_Q_START : self.RAW_IDX_Q_END]
        raw_vis   = raw_data[:, self.RAW_IDX_VIS_START : self.RAW_IDX_VIS_END]

        # Step 1: 孤岛清洗
        raw_vis_clean, drop_count = self.find_stable_islands(raw_vis, self.CLEAN_THRESHOLD_MM, self.MIN_ISLAND_LEN)
        print(f"   🧹 [Clean] Removed {drop_count} unstable frames")

        # Step 2: 原始时间切片
        dt_raw = np.diff(raw_times)
        split_indices = np.where(dt_raw > self.MAX_GAP_TOLERANCE)[0] + 1
        split_indices = np.concatenate([[0], split_indices, [len(raw_times)]])

        processed_blocks = []
        
        # Step 3: 循环处理
        for i in range(len(split_indices) - 1):
            idx_start = split_indices[i]
            idx_end = split_indices[i+1]
            
            chunk_t = raw_times[idx_start:idx_end]
            chunk_m = raw_motor[idx_start:idx_end]
            chunk_v = raw_vis_clean[idx_start:idx_end]

            if len(chunk_t) < 15: continue

            # [重要] 这里调用修复后的 split_by_validity
            # 内部会自动识别并切断 1200mm 的空间跳变
            sub_blocks = self.split_by_validity(chunk_t, chunk_m, chunk_v, self.MAX_GAP_TOLERANCE)
            
            for (sub_t, sub_m, sub_v) in sub_blocks:
                t_start_grid = sub_t[0]
                t_end_grid = sub_t[-1] - self.dynamic_latency_s 
                
                if t_end_grid <= t_start_grid: continue

                res = self.process_block_spline(sub_t, sub_m, sub_v, t_start_grid, t_end_grid)
                if res is not None:
                    processed_blocks.append(res)

        if not processed_blocks:
            print("❌ Failed: Data too fragmented.")
            return

        # Step 4: 合并
        clean_data_list = []
        segments = []
        current_idx = 0
        
        for block in processed_blocks:
            n_steps = len(block)
            clean_data_list.append(block)
            segments.append([current_idx, current_idx + n_steps])
            current_idx += n_steps
            
        final_data = np.vstack(clean_data_list)
        segments_arr = np.array(segments, dtype=int)

        # 保存
        os.makedirs(os.path.dirname(output_clean_path), exist_ok=True)
        np.save(output_clean_path, final_data)
        np.save(output_seg_path, segments_arr)

        print("="*40)
        print(f"✅ Processing Complete (V4)")
        print(f"   Segments: {len(segments_arr)}")
        print(f"   Total Frames: {current_idx}")
        print("="*40)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=str, required=True, help="Raw data .npy")
    parser.add_argument('--latency', type=str, required=True, help="Latency .json")
    parser.add_argument('--out_clean', type=str, required=True, help="Output clean .npy")
    parser.add_argument('--out_seg', type=str, required=True, help="Output segments .npy")
    args = parser.parse_args()
    
    cleaner = DataCleaner()
    cleaner.process_file(args.raw, args.latency, args.out_clean, args.out_seg)

if __name__ == "__main__":
    main()