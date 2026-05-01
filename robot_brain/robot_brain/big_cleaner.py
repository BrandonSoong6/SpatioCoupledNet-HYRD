import numpy as np
import os
import sys
import json
import argparse
import glob
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

class DataCleaner:
    def __init__(self):
        self.TARGET_FREQ = 100.0          
        self.DT_TARGET = 1.0 / self.TARGET_FREQ
        self.dynamic_latency_s = None
        
        # ================= 🔧 阈值配置 =================
        self.MAX_GAP_TOLERANCE = 0.05     
        self.CLEAN_THRESHOLD_MM = 100.0   
        self.CLEAN_THRESHOLD_DEG = 30.0    
        
        self.MIN_ISLAND_LEN = 10          
        
        # 索引配置
        self.RAW_IDX_TIME = 0 
        self.RAW_IDX_Q_START = 13
        self.RAW_IDX_Q_END = 23
        self.RAW_IDX_VIS_START = 44
        self.RAW_IDX_VIS_END = 59
        
        self.SG_WINDOW_LEN = 21  
        self.SG_POLY_ORDER = 3   

    def load_latency(self, json_path):
        default_val = 0.16
        if not os.path.exists(json_path):
            self.dynamic_latency_s = default_val
            return
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.dynamic_latency_s = float(data.get('recommended_latency_s', default_val))
        except:
            self.dynamic_latency_s = default_val

    def get_valid_islands(self, times, vision):
        """[Step 1] 识别孤岛 (含圆周角度修复)"""
        n_frames = len(times)
        if n_frames < 2: return []

        v_reshaped = vision.reshape(n_frames, 5, 3)
        
        is_nonzero = np.any(np.abs(v_reshaped[:, :, :2]) > 1e-5, axis=(1, 2))
        
        dt = np.diff(times)
        diff_vis = v_reshaped[1:] - v_reshaped[:-1]
        
        dist_xy = np.linalg.norm(diff_vis[:, :, :2], axis=2)
        
        # 圆周角度差值
        raw_diff_deg = diff_vis[:, :, 2]
        diff_deg = np.abs((raw_diff_deg + 180.0) % 360.0 - 180.0)
        
        max_dist = np.max(dist_xy, axis=1)
        max_deg = np.max(diff_deg, axis=1)
        
        is_connected = (
            (dt < self.MAX_GAP_TOLERANCE) & 
            (max_dist < self.CLEAN_THRESHOLD_MM) & 
            (max_deg < self.CLEAN_THRESHOLD_DEG) &
            (~np.isnan(max_dist)) &
            is_nonzero[1:] 
        )
        
        padded = np.concatenate(([False], is_connected, [False]))
        ranges = np.where(np.diff(padded))[0].reshape(-1, 2)
        
        valid_islands = []
        for start, end in ranges:
            real_end = end + 1
            if (real_end - start) >= self.MIN_ISLAND_LEN:
                valid_islands.append((start, real_end))
        return valid_islands

    def interpolate_island(self, times, motor, vision, start_t, end_t):
        """[Step 2] 插值 (含 Unwrap/Wrap 修复)"""
        grid_times = np.arange(start_t, end_t, self.DT_TARGET)
        if len(grid_times) < 5: return None, 0.0, 0.0

        try:
            # 1. 解缠绕
            angle_indices = [2, 5, 8, 11, 14]
            vision_unwrapped = vision.copy()
            vision_unwrapped[:, angle_indices] = np.degrees(
                np.unwrap(np.radians(vision[:, angle_indices]), axis=0)
            )

            # 2. 插值
            cs_motor = CubicSpline(times, motor, axis=0, bc_type='natural')
            cs_vision = CubicSpline(times, vision_unwrapped, axis=0, bc_type='natural')
            
            sample_times = grid_times + self.dynamic_latency_s
            
            if sample_times[-1] > times[-1] + 0.01: return None, 0.0, 0.0
            if sample_times[0] < times[0] - 0.01: return None, 0.0, 0.0 

            new_vision_unwrapped = cs_vision(sample_times)
            new_motor = cs_motor(grid_times)

            new_motor_s = savgol_filter(new_motor, self.SG_WINDOW_LEN, self.SG_POLY_ORDER, axis=0)
            new_vision_s = savgol_filter(new_vision_unwrapped, self.SG_WINDOW_LEN, self.SG_POLY_ORDER, axis=0)

            # 3. 重缠绕 (折叠回 -180, 180)
            new_vision_s[:, angle_indices] = (new_vision_s[:, angle_indices] + 180.0) % 360.0 - 180.0

            # 4. 统计插值后的最大跳变
            v_check = new_vision_s.reshape(-1, 5, 3)
            delta = v_check[1:] - v_check[:-1]
            
            # XY 跳变
            max_jump_xy = np.max(np.linalg.norm(delta[:, :, :2], axis=2))
            
            # 角度跳变 (使用圆周差值，避免误报 358度 跳变)
            raw_delta_deg = delta[:, :, 2]
            delta_deg = np.abs((raw_delta_deg + 180.0) % 360.0 - 180.0)
            max_jump_deg = np.max(delta_deg)
            
            return np.hstack([new_motor_s, new_vision_s]), max_jump_xy, max_jump_deg

        except:
            return None, 0.0, 0.0

    def process_file(self, raw_path, latency_path, out_clean, out_seg):
        self.load_latency(latency_path)
        if not os.path.exists(raw_path): return False, "No Raw"
        
        try:
            raw_data = np.load(raw_path).astype(np.float64)
        except: return False, "Load Fail"
        
        if raw_data.shape[0] < self.MIN_ISLAND_LEN: return False, "Too Short"

        # 1. 识别孤岛
        raw_times = raw_data[:, self.RAW_IDX_TIME]
        raw_vis = raw_data[:, self.RAW_IDX_VIS_START : self.RAW_IDX_VIS_END]
        islands = self.get_valid_islands(raw_times, raw_vis)
        
        # 统计原始有效帧数 (被识别为孤岛的总长度)
        total_raw_frames = sum([end - start for start, end in islands]) if islands else 0
        
        processed_blocks = []
        max_batch_xy = 0.0
        max_batch_deg = 0.0
        total_kept_frames = 0
        
        # 2. 插值处理
        for start, end in islands:
            chunk_t = raw_times[start:end]
            chunk_m = raw_data[start:end, self.RAW_IDX_Q_START : self.RAW_IDX_Q_END]
            chunk_v = raw_vis[start:end]
            
            t_start = chunk_t[0]
            t_end = chunk_t[-1] - self.dynamic_latency_s
            
            if t_end <= t_start: continue
            
            block_data, mj_xy, mj_deg = self.interpolate_island(
                chunk_t, chunk_m, chunk_v, t_start, t_end
            )
            
            if block_data is not None:
                processed_blocks.append(block_data)
                total_kept_frames += len(block_data)
                max_batch_xy = max(max_batch_xy, mj_xy)
                max_batch_deg = max(max_batch_deg, mj_deg)

        # 3. 清理与保存
        if os.path.exists(out_clean): os.remove(out_clean)
        if os.path.exists(out_seg): os.remove(out_seg)

        if not processed_blocks:
            return False, f"Raw: {total_raw_frames} | Kept: 0 (Loss: 100%)"

        data_list, segs, curr = [], [], 0
        for b in processed_blocks:
            data_list.append(b)
            segs.append([curr, curr + len(b)])
            curr += len(b)
            
        np.save(out_clean, np.vstack(data_list))
        np.save(out_seg, np.array(segs, dtype=int))
        
        # 计算损失比例
        loss_ratio = 100.0 * (1.0 - total_kept_frames / total_raw_frames) if total_raw_frames > 0 else 0.0
        
        info_str = (f"Raw: {total_raw_frames} | Kept: {total_kept_frames} (Loss: {loss_ratio:.1f}%) | "
                    f"MaxXY: {max_batch_xy:.1f}mm | MaxDeg: {max_batch_deg:.1f}°")
        return True, info_str

def run_big_cleaner(root_dir):
    cleaner = DataCleaner()
    search_pattern = os.path.join(root_dir, "**", "batch_*")
    batch_dirs = sorted(glob.glob(search_pattern, recursive=True))
    
    print(f"🚀 Starting BigCleaner (Full Stats: Raw vs Kept, Angle Wrap fix)...")
    print("-" * 120)
    print(f"{'Batch Name':<15} | {'Status':<5} | {'Details'}")
    print("-" * 120)
    
    success_count = 0
    fail_count = 0
    
    for b_dir in tqdm(batch_dirs, desc="Progress"):
        batch_name = os.path.basename(b_dir)
        raw_path = os.path.join(b_dir, "raw_data.npy")
        latency_path = os.path.join(b_dir, "latency.json")
        out_clean = os.path.join(b_dir, "clean_data.npy")
        out_seg = os.path.join(b_dir, "segments.npy")
        
        if not os.path.exists(raw_path): continue
        
        ok, msg = cleaner.process_file(raw_path, latency_path, out_clean, out_seg)
        
        status = "✅" if ok else "🗑️"
        # 监控警告：位置跳变 > 80mm 或 角度跳变 > 10度
        if "MaxXY" in msg:
            try:
                # 解析 msg 来判断是否警告
                parts = msg.split("|")
                m_xy = float(parts[2].split(":")[1].replace("mm", ""))
                m_deg = float(parts[3].split(":")[1].replace("°", ""))
                if m_xy > 80.0 or m_deg > 10.0: 
                    status = "⚠️"
            except: pass
            
        tqdm.write(f"{batch_name:<15} | {status:<5} | {msg}")
        
        if ok: success_count += 1
        else: fail_count += 1

    print("-" * 120)
    print(f"🏁 Finished.")
    print(f"✅ Kept: {success_count}")
    print(f"🗑️  Discarded: {fail_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/brandon/brandon/hyrd_robot/lifelong_data')
    args = parser.parse_args()
    run_big_cleaner(args.root)