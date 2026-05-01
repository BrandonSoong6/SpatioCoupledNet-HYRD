import numpy as np
import os
import glob
from tqdm import tqdm
import argparse

# ================= 配置 =================
# 根目录：你刚才展示的那个目录
DEFAULT_ROOT = "/home/brandon/brandon/hyrd_robot/lifelong_data/020401"

# 阈值：如果两帧之间 (20ms-50ms) 移动超过这个距离，绝对是数据错误
# 正常机器人最快也就 10-20mm/帧
JUMP_THRESHOLD_MM = 60.0 

# 数据结构定义 (基于 DataCleaner 的输出)
# clean_data 是 [Motor(10) | Vision(15)]
VIS_START_IDX = 10 
VIS_END_IDX = 25
# ========================================

def check_batch(batch_dir):
    batch_name = os.path.basename(batch_dir)
    clean_path = os.path.join(batch_dir, "clean_data.npy")
    seg_path = os.path.join(batch_dir, "segments.npy")

    # 1. 检查文件是否存在
    if not (os.path.exists(clean_path) and os.path.exists(seg_path)):
        # print(f"⚠️  Skipping {batch_name}: Files missing")
        return []

    try:
        # 2. 加载数据
        data = np.load(clean_path)     # [Total_Frames, 25]
        segments = np.load(seg_path)   # [Num_Segs, 2] -> [[start, end], ...]

        issues = []
        
        # 3. 逐个片段检查 (只检查片段内部!)
        for seg_idx, (start, end) in enumerate(segments):
            # 提取该段的视觉数据 [T, 15]
            # 15列 = 5个节点 * (x,y,theta)
            vis_chunk = data[start:end, VIS_START_IDX:VIS_END_IDX]
            
            if len(vis_chunk) < 2:
                continue

            # Reshape 成 [T, 5, 3] 以便计算每个节点的位移
            # 假设数据排列是: [x1, y1, th1, x2, y2, th2, ...]
            vis_reshaped = vis_chunk.reshape(-1, 5, 3)
            
            # 计算相邻帧差分
            # delta: [T-1, 5, 3]
            delta = vis_reshaped[1:] - vis_reshaped[:-1]
            
            # 计算欧几里得距离 (只看 xy，或者 xy+theta 引起的位移，这里粗略算 norm 即可)
            # dist: [T-1, 5]
            dist = np.linalg.norm(delta, axis=2)
            
            # 找最大值
            max_jump = np.max(dist)
            
            if max_jump > JUMP_THRESHOLD_MM:
                # 找到具体的帧和节点
                frame_idx_in_seg, node_idx = np.unravel_index(np.argmax(dist), dist.shape)
                abs_frame_idx = start + frame_idx_in_seg
                
                issues.append({
                    "seg_idx": seg_idx,
                    "abs_frame": abs_frame_idx,
                    "rel_frame": frame_idx_in_seg,
                    "node": node_idx,
                    "val": max_jump
                })

        return issues

    except Exception as e:
        print(f"❌ Error processing {batch_name}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=DEFAULT_ROOT)
    args = parser.parse_args()

    # 搜索所有 batch_xxx
    search_path = os.path.join(args.root, "batch_*")
    batch_dirs = sorted(glob.glob(search_path))
    
    print(f"🔍 Scanning {len(batch_dirs)} batches in: {args.root}")
    print(f"🎯 Threshold: > {JUMP_THRESHOLD_MM} mm")
    print("="*60)

    total_issues = 0
    dirty_batches = []

    pbar = tqdm(batch_dirs, unit="batch")
    for b_dir in pbar:
        b_name = os.path.basename(b_dir)
        pbar.set_description(f"Checking {b_name}")
        
        batch_issues = check_batch(b_dir)
        
        if batch_issues:
            pbar.write(f"\n🚨 {b_name} HAS ISSUES:")
            for issue in batch_issues:
                pbar.write(
                    f"   - Seg {issue['seg_idx']:<3} | "
                    f"Frame {issue['abs_frame']:<6} | "
                    f"Node {issue['node']} | "
                    f"Jump: {issue['val']:.2f} mm"
                )
            total_issues += len(batch_issues)
            dirty_batches.append(b_name)

    print("\n" + "="*60)
    if total_issues == 0:
        print("✅ ALL CLEAN! No internal jumps detected.")
        print("   这意味着你的 DataCleaner 工作完美，问题出在 DataGenerator 拼接 Segments 的地方。")
    else:
        print(f"❌ FOUND {total_issues} ILLEGAL JUMPS in {len(dirty_batches)} batches.")
        print(f"   Dirty Batches: {dirty_batches}")
        print("   这意味着这些 Batch 需要重新运行 DataCleaner (或者调低 Threshold)。")

if __name__ == "__main__":
    main()