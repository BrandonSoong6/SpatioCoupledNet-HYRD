import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# 默认路径，按照你之前的脚本配置
DEFAULT_DATA_PATH = "/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_smooth_strided.pt"

def inspect_jumps(data_path):
    print(f"📂 Loading dataset from: {data_path}")
    
    if not os.path.exists(data_path):
        print("❌ File not found! Please check the path.")
        return

    # 1. 加载数据
    data_dict = torch.load(data_path, map_location='cpu')
    
    # 检查 Key，RoboSeqDataset 里用的是 'data' 里的 'tgt_delta'
    if 'data' in data_dict:
        tensors = data_dict['data']
    else:
        # 兼容其他格式，直接尝试读取
        tensors = data_dict
        
    if 'tgt_delta' not in tensors:
        print("❌ 'tgt_delta' key not found in dataset. Available keys:", tensors.keys())
        return

    # [N, 5, 3] -> 每一帧、每个节点、XYZ位移
    # 注意：在 RoboSeqDataset 里，tgt_delta 是没有归一化的物理值，这非常方便
    deltas = tensors['tgt_delta'].float() 
    
    print(f"📊 Data Shape: {deltas.shape} (Frames, Nodes, XYZ)")
    
    # 2. 计算每一帧的“瞬时位移量” (L2 Norm)
    # shape: [N, 5] -> 每个节点这一步动了多少毫米
    jump_distances = torch.norm(deltas, dim=2).numpy()
    
    # 展平以便统计整体分布
    all_jumps = jump_distances.flatten()
    
    # 3. 统计基础指标
    max_jump = np.max(all_jumps)
    mean_jump = np.mean(all_jumps)
    median_jump = np.median(all_jumps)
    p99 = np.percentile(all_jumps, 99)
    p999 = np.percentile(all_jumps, 99.9)
    
    print("\n" + "="*40)
    print("       📉 位移分布统计 (Step Delta)       ")
    print("="*40)
    print(f"Mean Step:   {mean_jump:.4f} mm")
    print(f"Median Step: {median_jump:.4f} mm")
    print(f"99% Limit:   {p99:.4f} mm (99% 的动作都在这个范围内)")
    print(f"99.9% Limit: {p999:.4f} mm")
    print(f"⚠️ MAX JUMP:  {max_jump:.4f} mm <--- 重点看这个")
    print("="*40)

    # 4. 找出“飞了”的异常点 (Top 10 Outliers)
    print("\n🚨 Top 10 Largest Jumps (Potential Tracking Loss):")
    print(f"{'Frame Idx':<10} | {'Node Idx':<10} | {'Jump (mm)':<10}")
    print("-" * 36)
    
    # 获取 flatten 后的索引，然后转回二维索引
    flat_indices = np.argsort(all_jumps)[::-1][:10] # 取最大的10个
    for idx in flat_indices:
        frame_idx, node_idx = np.unravel_index(idx, jump_distances.shape)
        val = jump_distances[frame_idx, node_idx]
        print(f"{frame_idx:<10} | {node_idx:<10} | {val:.4f}")

    # 5. 绘制直方图
    plt.figure(figsize=(12, 6))
    
    # 主直方图 (Log scale y轴，因为异常值通常很少)
    plt.subplot(1, 2, 1)
    plt.hist(all_jumps, bins=100, color='skyblue', edgecolor='black', log=True)
    plt.title("Delta Magnitude Distribution (Log Scale)")
    plt.xlabel("Step Distance (mm)")
    plt.ylabel("Count (Log)")
    plt.axvline(p99, color='r', linestyle='--', label=f'99% ({p99:.2f}mm)')
    plt.legend()
    
    # 异常值特写 (只画 > 99.9% 分位数的部分)
    plt.subplot(1, 2, 2)
    outliers = all_jumps[all_jumps > p99]
    if len(outliers) > 0:
        plt.hist(outliers, bins=50, color='salmon', edgecolor='black')
        plt.title(f"Tail Distribution (Top 1%)")
        plt.xlabel("Step Distance (mm)")
        plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "No significant outliers", ha='center')
        
    save_path = "delta_distribution_check.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n🖼️  Histogram saved to: {os.path.abspath(save_path)}")
    print(f"💡 如果 Max Jump 远大于 99% Limit (比如 70mm vs 2mm)，那就是数据飞了。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=DEFAULT_DATA_PATH, help="Path to .pt dataset")
    args = parser.parse_args()
    
    inspect_jumps(args.path)