import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def analyze_data(file_path):
    print(f"📂 正在加载文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在 -> {file_path}")
        return

    # 加载数据 (映射到 CPU)
    data_dict = torch.load(file_path, map_location='cpu')
    
    # 提取 tgt_delta: Shape [N, 5, 3] -> (样本数, 5个单元, dx/dy/dtheta)
    if 'data' in data_dict and 'tgt_delta' in data_dict['data']:
        tgt_delta = data_dict['data']['tgt_delta'].numpy()
    else:
        print("❌ 错误: 数据字典中找不到 'tgt_delta' 键")
        return

    num_samples, num_segments, num_dims = tgt_delta.shape
    print(f"✅ 数据加载成功. 样本总数: {num_samples}, 单元数: {num_segments}, 维度: {num_dims}")

    # ================= 任务 1: 绘制 3x5 直方图 (分布概览) =================
    print("📊 正在绘制分布直方图...")
    
    fig_hist, axes = plt.subplots(3, 5, figsize=(24, 12))
    var_names = [r'$\delta x$ (mm)', r'$\delta y$ (mm)', r'$\delta \theta$ (rad)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 蓝, 橙, 绿

    # 遍历 3 个维度 (行)
    for dim in range(3):
        # 遍历 5 个单元 (列)
        for seg in range(5):
            ax = axes[dim, seg]
            data_col = tgt_delta[:, seg, dim]
            
            # 统计信息
            d_min, d_max = np.min(data_col), np.max(data_col)
            d_mean, d_std = np.mean(data_col), np.std(data_col)
            
            # 绘制直方图
            ax.hist(data_col, bins=100, color=colors[dim], alpha=0.7, log=True) # 使用对数坐标以便观察长尾分布
            
            # 设置标题和标签
            if dim == 0:
                ax.set_title(f"Segment {seg+1}", fontsize=12, fontweight='bold')
            if seg == 0:
                ax.set_ylabel(f"Count (Log Scale)\n{var_names[dim]}", fontsize=12)
            
            # 在图中标注统计数据
            stats_text = (f"Mean: {d_mean:.4f}\n"
                          f"Std:  {d_std:.4f}\n"
                          f"Min:  {d_min:.4f}\n"
                          f"Max:  {d_max:.4f}")
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right', 
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.suptitle(f"Distribution of Delta Pose (dx, dy, dtheta) across 5 Segments\n(N={num_samples})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ================= 任务 2: 随机抽取 1000 个连续点 (第5单元) =================
    print("📉 正在抽取第 5 单元的连续采样点...")
    
    SAMPLE_LEN = 1000
    TARGET_SEG_IDX = 4  # 索引4 对应 第5个单元
    
    if num_samples <= SAMPLE_LEN:
        start_idx = 0
        actual_len = num_samples
        print(f"⚠️ 警告: 总样本数少于 {SAMPLE_LEN}，将显示所有数据。")
    else:
        # 随机选择起始点
        start_idx = np.random.randint(0, num_samples - SAMPLE_LEN)
        actual_len = SAMPLE_LEN
        
    end_idx = start_idx + actual_len
    
    # 切片数据: [1000, 3]
    sampled_data = tgt_delta[start_idx:end_idx, TARGET_SEG_IDX, :]
    
    fig_ts, axes_ts = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    time_axis = np.arange(start_idx, end_idx)
    
    for dim in range(3):
        ax = axes_ts[dim]
        ax.plot(time_axis, sampled_data[:, dim], color=colors[dim], linewidth=1)
        
        ax.set_ylabel(var_names[dim], fontsize=12)
        ax.set_title(f"Segment 5 - {var_names[dim]} (Frame {start_idx} to {end_idx})", fontsize=10)
        ax.grid(True, alpha=0.5)
        
        # 标出该段数据的局部均值线
        local_mean = np.mean(sampled_data[:, dim])
        ax.axhline(local_mean, color='red', linestyle='--', alpha=0.6, label=f'Local Mean: {local_mean:.4f}')
        ax.legend(loc='upper right')

    axes_ts[-1].set_xlabel("Frame Index", fontsize=12)
    plt.suptitle(f"Continuous Sampling of Segment 5 Delta Pose (1000 Frames)", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Robot Delta Pose Data")
    # 默认路径设置为你之前代码中的输出路径
    parser.add_argument('--path', type=str, 
                        default='/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_smooth_strided.pt',
                        help='Path to the .pt file')
    
    args = parser.parse_args()
    
    analyze_data(args.path)