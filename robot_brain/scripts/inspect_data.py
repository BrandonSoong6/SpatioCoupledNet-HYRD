import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# ================= 配置 =================
# 请修改为你的实际文件路径
DATA_PATH = "/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_smooth_strided.pt"
SAMPLE_LEN = 1000  # 查看 1000 个点
FREQ = 100.0       # 采样频率 100Hz
DT = 1.0 / FREQ    # 0.01s

def inspect_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 文件不存在: {DATA_PATH}")
        return

    print(f"🔄 正在加载数据: {DATA_PATH} ...")
    # 加载 .pt 文件 (map_location 确保在 CPU 上也能跑)
    dataset = torch.load(DATA_PATH, map_location='cpu')
    
    # 提取 tgt_delta 张量
    # 结构应该在 dataset['data']['tgt_delta']
    # Shape: (Total_Frames, 5, 3) -> (N, Nodes, [dx, dy, dtheta])
    if 'data' in dataset:
        tgt_delta = dataset['data']['tgt_delta']
    else:
        # 兼容旧格式，直接可能是 tensor
        tgt_delta = dataset['tgt_delta']
        
    total_frames = tgt_delta.shape[0]
    print(f"✅ 数据加载完成. 总帧数: {total_frames}")
    
    if total_frames < SAMPLE_LEN:
        print("⚠️ 数据量少于采样长度，将显示所有数据")
        start_idx = 0
        end_idx = total_frames
    else:
        # 随机选择一个起始点
        start_idx = random.randint(0, total_frames - SAMPLE_LEN)
        end_idx = start_idx + SAMPLE_LEN
        print(f"🎲 随机采样区间: Index {start_idx} -> {end_idx}")

    # 提取第 5 个关节 (Index 4) 的数据
    # slice shape: (1000, 3)
    joint_idx = 0
    data_slice = tgt_delta[start_idx:end_idx, joint_idx, :].numpy()
    
    # 生成时间轴 (秒)
    time_axis = np.arange(data_slice.shape[0]) * DT
    
    # ================= 绘图 =================
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    titles = [
        f'Joint 5 Delta X (mm) - {FREQ}Hz',
        f'Joint 5 Delta Y (mm) - {FREQ}Hz',
        f'Joint 5 Delta Theta (rad) - {FREQ}Hz'
    ]
    
    colors = ['r', 'g', 'b']
    
    for i in range(3):
        ax = axes[i]
        # 绘制波形
        ax.plot(time_axis, data_slice[:, i], color=colors[i], linewidth=1.5, alpha=0.8)
        
        # 绘制 0 轴参考线
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # 标注统计信息
        mean_val = np.mean(data_slice[:, i])
        std_val = np.std(data_slice[:, i])
        max_val = np.max(np.abs(data_slice[:, i]))
        
        info_text = f"Mean: {mean_val:.4f} | Std: {std_val:.4f} | MaxAbs: {max_val:.4f}"
        ax.set_title(titles[i], fontsize=10, fontweight='bold')
        ax.text(0.02, 0.9, info_text, transform=ax.transAxes, 
                fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
        
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Increment")

    axes[2].set_xlabel("Time (seconds)")
    
    plt.tight_layout()
    save_path = "inspect_delta_plot.png"
    plt.savefig(save_path)
    print(f"📊 图表已保存至: {save_path}")
    
    # 打印部分数值供终端检查
    print("\n🔍 前 10 个数据点数值预览 (dx, dy, dtheta):")
    print(data_slice[:10])

if __name__ == "__main__":
    inspect_dataset()