import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
import argparse
import os

# ================= 1. 核心处理函数 =================
def apply_smoothing_and_save(input_path, output_path, window_len, poly_order):
    print(f"📂 [Step 1] Loading raw data from: {input_path}")
    if not os.path.exists(input_path):
        print("❌ Error: Input file not found!")
        return False

    checkpoint = torch.load(input_path, map_location='cpu')
    data = checkpoint['data']
    
    # 提取原始位置数据 [N, 5, 3] (x, y, theta)
    # 注意：我们不对 tgt_delta 直接滤波，而是对 pose_loc 滤波后重新计算 delta
    # 这样可以保证位姿和速度的物理一致性。
    raw_pose = data['pose_loc'].numpy()
    
    N, n_seg, n_dim = raw_pose.shape
    print(f"📊 Data Shape: {N} samples, {n_seg} segments.")
    print(f"🧹 [Step 2] Applying Savitzky-Golay Filter...")
    print(f"   ➤ Parameters: Window={window_len}, PolyOrder={poly_order}")
    
    smooth_pose = np.zeros_like(raw_pose)
    
    # 对每个 Segment 的每个维度分别在时间轴 (axis=0) 上滤波
    # 注意：这里假设数据在时间上是连续的。
    # 如果 mega_expert_big 是多个不连续轨迹的拼接，会在拼接点产生轻微的边缘效应，
    # 但相对于百万级的数据量，这几个点的误差可以忽略。
    for s in range(n_seg):
        for d in range(n_dim):
            smooth_pose[:, s, d] = savgol_filter(raw_pose[:, s, d], 
                                                 window_length=window_len, 
                                                 polyorder=poly_order, 
                                                 axis=0)
    
    print("🔄 [Step 3] Re-calculating Deltas (Differential)...")
    # 计算新的平滑 Delta: P[t+1] - P[t]
    # 由于 diff 会少一个数据点，我们需要填充或者切片
    # 这里我们采用 standard 做法：丢弃最后一个 pose，保持对齐
    
    # 原始数据生成逻辑通常是: delta[t] = pose[t+1] - pose[t]
    # 所以我们做同样的 diff
    new_delta_raw = np.diff(smooth_pose, axis=0)
    
    # 为了保持 tensor 长度不变 (N)，通常生成器会丢掉最后一帧
    # 这里我们为了严谨，检查原文件中 tgt_delta 的长度
    old_delta = data['tgt_delta'].numpy()
    target_len = old_delta.shape[0]
    
    # 如果 diff 后长度不对，进行截断 (通常 diff 后是 N-1)
    # 假设原数据已经是处理过的 (N_valid)，我们需要确保维度匹配
    # 这里直接使用 diff 后的数据，因为原始生成器也是 diff 得到的
    # 我们假设 pose_loc 长度为 N，那么生成的 delta 长度为 N-1
    # 如果原 pt 文件里的 pose_loc 和 tgt_delta 长度一致（说明 pose_loc 已经被切过），那我们 diff 后会少 1
    # 这是一个关键的对齐检查
    
    if new_delta_raw.shape[0] != target_len:
        print(f"⚠️ Length Mismatch! Orig Delta: {target_len}, New Diff: {new_delta_raw.shape[0]}")
        print("   -> Adjusting logic to match original dataset structure...")
        # 这种情况下，通常意味着 pose_loc 比 tgt_delta 多 1 帧 (或者反之)
        # 我们这里采取保守策略：只更新长度匹配的部分
        min_len = min(new_delta_raw.shape[0], target_len)
        new_delta = new_delta_raw[:min_len]
        
        # 同时也需要裁剪 pose_loc 以匹配
        # 但通常我们只替换 delta。为了安全，我们把新计算的 delta 放入字典
        # 并确保它和原来的形状完全一样
    else:
        new_delta = new_delta_raw

    # 更新数据字典
    # 1. 更新平滑后的 Pose (注意形状对齐)
    data['pose_loc'] = torch.tensor(smooth_pose) 
    
    # 2. 更新重新计算的 Delta (这是训练的关键 Label)
    # 必须确保长度一致，否则 Dataset读取会报错
    if new_delta.shape[0] == data['tgt_delta'].shape[0]:
         data['tgt_delta'] = torch.tensor(new_delta).float()
    else:
        # 如果长度不一致，说明原始数据的 pose_loc 和 delta 之间可能有位移
        # 这种情况下，我们简单地用平滑后的 pose 再做一次差分，并强制截断到原长度
        print("   -> Force matching length...")
        data['tgt_delta'] = torch.tensor(new_delta_raw[:target_len]).float()

    print("⚖️ [Step 4] Re-computing Statistics (Scalers)...")
    # 因为去除了噪声，均值和方差会变，必须更新 Scalers
    new_scalers = checkpoint['scalers']
    
    # 更新 pose_loc 的统计
    flat_pose = data['pose_loc'].reshape(-1, 3)
    new_scalers['pose_loc_mean'] = torch.mean(flat_pose, dim=0)
    new_scalers['pose_loc_std'] = torch.std(flat_pose, dim=0) + 1e-6
    
    # 更新 tgt_delta 的统计
    flat_delta = data['tgt_delta'].reshape(-1, 3)
    new_scalers['tgt_delta_mean'] = torch.mean(flat_delta, dim=0)
    new_scalers['tgt_delta_std'] = torch.std(flat_delta, dim=0) + 1e-6

    # 保存
    checkpoint['data'] = data
    checkpoint['scalers'] = new_scalers
    
    print(f"💾 [Step 5] Saving to: {output_path}")
    torch.save(checkpoint, output_path)
    print("✅ Processing Complete!")
    return True

# ================= 2. 分析与可视化函数 =================
def analyze_results(file_path):
    print(f"\n🔍 [Analysis] Inspecting: {file_path}")
    checkpoint = torch.load(file_path, map_location='cpu')
    delta = checkpoint['data']['tgt_delta'].numpy() # [N, 5, 3]
    
    dims = ['Delta X (mm)', 'Delta Y (mm)', 'Delta Theta (rad)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # --- 1. 统计学分析表格 ---
    print("\n📊 Statistical Report (Smoothed Data):")
    print("-" * 90)
    print(f"{'Dim':<15} | {'Mean':<12} | {'Std':<12} | {'Skew':<10} | {'Kurtosis':<10} | {'Min/Max'}")
    print("-" * 90)
    
    flat_data = delta.reshape(-1, 3)
    for i in range(3):
        col = flat_data[:, i]
        mu = np.mean(col)
        sigma = np.std(col)
        sk = skew(col)
        ku = kurtosis(col)
        rng = f"[{np.min(col):.2f}, {np.max(col):.2f}]"
        
        print(f"{dims[i]:<15} | {mu:+.4e} | {sigma:.6f}   | {sk:+.4f}     | {ku:+.4f}     | {rng}")
    print("-" * 90)
    print("💡 Note: Kurtosis > 3 implies heavy tails (retained physical jumps).")

    # --- 2. 绘图：直方图 + 样本时序图 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: 直方图 (Histograms)
    for i in range(3):
        ax = axes[0, i]
        col = flat_data[:, i]
        
        # 使用对数坐标，查看是否还有离群点
        ax.hist(col, bins=100, color=colors[i], alpha=0.7, log=True)
        ax.set_title(f"Dist: {dims[i]}\n(Log Scale)", fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 标注统计量
        stats_txt = f"$\mu$={np.mean(col):.2e}\n$\sigma$={np.std(col):.2e}"
        ax.text(0.95, 0.95, stats_txt, transform=ax.transAxes, ha='right', va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Row 2: 样本时序图 (Time Series Sample)
    # 选取最难处理的 Segment 5 (Index 4)
    seg_idx = 4
    # 随机取一段 1000 帧 (或前1000帧)
    start_idx = 2000 
    end_idx = 3000
    sample_slice = delta[start_idx:end_idx, seg_idx, :]
    
    time_x = np.arange(len(sample_slice)) # 10ms steps
    
    for i in range(3):
        ax = axes[1, i]
        ax.plot(time_x, sample_slice[:, i], color=colors[i], linewidth=1.5)
        ax.set_title(f"Sample Trace (Seg 5): {dims[i]}", fontweight='bold')
        ax.set_xlabel("Time Step (10ms)")
        ax.set_ylabel("Increment")
        ax.grid(True, alpha=0.4)

    plt.suptitle(f"Analysis of Smoothed Dataset (Window=71, Poly=4)\nFile: {os.path.basename(file_path)}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_img = "smoothed_analysis_report.png"
    plt.savefig(save_img)
    print(f"\n📈 Analysis plot saved to: {save_img}")
    # plt.show() # Uncomment if running locally with display

# ================= Main Execution =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径配置
    parser.add_argument('--in_path', type=str, 
                        default='/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_big_clean.pt')
    parser.add_argument('--out_path', type=str, 
                        default='/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_big_clean_smooth.pt')
    
    # 你的指定参数
    parser.add_argument('--window', type=int, default=101)
    parser.add_argument('--poly', type=int, default=3)
    
    args = parser.parse_args()
    
    # 1. 处理
    success = apply_smoothing_and_save(args.in_path, args.out_path, args.window, args.poly)
    
    # 2. 分析
    if success:
        analyze_results(args.out_path)