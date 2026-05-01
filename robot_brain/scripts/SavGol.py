import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis, norm
import argparse
import os

def smooth_and_analyze_residuals(file_path):
    print(f"📂 Loading: {file_path}")
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    data_dict = torch.load(file_path, map_location='cpu')
    pose_loc = data_dict['data']['pose_loc'].numpy() 
    
    seg_idx = 4
    start_f = 2000
    end_f = 3000
    
    # 确保数据长度足够
    if pose_loc.shape[0] < end_f:
        end_f = pose_loc.shape[0]
        print(f"⚠️ Data shorter than requested, clamping to {end_f}")

    sample_pose = pose_loc[start_f:end_f, seg_idx, :] 
    
    # ================= 参数设置 =================
    # 你设置的 71 (0.7秒) 其实对于 100Hz 来说非常大，可能会导致过平滑
    # 建议对比一下 31 和 71 的残差图
    WINDOW_LEN = 71 
    POLY_ORDER = 4
    
    smooth_pose = np.zeros_like(sample_pose)
    
    print(f"🧹 Applying Savitzky-Golay Filter (Window={WINDOW_LEN}, Poly={POLY_ORDER})...")
    for dim in range(3):
        smooth_pose[:, dim] = savgol_filter(sample_pose[:, dim], window_length=WINDOW_LEN, polyorder=POLY_ORDER)

    # ================= 计算差分 (Delta) =================
    # 我们分析的是 "速度的残差"，因为这是你训练神经网络的直接输入
    delta_raw = np.diff(sample_pose, axis=0)      # 原始噪声差分
    delta_smooth = np.diff(smooth_pose, axis=0)   # 平滑后差分
    
    # === 核心：计算残差 (Residuals) ===
    # Residual = Raw - Smooth
    # 如果平滑完美，这里剩下的应该纯粹是噪声
    residuals = delta_raw - delta_smooth

    dims_name = ['Delta X (mm)', 'Delta Y (mm)', 'Delta Theta (rad)']
    colors = ['r', 'g', 'b']

    # ================= 图表 1: 平滑效果对比 (保持你原来的图) =================
    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for i in range(3):
        ax = axes1[i]
        ax.plot(delta_raw[:, i], color='lightgray', label='Raw (Noisy)', linewidth=1.0, alpha=0.7)
        ax.plot(delta_smooth[:, i], color=colors[i], label='SavGol Smoothed', linewidth=2.0)
        ax.set_title(f"Smoothing Result: {dims_name[i]}")
        ax.set_ylabel("Increment / 10ms")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    fig1.suptitle(f"Signal Smoothing (Window: {WINDOW_LEN})", fontsize=16)
    fig1.tight_layout()

    # ================= 图表 2: 残差分析 (新增) =================
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 12)) 
    # 左列：残差时序图 (Time Series)
    # 右列：残差直方图 (Histogram)

    print("\n📊 Residual Analysis Statistics (Is it Gaussian White Noise?):")
    print("-" * 80)
    print(f"{'Dim':<12} | {'Mean':<10} | {'Std':<10} | {'Skewness':<10} | {'Kurtosis':<10} | {'Verdict'}")
    print("-" * 80)

    for i in range(3):
        res_data = residuals[:, i]
        
        # --- 统计量计算 ---
        mu = np.mean(res_data)
        sigma = np.std(res_data)
        sk = skew(res_data)   # 偏度：应该是 0
        ku = kurtosis(res_data) # 峰度 (Fisher)：应该是 0 (对应正态分布)
        
        # 简单判定
        is_centered = abs(mu) < 1e-4
        is_symmetric = abs(sk) < 0.5
        is_gaussian_shape = abs(ku) < 1.0 # 稍微放宽一点
        
        if is_centered and is_symmetric and is_gaussian_shape:
            verdict = "✅ Good Noise"
        else:
            verdict = "⚠️ Signal Leaked?"

        print(f"{dims_name[i]:<12} | {mu:+.2e} | {sigma:.4f}   | {sk:+.4f}    | {ku:+.4f}    | {verdict}")

        # --- 绘图：左侧时序图 ---
        ax_ts = axes2[i, 0]
        ax_ts.plot(res_data, color='black', alpha=0.6, linewidth=0.8)
        ax_ts.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax_ts.set_title(f"Residuals Time Series - {dims_name[i]}")
        ax_ts.set_ylabel("Residual (Raw - Smooth)")
        ax_ts.grid(True, alpha=0.2)
        
        # --- 绘图：右侧直方图 + 高斯拟合 ---
        ax_hist = axes2[i, 1]
        # 画直方图
        n, bins, patches = ax_hist.hist(res_data, bins=100, density=True, color=colors[i], alpha=0.5, label='Residuals')
        
        # 画理论高斯曲线
        xmin, xmax = ax_hist.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        ax_hist.plot(x, p, 'k--', linewidth=2, label=f'Normal Fit\n$\mu$={mu:.2e}, $\sigma$={sigma:.2f}')
        
        ax_hist.set_title(f"Residual Distribution - {dims_name[i]}")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.2)

    fig2.suptitle(f"Residual Analysis (Raw - Smooth)\nGoal: Zero Mean, No Pattern, Gaussian Shape", fontsize=16)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()

if __name__ == "__main__":
    # 请确保路径正确
    path = '/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_big_clean.pt'
    smooth_and_analyze_residuals(path)