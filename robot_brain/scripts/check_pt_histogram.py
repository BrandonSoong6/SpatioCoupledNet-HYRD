import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def check_pt_distribution(pt_path):
    print(f"🔍 Loading dataset: {pt_path}")
    if not os.path.exists(pt_path):
        print("❌ File not found!")
        return

    checkpoint = torch.load(pt_path)
    data = checkpoint['data']
    
    # 提取关键的差分数据
    # 1. 视觉位姿增量 (Target Delta): [N, 5, 3] -> (dx, dy, dtheta)
    tgt_delta = data['tgt_delta'].numpy()
    
    # 2. 电机指令增量 (Motor Command): [N, 5, 2] -> (dq_l, dq_r)
    dq_cmd = data['dq_cmd'].numpy()

    # --- 数据预处理 ---
    # 展平所有 Batch 和 Section，只看整体分布
    dx_all = tgt_delta[:, :, 0].flatten()
    dy_all = tgt_delta[:, :, 1].flatten()
    dth_all = tgt_delta[:, :, 2].flatten() # 原始单位是弧度 (Rad)
    
    # 将弧度转为度，方便直观检查
    dth_deg = np.degrees(dth_all)
    dq_all = dq_cmd.flatten() # 弧度

    # --- 统计异常值 ---
    print("\n📊 统计报告 (Statistics Report):")
    print("-" * 60)
    
    # 1. 检查角度跳变 (Vision Theta)
    # 正常运动下，10ms 内转动超过 10度 是几乎不可能的
    outlier_th = np.sum(np.abs(dth_deg) > 10.0)
    total_th = len(dth_deg)
    print(f"Angle Delta (Vision):")
    print(f"  Min: {np.min(dth_deg):.4f}° | Max: {np.max(dth_deg):.4f}° | Mean: {np.mean(dth_deg):.4f}°")
    print(f"  ⚠️ > 10° Jumps: {outlier_th} / {total_th} ({outlier_th/total_th*100:.4f}%)")
    
    # 2. 检查位移跳变 (Vision XY)
    dist_xy = np.sqrt(dx_all**2 + dy_all**2)
    outlier_xy = np.sum(dist_xy > 20.0) # 20mm 阈值
    print(f"\nPosition Delta (Vision XY):")
    print(f"  Max Step: {np.max(dist_xy):.4f} mm")
    print(f"  ⚠️ > 20mm Jumps: {outlier_xy} ({outlier_xy/len(dist_xy)*100:.4f}%)")

    print("-" * 60)

    # --- 绘图 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Differential Histograms Check: {os.path.basename(pt_path)}", fontsize=14)

    # Plot 1: Vision XY Delta
    axes[0].hist(dx_all, bins=100, alpha=0.5, label='dX (mm)', log=True) # 使用对数坐标看清异常值
    axes[0].hist(dy_all, bins=100, alpha=0.5, label='dY (mm)', log=True)
    axes[0].set_title("Vision Position Delta (XY)")
    axes[0].set_xlabel("Delta (mm)")
    axes[0].set_ylabel("Count (Log Scale)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Vision Theta Delta (最关键！)
    # 我们限制显示范围在 -5 到 5 度，看看核心分布
    axes[1].hist(dth_deg, bins=100, color='purple', alpha=0.7, log=True)
    axes[1].set_title("Vision Angle Delta (Theta)")
    axes[1].set_xlabel("Delta (Degrees)")
    # 画出 +/- 180 度的参考线，如果这附近有柱子，说明 Wrap 失败
    axes[1].axvline(x=180, color='red', linestyle='--', label='180 (Flip)')
    axes[1].axvline(x=-180, color='red', linestyle='--', label='-180 (Flip)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    # 动态调整视图：如果有大跳变，就看全貌；如果没有，就看细节
    if np.max(np.abs(dth_deg)) > 20:
        axes[1].set_xlim(-200, 200) # 广角视图查错
    else:
        axes[1].set_xlim(-5, 5)     # 微距视图查优

    # Plot 3: Motor Delta
    axes[2].hist(dq_all, bins=100, color='green', alpha=0.7, log=True)
    axes[2].set_title("Motor Command Delta (dq)")
    axes[2].set_xlabel("Delta (Rad)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "check_histogram_result.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Plot saved to: {os.path.abspath(save_path)}")
    print("💡 Interpret Guide:")
    print("   - Middle Plot (Theta): Should be a narrow spike centered at 0.")
    print("   - If you see bars near +/- 180 or +/- 360, the Angle Wrap failed.")
    print("   - If Max > 10 deg, check the outlier ratio.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', type=str, default='/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_big_clean_smooth.pt')
    args = parser.parse_args()
    
    check_pt_distribution(args.pt)