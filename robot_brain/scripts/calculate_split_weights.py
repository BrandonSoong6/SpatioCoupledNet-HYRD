import torch
import numpy as np
import argparse
import os

def analyze_split_weights(pt_path):
    print(f"📂 Loading dataset: {pt_path}")
    if not os.path.exists(pt_path):
        print("❌ Error: File not found.")
        return

    data = torch.load(pt_path, map_location='cpu')
    
    if 'tgt_delta' in data['data']:
        deltas = data['data']['tgt_delta']
    else:
        print("⚠️ 'tgt_delta' not found.")
        return

    # [N * Nodes, 3]
    flat_deltas = deltas.view(-1, 3).numpy()
    
    # 分离分量 (取绝对值计算均值，平方计算RMS)
    dx = flat_deltas[:, 0]
    dy = flat_deltas[:, 1]
    dth = flat_deltas[:, 2]
    
    # 过滤静止死区 (避免 0 拉低统计)
    active_mask = (np.abs(dx) > 1e-6) | (np.abs(dy) > 1e-6) | (np.abs(dth) > 1e-6)
    
    dx_active = dx[active_mask]
    dy_active = dy[active_mask]
    dth_active = dth[active_mask]
    
    print(f"📊 Analyzing {np.sum(active_mask)} active samples...")

    # ================= 1. 计算 RMS (均方根) =================
    # RMS^2 正比于 MSE Loss 的期望值
    rms_dx = np.sqrt(np.mean(dx_active**2))
    rms_dy = np.sqrt(np.mean(dy_active**2))
    rms_dth = np.sqrt(np.mean(dth_active**2))

    print("\n🔍 Signal Magnitude (RMS):")
    print("-" * 40)
    print(f"RMS_dx  : {rms_dx:.6f} mm")
    print(f"RMS_dy  : {rms_dy:.6f} mm")
    print(f"RMS_dth : {rms_dth:.6f} rad")
    
    # 检查 X/Y 不对称性
    xy_ratio = rms_dx / (rms_dy + 1e-8)
    print("-" * 40)
    print(f"⚖️  Asymmetry Check (X / Y): {xy_ratio:.2f}")
    if xy_ratio > 1.2:
        print("   👉 X varies significantly more than Y (X dominates)")
    elif xy_ratio < 0.8:
        print("   👉 Y varies significantly more than X (Y dominates)")
    else:
        print("   👉 X and Y are relatively balanced")

    # ================= 2. 计算建议权重 =================
    # 目标：Weight * RMS^2 ≈ Constant
    # 我们以 dy 为基准 (设 W_y = 1.0)，因为通常 Y 轴较稳定
    # 如果 x 波动大，Weights 应该小，以防止 Loss 被 x 噪声主导？
    # 不，通常为了让各分量"公平竞争"，我们会把它们缩放到同一量级。
    # 即：W_x * RMS_x^2 = W_y * RMS_y^2 = W_th * RMS_th^2
    
    # 以 RMS_dy 为基准 (Standard)
    base_energy = rms_dy**2
    
    w_y = 1.0
    w_x = base_energy / (rms_dx**2 + 1e-8)
    w_th = base_energy / (rms_dth**2 + 1e-8)

    print("\n🚀 FINAL SUGGESTED WEIGHTS (Normalized to Y=1.0):")
    print("=" * 50)
    print(f"W_X     = {w_x:.4f}")
    print(f"W_Y     = {w_y:.4f}")
    print(f"W_THETA = {w_th:.1f}")
    print("=" * 50)
    
    # 验证平衡后的贡献
    loss_contrib_x = w_x * (rms_dx**2)
    loss_contrib_y = w_y * (rms_dy**2)
    loss_contrib_th = w_th * (rms_dth**2)
    
    print(f"\nExpected Loss Contribution (Should be equal):")
    print(f"Lx: {loss_contrib_x:.5f} | Ly: {loss_contrib_y:.5f} | Lth: {loss_contrib_th:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, 
                        default='/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_smooth_strided.pt')
    args = parser.parse_args()
    
    analyze_split_weights(args.path)