#!/usr/bin/env python3
# File: scripts/visualize_xy_prediction.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# 引入配置和模型定义
from robot_brain.core.BigTrain_config import BigTrainConfig as CFG
from robot_brain.core.trainer import PhysicsGatedGNK

# 尝试设置一个好看的绘图风格
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('ggplot')

def load_mega_dataset():
    """直接加载聚合好的 Mega Dataset"""
    mega_path = os.path.join(CFG.DATA_ROOT, "mega_xy_dataset.pt")
    print(f"🔍 Loading Mega Dataset from: {mega_path}")
    
    if not os.path.exists(mega_path):
        raise FileNotFoundError(f"Dataset not found at {mega_path}. Run generate_xy_dataset.py first.")
        
    data = torch.load(mega_path, map_location='cpu')
    return data, data['scalers']

def visualize():
    device = CFG.DEVICE
    
    # 1. 加载数据
    data, scalers = load_mega_dataset()
    
    # 2. 准备模型 (注意维度修正)
    print("🧠 Loading Model...")
    # === 关键：必须与训练时的维度一致 ===
    model = PhysicsGatedGNK(
        state_dim=20,  # 10q + 10xy
        action_dim=10, # 10dq
        output_dim=10, # 10dx (XY Only)
        scalers=scalers
    ).to(device)
    
    model_path = CFG.get_model_path()
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}. Please train first.")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. 截取连续片段
    # Mega Dataset 是打乱的吗？通常 generate_xy_dataset 生成的是拼接的。
    # 如果是拼接的，我们可以取一段连续的。
    total_len = len(data['inputs_state'])
    seq_len = 1000
    
    if total_len > seq_len:
        # 随机取一段
        start_idx = random.randint(0, total_len - seq_len)
        indices = np.arange(start_idx, start_idx + seq_len)
        print(f"✂️ Slicing continuous segment: {start_idx} -> {start_idx + seq_len}")
    else:
        indices = np.arange(total_len)
    
    # 提取 Tensor
    s_n = data['inputs_state'][indices].to(device)
    a_n = data['inputs_action'][indices].to(device)
    a_p = data['inputs_action_phys'][indices].to(device)
    j_p = data['j_phys'][indices].to(device)
    tgt = data['targets'][indices].to(device)
    
    # 4. 推理
    with torch.no_grad():
        pred_norm, alpha = model(s_n, a_n, a_p, j_p)
        # 物理计算 (J * dq)
        dx_phys_raw = torch.bmm(j_p, a_p.unsqueeze(-1)).squeeze(-1)
    
    # 5. 去归一化
    tgt_mean = scalers['target_mean'].to(device)
    tgt_std = scalers['target_std'].to(device)
    
    pred_real = pred_norm * tgt_std + tgt_mean
    tgt_real = tgt * tgt_std + tgt_mean
    phys_real = dx_phys_raw 
    
    # 转 Numpy
    pred_np = pred_real.cpu().numpy()
    tgt_np = tgt_real.cpu().numpy()
    phys_np = phys_real.cpu().numpy()
    alpha_np = alpha.cpu().numpy()
    
    # 6. 提取末端 (Tip) 数据
    # 新的数据结构是 [x1, y1, x2, y2, ... x5, y5]
    # Tip X 是倒数第2个 (-2), Tip Y 是倒数第1个 (-1)
    
    gt_dx, gt_dy = tgt_np[:, -2], tgt_np[:, -1]
    md_dx, md_dy = pred_np[:, -2], pred_np[:, -1]
    ph_dx, ph_dy = phys_np[:, -2], phys_np[:, -1]
    
    # 单位转换 (如果数据是米，转毫米)
    unit_label = "mm"
    if np.std(gt_dx) < 0.1: # 简单启发式检测
        scale = 1000.0
        print("💡 Detected Unit: Meters. Converting to mm for plot.")
    else:
        scale = 1.0
        
    gt_dx *= scale; gt_dy *= scale
    md_dx *= scale; md_dy *= scale
    ph_dx *= scale; ph_dy *= scale
    
    # 7. 绘图 (3行1列: X, Y, Alpha)
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    x_axis = np.arange(len(indices))
    
    # 通用绘图
    def plot_row(ax, gt, phys, model, title):
        ax.plot(x_axis, gt, 'k-', label='Ground Truth', linewidth=2.0, alpha=0.4)
        ax.plot(x_axis, phys, 'b--', label='Physics (Jacobian)', linewidth=1.5, alpha=0.6)
        ax.plot(x_axis, model, 'r-', label='Hybrid Model', linewidth=1.2)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel(f"Delta ({unit_label})")
        ax.legend(loc='upper right')
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)

    # Subplot 1: X
    plot_row(axs[0], gt_dx, ph_dx, md_dx, 'Tip Delta X (Position)')
    
    # Subplot 2: Y
    plot_row(axs[1], gt_dy, ph_dy, md_dy, 'Tip Delta Y (Position)')
    
    # Subplot 3: Alpha
    axs[2].plot(x_axis, alpha_np, 'g-', linewidth=1.5, label='Alpha (Gate)')
    axs[2].set_title('Network Confidence (Alpha: 1=Physics, 0=Neural)', fontweight='bold', fontsize=12)
    axs[2].set_ylabel('Alpha Value')
    axs[2].set_ylim(0, 1.1)
    axs[2].axhline(0.5, color='gray', linestyle='--')
    axs[2].fill_between(x_axis, 0, alpha_np.flatten(), color='green', alpha=0.1)
    axs[2].legend(loc='lower right')
    axs[2].set_xlabel('Time Step (Sample Index)')

    plt.suptitle(f'XY-Only Model Analysis (Tip Response)\nScale: {unit_label}', fontsize=16)
    plt.tight_layout()
    
    save_path = "vis_xy_prediction.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved to: {os.path.abspath(save_path)}")
    
    # 打印误差统计
    print("-" * 50)
    print(f"📊 Stats (Mean Absolute Error):")
    print(f"   X-Axis | Physics: {np.mean(np.abs(gt_dx - ph_dx)):.4f} | Model: {np.mean(np.abs(gt_dx - md_dx)):.4f}")
    print(f"   Y-Axis | Physics: {np.mean(np.abs(gt_dy - ph_dy)):.4f} | Model: {np.mean(np.abs(gt_dy - md_dy)):.4f}")
    print("-" * 50)

if __name__ == "__main__":
    visualize()