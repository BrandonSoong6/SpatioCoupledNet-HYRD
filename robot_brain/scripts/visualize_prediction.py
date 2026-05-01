#!/usr/bin/env python3
# File: scripts/visualize_prediction.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from torch.utils.data import TensorDataset

# 引入配置和模型定义
from robot_brain.core.BigTrain_config import BigTrainConfig as CFG
from robot_brain.core.trainer import PhysicsGatedGNK

# 设置绘图风格，清晰一点
plt.style.use('ggplot')

def load_random_batch_data():
    """加载一个随机 Batch 的完整数据"""
    print(f"🔍 Scanning data in {CFG.DATA_ROOT}...")
    all_batch_dirs = sorted(glob.glob(os.path.join(CFG.DATA_ROOT, "batch_*")))
    
    if not all_batch_dirs:
        raise FileNotFoundError("No batch data found!")
        
    # 随机选一个 Batch
    target_dir = random.choice(all_batch_dirs)
    print(f"🎲 Selected Batch: {os.path.basename(target_dir)}")
    
    pt_path = os.path.join(target_dir, "train_data.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"train_data.pt missing in {target_dir}")
        
    data = torch.load(pt_path, map_location='cpu')
    return data, data['scalers']

def visualize():
    device = CFG.DEVICE
    
    # 1. 加载数据
    data, scalers = load_random_batch_data()
    
    # 2. 准备模型
    print("🧠 Loading Model...")
    model = PhysicsGatedGNK(scalers=scalers).to(device)
    model_path = CFG.get_model_path()
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}. Please train first.")
        return

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. 截取连续片段 (Contiguous Segment)
    # 我们取中间的一段，避开开头结尾可能的不稳定
    total_len = len(data['inputs_state'])
    seq_len = 1000
    
    if total_len > seq_len:
        start_idx = random.randint(0, total_len - seq_len)
        end_idx = start_idx + seq_len
        indices = np.arange(start_idx, end_idx)
        print(f"✂️ Slicing continuous segment: {start_idx} -> {end_idx}")
    else:
        indices = np.arange(total_len)
        print(f"⚠️ Batch size ({total_len}) < 1000, using all data.")
    
    # 提取 Tensor 并转到 Device
    s_n = data['inputs_state'][indices].to(device)
    a_n = data['inputs_action'][indices].to(device)
    a_p = data['inputs_action_phys'][indices].to(device)
    j_p = data['j_phys'][indices].to(device)
    tgt = data['targets'][indices].to(device)
    
    # 4. 推理 (Inference)
    with torch.no_grad():
        # Model Prediction (Hybrid)
        pred_norm, alpha = model(s_n, a_n, a_p, j_p)
        
        # Physics Prediction (Raw Jacobian)
        # J * dq = dx (Raw Physics)
        dx_phys_raw = torch.bmm(j_p, a_p.unsqueeze(-1)).squeeze(-1)
    
    # 5. 去归一化 (De-normalize)
    tgt_mean = scalers['target_mean'].to(device)
    tgt_std = scalers['target_std'].to(device)
    
    # Model: Norm -> Real
    pred_real = pred_norm * tgt_std + tgt_mean
    # GT: Norm -> Real
    tgt_real = tgt * tgt_std + tgt_mean
    # Physics: 已经是 Real Scale
    phys_real = dx_phys_raw 
    
    # 转为 Numpy CPU
    pred_np = pred_real.cpu().numpy()
    tgt_np = tgt_real.cpu().numpy()
    phys_np = phys_real.cpu().numpy()
    alpha_np = alpha.cpu().numpy()
    
    # 6. 数据切片 (最后 4 维: Tip X, Y, Sin, Cos)
    # 对应机器人第5节 (Tip) 的 Delta 变化量
    
    # [True Ground Truth]
    gt_dx = tgt_np[:, -4]
    gt_dy = tgt_np[:, -3]
    gt_dsin = tgt_np[:, -2]
    gt_dcos = tgt_np[:, -1]
    
    # [Model Prediction]
    md_dx = pred_np[:, -4]
    md_dy = pred_np[:, -3]
    md_dsin = pred_np[:, -2]
    md_dcos = pred_np[:, -1]
    
    # [Physics Prediction]
    ph_dx = phys_np[:, -4]
    ph_dy = phys_np[:, -3]
    ph_dsin = phys_np[:, -2]
    ph_dcos = phys_np[:, -1]
    
    # 7. 绘图 (4行1列)
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    x_axis = np.arange(len(indices))
    
    # 辅助绘图函数
    def plot_curve(ax, gt, phys, model, title, unit):
        ax.plot(x_axis, gt, 'k-', label='Ground Truth (Real)', linewidth=2.0, alpha=0.5)
        ax.plot(x_axis, phys, 'b--', label='Physics (Jacobian)', linewidth=1.5, alpha=0.6)
        ax.plot(x_axis, model, 'r-', label='Model (Hybrid)', linewidth=1.2)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(f"Delta ({unit})")
        ax.legend(loc='upper right', frameon=True)
        # 标出 0 线，方便看正负
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)

    # Plot 1: Delta X (mm)
    plot_curve(axs[0], gt_dx, ph_dx, md_dx, 'Tip Delta X (Position Change)', 'mm')
    
    # Plot 2: Delta Y (mm)
    plot_curve(axs[1], gt_dy, ph_dy, md_dy, 'Tip Delta Y (Position Change)', 'mm')
    
    # Plot 3: Delta Sin (Orientation Component 1)
    plot_curve(axs[2], gt_dsin, ph_dsin, md_dsin, 'Tip Delta Sin (Angle Change Component)', 'raw')
    
    # Plot 4: Delta Cos (Orientation Component 2)
    plot_curve(axs[3], gt_dcos, ph_dcos, md_dcos, 'Tip Delta Cos (Angle Change Component)', 'raw')

    # 添加 Alpha 值的颜色条到底部，展示模型什么时候介入
    # 这里用散点图模拟一条彩色带
    # ax_alpha = axs[3].twinx()
    # ax_alpha.fill_between(x_axis, 0, alpha_np.flatten(), color='green', alpha=0.1, label='Alpha Confidence')
    # ax_alpha.set_ylim(0, 1)
    # ax_alpha.set_yticks([]) # 隐藏坐标轴
    
    plt.xlabel(f"Time Step (10ms per step) - Batch Segment", fontsize=10)
    plt.suptitle(f'Prediction Analysis: Tip (5th Module) Response\n'
                 f'Unit: mm (Position) | Raw (Sin/Cos)', fontsize=14)
    
    plt.tight_layout()
    save_path = "vis_prediction_mm.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Visualization saved to: {os.path.abspath(save_path)}")
    
    # 打印一些统计信息
    print("-" * 50)
    print(f"📊 Statistics (Mean Absolute Error on this segment):")
    print(f"Tip Delta X | Phys: {np.mean(np.abs(gt_dx - ph_dx)):.4f} | Model: {np.mean(np.abs(gt_dx - md_dx)):.4f} mm")
    print(f"Tip Delta Y | Phys: {np.mean(np.abs(gt_dy - ph_dy)):.4f} | Model: {np.mean(np.abs(gt_dy - md_dy)):.4f} mm")
    print("-" * 50)

if __name__ == "__main__":
    visualize()