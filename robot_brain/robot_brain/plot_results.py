import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

def analyze_training(exp_id, root_path="/home/brandon/brandon/hyrd_robot/lifelong_data/experiments"):
    # 1. 定位路径
    exp_dir = os.path.join(root_path, f"{int(exp_id):03d}")
    log_path = os.path.join(exp_dir, "log.json")
    
    if not os.path.exists(log_path):
        print(f"❌ 找不到日志文件: {log_path}")
        return

    # 2. 读取数据
    with open(log_path, 'r') as f:
        history = json.load(f)

    epochs = history['epoch']
    
    # 3. 找出 Best Epoch (以 Val RMSE 最小为准)
    val_rmses = np.array(history['val_rmse'])
    best_idx = np.argmin(val_rmses)
    
    print("="*50)
    print(f"🚀 Experiment {exp_id} - Best Performance Analysis")
    print("="*50)
    print(f"🏆 Best Epoch: {history['epoch'][best_idx]}")
    print(f"📉 Min Val RMSE: {history['val_rmse'][best_idx]:.5f}")
    print(f"🧪 Physics Baseline: {history['phy_rmse'][best_idx]:.5f}")
    
    # 计算相对于物理基准的提升
    improvement = (history['phy_rmse'][best_idx] - history['val_rmse'][best_idx]) / history['phy_rmse'][best_idx] * 100
    print(f"📈 Improvement over Physics: {improvement:+.2f}%")
    
    if 'val_r2' in history and history['val_r2']:
        print(f"📊 R^2 Score at Best: {history['val_r2'][best_idx]:.4f}")
    
    print(f"💰 Training RMSE at Best: {history['train_rmse'][best_idx]:.5f}")
    print(f"⚡ Learning Rate: {history['lr'][best_idx]:.2e}")
    print("="*50)

    # 4. 绘图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.suptitle(f"Training Progress - Experiment {exp_id}", fontsize=16)

    # Subplot 1: RMSE Comparison
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_rmse'], label='Train RMSE', alpha=0.6)
    ax1.plot(epochs, history['val_rmse'], label='Val RMSE', linewidth=2)
    ax1.plot(epochs, history['phy_rmse'], 'r--', label='Physics Baseline', alpha=0.5)
    ax1.axvline(history['epoch'][best_idx], color='g', linestyle=':', label='Best')
    ax1.set_title("RMSE Convergence ($\downarrow$)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("RMSE (mm/rad)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Hybrid Loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_loss'], color='orange', label='Val Loss')
    ax2.set_title("Hybrid Loss (Target Optimization)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss Value")
    ax2.grid(True, alpha=0.3)

    # Subplot 3: R2 Score
    ax3 = axes[1, 0]
    if 'val_r2' in history and history['val_r2']:
        ax3.plot(epochs, history['val_r2'], color='green', label='Val $R^2$')
        ax3.set_ylim(0, 1.05)
        ax3.set_title(r"Model Fitting Capability ($R^2 \rightarrow 1$)")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("$R^2$ Score")
    else:
        ax3.text(0.5, 0.5, 'R2 Data Not Available', ha='center')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Learning Rate
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['lr'], color='purple')
    ax4.set_yscale('log')
    ax4.set_title("Learning Rate Schedule (Log Scale)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("LR")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图片到实验目录
    save_fig_path = os.path.join(exp_dir, f"summary_plot_{exp_id}.png")
    plt.savefig(save_fig_path)
    print(f"🖼️  分析图表已保存至: {save_fig_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True, help="Experiment ID (e.g., 015)")
    args = parser.parse_args()
    
    analyze_training(args.id)