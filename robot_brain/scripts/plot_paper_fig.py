import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ================= 配置区 =================
INPUT_CSV_PATH = '/home/brandon/brandon/hyrd_robot/src/robot_brain/scripts/raw_log.csv'
OUTPUT_DIR = os.path.dirname(INPUT_CSV_PATH)
OUTPUT_FILENAME = 'training_results_paper'

# 论文绘图风格设置
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.figsize": (10, 8),
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def plot_training_curves():
    # 1. 读取数据
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"❌ 找不到文件: {INPUT_CSV_PATH}")
        return
    
    df = pd.read_csv(INPUT_CSV_PATH)
    
    # 找到验证集 Loss 最低的 Epoch
    best_epoch_idx = df['Val_Loss'].idxmin()
    best_epoch = df.loc[best_epoch_idx, 'Epoch']
    min_val_loss = df.loc[best_epoch_idx, 'Val_Loss']
    
    print(f"🏆 Best Model found at Epoch {best_epoch} with Val Loss {min_val_loss:.5f}")

    # 2. 创建画布
    fig, axes = plt.subplots(2, 2, sharex=True)
    (ax_loss, ax_tip), (ax_alpha, ax_lr) = axes

    colors = sns.color_palette("deep")
    
    # 关键修改：使用 .to_numpy() 提取纯数组，避免 Pandas 版本冲突
    epochs = df['Epoch'].to_numpy()
    
    # --- 子图 1: Loss 曲线 ---
    ax_loss.plot(epochs, df['Train_Loss'].to_numpy(), label='Train Loss', color=colors[0], linestyle='-')
    ax_loss.plot(epochs, df['Val_Loss'].to_numpy(), label='Val Loss', color=colors[1], linestyle='--')
    
    # 标记最佳点
    ax_loss.scatter(best_epoch, min_val_loss, color='red', zorder=5, s=50, label='Best Checkpoint')
    ax_loss.annotate(f'Best: {min_val_loss:.3f}', xy=(best_epoch, min_val_loss), 
                     xytext=(best_epoch + 20, min_val_loss + 0.05),
                     arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.5))
    
    ax_loss.set_ylabel('Loss (MSE)')
    ax_loss.set_title('Training & Validation Loss')
    ax_loss.legend(loc='upper right', frameon=False)

    # --- 子图 2: Tip Error ---
    ax_tip.plot(epochs, df['Tip_X_mm'].to_numpy(), label='Tip-X Error', color=colors[2])
    ax_tip.plot(epochs, df['Tip_Y_mm'].to_numpy(), label='Tip-Y Error', color=colors[3])
    ax_tip.set_ylabel('Tip Error (mm)')
    ax_tip.set_title('Tip Position Error')
    ax_tip.legend(loc='upper right', frameon=False)
    
    # --- 子图 3: Alpha ---
    ax_alpha.plot(epochs, df['Alpha'].to_numpy(), label='Alpha', color=colors[4], linestyle='-.')
    ax_alpha.set_xlabel('Epoch')
    ax_alpha.set_ylabel('Alpha Value')
    ax_alpha.set_title('Alpha Evolution')
    
    # --- 子图 4: Learning Rate ---
    ax_lr.plot(epochs, df['LR'].to_numpy(), label='Learning Rate', color=colors[5])
    ax_lr.set_xlabel('Epoch')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.set_title('Learning Rate Schedule')
    ax_lr.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    # 3. 保存
    plt.tight_layout()
    
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME}.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ 绘图完成！")
    print(f"   📄 PDF: {pdf_path}")
    print(f"   🖼️ PNG: {png_path}")

if __name__ == "__main__":
    # 如果在服务器无头模式下运行报错，请取消下面这行的注释
    plt.switch_backend('Agg') 
    plot_training_curves()