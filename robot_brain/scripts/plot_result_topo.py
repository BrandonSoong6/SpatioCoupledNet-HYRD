import json
import matplotlib.pyplot as plt
import os

# 配置
LOG_PATH = "/home/brandon/brandon/hyrd_robot/lifelong_data/models/training_log.json"

def plot_training(log_path):
    if not os.path.exists(log_path):
        print("❌ Log file not found!")
        return

    with open(log_path, 'r') as f:
        history = json.load(f)

    epochs = history['epoch']
    
    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Topo-Impulse Training Analysis', fontsize=16)

    # 1. MSE Curve (最重要)
    ax = axes[0, 0]
    ax.plot(epochs, history['train_mse'], label='Train MSE', alpha=0.7)
    ax.plot(epochs, history['val_mse'], label='Val MSE', linewidth=2)
    ax.set_title('Global Position Error (MSE)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (mm²)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    # 这里的 scale 可以根据初期 loss 大小改成 'log'
    # ax.set_yscale('log') 

    # 2. Max Error (安全边界)
    ax = axes[0, 1]
    ax.plot(epochs, history['val_max_err'], color='red', label='Max Error (Val)')
    ax.set_title('Safety Bound: Max Deviation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (mm)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Learning Rate
    ax = axes[1, 0]
    ax.plot(epochs, history['lr'], color='orange', linestyle='--')
    ax.set_title('Learning Rate Schedule')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.grid(True, alpha=0.3)

    # 4. Composite Loss (优化器视角)
    ax = axes[1, 1]
    ax.plot(epochs, history['train_loss'], label='Train Loss')
    ax.plot(epochs, history['val_loss'], label='Val Loss')
    ax.set_title('Composite Loss (Optimization)')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(log_path.replace('.json', '.png'), dpi=300)
    print(f"📈 Plot saved to {log_path.replace('.json', '.png')}")
    plt.show()

if __name__ == "__main__":
    plot_training(LOG_PATH)