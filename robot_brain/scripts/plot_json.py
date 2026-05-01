import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_academic_mse_analysis(log_path):
    with open(log_path, 'r') as f:
        h = json.load(f)
    
    epochs = h['epoch']
    # 提取 RMSE，如果不存在则从 MSE 计算
    train_rmse = h.get('train_rmse', np.sqrt(h['train_mse']))
    val_rmse = h.get('val_rmse', np.sqrt(h['val_mse']))
    
    # 设置 IEEE/ACM 风格
    plt.style.use('seaborn-v0_8-paper') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # --- 左图：RMSE 核心收敛曲线 (对数坐标) ---
    # 机器人位姿预测中，后期细微的 mm 级提升才是核心竞争力
    ax1.plot(epochs, train_rmse, color='#4C72B0', label='Training', lw=1.5, alpha=0.8)
    ax1.plot(epochs, val_rmse, color='#DD8452', label='Validation', lw=1.5)
    
    ax1.set_yscale('log') # 科研论文常用对数轴展示误差
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('RMSE (mm)', fontweight='bold')
    ax1.set_title('Global Convergence Profile', fontsize=12, pad=15)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(frameon=True)

    # --- 右图：泛化间隙与收敛稳定性 (Generalization Gap) ---
    # 绘制 (Val RMSE - Train RMSE)，反映模型是否过拟合
    gap = np.array(val_rmse) - np.array(train_rmse)
    ax2.fill_between(epochs, 0, gap, color='#55A868', alpha=0.3, label='Generalization Gap')
    ax2.plot(epochs, gap, color='#55A868', lw=1)
    
    # 计算移动平均以展示趋势平滑线
    if len(gap) > 20:
        smooth_gap = np.convolve(gap, np.ones(10)/10, mode='valid')
        ax2.plot(epochs[9:], smooth_gap, color='#C44E52', lw=2, label='Gap Trend')

    ax2.set_xlabel('Epochs', fontweight='bold')
    ax2.set_ylabel('$\Delta$ RMSE (Val - Train)', fontweight='bold')
    ax2.set_title('Generalization & Stability Analysis', fontsize=12, pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('model_performance_analysis.pdf') # 建议保存为 PDF 矢量图用于论文
    plt.show()

# 运行
plot_academic_mse_analysis("/home/brandon/brandon/hyrd_robot/lifelong_data/models/training_log.json")