import numpy as np
import os
import matplotlib.pyplot as plt

# ================= 🔧 目标路径 =================
# 指向那个让你怀疑人生的 batch_002
TARGET_BATCH = "/home/brandon/brandon/hyrd_robot/lifelong_data/020401/batch_032"
# ==============================================

def plot_id_angles(batch_dir):
    print(f"📊 正在提取原始角度数据 (单位已确认为：度): {batch_dir}")
    raw_path = os.path.join(batch_dir, "raw_data.npy")
    
    if not os.path.exists(raw_path):
        print("❌ 找不到原始数据文件，请检查路径是否正确。")
        return

    # 加载原始数据
    try:
        data = np.load(raw_path)
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return

    # 视觉数据索引 44-59。ID 的角度列分别是：
    # ID1:46, ID2:49, ID3:52, ID4:55, ID5:58
    angle_cols = [46, 49, 52, 55, 58]
    
    # 直接提取原始数据，不再做任何单位转换
    angles = data[:, angle_cols]
    
    # 创建 5 个子图，看清每一个节的表现
    fig, axes = plt.subplots(5, 1, figsize=(16, 24), sharex=True)
    fig.suptitle(f"Raw Angle Time Series (Direct Degrees) - {os.path.basename(batch_dir)}", fontsize=20, y=0.98)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 默认查看前 3000 帧（30秒），如果想看全量，把 3000 改为 len(angles)
    display_len = min(3000, len(angles))
    frames = np.arange(display_len)

    for i in range(5):
        ax = axes[i]
        # 绘制连续折线
        ax.plot(frames, angles[:display_len, i], 
                color=colors[i], linewidth=1.2, label=f"Visual ID {i+1}")
        
        # 叠加散点，防止采样点太稀疏漏掉瞬时跳变
        ax.scatter(frames, angles[:display_len, i], 
                   color=colors[i], s=2, alpha=0.4)

        # 标注 -180, 0, 180 参考线
        ax.axhline(y=180, color='red', linestyle='--', alpha=0.3, label='180 deg')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
        ax.axhline(y=-180, color='red', linestyle='--', alpha=0.3, label='-180 deg')

        ax.set_ylabel("Angle (Deg)", fontsize=12)
        ax.set_title(f"Visual ID {i+1} Raw Data", fontsize=14, loc='left')
        ax.set_ylim(-210, 210) # 方便观察 +/- 180 附近的跳变
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Frame Index (100Hz)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存图片
    save_path = "angle_analysis_result.png"
    plt.savefig(save_path, dpi=120)
    print(f"\n✅ 分析完成！结果已保存至: {os.path.abspath(save_path)}")
    print("💡 建议：打开图片后，重点看 ID 2 和 ID 3 是否在 180 和 -180 之间反复横跳。")

if __name__ == "__main__":
    plot_id_angles(TARGET_BATCH)