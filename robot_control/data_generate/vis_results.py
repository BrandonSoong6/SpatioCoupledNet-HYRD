import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import glob

# ==========================================
# 1. 物理参数配置 (来自 Loom Pro)
# ==========================================
LIMIT_MIN, LIMIT_MAX = 10.0, 160.0
CPL_OFFSET, CPL_COEFF, CPL_EXP = 10.0, 0.17, 1.1

# ==========================================
# 2. 核心几何算法 (复用你的代码)
# ==========================================
def get_dynamic_bounds_vec(q_p):
    """计算动态边界 (q_p 是主导关节，计算从动关节的范围)"""
    zero_clip = np.maximum(0, q_p - CPL_OFFSET)
    d_min = CPL_OFFSET + CPL_COEFF * np.power(zero_clip, CPL_EXP)
    d_max = CPL_OFFSET + np.power(zero_clip / CPL_COEFF, 1.0/CPL_EXP)
    return np.maximum(LIMIT_MIN, d_min), np.minimum(LIMIT_MAX, d_max)

# ==========================================
# 3. 绘图辅助函数
# ==========================================
def plot_phase_trajectory(ax, q1_data, q2_data, style, label=None, cmap_name=None):
    """
    绘制带颜色渐变的轨迹，显示时间流逝方向
    """
    if cmap_name:
        # 创建彩色线段 (表示时间进度)
        points = np.array([q1_data, q2_data]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 创建颜色映射
        norm = plt.Normalize(0, len(q1_data))
        lc = LineCollection(segments, cmap=cmap_name, norm=norm, alpha=0.8)
        lc.set_array(np.arange(len(q1_data)))
        lc.set_linewidth(1.5)
        ax.add_collection(lc)
        # 添加一个不可见的点用于生成图例
        ax.plot([], [], color=plt.get_cmap(cmap_name)(0.8), label=label)
    else:
        # 普通线条
        ax.plot(q1_data, q2_data, style, linewidth=1.0, alpha=0.6, label=label)

def calculate_rmse(tgt, act):
    return np.sqrt(np.mean((tgt - act)**2))

# ==========================================
# 4. 主处理逻辑
# ==========================================
def visualize_robot_log(file_path):
    print(f"📂 Loading: {file_path}")
    data = np.load(file_path)
    
    # 按照 RobustRecorder 的数据结构解析
    # [0:t, 1:t_m, 2:t_v, 3-12:tgt, 13-22:act, ...]
    
    # 提取目标 (Target) 和 实际 (Actual)
    # 形状都是 (N, 10)
    targets = data[:, 3:13]  
    actuals = data[:, 13:23]
    
    # 创建画布：1行5列，或者 2行3列 (为了看得清楚，建议 1行5列长图)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)
    fig.suptitle(f"Loom Pro Trajectory Tracking Analysis\nFile: {os.path.basename(file_path)}", fontsize=16)

    # 预计算边界区域 (背景)
    bx = np.linspace(LIMIT_MIN, LIMIT_MAX, 300)
    b_mn, b_mx = get_dynamic_bounds_vec(bx)

    # 循环绘制 5 个模组
    # 假设 ID 排列是 [1,2], [3,4], [5,6], [7,8], [9,10]
    module_names = ["Mod 1 (Tip)", "Mod 2", "Mod 3", "Mod 4", "Mod 5 (Base)"] # 根据你的Recorder ID顺序调整
    
    for i in range(5):
        ax = axes[i]
        idx_q1 = i * 2
        idx_q2 = i * 2 + 1
        
        # 1. 绘制可行域 (Feasible Region)
        ax.fill_between(bx, b_mn, b_mx, color='#E0E0E0', label='Constraint' if i==0 else "")
        ax.plot(bx, b_mn, 'k-', lw=0.5, alpha=0.3)
        ax.plot(bx, b_mx, 'k-', lw=0.5, alpha=0.3)
        
        # 2. 提取数据
        tgt_q1, tgt_q2 = targets[:, idx_q1], targets[:, idx_q2]
        act_q1, act_q2 = actuals[:, idx_q1], actuals[:, idx_q2]
        
        # 3. 计算误差指标 (RMSE) - 综合两个关节
        rmse_q1 = calculate_rmse(tgt_q1, act_q1)
        rmse_q2 = calculate_rmse(tgt_q2, act_q2)
        avg_rmse = (rmse_q1 + rmse_q2) / 2.0
        
        # 4. 绘制轨迹
        # Target: 蓝色虚线
        ax.plot(tgt_q1, tgt_q2, 'b--', lw=1, alpha=0.4, label='Ref')
        
        # Actual: 红色渐变实线 (深红表示结束，浅红表示开始)
        plot_phase_trajectory(ax, act_q1, act_q2, style='-', label='Act', cmap_name='autumn_r')

        # 5. 格式化图表
        ax.set_title(f"{module_names[i]}\nRMSE: {avg_rmse:.2f}", fontsize=12, fontweight='bold')
        ax.set_xlabel(f"Joint {idx_q1+1} (deg)")
        if i == 0:
            ax.set_ylabel(f"Joint {idx_q2+1} (deg)")
        
        ax.set_xlim(0, 170)
        ax.set_ylim(0, 170)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal')
        
        # 在第一张图显示图例
        if i == 0:
            ax.legend(loc='upper left', fontsize='small', framealpha=0.9)

    # 保存结果
    save_path = file_path.replace('.npy', '_phase_plot.png')
    plt.savefig(save_path, dpi=150)
    print(f"✅ Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    file_path = '/home/brandon/brandon/hyrd_robot/training_data/test_001.npy'
    visualize_robot_log(file_path)