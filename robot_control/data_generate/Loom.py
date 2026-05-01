"""
Project: Loom (织机) - Pro Edition
Description: Sobol Sampling with Edge Projection + Inverted Weights Greedy Search.
Author: Brandon (Song Yuli) @ NUS
Date: 2026-01-29
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.spatial.distance import cdist
import seaborn as sns
import os
import time

# === 1. 参数与配置 ===
OUTPUT_DIR = "Loom_Pro_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 物理参数
LIMIT_MIN, LIMIT_MAX = 10.0, 160.0
CPL_OFFSET, CPL_COEFF, CPL_EXP = 10.0, 0.17, 1.1

# 采样参数
TARGET_POINTS = 2000
SOBOL_POWER = 15 # 稍微加大一点采样量，因为我们要进行投影筛选
RELAX_MARGIN = 3.0 # 关键修改：允许超出的宽容度 (mm)

# 舵机权重 (Base -> Tip: Mod1 -> Mod5)
# 修正：Base (Mod 1) 是 9-10号舵机，最累，权重最大(8.0)
# Tip (Mod 5) 是 1-2号舵机，最轻，权重最小(1.0)
MODULE_WEIGHTS = np.array([8.0, 4.0, 2.5, 1.5, 1.0]) 

# === 2. 几何核心 (向量化投影版) ===
def get_dynamic_bounds_vec(q_p):
    """计算动态边界"""
    zero_clip = np.maximum(0, q_p - CPL_OFFSET)
    d_min = CPL_OFFSET + CPL_COEFF * np.power(zero_clip, CPL_EXP)
    d_max = CPL_OFFSET + np.power(zero_clip / CPL_COEFF, 1.0/CPL_EXP)
    return np.maximum(LIMIT_MIN, d_min), np.minimum(LIMIT_MAX, d_max)

def project_points_to_feasible(points_10d):
    """
    核心升级：将点投影到可行域内。
    逻辑：如果点在 [Min-3mm, Max+3mm] 之间，强制 Clip 到 [Min, Max]。
    这样可以显著增加边界上的点密度。
    """
    # 迭代 3 次以解决耦合约束的相互依赖
    projected = points_10d.copy()
    
    for _ in range(3):
        modules = projected.reshape(-1, 5, 2)
        q1 = modules[:, :, 0]
        q2 = modules[:, :, 1]
        
        # 1. 先满足全局限位 [10, 160]
        q1 = np.clip(q1, LIMIT_MIN, LIMIT_MAX)
        q2 = np.clip(q2, LIMIT_MIN, LIMIT_MAX)
        
        # 2. q2 限制 q1
        mn1, mx1 = get_dynamic_bounds_vec(q2)
        q1 = np.clip(q1, mn1, mx1)
        
        # 3. q1 限制 q2
        mn2, mx2 = get_dynamic_bounds_vec(q1)
        q2 = np.clip(q2, mn2, mx2)
        
        projected = np.stack([q1, q2], axis=2).reshape(-1, 10)
        
    return projected

def check_feasibility_relaxed(points_10d, margin=0.0):
    """
    宽容度检查：检查点是否在 '可行域 + Margin' 的范围内
    """
    modules = points_10d.reshape(-1, 5, 2)
    q1, q2 = modules[:, :, 0], modules[:, :, 1]
    
    # 放宽的全局限位
    valid_range = (q1 >= LIMIT_MIN - margin) & (q1 <= LIMIT_MAX + margin) & \
                  (q2 >= LIMIT_MIN - margin) & (q2 <= LIMIT_MAX + margin)
    
    # 放宽的动态约束
    mn2, mx2 = get_dynamic_bounds_vec(q1)
    valid_cpl_1 = (q2 >= mn2 - margin) & (q2 <= mx2 + margin)
    
    mn1, mx1 = get_dynamic_bounds_vec(q2)
    valid_cpl_2 = (q1 >= mn1 - margin) & (q1 <= mx1 + margin)
    
    return np.all(valid_range & valid_cpl_1 & valid_cpl_2, axis=1)

# === 3. Sobol 采样 (带投影) ===
def generate_enhanced_sobol(target_n, power=15, margin=3.0):
    print(f"1. Generating {2**power} Sobol candidates (Relaxed Margin: {margin}mm)...")
    sampler = qmc.Sobol(d=10, scramble=True)
    
    # 生成 [0, 1]
    sample_norm = sampler.random_base2(m=power)
    
    # 映射到物理空间，甚至可以稍微映射大一点点，让更多点落在外面
    # 这里我们映射到 [LIMIT_MIN - margin, LIMIT_MAX + margin]
    phys_min = LIMIT_MIN - margin
    phys_max = LIMIT_MAX + margin
    sample_phys = phys_min + sample_norm * (phys_max - phys_min)
    
    # 1. 宽容筛选：保留在 3mm 缓冲带里的所有点
    mask_relaxed = check_feasibility_relaxed(sample_phys, margin=margin)
    candidates = sample_phys[mask_relaxed]
    print(f"   In Relaxed Region: {len(candidates)}")
    
    # 2. 投影：把这些点按死在墙上
    projected_points = project_points_to_feasible(candidates)
    
    # 3. 最终确认：确保投影后的点真的在合法区域内 (通常都在，但以防万一)
    # 使用严格检查 margin=0
    mask_strict = check_feasibility_relaxed(projected_points, margin=1e-5)
    final_valid = projected_points[mask_strict]
    
    print(f"   After Projection: {len(final_valid)} valid points.")
    
    if len(final_valid) < target_n:
        raise ValueError("Not enough points after projection. Increase SOBOL_POWER.")
        
    return final_valid[:target_n]

# === 4. 手写加权贪婪算法 (Instant Solver) ===
def solve_greedy_weighted_nn(points):
    N = len(points)
    print(f"2. Computing Weighted Matrix for {N} points...")
    print(f"   Weights applied (Base->Tip): {MODULE_WEIGHTS}")
    print("   (Note: High weight = Hard to move = Low Average Step Size)")
    
    sqrt_weights = np.sqrt(MODULE_WEIGHTS)
    scale_factors = np.repeat(sqrt_weights, 2)
    weighted_points = points * scale_factors
    
    dist_matrix = cdist(weighted_points, weighted_points, metric='euclidean')
    np.fill_diagonal(dist_matrix, np.inf)
    
    print("3. Running Manual Greedy Search...")
    t_start = time.time()
    
    current_idx = 0 
    path_indices = [current_idx]
    visited_mask = np.zeros(N, dtype=bool)
    visited_mask[current_idx] = True
    
    for _ in range(N - 1):
        dists = dist_matrix[current_idx]
        dists[visited_mask] = np.inf
        next_idx = np.argmin(dists)
        path_indices.append(next_idx)
        visited_mask[next_idx] = True
        current_idx = next_idx
        
    print(f"   [Done] Sorted in {time.time() - t_start:.4f}s")
    return points[path_indices]

# === 5. 分析与保存 ===
def analyze_and_save(ordered_points):
    print("4. Saving and Plotting...")
    final_data = ordered_points.reshape(-1, 5, 2)
    
    np.save(os.path.join(OUTPUT_DIR, "loom_pro_10d.npy"), final_data)
    np.savetxt(os.path.join(OUTPUT_DIR, "loom_pro_10d.csv"), ordered_points, delimiter=",", fmt="%.4f",
               header="q1_m1,q2_m1,q1_m2,q2_m2,q1_m3,q2_m3,q1_m4,q2_m4,q1_m5,q2_m5")

    # Fig 1: Path with Edge Highlight
    plt.figure(figsize=(6, 6))
    bx = np.linspace(LIMIT_MIN, LIMIT_MAX, 300)
    b_mn, b_mx = get_dynamic_bounds_vec(bx)
    plt.fill_between(bx, b_mn, b_mx, color='#EEEEEE', label='Feasible')
    
    # 特别标记出边界上的点 (距离边界 < 0.1mm)
    # 简单算一下到下边界的距离作为示意
    q1, q2 = final_data[:, 0, 0], final_data[:, 0, 1]
    
    plt.plot(q1, q2, 'k-', lw=0.1, alpha=0.3)
    plt.scatter(q1, q2, c=np.arange(len(q1)), cmap='turbo', s=3, label='Inner')
    
    plt.title("Loom Pro Path (With Edge Projection)")
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig1_Pro_Path.png"), bbox_inches='tight')
    plt.close()
    
    # Fig 2: Energy Cost
    diffs = np.abs(np.diff(ordered_points, axis=0)).reshape(-1, 5, 2).sum(axis=2)
    avg_disp = diffs.mean(axis=0)
    
    plt.figure(figsize=(8, 4))
    # 这里的标签顺序要对应 weights 的顺序
    bars = plt.bar(["Mod 1\n(Base/Heavy)", "Mod 2", "Mod 3", "Mod 4", "Mod 5\n(Tip/Light)"], avg_disp, 
                   color=sns.color_palette("magma_r", 5)) # magma_r 颜色从深到浅
    plt.title("Average Step Size (Corrected Weights)")
    plt.ylabel("Mean Step Size (mm)")
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold')
        
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig2_Pro_Cost.png"), bbox_inches='tight')
    plt.close()

# === 主函数 ===
def main():
    t0 = time.time()
    print(f"=== Loom Pro Edition (Edge Boost + Corrected Weights) ===")
    
    points = generate_enhanced_sobol(TARGET_POINTS, SOBOL_POWER, RELAX_MARGIN)
    ordered = solve_greedy_weighted_nn(points)
    analyze_and_save(ordered)
    
    print(f"\nTotal Time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()