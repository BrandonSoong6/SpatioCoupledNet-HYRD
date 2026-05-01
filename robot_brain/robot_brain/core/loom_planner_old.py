# File: ~/brandon/hyrd_robot/src/robot_brain/robot_brain/core/loom_planner.py
import numpy as np
import os
import time
from scipy.stats import qmc
from scipy.spatial.distance import cdist

class LoomPlanner:
    def __init__(self):
        # === 1. 物理参数配置 (完全复刻 Loom.py) ===
        self.LIMIT_MIN = 10.0
        self.LIMIT_MAX = 160.0
        self.CPL_OFFSET = 10.0
        self.CPL_COEFF = 0.17
        self.CPL_EXP = 1.1
        
        # 权重 (Base -> Tip: Mod 1 -> Mod 5)
        self.MODULE_WEIGHTS = np.array([8.0, 4.0, 2.5, 1.5, 1.0]) 

    def get_dynamic_bounds_vec(self, q_p):
        """核心几何算法 (保持不变)"""
        zero_clip = np.maximum(0, q_p - self.CPL_OFFSET)
        d_min = self.CPL_OFFSET + self.CPL_COEFF * np.power(zero_clip, self.CPL_EXP)
        d_max = self.CPL_OFFSET + np.power(zero_clip / self.CPL_COEFF, 1.0/self.CPL_EXP)
        return np.maximum(self.LIMIT_MIN, d_min), np.minimum(self.LIMIT_MAX, d_max)

    def project_points_to_feasible(self, points_10d):
        """投影逻辑 (保持不变)"""
        projected = points_10d.copy()
        for _ in range(3):
            modules = projected.reshape(-1, 5, 2)
            q1 = modules[:, :, 0]
            q2 = modules[:, :, 1]
            
            # 1. 全局限位
            q1 = np.clip(q1, self.LIMIT_MIN, self.LIMIT_MAX)
            q2 = np.clip(q2, self.LIMIT_MIN, self.LIMIT_MAX)
            
            # 2. q2 限制 q1
            mn1, mx1 = self.get_dynamic_bounds_vec(q2)
            q1 = np.clip(q1, mn1, mx1)
            
            # 3. q1 限制 q2
            mn2, mx2 = self.get_dynamic_bounds_vec(q1)
            q2 = np.clip(q2, mn2, mx2)
            
            projected = np.stack([q1, q2], axis=2).reshape(-1, 10)
        return projected

    def check_feasibility_relaxed(self, points_10d, margin=0.0):
        """宽容度检查 (保持不变)"""
        modules = points_10d.reshape(-1, 5, 2)
        q1, q2 = modules[:, :, 0], modules[:, :, 1]
        
        valid_range = (q1 >= self.LIMIT_MIN - margin) & (q1 <= self.LIMIT_MAX + margin) & \
                      (q2 >= self.LIMIT_MIN - margin) & (q2 <= self.LIMIT_MAX + margin)
        
        mn2, mx2 = self.get_dynamic_bounds_vec(q1)
        valid_cpl_1 = (q2 >= mn2 - margin) & (q2 <= mx2 + margin)
        
        mn1, mx1 = self.get_dynamic_bounds_vec(q2)
        valid_cpl_2 = (q1 >= mn1 - margin) & (q1 <= mx1 + margin)
        
        return np.all(valid_range & valid_cpl_1 & valid_cpl_2, axis=1)

    def solve_greedy_weighted_nn(self, points):
        """贪婪排序 (保持不变)"""
        N = len(points)
        if N == 0: return points
        
        sqrt_weights = np.sqrt(self.MODULE_WEIGHTS)
        scale_factors = np.repeat(sqrt_weights, 2)
        weighted_points = points * scale_factors
        
        dist_matrix = cdist(weighted_points, weighted_points, metric='euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        
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
            
        return points[path_indices]

    def generate_trajectory(self, n_points=150, seed=None, save_path=None):
        """
        主入口: 生成轨迹并在末尾追加归位点
        """
        print(f"[LoomPlanner] Generating {n_points} points (Seed: {seed})...")
        
        sobol_power = int(np.ceil(np.log2(n_points * 10))) 
        if sobol_power < 8: sobol_power = 8
        
        sampler = qmc.Sobol(d=10, scramble=True, seed=seed)
        margin = 3.0
        
        # 1. 生成与筛选
        sample_norm = sampler.random_base2(m=sobol_power)
        phys_min = self.LIMIT_MIN - margin
        phys_max = self.LIMIT_MAX + margin
        sample_phys = phys_min + sample_norm * (phys_max - phys_min)
        
        mask_relaxed = self.check_feasibility_relaxed(sample_phys, margin=margin)
        candidates = sample_phys[mask_relaxed]
        
        # 2. 投影与严格检查
        projected_points = self.project_points_to_feasible(candidates)
        mask_strict = self.check_feasibility_relaxed(projected_points, margin=1e-5)
        final_valid = projected_points[mask_strict]
        
        # 3. 截断
        if len(final_valid) < n_points:
            print(f"Warning: Only found {len(final_valid)} valid points (Target: {n_points})")
            targets = final_valid
        else:
            targets = final_valid[:n_points]
            
        # 4. 排序 (最小化路径代价)
        ordered_targets = self.solve_greedy_weighted_nn(targets)

        # === 🚀 关键修改: 强制追加归位点 (Safe Homing) ===
        # 创建一个全为 10.0 (LIMIT_MIN) 的状态点
        home_pose = np.full((1, 10), self.LIMIT_MIN)
        
        # 将其堆叠到轨迹末尾
        # 机器人执行完 ordered_targets 后，会执行这个 home_pose，从而缩回安全状态
        ordered_targets = np.vstack([ordered_targets, home_pose])
        # ===============================================
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, ordered_targets)
            
        return ordered_targets