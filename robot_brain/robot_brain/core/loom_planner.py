# File: robot_brain/core/loom_planner.py
import numpy as np
import os
import time
from scipy.stats import qmc
from scipy.spatial.distance import cdist

class LoomPlanner:
    def __init__(self):
        # === 1. 物理参数配置 (保持不变) ===
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
        
        # 结合全局物理限位
        final_min = np.maximum(self.LIMIT_MIN, d_min)
        final_max = np.minimum(self.LIMIT_MAX, d_max)
        return final_min, final_max

    def project_points_to_feasible(self, points_10d):
        """
        标准投影: 将不可行点拉回可行域边界 (Clip)
        (保持不变: 具体的吸附偏好由 generate_trajectory 中的概率采样层负责)
        """
        projected = points_10d.copy()
        for _ in range(3):
            modules = projected.reshape(-1, 5, 2)
            q1 = modules[:, :, 0]
            q2 = modules[:, :, 1]
            
            # 1. 全局限位
            q1 = np.clip(q1, self.LIMIT_MIN, self.LIMIT_MAX)
            q2 = np.clip(q2, self.LIMIT_MIN, self.LIMIT_MAX)
            
            # 2. CPL 约束投影
            mn1, mx1 = self.get_dynamic_bounds_vec(q2)
            q1 = np.clip(q1, mn1, mx1)
            
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

    def _calculate_boundary_margin_sum(self, points):
        """
        [新增] 计算距离场: 每个点距离可行域边界的"安全边距"之和。
        Margin 越小 -> 离边界越近。
        """
        N = len(points)
        total_margins = np.zeros(N)
        modules = points.reshape(-1, 5, 2)
        
        for i in range(5):
            q1 = modules[:, i, 0]
            q2 = modules[:, i, 1]
            
            # 计算 q1 维度的边界距离
            mn1, mx1 = self.get_dynamic_bounds_vec(q2)
            dist_q1 = np.minimum(np.abs(q1 - mn1), np.abs(mx1 - q1))
            
            # 计算 q2 维度的边界距离
            mn2, mx2 = self.get_dynamic_bounds_vec(q1)
            dist_q2 = np.minimum(np.abs(q2 - mn2), np.abs(mx2 - q2))
            
            # 该模块距离边界的最近距离
            module_min_dist = np.minimum(dist_q1, dist_q2)
            total_margins += module_min_dist
            
        return total_margins

    def _generate_extreme_points(self, n_extreme):
        """[新增] 生成极端测试点: 单舵机极限 (One Max, One Min)"""
        extreme_points = np.zeros((n_extreme, 10))
        for i in range(n_extreme):
            for m in range(5):
                is_q1_max = np.random.rand() > 0.5
                idx_q1 = m * 2
                idx_q2 = m * 2 + 1
                if is_q1_max:
                    val_q1 = self.LIMIT_MAX
                    mn2, _ = self.get_dynamic_bounds_vec(np.array([val_q1]))
                    val_q2 = mn2[0] 
                else:
                    val_q2 = self.LIMIT_MAX
                    mn1, _ = self.get_dynamic_bounds_vec(np.array([val_q2]))
                    val_q1 = mn1[0]
                extreme_points[i, idx_q1] = val_q1
                extreme_points[i, idx_q2] = val_q2
        return extreme_points

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
        主入口: 使用新版逻辑 (超量采样 + 距离场筛选 + 极限注入)
        接口与旧版完全一致。
        """
        print(f"[LoomPlanner] Generating {n_points} points (Seed: {seed})...")
        
        # === 1. 配额划分 ===
        # 5% 极端点 (Corner Cases)
        n_extreme = int(np.ceil(n_points * 0.05)) 
        # 95% 待筛选点 (Sobol with Distance Field)
        n_sobol_target = n_points - n_extreme     
        
        # === 2. Sobol 宽域超量采样 (Wide Oversampling) ===
        oversample_factor = 5
        n_pool = n_sobol_target * oversample_factor
        
        # 计算 Sobol 阶数
        sobol_power = int(np.ceil(np.log2(n_pool * 10))) 
        if sobol_power < 8: sobol_power = 8
        
        sampler = qmc.Sobol(d=10, scramble=True, seed=seed)
        sample_norm = sampler.random_base2(m=sobol_power)
        
        # 宽域生成 (Margin=10.0)
        gen_margin = 10.0 
        phys_min = self.LIMIT_MIN - gen_margin
        phys_max = self.LIMIT_MAX + gen_margin
        sample_phys = phys_min + sample_norm * (phys_max - phys_min)
        
        # === 3. 第一轮拒绝 + 投影 ===
        snap_threshold = 5.0 # (可调参数)
        
        # 快速判断: 离边界太远的直接扔掉
        is_close_enough = self.check_feasibility_relaxed(sample_phys, margin=snap_threshold)
        candidates_near = sample_phys[is_close_enough]
        print(f"   🗑️ Round 1 Filter: Dropped {len(sample_phys) - len(candidates_near)} points")
        
        # 投影剩下的
        projected_pool = self.project_points_to_feasible(candidates_near)
        
        # 严格检查
        mask_strict = self.check_feasibility_relaxed(projected_pool, margin=1e-5)
        valid_pool = projected_pool[mask_strict]
        
        if len(valid_pool) < n_sobol_target:
            print(f"⚠️ Warning: Pool too small ({len(valid_pool)} < {n_sobol_target}). Using all valid.")
            sobol_final = valid_pool
        else:
            # === 4. 第二轮拒绝：基于距离场的概率重采样 ===
            margins = self._calculate_boundary_margin_sum(valid_pool)
            
            epsilon = 1.0 
            sharpness = 5.0 # (可调参数: 越高越倾向于边界)
            weights = 1.0 / np.power(margins + epsilon, sharpness)
            probs = weights / np.sum(weights)
            
            indices = np.random.choice(
                len(valid_pool), 
                size=n_sobol_target, 
                replace=False, 
                p=probs
            )
            sobol_final = valid_pool[indices]
            print(f"   ✨ Round 2 Resampling: Picked {n_sobol_target} points (Boundary Biased).")

        # === 5. 注入极端点 ===
        extreme_final = self._generate_extreme_points(n_extreme)
        
        # === 6. 合并、排序 ===
        combined_points = np.vstack([sobol_final, extreme_final])
        ordered_targets = self.solve_greedy_weighted_nn(combined_points)
        
        # === 7. 强制追加归位点 (Critical for Safety) ===
        home_pose = np.full((1, 10), self.LIMIT_MIN)
        ordered_targets = np.vstack([ordered_targets, home_pose])
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, ordered_targets)
            
        return ordered_targets