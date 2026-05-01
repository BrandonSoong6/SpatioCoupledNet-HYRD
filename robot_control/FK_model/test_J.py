#!/usr/bin/env python3
import numpy as np
import time
from numba import jit

# ================= ⚙️ 配置 =================
C_LIST_CONFIG = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL = 22.0
H0_VAL = 52.0
M_VAL = H0_VAL - 2 * N_VAL

# ================= ⚡ Numba FK (只算坐标) =================
@jit(nopython=True, cache=True)
def get_section_transform_numba(q_l, q_r, c_val, n, m):
    if c_val < 1e-5: c_val = 100.0
    delta_q = q_l - q_r; sum_q = q_l + q_r
    theta = delta_q / c_val
    L_c = m + sum_q / 2.0
    
    if abs(theta) < 1e-6:
        lx, ly, c, s = 0.0, L_c, 1.0, 0.0
    else:
        rho = L_c / theta
        c = np.cos(-theta); s = np.sin(-theta)
        lx = rho * (1.0 - np.cos(theta)); ly = rho * np.sin(theta)
    
    # 构造 T矩阵
    T = np.eye(3, dtype=np.float64)
    T[0,0]=c; T[0,1]=-s; T[0,2]=lx - s*n
    T[1,0]=s; T[1,1]=c;  T[1,2]=ly + c*n + n
    return T

@jit(nopython=True, cache=True)
def get_xy_state_fast(q_vec, c_list, n_val, m_val):
    state = np.zeros(10, dtype=np.float64)
    T_curr = np.eye(3, dtype=np.float64)
    num_sections = len(c_list)
    for i in range(num_sections):
        idx = (num_sections - 1 - i) * 2
        T_sec = get_section_transform_numba(q_vec[idx], q_vec[idx+1], c_list[i], n_val, m_val)
        T_curr = T_curr @ T_sec 
        state[i*2] = T_curr[0, 2]; state[i*2+1] = T_curr[1, 2]
    return state

# ================= ⚡ 局部求解 (死磕位置) =================
@jit(nopython=True, cache=True)
def solve_local_XY_only(target_local_xy, c_val, q_guess, n, m):
    q_curr = q_guess.copy()
    J = np.zeros((2, 2), dtype=np.float64)
    
    for k in range(150): # 给够迭代次数
        # 1. FK
        T = get_section_transform_numba(q_curr[0], q_curr[1], c_val, n, m)
        curr_xy = np.array([T[0,2], T[1,2]], dtype=np.float64)
        
        # 2. 误差
        err = target_local_xy - curr_xy
        if np.linalg.norm(err) < 0.05: break 
        
        # 3. 雅可比
        eps = 1e-4
        for d in range(2):
            q_p = q_curr.copy(); q_p[d] += eps
            Tp = get_section_transform_numba(q_p[0], q_p[1], c_val, n, m)
            yp = np.array([Tp[0,2], Tp[1,2]])
            J[:, d] = (yp - curr_xy) / eps
            
        # 4. 求解
        # 阻尼设小一点(1e-5)，让它对 X 轴误差更敏感
        dq = np.linalg.solve(J.T @ J + 1e-5 * np.eye(2), J.T @ err)
        
        # 步长裁剪
        max_step = 5.0
        if np.max(np.abs(dq)) > max_step:
            dq = dq * (max_step / np.max(np.abs(dq)))
            
        q_curr += dq
        
        # 限位
        for x in range(2):
            if q_curr[x] < 0.0: q_curr[x] = 0.0
            if q_curr[x] > 165.0: q_curr[x] = 165.0
            
    return q_curr

@jit(nopython=True, cache=True)
def solve_sequential_XY(target_points_flat, c_list, n, m):
    q_solved = np.zeros(10, dtype=np.float64)
    T_acc = np.eye(3, dtype=np.float64)
    
    for i in range(5):
        # 1. 转局部坐标
        tgt_global = np.array([target_points_flat[i*2], target_points_flat[i*2+1], 1.0])
        # 手动求逆 T_acc (3x3)
        # 既然 T_acc 是刚体变换，用 np.linalg.inv 也没问题
        p_local_homo = np.linalg.inv(T_acc) @ tgt_global
        tgt_local_xy = p_local_homo[:2]
        
        # 2. 智能初值 + 🔥🔥🔥 踢一脚 (打破对称性) 🔥🔥🔥
        q_init = np.array([30.0, 30.0], dtype=np.float64)
        if i > 0:
            prev_idx = (5 - 1 - (i-1)) * 2
            q_init[0] = q_solved[prev_idx]
            q_init[1] = q_solved[prev_idx+1]
        
        # [关键]：如果有横向偏移(X!=0)，强制给一个非对称初值
        # 这样 Jacobian 里的 X 分量就不会是 0
        if abs(tgt_local_xy[0]) > 1.0:
            # 这里的方向很重要：
            # 如果目标在左边(X<0)，通常需要 qL > qR (弯曲) 或者 qL != qR
            # 随便给一个偏移，梯度下降会自动修正方向
            q_init[0] += 2.0 
            q_init[1] -= 2.0
            
        # 3. 求解
        q_res = solve_local_XY_only(tgt_local_xy, c_list[i], q_init, n, m)
        
        # 4. 存
        save_idx = (5 - 1 - i) * 2
        q_solved[save_idx] = q_res[0]
        q_solved[save_idx+1] = q_res[1]
        
        # 5. 更新 T_acc
        T_sec = get_section_transform_numba(q_res[0], q_res[1], c_list[i], n, m)
        T_acc = T_acc @ T_sec
        
    return q_solved

# ================= 🚀 接口 =================
class FastIKSolver:
    def __init__(self):
        print("🔥 Warming up Numba JIT...")
        dummy_pts = np.zeros(10, dtype=np.float64)
        solve_sequential_XY(dummy_pts, C_LIST_CONFIG, N_VAL, M_VAL)
        print("✅ Numba Ready!")

    def solve(self, points_10d):
        return solve_sequential_XY(points_10d, C_LIST_CONFIG, N_VAL, M_VAL)

# ================= 🧪 验证 =================
if __name__ == "__main__":
    solver = FastIKSolver()
    
    target_points = np.array([
        0.0, 62.0, 
        -70.893196, 191.89363, 
        -242.0456, 253.83846, 
        -303.5778, 261.44028, 
        -365.11002, 269.0421
    ], dtype=np.float64)
    
    print("\n🚀 Running Pure XY Sequential IK (Symmetry Broken)...")
    start = time.time()
    q_solved = solver.solve(target_points)
    end = time.time()
    
    # FK 验证
    current_xy = get_xy_state_fast(q_solved, C_LIST_CONFIG, N_VAL, M_VAL)
    
    print("-" * 65)
    print(f"⏱️  Solve Time: {(end - start)*1000:.4f} ms")
    print(f"🧩  Solved Q: {np.round(q_solved, 1)}")
    print("-" * 65)
    print(f"{'Sec':<3} | {'Target (X, Y)':<20} | {'Solved (X, Y)':<20} | {'Err(mm)':<8}")
    
    max_err = 0.0
    for i in range(5):
        t_x, t_y = target_points[i*2], target_points[i*2+1]
        s_x, s_y = current_xy[i*2], current_xy[i*2+1]
        err = np.sqrt((t_x - s_x)**2 + (t_y - s_y)**2)
        max_err = max(max_err, err)
        print(f"{i+1:<3} | ({t_x:>7.2f}, {t_y:>7.2f}) | ({s_x:>7.2f}, {s_y:>7.2f}) | {err:>8.4f}")
    
    print("-" * 65)
    print(f"🔥 Max Tracking Error: {max_err:.4f} mm")
    if max_err < 1.0:
        print("✅ Status: PERFECT. You can sleep now.")
    else:
        print("⚠️ Status: Still divergent? Check q limits.")    