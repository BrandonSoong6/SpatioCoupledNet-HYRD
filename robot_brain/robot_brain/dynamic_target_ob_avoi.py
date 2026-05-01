import numpy as np
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 根据需要导入 ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    from geometry_msgs.msg import Point
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("⚠️ ROS 2 imports failed. Forcing SIM mode.")

# =========================================================
# 0. 模式选择配置
# =========================================================
MODE = "RUN"  # "SIM" 或 "RUN"

# =========================================================
# 1. 物理参数与运动学 (与主程序严格同步)
# =========================================================
C_LIST = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL, H0_VAL = 22.0, 52.0
M_VAL = H0_VAL - 2 * N_VAL
SEG_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def forward_kinematics_full_chain(q_all, return_arc_points=False):
    T_curr = np.eye(3); node_poses = []; segment_points_list = []
    for i in range(5):
        q_l, q_r = q_all[i]; theta = (q_l - q_r) / C_LIST[i]; L_c = M_VAL + (q_l + q_r) / 2.0
        if return_arc_points:
            num_samples = 25; total_len = 2 * N_VAL + L_c; s_vals = np.linspace(0, total_len, num_samples)
            current_seg_pts = []
            for s in s_vals:
                if abs(theta) < 1e-12: lx, ly = 0.0, s
                else:
                    rho = L_c / theta
                    if s <= N_VAL: lx, ly = 0.0, s
                    elif s <= (N_VAL + L_c):
                        arc_s = s - N_VAL; curr_th = (arc_s / L_c) * theta
                        lx, ly = rho * (1.0 - np.cos(curr_th)), N_VAL + rho * np.sin(curr_th)
                    else:
                        rem_s = s - (N_VAL + L_c); arc_end_x, arc_end_y = rho * (1.0 - np.cos(theta)), N_VAL + rho * np.sin(theta)
                        lx, ly = arc_end_x + np.sin(theta)*rem_s, arc_end_y + np.cos(theta)*rem_s
                p_glob = T_curr @ np.array([lx, ly, 1]); current_seg_pts.append(p_glob[:2])
            segment_points_list.append(np.array(current_seg_pts))
        th_l = (q_l - q_r) / C_LIST[i]; lc_l = M_VAL + (q_l + q_r) / 2.0
        if abs(th_l) < 1e-12: lx_l, ly_l = 0.0, 2*N_VAL + lc_l
        else:
            rho_l = lc_l / th_l; lx_arc, ly_arc = rho_l * (1.0 - np.cos(th_l)), rho_l * np.sin(th_l)
            lx_l = np.sin(th_l)*N_VAL + lx_arc; ly_l = np.cos(th_l)*N_VAL + ly_arc + N_VAL
        c, s = np.cos(-th_l), np.sin(-th_l)
        T_local = np.array([[c, -s, lx_l], [s, c, ly_l], [0, 0, 1]])
        T_curr = T_curr @ T_local
        node_poses.append([T_curr[0, 2], T_curr[1, 2], np.arctan2(T_curr[1, 0], T_curr[0, 0])])
    return np.array(node_poses), segment_points_list

def enforce_constraints(q_in):
    q_out = np.zeros((5, 2))
    LIMIT_MIN_MM = 13.0
    LIMIT_MAX_MM = 157.0
    CPL_OFFSET, CPL_COEFF, CPL_EXP = 10.0, 0.17, 1.1
    CPL_INV_EXP = 1.0 / 1.1
    def safe_pow(base, exp): return max(base, 0.0) ** exp
    for i in range(5):
        ql, qr = q_in[i]
        max_delta_q = np.radians(50.0) * C_LIST[i]
        current_delta = ql - qr
        if abs(current_delta) > max_delta_q:
            mean_q = (ql + qr) / 2.0
            sign = 1.0 if current_delta > 0 else -1.0
            ql = mean_q + sign * (max_delta_q / 2.0)
            qr = mean_q - sign * (max_delta_q / 2.0)
        ql, qr = max(LIMIT_MIN_MM, min(LIMIT_MAX_MM, ql)), max(LIMIT_MIN_MM, min(LIMIT_MAX_MM, qr))
        dyn_min_r = CPL_OFFSET + CPL_COEFF * safe_pow(ql - CPL_OFFSET, CPL_EXP)
        dyn_max_r = CPL_OFFSET + safe_pow((ql - CPL_OFFSET) / CPL_COEFF, CPL_INV_EXP)
        qr = max(max(LIMIT_MIN_MM, dyn_min_r), min(min(LIMIT_MAX_MM, dyn_max_r), qr))
        dyn_min_l = CPL_OFFSET + CPL_COEFF * safe_pow(qr - CPL_OFFSET, CPL_EXP)
        dyn_max_l = CPL_OFFSET + safe_pow((qr - CPL_OFFSET) / CPL_COEFF, CPL_INV_EXP)
        ql = max(max(LIMIT_MIN_MM, dyn_min_l), min(min(LIMIT_MAX_MM, dyn_max_l), ql))
        q_out[i] = [ql, qr]
    return q_out

# =========================================================
# 2. 核心避障优化逻辑
# =========================================================
class DynamicTargetApp:
    def __init__(self, mode="RUN"):
        self.mode = mode if ROS_AVAILABLE else "SIM"
        
        # --- APF 避障暴力调优参数 ---
        self.R_ROBOT = 200.0     # 骨架膨胀半径 (流体厚度)
        self.R_REP = 200.0      # 距离【表面】的感应半径
        self.K_REP = 1600.0     # 斥力极值
        self.K_ATT = 0.03       # 身体姿态恢复力
        self.MAX_STEP = 5.0     # 迭代最大步长
        
        self.obs_xy = None
        self.q_nom = np.ones((5, 2)) * 100.0  # 固定名义目标
        self.q_dyn = self.q_nom.copy()
        self.is_running = True
        
        nom_nodes, _ = forward_kinematics_full_chain(self.q_nom)
        self.tip_nom = nom_nodes[-1, :2] 
        
        if self.mode == "RUN":
            rclpy.init()
            self.node = Node('dynamic_target_optimizer')
            # 订阅 vision_node.py 发布的障碍物信息
            self.sub_obs = self.node.create_subscription(Point, '/vision/obstacle_pos', self.obs_cb, 10)
            self.pub_dyn = self.node.create_publisher(Float32MultiArray, '/dynamic_target_cmd', 10)
            self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
            self.ros_thread.start()
            print(f"🚀 RUN Mode Active. Listening to 'vision/obstacle_pos'...")

        # --- 可视化初始化 ---
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-400, 400); self.ax.set_ylim(-100, 1200)
        self.ax.grid(True, ls=':', alpha=0.5)
        
        # 静态背景：名义直线
        _, nom_segs = forward_kinematics_full_chain(self.q_nom, return_arc_points=True)
        for i in range(5):
            self.ax.plot(nom_segs[i][:, 0], nom_segs[i][:, 1], '--', color='gray', lw=2, alpha=0.2)
            
        # 动态目标线条
        self.lines_dyn = [self.ax.plot([], [], '-', lw=5, color=SEG_COLORS[i], solid_capstyle='round')[0] for i in range(5)]
        
        # 固定末端标记
        self.ax.plot(self.tip_nom[0], self.tip_nom[1], 'k*', ms=15, label="FIXED TIP TARGET")
        
        # 🔥 新增：斥力点可视化组件
        self.obs_visual = self.ax.plot([], [], 'ro', ms=10, zorder=20)[0] # 红色中心点
        self.rep_circle = Circle((0, 0), self.R_REP, color='red', alpha=0.15, fill=True, zorder=5)
        self.ax.add_patch(self.rep_circle)
        self.rep_circle.set_visible(False)
        self.obs_visual.set_visible(False)

        self.ax.legend(loc='upper right')
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        self.timer = self.fig.canvas.new_timer(interval=20)
        self.timer.add_callback(self.optimization_loop)
        self.timer.start()
        plt.show()

    def obs_cb(self, msg):
        # 接收来自 ROS 的实时障碍物坐标
        self.obs_xy = np.array([msg.x, msg.y])
        print(f"📡 Received Obstacle Position: ({msg.x:.1f}, {msg.y:.1f})")

    def on_mouse_move(self, event):
        # 🔥 修复：RUN 模式下禁用鼠标干扰
        if self.mode == "SIM" and event.inaxes:
            self.obs_xy = np.array([event.xdata, event.ydata])

    def optimization_loop(self):
        if not self.is_running: return
        
        dyn_nodes, dyn_segs = forward_kinematics_full_chain(self.q_dyn, return_arc_points=True)
        tip_dyn = dyn_nodes[-1, :2]
        base_pts = np.vstack(dyn_segs)
        
        # 1. 计算 Jacobian (Null-space 投影基础)
        J_tip = np.zeros((2, 10))
        J_tensor = np.zeros((len(base_pts), 2, 10))
        eps = 1e-4
        for i in range(10):
            q_p = self.q_dyn.flatten(); q_p[i] += eps
            n_p, s_p = forward_kinematics_full_chain(q_p.reshape(5, 2), return_arc_points=True)
            J_tip[:, i] = (n_p[-1, :2] - tip_dyn) / eps
            J_tensor[:, :, i] = (np.vstack(s_p) - base_pts) / eps

        J_tip_pinv = np.linalg.pinv(J_tip, rcond=1e-3)

        # 2. 最近点避障策略 (APF)
        grad_rep = np.zeros(10)
        att_scale = 1.0
        if self.obs_xy is not None:
            # 计算障碍物到所有骨架点的向量及中心距离
            delta_vecs = base_pts - self.obs_xy
            dists_center = np.linalg.norm(delta_vecs, axis=1)
            min_idx = np.argmin(dists_center)
            d_center = dists_center[min_idx]
            
            # 核心修改：将距离度量从中心距离转化为表面距离
            d_surface = d_center - self.R_ROBOT
            
            # 判断是否进入表面感应区
            if d_surface < self.R_REP:
                p_closest = base_pts[min_idx]
                
                # 奇异性与穿模保护
                # 若障碍物侵入流体内部 (d_surface <= 0)，将其截断为一个极小的正值
                # 这样势场公式会输出一个极大的斥力，强行将机器人推离障碍物
                d_safe = max(d_surface, 1.0)
                
                # 斥力大小：使用表面距离 d_safe 进行计算
                F_mag = self.K_REP * (1.0/d_safe - 1.0/self.R_REP) * (self.R_REP / d_safe)**2
                
                # 斥力方向：表面法线方向与中心连线方向严格共线
                # 注意这里除以的是 d_center 而不是 d_safe，以获取准确的单位方向向量
                F_dir = (p_closest - self.obs_xy) / max(d_center, 1e-3) 
                
                # 映射到关节空间的斥力梯度
                grad_rep = J_tensor[min_idx].T @ (F_mag * F_dir)
                
                # 姿态恢复力缩放：同样基于表面距离进行衰减
                att_scale = np.clip((d_surface - 20.0) / (self.R_REP - 20.0), 0.0, 1.0)

        # 3. 分离主副任务并进行零空间投影 (保证 Tip 不动)
        dq_secondary = grad_rep + att_scale * self.K_ATT * (self.q_nom.flatten() - self.q_dyn.flatten())
        
        # 修复1：仅对次要任务（避障+姿态）进行限幅，绝不能缩放主任务（末端位置补偿）
        norm_sec = np.linalg.norm(dq_secondary)
        if norm_sec > self.MAX_STEP: 
            dq_secondary = dq_secondary * (self.MAX_STEP / norm_sec)

        N_proj = np.eye(10) - J_tip_pinv @ J_tip
        
        # 主任务：利用伪逆计算消除当前末端误差的理论增量
        dq_primary = J_tip_pinv @ (self.tip_nom - tip_dyn)
        
        # 合成最终的理论关节增量
        dq = dq_primary + N_proj @ dq_secondary
        
        # 4. 更新初始动态关节角
        # 注意：这里暂不直接调用 enforce_constraints，留到内循环的 IK 求解中统一处理
        self.q_dyn = (self.q_dyn.flatten() + dq).reshape(5, 2)

        # 5. 严格的末端补偿修正与约束闭环 (内循环)
        # 修复2：增加迭代次数上限，提高收敛标准，并将物理约束融合到迭代过程中
        max_iters = 25
        tolerance = 0.01  # 将精度提高到 0.01 毫米级别
        
        for _ in range(max_iters):
            curr_nodes, _ = forward_kinematics_full_chain(self.q_dyn)
            err = self.tip_nom - curr_nodes[-1, :2]
            
            # 满足极高的精度要求后方可跳出循环
            if np.linalg.norm(err) < tolerance: 
                break
                
            # 重新计算当前局部状态下的雅可比矩阵
            J_corr = np.zeros((2, 10))
            eps_corr = 1e-4
            q_flat = self.q_dyn.flatten()
            for i in range(10):
                q_p = q_flat.copy()
                q_p[i] += eps_corr
                n_p, _ = forward_kinematics_full_chain(q_p.reshape(5, 2))
                J_corr[:, i] = (n_p[-1, :2] - curr_nodes[-1, :2]) / eps_corr
                
            # 计算纯粹的末端补偿增量
            dq_corr = np.linalg.pinv(J_corr) @ err
            
            # 将增量应用后，执行非线性约束截断。
            # 如果截断导致新的末端误差，下一次迭代会继续修正，直到找到一个既满足约束又保证末端不动的解。
            self.q_dyn = enforce_constraints((self.q_dyn.flatten() + dq_corr).reshape(5, 2))

        # 6. 发送数据给 Solver
        if self.mode == "RUN":
            msg = Float32MultiArray()
            msg.data = self.q_dyn.flatten().tolist()
            self.pub_dyn.publish(msg)

        # 7. 更新 UI 可视化
        _, dyn_segs_plot = forward_kinematics_full_chain(self.q_dyn, return_arc_points=True)
        for i in range(5):
            self.lines_dyn[i].set_data(dyn_segs_plot[i][:, 0], dyn_segs_plot[i][:, 1])
            
        # 更新红色斥力点和光圈的位置
        if self.obs_xy is not None:
            self.obs_visual.set_data([self.obs_xy[0]], [self.obs_xy[1]])
            self.rep_circle.center = (self.obs_xy[0], self.obs_xy[1])
            self.obs_visual.set_visible(True)
            self.rep_circle.set_visible(True)
        else:
            self.obs_visual.set_visible(False)
            self.rep_circle.set_visible(False)
            
        self.fig.canvas.draw_idle()

def main(args=None):
    app = DynamicTargetApp(mode=MODE)

if __name__ == '__main__':
    main()