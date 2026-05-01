import numpy as np
import time
import threading
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    from sensor_msgs.msg import Joy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS 2 imports failed. Forcing SIM mode.")

MODE = "RUN" 

# =========================================================
# 1. 物理参数与运动学 (保持不变)
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
    LIMIT_MIN_MM, LIMIT_MAX_MM = 13.0, 157.0
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
# 2. 核心遥操作与优化逻辑
# =========================================================
class DynamicTargetApp:
    def __init__(self, mode="RUN"):
        self.mode = mode if ROS_AVAILABLE else "SIM"
        
        self.JOY_SPEED_TIP = 8.0    
        self.JOY_SPEED_REP = 12.0   
        self.R_REP = 80.0    
        self.K_REP = 50.0    
        self.MAX_DQ_STEP = 2.0    
        
        self.q_init = np.ones((5, 2)) * 100.0 
        self.q_dyn = self.q_init.copy()
        self.is_running = True
        
        self.joy_axes = []
        self.joy_buttons = []
        self.prev_buttons = []
        self.dq_sec_filtered = np.zeros(10) 
        
        self.b_press_start_time = 0.0
        
        self.rep_points = [np.array([200.0, 500.0])] 
        self.active_rep_idx = 0
        
        nom_nodes, _ = forward_kinematics_full_chain(self.q_init)
        self.tip_target = nom_nodes[-1, :2].copy()  
        
        if self.mode == "RUN":
            rclpy.init()
            self.node = Node('dynamic_target_teleop')
            self.sub_joy = self.node.create_subscription(Joy, '/joy', self.joy_cb, 10)
            self.pub_dyn = self.node.create_publisher(Float32MultiArray, '/dynamic_target_cmd', 10)
            self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
            self.ros_thread.start()

        # UI 初始化
        self.fig, self.ax = plt.subplots(figsize=(10, 8)) 
        self.fig.patch.set_facecolor('#f4f4f9') 
        self.ax.set_facecolor('#ffffff')        
        self.ax.set_aspect('equal')
        
        # 🔥 优化视野范围：锚点上移后，Y轴范围设为 -100 到 1100 最完美
        self.ax.set_xlim(-1200, 1200)
        self.ax.set_ylim(-100, 1700)
        self.ax.grid(True, ls='--', alpha=0.6, color='#d3d3d3')
        
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#cccccc')
            
        # 🔥 所有视觉映射从 600 改为 800
        _, init_segs = forward_kinematics_full_chain(self.q_init, return_arc_points=True)
        for i in range(5):
            self.ax.plot(init_segs[i][:, 0], 1200.0 - init_segs[i][:, 1], '--', color='gray', lw=2, alpha=0.2)
            
        self.lines_dyn = [self.ax.plot([], [], '-', lw=7, color=SEG_COLORS[i], solid_capstyle='round')[0] for i in range(5)]
        self.tip_actual_visual = self.ax.plot([], [], 'ko', ms=8, zorder=10)[0] 
        self.target_visual = self.ax.plot(self.tip_target[0], 1200.0 - self.tip_target[1], '*', color='#ff00ff', ms=18, zorder=11)[0] 
        self.tether_line = self.ax.plot([], [], 'r--', lw=2.5, alpha=0.7, zorder=9)[0]
        
        self.rep_visual_dots = []
        self.rep_visual_circles = []
        for i in range(3):
            dot = self.ax.plot([], [], 'o', ms=14, zorder=20)[0]
            circle = Circle((0, 0), self.R_REP, color='red', alpha=0.1, fill=True, zorder=5)
            self.ax.add_patch(circle)
            self.rep_visual_dots.append(dot)
            self.rep_visual_circles.append(circle)
            
        bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc", alpha=0.9)
        self.hud_controls = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                         va='top', ha='left', fontsize=10, family='monospace', bbox=bbox_props)
        self.hud_status = self.ax.text(0.98, 0.98, 'SYSTEM: INITIALIZING', transform=self.ax.transAxes, 
                                       va='top', ha='right', fontsize=12, fontweight='bold', family='monospace', bbox=bbox_props)

        # 🔥 长按 B 键退出组件：往上挪到 950 避免遮挡
        self.cx, self.cy = 1000, 1400
        self.ring_r = 70
        self.b_ring_bg = self.ax.plot([], [], color='#cccccc', lw=4, zorder=30)[0]
        self.b_ring_fg = self.ax.plot([], [], color='#d62728', lw=5, zorder=31)[0]
        self.b_ring_text = self.ax.text(self.cx, self.cy, 'EXIT', color='#d62728',
                                        ha='center', va='center', fontweight='bold', fontsize=12, zorder=32)
        self.b_ring_text.set_visible(False)

        self.fig.tight_layout()
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        self.timer = self.fig.canvas.new_timer(interval=20)
        self.timer.add_callback(self.optimization_loop)
        self.timer.start()
        plt.show()

    def on_close(self, event):
        self.is_running = False
        if self.mode == "RUN":
            rclpy.shutdown()
        sys.exit(0)

    def joy_cb(self, msg):
        if not self.prev_buttons:
            self.prev_buttons = list(msg.buttons)
        self.joy_buttons = msg.buttons
        self.joy_axes = msg.axes
        
        btn_B_pressed = (msg.buttons[1] == 1 and self.prev_buttons[1] == 0)
        btn_B_released = (msg.buttons[1] == 0 and self.prev_buttons[1] == 1)
        btn_X_pressed = (msg.buttons[2] == 1 and self.prev_buttons[2] == 0)
        btn_Y_pressed = (msg.buttons[3] == 1 and self.prev_buttons[3] == 0)
        
        if btn_B_pressed:
            self.b_press_start_time = time.time()
        elif btn_B_released:
            self.b_press_start_time = 0.0
        
        if btn_Y_pressed and len(self.rep_points) < 3:
            new_pt = self.rep_points[self.active_rep_idx].copy() + np.array([100.0, 0.0])
            self.rep_points.append(new_pt)
            self.active_rep_idx = len(self.rep_points) - 1 
            
        if btn_X_pressed and len(self.rep_points) > 1:
            self.active_rep_idx = (self.active_rep_idx + 1) % len(self.rep_points)
            
        if msg.buttons[0] == 1:
            self.q_dyn = self.q_init.copy()
            nom_nodes, _ = forward_kinematics_full_chain(self.q_init)
            self.tip_target = nom_nodes[-1, :2].copy()
            self.rep_points = [np.array([200.0, 500.0])]
            self.active_rep_idx = 0
            
        self.prev_buttons = list(msg.buttons)

    def get_btn_state(self, idx):
        return len(self.joy_buttons) > idx and self.joy_buttons[idx] == 1

    def _quick_ik_search(self, target_pos, current_q, num_seeds=8, max_iters=15):
        for _ in range(num_seeds):
            noise = np.random.normal(0, 20.0, (5, 2))
            q_rand = enforce_constraints(current_q + noise)
            
            for _ in range(max_iters):
                nodes, _ = forward_kinematics_full_chain(q_rand)
                err = target_pos - nodes[-1, :2]
                if np.linalg.norm(err) < 2.0:
                    return q_rand, True
                    
                J_corr = np.zeros((2, 10))
                q_flat = q_rand.flatten()
                eps = 1e-4
                for i in range(10):
                    q_p = q_flat.copy(); q_p[i] += eps
                    n_p, _ = forward_kinematics_full_chain(q_p.reshape(5, 2))
                    J_corr[:, i] = (n_p[-1, :2] - nodes[-1, :2]) / eps
                dq_corr = np.linalg.pinv(J_corr) @ err
                q_rand = enforce_constraints((q_rand.flatten() + dq_corr).reshape(5, 2))
        return None, False

    def optimization_loop(self):
        if not self.is_running: return
        
        prev_q_dyn = self.q_dyn.copy()
        
        if self.b_press_start_time > 0.0:
            elapsed = time.time() - self.b_press_start_time
            progress = min(1.0, elapsed / 1.0)
            
            theta_bg = np.linspace(0, 2*np.pi, 100)
            self.b_ring_bg.set_data(self.cx + self.ring_r*np.cos(theta_bg), self.cy + self.ring_r*np.sin(theta_bg))
            theta_fg = np.linspace(np.pi/2, np.pi/2 - 2*np.pi*progress, max(2, int(100*progress)))
            self.b_ring_fg.set_data(self.cx + self.ring_r*np.cos(theta_fg), self.cy + self.ring_r*np.sin(theta_fg))
            
            self.b_ring_bg.set_visible(True)
            self.b_ring_fg.set_visible(True)
            self.b_ring_text.set_visible(True)
            
            if progress >= 1.0:
                print("Exiting safely...")
                plt.close(self.fig) 
                return
        else:
            self.b_ring_bg.set_visible(False)
            self.b_ring_fg.set_visible(False)
            self.b_ring_text.set_visible(False)
        
        deadzone = 0.15
        if len(self.joy_axes) >= 5:
            lx = self.joy_axes[0] if abs(self.joy_axes[0]) > deadzone else 0.0
            ly = self.joy_axes[1] if abs(self.joy_axes[1]) > deadzone else 0.0
            self.tip_target[0] += -lx * self.JOY_SPEED_TIP 
            self.tip_target[1] -= ly * self.JOY_SPEED_TIP
            
            rx = self.joy_axes[3] if abs(self.joy_axes[3]) > deadzone else 0.0
            ry = self.joy_axes[4] if abs(self.joy_axes[4]) > deadzone else 0.0
            self.rep_points[self.active_rep_idx][0] += -rx * self.JOY_SPEED_REP
            self.rep_points[self.active_rep_idx][1] -= ry * self.JOY_SPEED_REP

        dyn_nodes, dyn_segs = forward_kinematics_full_chain(self.q_dyn, return_arc_points=True)
        tip_dyn = dyn_nodes[-1, :2]
        base_pts = np.vstack(dyn_segs)
        
        J_tip = np.zeros((2, 10))
        J_tensor = np.zeros((len(base_pts), 2, 10))
        eps = 1e-4
        for i in range(10):
            q_p = self.q_dyn.flatten(); q_p[i] += eps
            n_p, s_p = forward_kinematics_full_chain(q_p.reshape(5, 2), return_arc_points=True)
            J_tip[:, i] = (n_p[-1, :2] - tip_dyn) / eps
            J_tensor[:, :, i] = (np.vstack(s_p) - base_pts) / eps

        J_tip_pinv = np.linalg.pinv(J_tip, rcond=1e-3)
        N_proj = np.eye(10) - J_tip_pinv @ J_tip
        
        dq_secondary = np.zeros(10)
        for pt in self.rep_points:
            delta_vecs = base_pts - pt
            dists_center = np.linalg.norm(delta_vecs, axis=1)
            active_indices = np.where(dists_center < self.R_REP)[0]
            for idx in active_indices:
                d_center = dists_center[idx]
                F_dir = (base_pts[idx] - pt) / (d_center + 1e-6)
                magnitude = self.K_REP * ((self.R_REP - d_center) / self.R_REP)**2 
                dq_secondary += J_tensor[idx].T @ (magnitude * F_dir)   
        
        self.dq_sec_filtered = 0.7 * self.dq_sec_filtered + 0.3 * dq_secondary

        dq_primary = J_tip_pinv @ (self.tip_target - tip_dyn)
        dq_raw = dq_primary + N_proj @ self.dq_sec_filtered
        dq_clamped = np.clip(dq_raw, -self.MAX_DQ_STEP, self.MAX_DQ_STEP)

        self.q_dyn = enforce_constraints((self.q_dyn.flatten() + dq_clamped).reshape(5, 2))

        max_iters = 25
        for _ in range(max_iters):
            curr_nodes, _ = forward_kinematics_full_chain(self.q_dyn)
            err = self.tip_target - curr_nodes[-1, :2]
            if np.linalg.norm(err) < 0.05: 
                break
            J_corr = np.zeros((2, 10))
            q_flat = self.q_dyn.flatten()
            for i in range(10):
                q_p = q_flat.copy(); q_p[i] += eps
                n_p, _ = forward_kinematics_full_chain(q_p.reshape(5, 2))
                J_corr[:, i] = (n_p[-1, :2] - curr_nodes[-1, :2]) / eps
            dq_corr = np.linalg.pinv(J_corr) @ err
            self.q_dyn = enforce_constraints((self.q_dyn.flatten() + dq_corr).reshape(5, 2))
            
        final_nodes, _ = forward_kinematics_full_chain(self.q_dyn)
        actual_tip = final_nodes[-1, :2]
        final_err = np.linalg.norm(self.tip_target - actual_tip)
        
        if final_err > 5.0: 
            self.q_dyn = prev_q_dyn.copy()
            final_nodes, _ = forward_kinematics_full_chain(self.q_dyn)
            actual_tip = final_nodes[-1, :2]
            
            q_global_goal, found_escape = self._quick_ik_search(self.tip_target, self.q_dyn, num_seeds=8, max_iters=15)
            
            if found_escape:
                self.q_dyn = q_global_goal.copy()
                self.dq_sec_filtered = np.zeros(10) 
                final_nodes, _ = forward_kinematics_full_chain(self.q_dyn)
                actual_tip = final_nodes[-1, :2]
                
                self.hud_status.set_text("TELEPORTED TO VALID STATE")
                self.hud_status.set_color('#9467bd') 
                self.tether_line.set_data([], [])
            else:
                self.hud_status.set_text(f"STUCK: LIMIT REACHED\nERR: {final_err:.1f} mm")
                self.hud_status.set_color('#d62728') 
                # 🔥 这里也改成 800.0 - Y
                self.tether_line.set_data([actual_tip[0], self.tip_target[0]], [1200.0 - actual_tip[1], 1200.0 - self.tip_target[1]])
        else:
            self.hud_status.set_text("SYSTEM: NORMAL\nTRACKING OK")
            self.hud_status.set_color('#2ca02c') 
            self.tether_line.set_data([], [])

        if self.mode == "RUN":
            q_send = np.zeros_like(self.q_dyn)
            q_send[:, 0] = self.q_dyn[:, 0]
            q_send[:, 1] = self.q_dyn[:, 1]
            
            msg = Float32MultiArray()
            msg.data = q_send.flatten().tolist()
            self.pub_dyn.publish(msg)

        # 🔥 UI 更新，全部统一为 800.0 - Y
        _, dyn_segs_plot = forward_kinematics_full_chain(self.q_dyn, return_arc_points=True)
        for i in range(5):
            self.lines_dyn[i].set_data(dyn_segs_plot[i][:, 0], 1200.0 - dyn_segs_plot[i][:, 1])
            
        self.tip_actual_visual.set_data([actual_tip[0]], [1200.0 - actual_tip[1]])
        self.target_visual.set_data([self.tip_target[0]], [1200.0 - self.tip_target[1]])
        
        for i in range(3):
            if i < len(self.rep_points):
                pt = self.rep_points[i]
                self.rep_visual_dots[i].set_data([pt[0]], [1200.0 - pt[1]])
                self.rep_visual_circles[i].center = (pt[0], 1200.0 - pt[1])
                self.rep_visual_dots[i].set_visible(True)
                self.rep_visual_circles[i].set_visible(True)
                
                if i == self.active_rep_idx:
                    self.rep_visual_dots[i].set_color('#ff7f0e') 
                    self.rep_visual_circles[i].set_color('#ff7f0e')
                    self.rep_visual_circles[i].set_alpha(0.25)
                else:
                    self.rep_visual_dots[i].set_color('#999999')
                    self.rep_visual_circles[i].set_color('#999999')
                    self.rep_visual_circles[i].set_alpha(0.08)
            else:
                self.rep_visual_dots[i].set_visible(False)
                self.rep_visual_circles[i].set_visible(False)
                
        btn_A_str = "[A Btn]  : Reset Pose" + (" [PRESSED]" if self.get_btn_state(0) else "")
        btn_B_str = "[HOLD B] : Exit Program" + (" [PRESSED]" if self.get_btn_state(1) else "")
        btn_X_str = "[X Btn]  : Switch Active" + (" [PRESSED]" if self.get_btn_state(2) else "")
        btn_Y_str = "[Y Btn]  : Add Point" + (" [PRESSED]" if self.get_btn_state(3) else "")
        
        controls_text = (
            "--- CONTROLS ---\n"
            "[L-Stick]: Move Target\n"
            "[R-Stick]: Move Repulsion\n"
            f"{btn_Y_str}\n"
            f"{btn_X_str}\n"
            f"{btn_A_str}\n"
            f"{btn_B_str}\n"
            "----------------\n"
            f"Active Point: {self.active_rep_idx + 1} / {len(self.rep_points)}"
        )
        self.hud_controls.set_text(controls_text)
                
        self.fig.canvas.draw_idle()

def main(args=None):
    app = DynamicTargetApp(mode=MODE)

if __name__ == '__main__':
    main()