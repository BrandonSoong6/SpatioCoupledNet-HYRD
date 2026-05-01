#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point  # ✨ 新增：适配 Controller
from robot_interfaces.msg import VisionState

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pynput import keyboard
import threading
import sys
import math

# ================= 🚫 屏蔽 Matplotlib 默认快捷键 =================
for key in plt.rcParams:
    if key.startswith('keymap.'):
        plt.rcParams[key] = []

# ================= ⚙️ 配置参数 =================
C_LIST_CONFIG = [92.0, 108.0, 123.5, 140.0, 156.0] 
RACK_TOTAL_LEN = 240.0                             
VIS_FLEX_LINE_WIDTH = 3.0   
VIS_SHOW_GHOST = True       

# 动力学耦合参数
CPL_OFFSET = 10.0
CPL_COEFF = 0.17
CPL_EXP = 1.1

# ================= 🧮 算法函数 =================
def calculate_dynamic_bounds(q_partner):
    val_corrected = max(0.0, q_partner - CPL_OFFSET)
    dyn_min = CPL_OFFSET + CPL_COEFF * pow(val_corrected, CPL_EXP)
    dyn_max = CPL_OFFSET + pow(val_corrected / CPL_COEFF, 1.0 / CPL_EXP)
    return dyn_min, dyn_max

def enforce_coupling_constraints(q_vec):
    q_new = q_vec.copy()
    for _ in range(2):
        for i in range(0, 10, 2):
            q_l = q_new[i]; q_r = q_new[i+1]
            l_min, l_max = calculate_dynamic_bounds(q_r)
            q_l = np.clip(q_l, max(0.0, l_min), min(160.0, l_max))
            r_min, r_max = calculate_dynamic_bounds(q_l)
            q_r = np.clip(q_r, max(0.0, r_min), min(160.0, r_max))
            q_new[i] = q_l; q_new[i+1] = q_r
    return q_new

def wrap_angle_deg(angle):
    """ 将角度归一化到 [-180, 180] """
    return (angle + 180.0) % 360.0 - 180.0

# ================= 🤖 机器人模型 =================
class DiscreteContinuumRobot:
    def __init__(self, n=22.0, H0=52.0, c_list=None, num_sections=5):
        if c_list is None: c_list = [200.0] * num_sections
        self.n = n; self.H0 = H0; self.m = H0 - 2 * n
        self.c_list = c_list; self.num_sections = num_sections

    def _get_arc_params(self, q_left, q_right, current_c):
        delta_q = q_left - q_right; sum_q = q_left + q_right
        theta = delta_q / current_c; L_center = self.m + sum_q / 2.0
        if abs(theta) < 1e-6: return None, 0.0, L_center, True
        rho = L_center / theta
        return rho, theta, L_center, False

    def get_section_mesh(self, T_start, q_left, q_right, section_idx):
        current_c = self.c_list[section_idx]
        rho, theta, L_c, is_straight = self._get_arc_params(q_left, q_right, current_c)
        T_n = np.eye(3); T_n[1, 2] = self.n
        T_flex_start = T_start @ T_n
        if is_straight:
            T_arc = np.eye(3); T_arc[1, 2] = L_c
        else:
            lx = rho*(1-np.cos(theta)); ly = rho*np.sin(theta)
            c, s = np.cos(-theta), np.sin(-theta)
            T_arc = np.array([[c,-s,lx],[s,c,ly],[0,0,1]])
        T_flex_end = T_flex_start @ T_arc
        T_end = T_flex_end @ T_n
        steps_m = 10 
        left_pts, right_pts = [], []
        half_w = current_c / 2.0
        for i in range(steps_m):
            s = i / (steps_m - 1)
            if is_straight:
                T_curr = T_flex_start @ np.array([[1,0,0],[0,1,s*L_c],[0,0,1]])
            else:
                ang = s * theta
                lx, ly = rho*(1-np.cos(ang)), rho*np.sin(ang)
                c_p, s_p = np.cos(-ang), np.sin(-ang)
                T_curr = T_flex_start @ np.array([[c_p,-s_p,lx],[s_p,c_p,ly],[0,0,1]])
            left_pts.append((T_curr @ [-half_w, 0, 1])[:2])
            right_pts.append((T_curr @ [half_w, 0, 1])[:2])
        return {'T_start': T_start, 'T_end': T_end, 'flex_left': np.array(left_pts), 'flex_right': np.array(right_pts), 'c': current_c}

    def forward_visualize(self, q_array):
        all_sections = []
        T_curr = np.eye(3)
        collected_data = [] # Stores [x, y, theta_deg] for each section
        
        for i in range(self.num_sections):
            idx_base = (self.num_sections - 1 - i) * 2
            data = self.get_section_mesh(T_curr, q_array[idx_base], q_array[idx_base+1], i)
            data['id'] = i + 1
            all_sections.append(data)
            T_curr = data['T_end']
            
            # ✨ 核心修改：计算全局坐标 (x, y) 和 全局角度 (theta)
            x_global = T_curr[0, 2]
            y_global = T_curr[1, 2]
            # 计算全局旋转角 (相对于 Global X 轴)
            theta_rad = np.arctan2(T_curr[1, 0], T_curr[0, 0])
            theta_deg = wrap_angle_deg(np.degrees(theta_rad))
            
            collected_data.extend([x_global, y_global, theta_deg])

        return all_sections, T_curr, np.array(collected_data)

# ================= 🎨 绘图函数 =================
def draw_frame(ax, robot, q_target, xy_feedback=None):
    ax.clear()
    # xy_15d_target 包含 [x1, y1, th1, x2, y2, th2, ...]
    sections_data, T_tip_global, xy_15d_target = robot.forward_visualize(q_target)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    all_x, all_y = [], []
    def rot180(p): return -p

    def draw_poly(vis_pts, fc, ec, alpha=1.0, z=20, ls='-'):
        ax.add_patch(Polygon(vis_pts, facecolor=fc, edgecolor=ec, alpha=alpha, zorder=z, linestyle=ls))
        all_x.extend(vis_pts[:, 0]); all_y.extend(vis_pts[:, 1])

    def draw_line(pts, c, alpha=1.0, z=10, lw=VIS_FLEX_LINE_WIDTH):
        if len(pts) < 2: return
        v_pts = rot180(np.array(pts))
        ax.plot(v_pts[:, 0], v_pts[:, 1], color=c, alpha=alpha, linewidth=lw, zorder=z, solid_capstyle='round')
        all_x.extend(v_pts[:, 0]); all_y.extend(v_pts[:, 1])

    def get_path_len(pts):
        if len(pts) < 2: return 0.0
        return np.sum(np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1)))

    # 1. 绘制基座
    base_c = robot.c_list[0]; base_w, base_h = base_c * 1.6, 20
    vis_base = rot180(np.array([[-base_w/2, -base_h], [base_w/2, -base_h], [base_w/2, 0], [-base_w/2, 0]]))
    draw_poly(vis_base, '#333333', 'k')

    # 2. 绘制本体
    for i, data in enumerate(sections_data):
        color = colors[i % len(colors)]
        cur_c = data['c']; real_w = 1.3 * cur_c; h2 = robot.n 
        
        # 刚性块
        block_local = np.array([[-real_w/2, -h2, 1], [real_w/2, -h2, 1], [real_w/2, h2, 1], [-real_w/2, h2, 1]])
        draw_poly(rot180((data['T_start'] @ block_local.T).T[:, :2]), color, 'k')
        
        # 弹性体
        draw_line(data['flex_left'], color, 1.0, 3)
        draw_line(data['flex_right'], color, 1.0, 3)

        # 齿条
        sides = [('left', -cur_c/2.0), ('right', cur_c/2.0)]
        for side_name, x_off in sides:
            g_out = (data['T_start'] @ [x_off, -h2, 1])[:2]
            flex_pts_raw = data['flex_left'] if side_name == 'left' else data['flex_right']
            l_used = get_path_len(flex_pts_raw) + 2*h2
            l_rem = RACK_TOTAL_LEN - l_used
            if l_rem > 0:
                vec_down = (data['T_start'] @ [0, -1, 0])[:2] - (data['T_start'] @ [0, 0, 0])[:2]
                vec_down /= np.linalg.norm(vec_down)
                p_end = g_out + vec_down * l_rem
                draw_line([g_out, p_end], color, 1.0, 10)

    # 3. 绘制目标 Tip
    last_w = 1.3 * sections_data[-1]['c']
    tip_solid = np.array([[-last_w/2,-robot.n,1], [last_w/2,-robot.n,1], [last_w/2,robot.n,1], [-last_w/2,robot.n,1]])
    draw_poly(rot180((T_tip_global @ tip_solid.T).T[:, :2]), '#333333', 'k')
    
    # 坐标系箭头
    tip = -T_tip_global[:2, 2]; phi = np.arctan2(T_tip_global[1, 0], T_tip_global[0, 0]) + np.pi 
    ax.plot(tip[0], tip[1], 'o', color='white', mec='k', ms=5, zorder=22)
    L_arrow = robot.H0 * 0.5
    ax.arrow(tip[0], tip[1], L_arrow*np.cos(phi), L_arrow*np.sin(phi), head_width=L_arrow/6, color='red', zorder=22)
    ax.arrow(tip[0], tip[1], L_arrow*np.cos(phi-np.pi/2), L_arrow*np.sin(phi-np.pi/2), head_width=L_arrow/6, color='blue', zorder=22)

    # 4. 绘制视觉反馈
    if xy_feedback is not None and len(xy_feedback) >= 10:
        fb_pts_x = [0.0]; fb_pts_y = [0.0]
        # VisionState 是 (x, y, th) 但这里我们只拿到了 x, y
        # xy_feedback 长度如果是 10 (5x2) 或者 15 (5x3)
        stride = 3 if len(xy_feedback) == 15 else 2
        for i in range(5):
            p = rot180(np.array([xy_feedback[stride*i], xy_feedback[stride*i+1]]))
            fb_pts_x.append(p[0]); fb_pts_y.append(p[1])
        
        ax.plot(fb_pts_x, fb_pts_y, 'r-o', linewidth=2.0, markersize=5, mfc='white', mec='r', label='Vision', zorder=100)
        ax.plot(fb_pts_x[-1], fb_pts_y[-1], 'r*', markersize=12, mfc='yellow', mec='r', zorder=101)
        all_x.extend(fb_pts_x); all_y.extend(fb_pts_y)

    # 5. UI 布局
    if not all_x: all_x=[-100,100]; all_y=[-100,100]
    min_x, max_x = min(all_x), max(all_x); min_y, max_y = min(all_y), max(all_y)
    PAD = 20.0
    text_x, text_y = min_x - PAD, max_y + PAD - 20
    
    # 显示状态信息：末端位置 + 角度
    tip_x = xy_15d_target[-3]
    tip_y = xy_15d_target[-2]
    tip_th = xy_15d_target[-1]
    
    status = f"Tip X: {tip_x:.1f} | Y: {tip_y:.1f} | Th: {tip_th:.1f}°"
    if xy_feedback is not None:
        # 简单计算末端位置误差
        vis_x = xy_feedback[-3] if len(xy_feedback)==15 else xy_feedback[-2]
        vis_y = xy_feedback[-2] if len(xy_feedback)==15 else xy_feedback[-1]
        err = np.linalg.norm([vis_x - tip_x, vis_y - tip_y])
        status += f"\nErr: {err:.1f}mm"
    
    ax.text(text_x, text_y, status, fontsize=12, family='monospace', va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.9, ec='gray'), zorder=100)
    
    ax.set_aspect('equal'); ax.grid(True, ls=':', alpha=0.4)
    ax.set_xlim(text_x - 150, max_x + PAD); ax.set_ylim(min_y - PAD, max_y + PAD)
    ax.set_title("Robot Target Generator")

    return xy_15d_target

# ================= 🎮 ROS 通信类 =================
class RosComms(Node):
    def __init__(self):
        super().__init__('target_generator_node')
        # ✨ 话题 1: 完整的 15D 目标 (x, y, theta_deg) - 给高级控制器/数据记录
        self.pub_15d = self.create_publisher(Float32MultiArray, '/robot/target_pose_15d', 10)
        # ✨ 话题 2: 简单的 Tip 目标 (x, y) - 给简单的 Controller 节点
        self.pub_tip = self.create_publisher(Point, '/robot/target_pose', 10)
        
        self.sub_vision = self.create_subscription(VisionState, 'vision/state', self.vision_callback, 10)
        self.latest_vision_data = None
        self.get_logger().info("📡 Node Ready: Publishing to /robot/target_pose & /robot/target_pose_15d")

    def vision_callback(self, msg):
        # 存储 15D 数据以对齐显示 (x, y, th)
        temp = np.zeros(15)
        target_ids = [1, 2, 3, 4, 5]
        cnt = 0
        for i, vid in enumerate(target_ids):
            if vid in msg.ids:
                idx = msg.ids.index(vid)
                # 注意：这里 VisionState 通常是 x_local, y_local, theta(deg)
                # 我们假设 Vision 坐标系 x 也是反的（根据原始代码的 -msg.x_local）
                temp[3*i]   = -msg.x_local[idx]
                temp[3*i+1] = msg.y_local[idx]
                temp[3*i+2] = msg.theta[idx] # 角度直接存度数
                cnt += 1
        if cnt > 0: self.latest_vision_data = temp

    def send_target(self, xy_th_vector):
        if xy_th_vector is None: return
        
        # 1. 发送 15D 完整数据
        msg_15d = Float32MultiArray()
        msg_15d.data = xy_th_vector.tolist()
        self.pub_15d.publish(msg_15d)
        
        # 2. 发送简单的 Tip Point (只取最后3个值中的 x, y)
        msg_tip = Point()
        msg_tip.x = float(xy_th_vector[-3])
        msg_tip.y = float(xy_th_vector[-2])
        msg_tip.z = float(xy_th_vector[-1]) # 借用 z 存角度 (Optional)
        self.pub_tip.publish(msg_tip)
        
        print(f"\n>>>> 🚀 SENT TARGET (Tip X: {msg_tip.x:.1f}, Th: {msg_tip.z:.1f}°) <<<<\n")

# ================= 🎮 全局控制 =================
current_q = np.ones(10) * 10.0; STEP_SIZE = 10.0 
current_target_15d = None
ros_node = None
is_running = True
KEY_MAP = {'e':(8,1), 'd':(8,-1), 'w':(9,1), 's':(9,-1), 't':(6,1), 'g':(6,-1), 'r':(7,1), 'f':(7,-1), 'u':(4,1), 'j':(4,-1), 'y':(5,1), 'h':(5,-1), 'o':(2,1), 'l':(2,-1), 'i':(3,1), 'k':(3,-1), '[':(0,1), "'":(0,-1), 'p':(1,1), ';':(1,-1)}
pressed_keys = set()

def on_press(key):
    global is_running, ros_node, current_target_15d
    try:
        if hasattr(key, 'char'): 
            k = key.char.lower()
            if k in KEY_MAP: pressed_keys.add(k)
            if k == 'q': is_running = False
    except: pass
    
    if key == keyboard.Key.esc: is_running = False
    if key == keyboard.Key.space:
        if ros_node and current_target_15d is not None:
            ros_node.send_target(current_target_15d)

def on_release(key):
    try:
        if hasattr(key, 'char') and key.char.lower() in KEY_MAP: 
            pressed_keys.discard(key.char.lower())
    except: pass

def ros_spin_thread():
    rclpy.spin(ros_node)

def main(args=None):
    global ros_node, current_target_15d, current_q, is_running
    rclpy.init(args=args)
    ros_node = RosComms()
    
    t = threading.Thread(target=ros_spin_thread, daemon=True)
    t.start()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    print(f"🚀 仿真启动 (Target Angle Enabled)... 按 [SPACE] 发送")
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 10))
    robot = DiscreteContinuumRobot(n=22.0, H0=52.0, c_list=C_LIST_CONFIG, num_sections=5)
    
    try:
        while is_running and plt.fignum_exists(fig.number):
            if pressed_keys:
                for char in pressed_keys:
                    idx, dr = KEY_MAP[char]
                    current_q[idx] += dr * STEP_SIZE
                current_q = enforce_coupling_constraints(current_q)
            
            vis_data = ros_node.latest_vision_data
            # 返回 15D 数据
            current_target_15d = draw_frame(ax, robot, current_q, vis_data)
            
            plt.pause(0.02)
            
    except KeyboardInterrupt: pass
    finally:
        is_running = False
        listener.stop()
        plt.close('all')
        if rclpy.ok(): rclpy.shutdown()
        print("👋 结束")

if __name__ == "__main__":
    main()