import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from pynput import keyboard
import platform

# ================= 🚫 屏蔽 Matplotlib 默认快捷键 =================
for key in plt.rcParams:
    if key.startswith('keymap.'):
        plt.rcParams[key] = []

# ================= ⚙️ 视觉与几何配置 =================
C_LIST_CONFIG = [92.0, 108.0, 123.5, 140.0, 156.0] 
RACK_TOTAL_LEN = 240.0                             

VIS_FLEX_LINE_WIDTH = 3.0   
VIS_SHOW_GHOST = True       

# 动力学耦合参数
CPL_OFFSET = 10.0
CPL_COEFF = 0.17
CPL_EXP = 1.1

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
        steps_m = 15; left_pts, right_pts = [], []
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
        for i in range(self.num_sections):
            idx_base = (self.num_sections - 1 - i) * 2
            data = self.get_section_mesh(T_curr, q_array[idx_base], q_array[idx_base+1], i)
            data['id'] = i + 1
            all_sections.append(data)
            T_curr = data['T_end']
        return all_sections, T_curr

# ================= 🎨 绘图函数 (Root-to-Contact Logic) =================
def draw_frame(ax, robot, q):
    ax.clear()
    sections_data, T_tip_global = robot.forward_visualize(q)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    all_x, all_y = [], []
    def rot180(p): return -p

    # --- 基础工具 ---
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

    # 替换原来的 generate_bezier 或 generate_bezier_bridge
    def generate_c_curve(p_root, p_target, tangent_target, steps=20):
        p0 = np.array(p_root)   # 起点
        p2 = np.array(p_target) # 终点
        
        # --- 计算唯一的控制点 P1 (二次贝塞尔) ---
        # 1. P1 的 X 坐标必须与 P0 相同，以保证根部垂直出射
        p1_x = p0[0]
        
        # 2. P1 的 Y 坐标：通过目标点的切线反向推算
        # 斜率 k = dy/dx
        # 直线方程: y - p2_y = k * (x - p2_x)
        # 代入 x = p1_x 求 y
        
        # 防止除零 (垂直切线)
        if abs(tangent_target[0]) < 1e-5:
            # 如果目标切线也是垂直的，P1 取垂直中点
            p1_y = (p0[1] + p2[1]) / 2
        else:
            k = tangent_target[1] / tangent_target[0]
            p1_y = p2[1] + k * (p1_x - p2[0])
            
        # [安全限制] 
        # 强制 P1 的 Y 坐标位于 P0 和 P2 之间 (约 20%~80% 处)
        # 防止因切线角度过大导致曲线“飞”出去或折回
        y_min = min(p0[1], p2[1])
        y_max = max(p0[1], p2[1])
        dy = y_max - y_min
        
        # 限制范围：离两端至少保留 10% 的垂直距离
        p1_y = np.clip(p1_y, y_min + 0.1*dy, y_max - 0.1*dy)
        
        p1 = np.array([p1_x, p1_y])
        
        # --- 生成二次贝塞尔曲线点集 ---
        # B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
        t = np.linspace(0, 1, steps)
        curve_pts = []
        for ti in t:
            pt = (1-ti)**2 * p0 + 2*(1-ti)*ti * p1 + ti**2 * p2
            curve_pts.append(pt)
            
        return curve_pts

    # --- 绘制基座 ---
    base_c = robot.c_list[0]; base_w, base_h = base_c * 1.6, 20
    vis_base = rot180(np.array([[-base_w/2, -base_h], [base_w/2, -base_h], [base_w/2, 0], [-base_w/2, 0]]))
    draw_poly(vis_base, '#333333', 'k')

    # --- 核心：多级齿条绘制 ---
    rack_full_paths = {'left': None, 'right': None}
    
    # ⚠️ 1倍线宽偏移
    OFFSET_VAL = 1.0 * VIS_FLEX_LINE_WIDTH 

    for i, data in enumerate(sections_data):
        color = colors[i % len(colors)]
        cur_c = data['c']; real_w = 1.3 * cur_c; h2 = robot.n 
        
        # 1. 绘制本体
        block_local = np.array([[-real_w/2, -h2, 1], [real_w/2, -h2, 1], [real_w/2, h2, 1], [-real_w/2, h2, 1]])
        draw_poly(rot180((data['T_start'] @ block_local.T).T[:, :2]), color, 'k')
        if VIS_SHOW_GHOST:
            mw = max([1.3*c for c in C_LIST_CONFIG]); gw = (mw - real_w)/2
            if gw > 0.1:
                off = real_w/2 + gw/2
                lg = np.array([[-off-gw/2,-h2,1], [-off+gw/2,-h2,1], [-off+gw/2,h2,1], [-off-gw/2,h2,1]])
                rg = np.array([[off-gw/2,-h2,1], [off+gw/2,-h2,1], [off+gw/2,h2,1], [off-gw/2,h2,1]])
                draw_poly(rot180((data['T_start'] @ lg.T).T[:, :2]), color, 'k', 0.3, 15, '--')
                draw_poly(rot180((data['T_start'] @ rg.T).T[:, :2]), color, 'k', 0.3, 15, '--')
        draw_line(data['flex_left'], color, 1.0, 3)
        draw_line(data['flex_right'], color, 1.0, 3)

        # 2. 齿条逻辑
        sides = [('left', -cur_c/2.0, -1), ('right', cur_c/2.0, 1)]
        
        for side_name, x_off, sign in sides:
            # A. 根部起点
            g_in = (data['T_start'] @ [x_off, h2, 1])[:2]
            g_out = (data['T_start'] @ [x_off, -h2, 1])[:2] # 根部
            draw_line([g_in, g_out], color, 0.4, 21)

            # B. 计算尾巴总长
            flex_pts_raw = data['flex_left'] if side_name == 'left' else data['flex_right']
            l_used = get_path_len(flex_pts_raw) + 2*h2
            l_rem = RACK_TOTAL_LEN - l_used
            
            tail_final = []

            if l_rem > 0:
                # C. 生成自然路径 (直线，用于检测碰撞)
                vec_down = (data['T_start'] @ [0, -1, 0])[:2] - (data['T_start'] @ [0, 0, 0])[:2]
                vec_down /= np.linalg.norm(vec_down)
                natural_path = [g_out + vec_down * d for d in np.linspace(0, l_rem, int(l_rem*2)+5)] # 0.5mm step
                
                final_path = natural_path
                
                # D. 干涉检测
                is_collision = False
                collision_idx = -1
                ref_collision_idx = -1
                
                if i > 0 and rack_full_paths[side_name] is not None:
                    ref_path = np.array(rack_full_paths[side_name])
                    
                    # 遍历自然路径，寻找【最早】的干涉点
                    for k, pt in enumerate(natural_path):
                        # 寻找同Y参考点
                        y_diff = np.abs(ref_path[:, 1] - pt[1])
                        # 搜索范围放宽一点点避免漏判
                        valid_indices = np.where(y_diff < 1.0)[0]
                        
                        if len(valid_indices) > 0:
                            nearest_idx = valid_indices[np.argmin(y_diff[valid_indices])]
                            ref_pt = ref_path[nearest_idx]
                            
                            # 判定
                            GAP = 0.5
                            if (sign == -1 and pt[0] > ref_pt[0] - GAP) or \
                               (sign == 1 and pt[0] < ref_pt[0] + GAP):
                                is_collision = True
                                collision_idx = k
                                ref_collision_idx = nearest_idx
                                break
                
                if is_collision:
                    # === 触发根部拟合逻辑 ===
                    # 1. 确定拟合的终点 P_target
                    # P_target = 干涉点在参考路径上的位置 + 平移量
                    p_ref_hit = ref_path[ref_collision_idx]
                    p_target = p_ref_hit + np.array([sign * OFFSET_VAL, 0])
                    
                    # 2. 确定 P_target 处的切线
                    # 只需要简单的相邻点差分
                    p_prev = ref_path[max(0, ref_collision_idx-1)]
                    p_next = ref_path[min(len(ref_path)-1, ref_collision_idx+1)]
                    tangent = p_next - p_prev
                    
                    # 3. 生成根部曲线 (Root -> P_target)
                    # 直接舍弃 collision_idx 之前的所有自然路径
                    curve_fit = generate_c_curve(g_out, p_target, tangent, steps=20)
                    
                    # 4. 拼接后续路径 (沿着参考路径平移)
                    len_curve = get_path_len(curve_fit)
                    len_need = l_rem - len_curve
                    
                    track_segment = []
                    if len_need > 0:
                        curr_l = 0
                        # 确定方向：ref_path 存储时应该是从上到下的，所以 index 增加就是向下
                        # 简单检查一下
                        step_dir = 1
                        if ref_path[0][1] < ref_path[-1][1]: step_dir = -1
                        
                        idx_curr = ref_collision_idx
                        while 0 <= idx_curr < len(ref_path)-1:
                            next_idx = idx_curr + step_dir
                            if not (0 <= next_idx < len(ref_path)): break
                            
                            p1, p2 = ref_path[idx_curr], ref_path[next_idx]
                            # 忽略异常向上跳动
                            if p2[1] > p1[1]: 
                                idx_curr += step_dir; continue
                                
                            seg = np.linalg.norm(p2 - p1)
                            if curr_l + seg > len_need:
                                ratio = (len_need - curr_l) / seg
                                p_interp = p1 + (p2 - p1) * ratio
                                track_segment.append(p_interp + np.array([sign * OFFSET_VAL, 0]))
                                break
                            
                            track_segment.append(p2 + np.array([sign * OFFSET_VAL, 0]))
                            curr_l += seg
                            idx_curr += step_dir
                            
                    final_path = curve_fit + track_segment
                    
                    # 视觉装饰：在汇合点画个小圆点
                    ax.add_patch(Circle(rot180(p_target), VIS_FLEX_LINE_WIDTH*0.2, color=color, zorder=12))

                tail_final = final_path
                draw_line(tail_final, color, 1.0, 10)

            # 3. 存储全路径
            flex_reversed = flex_pts_raw[::-1] 
            block_path = [g_in, g_out]
            if len(tail_final) > 0:
                full_path_combined = np.vstack([flex_reversed, block_path, tail_final])
            else:
                full_path_combined = np.vstack([flex_reversed, block_path])
            rack_full_paths[side_name] = full_path_combined

    # 3. 终端 UI
    last_w = 1.3 * sections_data[-1]['c']
    tip_solid = np.array([[-last_w/2,-robot.n,1], [last_w/2,-robot.n,1], [last_w/2,robot.n,1], [-last_w/2,robot.n,1]])
    draw_poly(rot180((T_tip_global @ tip_solid.T).T[:, :2]), '#333333', 'k')
    
    tip = -T_tip_global[:2, 2]; phi = np.arctan2(T_tip_global[1, 0], T_tip_global[0, 0]) + np.pi 
    ax.plot(tip[0], tip[1], 'o', color='white', mec='k', ms=5, zorder=22)
    L = robot.H0 * 0.5
    ax.arrow(tip[0], tip[1], L*np.cos(phi), L*np.sin(phi), head_width=L/6, color='red', zorder=22, lw=2)
    ax.arrow(tip[0], tip[1], L*np.cos(phi-np.pi/2), L*np.sin(phi-np.pi/2), head_width=L/6, color='blue', zorder=22, lw=2)

    # UI 布局
    if not all_x: all_x=[-100,100]; all_y=[-100,100]
    min_x, max_x = min(all_x), max(all_x); min_y, max_y = min(all_y), max(all_y)
    PAD = 20.0
    text_x, text_y = min_x - PAD, max_y + PAD - 20
    
    guide_text = "      CONTROLS\n-------------------\nLeft Rack | Right Rack\n ( + / - ) | ( + / - )\n-------------------\nJ5 (Tip) : P/; | [/'\nJ4       : I/K | O/L\nJ3       : Y/H | U/J\nJ2       : R/F | T/G\nJ1 (Base): W/S | E/D\n-------------------\nQUIT: Q/Esc"
    ax.text(text_x, text_y, guide_text, fontsize=10, family='monospace', va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.9, ec='gray'), zorder=100)
    
    ax.set_aspect('equal'); ax.grid(True, ls=':', alpha=0.4)
    ax.set_xlim(text_x - 220, max_x + PAD)
    ax.set_ylim(min_y - PAD, max_y + PAD)
    ax.set_title(f"Coupled Mode | Root-to-Contact Fitting\nOffset: {OFFSET_VAL:.1f}mm | q: {np.round(q, 1)}")

# ================= 🎮 控制逻辑 =================
current_q = np.ones(10) * 10.0; STEP_SIZE = 2.0 
KEY_MAP = {'e':(8,1), 'd':(8,-1), 'w':(9,1), 's':(9,-1), 't':(6,1), 'g':(6,-1), 'r':(7,1), 'f':(7,-1), 'u':(4,1), 'j':(4,-1), 'y':(5,1), 'h':(5,-1), 'o':(2,1), 'l':(2,-1), 'i':(3,1), 'k':(3,-1), '[':(0,1), "'":(0,-1), 'p':(1,1), ';':(1,-1)}
pressed_keys, is_running = set(), True

def on_press(key):
    global is_running
    try:
        if hasattr(key, 'char'): k = key.char.lower(); 
        if k in KEY_MAP: pressed_keys.add(k)
        if k == 'q': is_running = False
    except: pass
    if key == keyboard.Key.esc: is_running = False

def on_release(key):
    try:
        if hasattr(key, 'char') and key.char.lower() in KEY_MAP: pressed_keys.discard(key.char.lower())
    except: pass

if __name__ == "__main__":
    print(f"🚀 系统启动... OS: {platform.system()}")
    listener = keyboard.Listener(on_press=on_press, on_release=on_release); listener.start()
    plt.ion(); fig, ax = plt.subplots(figsize=(12, 10))
    robot = DiscreteContinuumRobot(n=22.0, H0=52.0, c_list=C_LIST_CONFIG, num_sections=5)
    try:
        while is_running:
            if pressed_keys:
                for char in pressed_keys: idx, dr = KEY_MAP[char]; current_q[idx] += dr * STEP_SIZE
                current_q = enforce_coupling_constraints(current_q)
            draw_frame(ax, robot, current_q); plt.pause(0.02)
    except KeyboardInterrupt: pass
    finally: listener.stop(); plt.close(); print("👋 仿真结束")