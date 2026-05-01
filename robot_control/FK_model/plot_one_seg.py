import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import matplotlib.path as mpath

# ================= 🎨 配色方案 =================
PALETTE = {
    'light_blue': '#bce4e9', 
    'grey':       '#818486', 
    'red':        '#c61c22', 
    'teal':       '#00b0b0', 
}

AUX_COLORS = {
    'ochre':      '#D68910', 
    'flat_blue':  '#2980B9', 
}

# --- 实体颜色 ---
COLOR_BLOCK_FILL = PALETTE['light_blue'] 
COLOR_BLOCK_EDGE = PALETTE['grey']       
COLOR_RACK       = PALETTE['grey']       

# --- 🌟 FK 核心变量颜色 ---
COLOR_CI         = PALETTE['teal']       # c_i: 间距
COLOR_QIL        = PALETTE['red']        # q_{i,L}: 左齿条
COLOR_QIR        = AUX_COLORS['ochre']   # q_{i,R}: 右齿条
COLOR_THETA      = AUX_COLORS['flat_blue']  # \theta_i: 弯折角

# ================= ⚙️ 机器人模型 (FK 核心逻辑) =================
class DiscreteContinuumRobot:
    def __init__(self, n=12.0, H0=180.0, c_list=None, num_sections=1):
        if c_list is None: c_list = [50.0] * num_sections
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
            c_ang, s_ang = np.cos(-theta), np.sin(-theta)
            T_arc = np.array([[c_ang, -s_ang, lx], [s_ang, c_ang, ly], [0, 0, 1]])
        
        T_flex_end = T_flex_start @ T_arc
        T_end = T_flex_end @ T_n
        
        steps_m = 40; left_pts, right_pts, T_flex_list = [], [], []
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
            
            T_flex_list.append(T_curr)
            left_pts.append((T_curr @ [-half_w, 0, 1])[:2])
            right_pts.append((T_curr @ [half_w, 0, 1])[:2])

        return {'T_start': T_start, 'T_end': T_end, 'T_flex_start': T_flex_start, 'T_flex_end': T_flex_end,
                'flex_left': np.array(left_pts), 'flex_right': np.array(right_pts), 
                'T_flex_list': T_flex_list, 'c': current_c, 'theta': theta}

    def forward_visualize(self, q_array):
        T_curr = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        data = self.get_section_mesh(T_curr, q_array[0], q_array[1], 0)
        return [data]

# ================= 🎨 极简绘图函数 =================
def draw_fk_schematic(ax, robot, q):
    ax.clear()
    data = robot.forward_visualize(q)[0]
    all_points = []

    def draw_poly(vis_pts, fc, ec, alpha=0.8): 
        ax.add_patch(Polygon(vis_pts, facecolor=fc, edgecolor=ec, alpha=alpha, zorder=2, lw=1.5))
        all_points.extend(vis_pts)
        
    def draw_line(pts, c, lw=3.0, ls='-', z=3):
        ax.plot(pts[:, 0], pts[:, 1], color=c, linewidth=lw, linestyle=ls, zorder=z, solid_capstyle='round')
        all_points.extend(pts)

    # 1. 绘制刚性基座
    cur_c = data['c']; real_w = 1.2 * cur_c; h2 = robot.n
    block_local = np.array([[-real_w/2, -h2, 1], [real_w/2, -h2, 1], [real_w/2, h2, 1], [-real_w/2, h2, 1]])
    draw_poly((data['T_start'] @ block_local.T).T[:, :2], COLOR_BLOCK_FILL, COLOR_BLOCK_EDGE)
    draw_poly((data['T_end'] @ block_local.T).T[:, :2], COLOR_BLOCK_FILL, COLOR_BLOCK_EDGE)
    
    # 2. 绘制齿条
    draw_line(data['flex_left'], COLOR_RACK, lw=4, z=4)
    draw_line(data['flex_right'], COLOR_RACK, lw=4, z=4)
    half_w = cur_c / 2.0; tooth_len = 5.0
    for T in data['T_flex_list'][2:-2:3]:
        p_l = T @ [-half_w, 0, 1]; p_l_out = T @ [-half_w - tooth_len, 0, 1]
        ax.plot([p_l[0], p_l_out[0]], [p_l[1], p_l_out[1]], color=COLOR_RACK, lw=2, zorder=3)
        p_r = T @ [half_w, 0, 1]; p_r_out = T @ [half_w + tooth_len, 0, 1]
        ax.plot([p_r[0], p_r_out[0]], [p_r[1], p_r_out[1]], color=COLOR_RACK, lw=2, zorder=3)
        all_points.extend([p_l[:2], p_l_out[:2], p_r[:2], p_r_out[:2]])

    text_bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.95)

    # ================= 🎯 纯粹的 FK 变量标注 =================
    # A. c_i (结构宽度)
    p_c_top = (data['T_start'] @ [-half_w, -h2/2, 1])[:2]
    p_c_bot = (data['T_start'] @ [half_w, -h2/2, 1])[:2]
    ax.add_patch(FancyArrowPatch(posA=p_c_bot, posB=p_c_top, arrowstyle='<|-|>', mutation_scale=12, color=COLOR_CI, lw=1.5, zorder=10))
    p_ci_text = np.array([p_c_top[0] - 15, (p_c_top[1]+p_c_bot[1])/2])
    ax.text(p_ci_text[0], p_ci_text[1], r'$c_i$', fontsize=16, color=COLOR_CI, va='center', ha='right', zorder=15, fontweight='bold', bbox=text_bbox)
    all_points.extend([p_c_top, p_c_bot, p_ci_text])

    # B. q_{i,L} & q_{i,R} (直接贴合齿条的驱动量)
    offset_q = 18.0; ext_len = 4.0
    mid_idx = len(data['flex_left']) // 2; T_mid = data['T_flex_list'][mid_idx]

    # q_L
    for T_bound in [data['T_flex_start'], data['T_flex_end']]:
        p_in = (T_bound @ [-half_w, 0, 1])[:2]
        p_out = (T_bound @ [-half_w - offset_q - ext_len, 0, 1])[:2]
        ax.plot([p_in[0], p_out[0]], [p_in[1], p_out[1]], color=COLOR_QIL, lw=1.2, ls='--', alpha=0.6, zorder=4)
        all_points.extend([p_in, p_out])
        
    qL_pts = np.array([(T @ [-half_w - offset_q, 0, 1])[:2] for T in data['T_flex_list']])
    ax.add_patch(FancyArrowPatch(path=mpath.Path(qL_pts), arrowstyle='<|-|>', mutation_scale=10, color=COLOR_QIL, lw=1.8, zorder=5))
    p_qL = T_mid @ [-half_w - offset_q, 0, 1]
    ax.text(p_qL[0], p_qL[1], r'$q_{i,L}$', fontsize=18, color=COLOR_QIL, va='center', ha='center', zorder=15, fontweight='bold', bbox=text_bbox)
    all_points.extend(qL_pts); all_points.append(p_qL[:2])

    # q_R
    for T_bound in [data['T_flex_start'], data['T_flex_end']]:
        p_in = (T_bound @ [half_w, 0, 1])[:2]
        p_out = (T_bound @ [half_w + offset_q + ext_len, 0, 1])[:2]
        ax.plot([p_in[0], p_out[0]], [p_in[1], p_out[1]], color=COLOR_QIR, lw=1.2, ls='--', alpha=0.6, zorder=4)
        all_points.extend([p_in, p_out])
        
    qR_pts = np.array([(T @ [half_w + offset_q, 0, 1])[:2] for T in data['T_flex_list']])
    ax.add_patch(FancyArrowPatch(path=mpath.Path(qR_pts), arrowstyle='<|-|>', mutation_scale=10, color=COLOR_QIR, lw=1.8, zorder=5))
    p_qR = T_mid @ [half_w + offset_q, 0, 1]
    ax.text(p_qR[0], p_qR[1], r'$q_{i,R}$', fontsize=18, color=COLOR_QIR, va='center', ha='center', zorder=15, fontweight='bold', bbox=text_bbox)
    all_points.extend(qR_pts); all_points.append(p_qR[:2])

    # C. \theta_i (末端输出角度)
    theta = data['theta']
    if abs(theta) > 1e-4:
        P1_center = (data['T_flex_end'] @ [0, 0, 1])[:2]
        d0 = data['T_flex_start'][:2, 1]; d1 = data['T_flex_end'][:2, 1]
        ext_L = 50
        p_d0_end = P1_center + d0*ext_L; p_d1_end = P1_center + d1*ext_L
        ax.plot([P1_center[0], p_d0_end[0]], [P1_center[1], p_d0_end[1]], color=COLOR_THETA, lw=1.5, ls='-.', alpha=0.6, zorder=4)
        ax.plot([P1_center[0], p_d1_end[0]], [P1_center[1], p_d1_end[1]], color=COLOR_THETA, lw=1.5, ls='-.', alpha=0.6, zorder=4)
        
        a0 = np.arctan2(d0[1], d0[0]); a1 = np.arctan2(d1[1], d1[0]); arc_r = 35
        angles = np.linspace(a0, a1, 30)
        arc_pts = P1_center + arc_r * np.column_stack((np.cos(angles), np.sin(angles)))
        ax.plot(arc_pts[:, 0], arc_pts[:, 1], color=COLOR_THETA, lw=2.0, zorder=6)
        ax.add_patch(FancyArrowPatch(posA=arc_pts[1], posB=arc_pts[0], arrowstyle='-|>', mutation_scale=12, color=COLOR_THETA, zorder=6))
        ax.add_patch(FancyArrowPatch(posA=arc_pts[-2], posB=arc_pts[-1], arrowstyle='-|>', mutation_scale=12, color=COLOR_THETA, zorder=6))
        
        mid_angle = (a0 + a1) / 2
        text_pos_theta = P1_center + (arc_r + 15) * np.array([np.cos(mid_angle), np.sin(mid_angle)])
        ax.text(text_pos_theta[0], text_pos_theta[1], r'$\theta_i$', fontsize=18, color=COLOR_THETA, va='center', ha='center', zorder=15, fontweight='bold', bbox=text_bbox)
        all_points.extend([P1_center, p_d0_end, p_d1_end, text_pos_theta])
        all_points.extend(arc_pts)

    # ================= 🌟 自动裁剪与高保真输出 =================
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 1. 获取当前所有点的边界
    points_np = np.array(all_points)
    min_x, min_y = np.min(points_np, axis=0)
    max_x, max_y = np.max(points_np, axis=0)
    
    width = max_x - min_x
    height = max_y - min_y

    # 2. 设定横向留白系数 (可调)
    # 比如 0.5 表示左右各增加 50% 宽度的留白
    h_margin_ratio = 0.25 
    
    new_min_x = min_x - width * h_margin_ratio
    new_max_x = max_x + width * h_margin_ratio
    
    # 3. 关键：添加一个透明矩形来锁定 Bounding Box
    # 我们让矩形的高度精确等于数据的 min_y 到 max_y
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (new_min_x, min_y), 
        new_max_x - new_min_x, 
        height, 
        fill=False, 
        edgecolor='none', # 完全透明
        zorder=0
    )
    ax.add_patch(rect)

    # 4. 强制设置坐标轴范围
    ax.set_xlim(new_min_x, new_max_x)
    ax.set_ylim(min_y, max_y)

if __name__ == "__main__":
    robot = DiscreteContinuumRobot()
    current_q = np.array([-20.0, -60.0]) 
    
    # figsize 比例不一定要精准，因为 bbox_inches='tight' 会重新调整它
    fig, ax = plt.subplots(figsize=(10, 4)) 
    draw_fk_schematic(ax, robot, current_q)
    
    output_filename = "fk_schematic_clean.png"
    plt.savefig(
        output_filename, 
        dpi=600, 
        bbox_inches='tight', 
        # 必须设为 0，否则 Matplotlib 会在四个方向都默认加 0.1 inch 的白边
        pad_inches=0, 
        transparent=False, 
        facecolor='white'
    )
    print(f"✅ 已保存！")