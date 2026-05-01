import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Circle
from matplotlib.lines import Line2D

# ================= 1. 物理参数 =================
C_LIST = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL, H0_VAL = 22.0, 52.0
M_VAL = H0_VAL - 2 * N_VAL
SEG_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# ================= 2. 网络架构 =================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.gelu = nn.GELU()
    def forward(self, x): return self.gelu(x + self.net(x))

class TopoImpulseNet(nn.Module):
    def __init__(self, scalers, input_dim=15, feature_dim=128, gru_dim=128, dropout=0.2):
        super().__init__()
        self.register_buffer('jac_mean', scalers['jacobian_mean'])
        self.register_buffer('jac_std', scalers['jacobian_std'])
        self.register_buffer('cmd_mean', scalers['dq_cmd_mean'])
        self.register_buffer('cmd_std', scalers['dq_cmd_std'])
        self.num_nodes = 5
        
        self.stage1_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, feature_dim), nn.LayerNorm(feature_dim),
                nn.GELU(), nn.Dropout(dropout),
                ResidualBlock(feature_dim, dropout), ResidualBlock(feature_dim, dropout) 
            ) for _ in range(self.num_nodes)
        ])
        
        self.stage2_gru = nn.GRU(
            input_size=feature_dim, hidden_size=gru_dim,
            num_layers=2, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.gru_dropout = nn.Dropout(dropout)
        
        head_input_dim = feature_dim + (gru_dim * 2) + 6 
        
        self.net_expert_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, 128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3) 
            ) for _ in range(self.num_nodes)
        ])
        
        self.confidence_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, 64), nn.Tanh(),
                nn.Linear(64, 3), nn.Sigmoid() 
            ) for _ in range(self.num_nodes)
        ])

    def compute_nominal_delta(self, jac_norm, dq_cmd_norm):
        B, N, _ = jac_norm.shape
        jac_real = jac_norm * self.jac_std + self.jac_mean
        dq_real = dq_cmd_norm * self.cmd_std + self.cmd_mean
        J_matrix = jac_real.view(B, N, 3, 2)
        nominal_delta = torch.matmul(J_matrix, dq_real.unsqueeze(-1)).squeeze(-1)
        return nominal_delta

    def forward(self, x_inputs, jacobian_norm, dq_cmd_norm, return_internals=True, debug_mode=False):
        B = x_inputs.shape[0]
        local_features = [self.stage1_experts[i](x_inputs[:, i, :]) for i in range(self.num_nodes)]
        gru_out, _ = self.stage2_gru(torch.stack(local_features, dim=1))
        gru_out = self.gru_dropout(gru_out)
        nominal_delta = self.compute_nominal_delta(jacobian_norm, dq_cmd_norm)
        
        outputs, betas = [], []
        for i in range(self.num_nodes):
            feat = torch.cat([local_features[i], gru_out[:, i, :], jacobian_norm[:, i, :]], dim=1)
            net_prediction = self.net_expert_head[i](feat)
            beta = self.confidence_gate[i](feat)
            if debug_mode: outputs.append(nominal_delta[:, i, :])
            else: outputs.append(beta * nominal_delta[:, i, :] + (1.0 - beta) * net_prediction)
            betas.append(beta)
            
        final_stack = torch.stack(outputs, dim=1)
        return (final_stack, nominal_delta, torch.stack(betas, dim=1)) if return_internals else final_stack

# ================= 3. 辅助算法 =================
def compute_local_jacobian(q_pair, idx):
    eps = 1e-4; c_val = C_LIST[idx]
    def get_lp(ql, qr):
        th = (ql - qr) / c_val; lc = M_VAL + (ql + qr) / 2.0
        if abs(th) < 1e-6: return np.array([0.0, 2*N_VAL + lc, 0.0])
        rho = lc / th
        lx, ly = rho * (1.0 - np.cos(th)), rho * np.sin(th)
        return np.array([np.sin(th)*N_VAL + lx, np.cos(th)*N_VAL + ly + N_VAL, -th])
    
    curr = get_lp(q_pair[0], q_pair[1])
    j = np.zeros((3, 2))
    j[:, 0] = (get_lp(q_pair[0]+eps, q_pair[1]) - get_lp(q_pair[0]-eps, q_pair[1])) / (2*eps)
    j[:, 1] = (get_lp(q_pair[0], q_pair[1]+eps) - get_lp(q_pair[0], q_pair[1]-eps) ) / (2*eps)
    return j.flatten()

# ================= 4. 高保真渲染 =================
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
                        rem_s = s - (N_VAL + L_c)
                        arc_end_x, arc_end_y = rho * (1.0 - np.cos(theta)), N_VAL + rho * np.sin(theta)
                        lx, ly = arc_end_x + np.sin(theta)*rem_s, arc_end_y + np.cos(theta)*rem_s
                p_glob = T_curr @ np.array([lx, ly, 1]); current_seg_pts.append(p_glob[:2])
            segment_points_list.append(np.array(current_seg_pts))
        
        th_l = (q_l - q_r) / C_LIST[i]; lc_l = M_VAL + (q_l + q_r) / 2.0
        if abs(th_l) < 1e-12: lx_l, ly_l = 0.0, 2*N_VAL + lc_l
        else:
            rho_l = lc_l / th_l
            lx_arc, ly_arc = rho_l * (1.0 - np.cos(th_l)), rho_l * np.sin(th_l)
            lx_l = np.sin(th_l)*N_VAL + lx_arc
            ly_l = np.cos(th_l)*N_VAL + ly_arc + N_VAL
        
        c, s = np.cos(-th_l), np.sin(-th_l)
        T_local = np.array([[c, -s, lx_l], [s, c, ly_l], [0, 0, 1]])
        T_curr = T_curr @ T_local
        node_poses.append([T_curr[0, 2], T_curr[1, 2], np.arctan2(T_curr[1, 0], T_curr[0, 0])])
            
    return np.array(node_poses), segment_points_list

# ================= 5. 仿真逻辑 =================
class HybridPhantomSimV11:
    def __init__(self, m_path, d_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_resources(m_path, d_path)
        
        self.q_robot = np.ones((5, 2)) * 10.0
        self.q_target = np.ones((5, 2)) * 10.0
        self.q_prev = self.q_robot.copy()
        self.use_neural = False
        self.selected_idx = 4
        
        # 🔥 动态权重引擎状态
        self.peak_mu = 0.0 

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.ax.set_aspect('equal'); self.ax.set_xlim(-400, 400); self.ax.set_ylim(-100, 700); self.ax.grid(True, ls=':', alpha=0.5)
        
        self.lines_robot = [self.ax.plot([], [], '-', lw=5, color=SEG_COLORS[i], alpha=0.9, solid_capstyle='round')[0] for i in range(5)]
        self.lines_target = [self.ax.plot([], [], '--', lw=2, color=SEG_COLORS[i], alpha=0.6)[0] for i in range(5)]
        self.joints_robot, = self.ax.plot([], [], 'ko', ms=4, zorder=20)
        self.pt_selected, = self.ax.plot([], [], 'ro', markersize=12, mec='k', mew=2, zorder=25)
        
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.timer = self.fig.canvas.new_timer(interval=30); self.timer.add_callback(self.control_loop); self.timer.start()
        print("💡 Press 'N' to toggle Neural Mode | 1-5 to select segment")
        plt.show()

    def load_resources(self, m_path, d_path):
        data = torch.load(d_path, map_location=self.device)
        self.scalers = data['scalers']
        self.model = TopoImpulseNet(self.scalers).to(self.device)
        checkpoint = torch.load(m_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        self.model.eval()
        print(f"✅ Model Loaded Successfully from {m_path}")

    def on_key(self, event):
        if event.key in ['1', '2', '3', '4', '5']: self.selected_idx = int(event.key) - 1
        if event.key == 'n': self.use_neural = not self.use_neural

    def on_move(self, event):
        if event.inaxes != self.ax or event.button != 1: return
        poses, _ = forward_kinematics_full_chain(self.q_target)
        prev_p = poses[self.selected_idx-1][:3] if self.selected_idx > 0 else [0,0,0]
        dx, dy = event.xdata - prev_p[0], event.ydata - prev_p[1]
        c, s = np.cos(prev_p[2]), np.sin(prev_p[2])
        local_x, local_y = c*dx + s*dy, -s*dx + c*dy
        
        q = self.q_target[self.selected_idx].copy()
        for _ in range(5):
            th = (q[0]-q[1])/C_LIST[self.selected_idx]; lc = M_VAL+(q[0]+q[1])/2.0
            if abs(th)<1e-12: p = np.array([0.0, 2*N_VAL+lc])
            else:
                rho = lc/th; lx, ly = rho*(1-np.cos(th)), rho*np.sin(th)
                p = np.array([np.sin(th)*N_VAL+lx, np.cos(th)*N_VAL+ly+N_VAL])
            err = np.array([local_x, local_y]) - p
            if np.linalg.norm(err) < 0.1: break
            j = np.zeros((2, 2)); eps = 1e-3
            for i in range(2):
                q_p = q.copy(); q_p[i] += eps
                th_p = (q_p[0]-q_p[1])/C_LIST[self.selected_idx]; lc_p = M_VAL+(q_p[0]+q_p[1])/2.0
                if abs(th_p)<1e-12: p_p = np.array([0.0, 2*N_VAL+lc_p])
                else:
                    rho_p = lc_p/th_p; lx_p, ly_p = rho_p*(1-np.cos(th_p)), rho_p*np.sin(th_p)
                    p_p = np.array([np.sin(th_p)*N_VAL+lx_p, np.cos(th_p)*N_VAL+ly_p+N_VAL])
                j[:, i] = (p_p - p) / eps
            try: q += np.linalg.inv(j) @ (err * 0.8)
            except: break
        self.q_target[self.selected_idx] = np.clip(q, 0, 160)

    def global_to_local_poses(self, global_poses):
        p_l = []
        prev_p = np.array([0.0, 0.0, 0.0])
        for i in range(5):
            curr_p = global_poses[i]
            th_p = prev_p[2]
            dx, dy = curr_p[0] - prev_p[0], curr_p[1] - prev_p[1]
            lx = dx * np.cos(th_p) + dy * np.sin(th_p)
            ly = -dx * np.sin(th_p) + dy * np.cos(th_p)
            lth = curr_p[2] - prev_p[2]
            p_l.append([lx, ly, lth])
            prev_p = curr_p
        return np.array(p_l)

    def control_loop(self):
        vision_rob_poses, rob_segs = forward_kinematics_full_chain(self.q_robot, return_arc_points=True)
        vision_targ_poses, targ_segs = forward_kinematics_full_chain(self.q_target, return_arc_points=True)
        
        base_xy = vision_rob_poses[:, :2].flatten()
        error = vision_targ_poses[:, :2].flatten() - base_xy
        
        eps = 1e-3; J_phy = np.zeros((10, 10))
        for i in range(10):
            q_f = self.q_robot.flatten(); q_f[i] += eps
            new_p, _ = forward_kinematics_full_chain(q_f.reshape(5, 2))
            J_phy[:, i] = (new_p[:, :2].flatten() - base_xy) / eps
        
        curr_J = J_phy; beta_val = 1.0

        if self.use_neural:
            def norm(v, k): return (torch.from_numpy(v).float() - self.scalers[f'{k}_mean'].cpu()) / self.scalers[f'{k}_std'].cpu()
            
            real_local_poses = self.global_to_local_poses(vision_rob_poses)
            
            q_t = norm(self.q_robot, 'q_curr').to(self.device).unsqueeze(0)
            dq_h = norm(self.q_robot - self.q_prev, 'dq_hist').to(self.device).unsqueeze(0)
            p_l_t = norm(real_local_poses, 'pose_loc').to(self.device).unsqueeze(0)
            
            jac_feats = np.array([compute_local_jacobian(self.q_robot[i], i) for i in range(5)])
            j_f_t = norm(jac_feats, 'jacobian').to(self.device).unsqueeze(0)
            
            batch_size = 11; dq_eps = 1e-3
            dq_cmd_batch = torch.zeros((batch_size, 5, 2), device=self.device)
            for i in range(10):
                joint_idx, motor_idx = i // 2, i % 2
                dq_cmd_batch[i+1, joint_idx, motor_idx] = dq_eps 
            
            q_t_b = q_t.repeat(batch_size, 1, 1)
            dq_h_b = dq_h.repeat(batch_size, 1, 1)
            p_l_t_b = p_l_t.repeat(batch_size, 1, 1)
            j_f_t_b = j_f_t.repeat(batch_size, 1, 1)
            
            x_in_batch = torch.cat([q_t_b, dq_h_b, p_l_t_b, dq_cmd_batch, j_f_t_b], dim=2)
            
            with torch.no_grad():
                pred_local_batch, _, beta_batch = self.model(x_in_batch, j_f_t_b, dq_cmd_batch, return_internals=True)
                current_local_raw_b = torch.from_numpy(real_local_poses).float().to(self.device).unsqueeze(0).repeat(batch_size, 1, 1)
                next_local_batch = current_local_raw_b + pred_local_batch 
                
                T_curr = torch.eye(3, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
                global_points = []
                for i in range(5):
                    lx, ly, lth = next_local_batch[:, i, 0], next_local_batch[:, i, 1], next_local_batch[:, i, 2]
                    cos_t, sin_t = torch.cos(lth), torch.sin(lth)
                    
                    r0 = torch.stack([cos_t, -sin_t, lx], dim=1)
                    r1 = torch.stack([sin_t, cos_t, ly], dim=1)
                    r2 = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(batch_size, 1)
                    T_local = torch.stack([r0, r1, r2], dim=1)
                    
                    T_curr = torch.bmm(T_curr, T_local)
                    global_points.append(T_curr[:, 0, 2])
                    global_points.append(T_curr[:, 1, 2])
                
                full_global_xy_batch = torch.stack(global_points, dim=1) 
            
            nominal_xy = full_global_xy_batch[0]
            perturbed_xy = full_global_xy_batch[1:]
            
            J_net_norm = ((perturbed_xy - nominal_xy) / dq_eps).T.cpu().numpy()
            
            dq_std_raw = self.scalers['dq_cmd_std'].cpu().numpy().flatten()
            if dq_std_raw.size == 2: dq_std_full = np.tile(dq_std_raw, 5)
            else: dq_std_full = dq_std_raw
                
            J_net_global = J_net_norm / dq_std_full.reshape(1, 10)
            
            beta_val = beta_batch[0, 4, 0].item()
            curr_J = beta_val * J_phy + (1.0 - beta_val) * J_net_global

        # ==========================================================
        # 🔥 动态高斯权重生成引擎
        # ==========================================================
        # 1. 计算每个节点的 L2 位置误差
        node_errors = np.linalg.norm(error.reshape(5, 2), axis=1)
        
        # 2. 寻找目标峰值位置 (寻找第一个误差越界的节点)
        convergence_threshold = 5.0 # 收敛容忍度 (mm)，可微调
        target_mu = 4.0 # 默认峰值在最末端
        for i in range(5):
            if node_errors[i] > convergence_threshold:
                target_mu = float(i)
                break
                
        # 3. 峰值平滑漂移 (一阶低通滤波)
        self.peak_mu = 0.85 * self.peak_mu + 0.15 * target_mu
        
        # 4. 生成高斯权重分布
        sigma = 1.0 
        w_min = 0.5 # 保证收敛的节点仍留有微小的修正能力
        gaussian_weights = np.exp(-((np.arange(5) - self.peak_mu)**2) / (2 * sigma**2)) + w_min
        gaussian_weights = gaussian_weights / np.max(gaussian_weights) # 归一化，最大权重始终为 1.0
        
        # 扩展至 10 维对角矩阵 (每个节点复制两次，对应 X 和 Y)
        weight_array = np.repeat(gaussian_weights, 2)
        W = np.diag(weight_array)

        # ==========================================================
        # 5. 加权稳定性控制
        # ==========================================================
        try:
            lambda_val = 0.2 if self.use_neural else 0.05
            gain_val = 0.2 if self.use_neural else 0.8
            # 引入 W 矩阵
            dq = np.linalg.inv(curr_J.T @ W @ curr_J + lambda_val * np.eye(10)) @ curr_J.T @ W @ (error * gain_val)
        except: dq = np.zeros(10)
        
        self.q_prev = self.q_robot.copy(); self.q_robot += dq.reshape(5, 2); self.q_robot = np.clip(self.q_robot, 0, 160)
        
        # 渲染更新
        for i in range(5):
            self.lines_robot[i].set_data(rob_segs[i][:, 0], rob_segs[i][:, 1])
            self.lines_target[i].set_data(targ_segs[i][:, 0], targ_segs[i][:, 1])
        j_r = np.array([np.array([0,0])] + [p[-1] for p in rob_segs])
        self.joints_robot.set_data(j_r[:, 0], j_r[:, 1])
        self.pt_selected.set_data([vision_targ_poses[self.selected_idx, 0]], [vision_targ_poses[self.selected_idx, 1]])
        
        # UI 显示当前的动态峰值位置 (0.0=根部，4.0=末端)
        self.ax.set_title(f"V11.7 Dynamic W | Mode: {'NEURAL' if self.use_neural else 'PHY'} | Peak: Node {self.peak_mu:.1f}")
        self.fig.canvas.draw()

if __name__ == "__main__":
    m_p = "/home/brandon/brandon/hyrd_robot/lifelong_data/experiments/001/best_model.pth"
    d_p = "/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_smooth_strided.pt"
    HybridPhantomSimV11(m_p, d_p)