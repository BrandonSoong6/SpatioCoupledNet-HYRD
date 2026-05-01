import numpy as np
import torch
import torch.nn as nn
import os
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Circle
from matplotlib.lines import Line2D
from pathlib import Path

# ================= 1. ROS 2 环境与通信桥梁 =================
try:
    import rclpy
    from rclpy.node import Node
    from robot_interfaces.msg import MotorCommand, MotorState, VisionState
    ROS_ENABLED = True
except ImportError:
    ROS_ENABLED = False
    print("⚠️ ROS 2 imports failed. Running in Pure Simulation Mode.")

class RealRobotBridge(Node if ROS_ENABLED else object):
    def __init__(self, parent_sim):
        if not ROS_ENABLED: return
        super().__init__('sim2real_orchestrator')
        self.parent = parent_sim
        
        self.LOGICAL_TO_PHYSICAL_IDS = [9, 10, 7, 8, 5, 6, 3, 4, 1, 2]
        self.EXPECTED_VISION_IDS = [1, 2, 3, 4, 5]
        self.MIRROR_X = False 
        
        self.pub_cmd = self.create_publisher(MotorCommand, '/motor_cmd', 10)
        self.sub_state = self.create_subscription(MotorState, '/motor_state', self.motor_cb, 10)
        self.sub_vision = self.create_subscription(VisionState, '/vision/state', self.vision_cb, 10)

    def motor_cb(self, msg):
        q_logical = np.zeros(10)
        m_dict = {mid: pos for mid, pos in zip(msg.ids, msg.positions)}
        
        all_present = True
        for i, expected_id in enumerate(self.LOGICAL_TO_PHYSICAL_IDS):
            if expected_id in m_dict:
                q_logical[i] = m_dict[expected_id]
            else:
                all_present = False
                
        if all_present:
            self.parent.real_q_robot = q_logical.reshape(5, 2)

    def vision_cb(self, msg):
        if not msg.is_plane_locked: return
        
        poses = np.zeros((5, 3))
        v_dict = {}
        for i, vid in enumerate(msg.ids):
            x = msg.x_local[i]
            y = msg.y_local[i]
            th = np.radians(msg.theta[i])
            if self.MIRROR_X:
                x = -x; th = -th
            v_dict[vid] = (x, y, th)
        
        all_present = True
        for i, expected_id in enumerate(self.EXPECTED_VISION_IDS):
            if expected_id in v_dict:
                poses[i] = v_dict[expected_id]
            else:
                all_present = False
                
        if all_present:
            self.parent.real_vision_poses = poses

    def send_rpm_command(self, rpms):
        msg = MotorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ids = self.LOGICAL_TO_PHYSICAL_IDS
        msg.target_rpms = rpms.tolist()
        self.pub_cmd.publish(msg)

# ================= 2. 物理参数与网络架构 =================
C_LIST = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL, H0_VAL = 22.0, 52.0
M_VAL = H0_VAL - 2 * N_VAL
SEG_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
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
        self.stage1_experts = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, feature_dim), nn.LayerNorm(feature_dim), nn.GELU(), nn.Dropout(dropout), ResidualBlock(feature_dim, dropout), ResidualBlock(feature_dim, dropout)) for _ in range(self.num_nodes)])
        self.stage2_gru = nn.GRU(input_size=feature_dim, hidden_size=gru_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.gru_dropout = nn.Dropout(dropout)
        head_input_dim = feature_dim + (gru_dim * 2) + 6 
        self.net_expert_head = nn.ModuleList([nn.Sequential(nn.Linear(head_input_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3)) for _ in range(self.num_nodes)])
        self.confidence_gate = nn.ModuleList([nn.Sequential(nn.Linear(head_input_dim, 64), nn.Tanh(), nn.Linear(64, 3), nn.Sigmoid()) for _ in range(self.num_nodes)])

    def compute_nominal_delta(self, jac_norm, dq_cmd_norm):
        B, N, _ = jac_norm.shape
        jac_real = jac_norm * self.jac_std + self.jac_mean
        dq_real = dq_cmd_norm * self.cmd_std + self.cmd_mean
        J_matrix = jac_real.view(B, N, 3, 2)
        return torch.matmul(J_matrix, dq_real.unsqueeze(-1)).squeeze(-1)

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

def compute_local_jacobian(q_pair, idx):
    eps = 1e-4; c_val = C_LIST[idx]
    def get_lp(ql, qr):
        th = (ql - qr) / c_val; lc = M_VAL + (ql + qr) / 2.0
        if abs(th) < 1e-6: return np.array([0.0, 2*N_VAL + lc, 0.0])
        rho = lc / th; lx, ly = rho * (1.0 - np.cos(th)), rho * np.sin(th)
        return np.array([np.sin(th)*N_VAL + lx, np.cos(th)*N_VAL + ly + N_VAL, -th])
    curr = get_lp(q_pair[0], q_pair[1]); j = np.zeros((3, 2))
    j[:, 0] = (get_lp(q_pair[0]+eps, q_pair[1]) - get_lp(q_pair[0]-eps, q_pair[1])) / (2*eps)
    j[:, 1] = (get_lp(q_pair[0], q_pair[1]+eps) - get_lp(q_pair[0], q_pair[1]-eps) ) / (2*eps)
    return j.flatten()

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

# ================= 3. Sim2Real 控制逻辑 =================
class HybridPhantomSimV11:
    def __init__(self, m_path, d_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_resources(m_path, d_path)
        
        self.q_robot = np.ones((5, 2)) * 10.0
        self.q_target = np.ones((5, 2)) * 10.0
        self.q_prev = self.q_robot.copy()

        self.PRESET_TARGETS = {
            'easy': np.array([
                [80.36218746, 100.18070362],
                [100.31108559, 80.55384305],
                [100.0, 70.05793281],
                [90.0, 70.73016357],
                [80.0, 60.88721918]
            ]),
            'medium': np.array([
                [146.64968812,  66.36454253],
                [ 100.18175405, 150.42953366],
                [ 68.00878881, 157.        ],
                [157.        ,  130.25337291],
                [157.        ,  51.16214419]]),
            'extreme': np.array([
                [134.17019636, 53.88505077],
                [156.48704, 62.23926039],
                [157.0, 60.73030101],
                [70.03693031, 157.0],
                [64.44033214, 157.0]
            ])
        }
        
        self.CONTROL_MODES = ["PHY", "HYBRID", "PURE_NN"]
        self.control_mode_idx = 0 
        self.use_real_robot = False   
        self.real_q_robot = None      
        self.real_vision_poses = None 
        
        self.selected_idx = 4
        self.peak_mu = 0.0 
        self.is_shutting_down = False 

        # === 录制好的完整 11 帧目标序列 (严格不删减) ===
        self.seq_idx = 0
        self.TARGET_SEQUENCE = [
            # 1. 初始状态 (Base 10)
            np.array([[10.00, 10.00], [10.00, 10.00], [10.00, 10.00], [10.00, 10.00], [10.00, 10.00]]),
            
            # 2. 全段均匀伸长
            np.array([[29.84, 29.84], [29.84, 29.84], [29.84, 29.84], [29.84, 29.84], [29.84, 29.84]]),
            
            # 3. 根部(Node 1&2)初步弯曲
            np.array([[36.13, 54.94], [33.20, 50.94], [29.84, 29.84], [29.84, 29.84], [29.84, 29.84]]),
            
            # 4. 整体协调变形 (第一次)
            np.array([[43.79, 84.03], [41.31, 90.10], [53.50, 55.71], [45.76, 49.58], [47.13, 40.35]]),
            
            # 5. 整体协调变形 (第二次，幅度加大)
            np.array([[60.73, 120.98], [71.18, 115.95], [94.35, 94.76], [94.71, 138.10], [108.40, 135.04]]),
            
            # 6. 根部强化弯曲 (Node 1&2 到达 150+ 区域)
            np.array([[78.14, 157.00], [90.69, 153.79], [94.35, 94.76], [94.71, 138.10], [108.40, 135.04]]),
            
            # 7. 中间节(Node 3)大幅压缩
            np.array([[78.14, 157.00], [90.69, 153.79], [68.60, 156.69], [94.71, 138.10], [108.40, 135.04]]),
            
            # 8. 全段反向耦合调整 (Node 4&5 加入)
            np.array([[78.14, 157.00], [90.69, 153.79], [54.01, 157.00], [57.41, 157.00], [27.31, 76.90]]),
            
            # 9. 根部 Node 1 回拉微调
            np.array([[88.03, 103.85], [90.69, 153.79], [54.01, 157.00], [57.41, 157.00], [27.31, 76.90]]),
            
            # 10. 全段极限位置调整 (J-Shape 雏形)
            np.array([[157.00, 79.72], [90.69, 153.79], [54.01, 157.00], [57.41, 157.00], [27.31, 76.90]]),
            
            # 11. 最终极端复合位姿 (Extreme Coupling)
            np.array([[157.00, 79.72], [157.00, 157.00], [54.01, 157.00], [157.00, 51.16], [157.00, 121.15]])
        ]
        # ================================

        # === 核心修改区 1：扩充离线数据记录结构 ===
        self.control_active = False
        self.history_dq = []
        self.history_err = []
        self.history_all_poses = []
        self.history_q = []      # 新增: 记录真实的关节配置空间 Q
        self.history_betas = []  # 新增: 记录控制过程中的 10 维置信度门控
        
        # 保存录制开始时的静态快照
        self.rec_q_target = None
        self.rec_target_poses = None
        self.start_ee_pos = None
        self.target_ee_pos = None
        # ==========================================

        if ROS_ENABLED:
            rclpy.init()
            self.ros_node = RealRobotBridge(self)
            self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.ros_node,), daemon=True)
            self.ros_thread.start()
            print("🟢 ROS 2 Bridge Started in Background.")
            from std_msgs.msg import Float32MultiArray
            # 订阅动态生成的避障目标
            self.sub_dyn_target = self.ros_node.create_subscription(
                Float32MultiArray, 
                '/dynamic_target_cmd', 
                self.dynamic_target_cb, 
                10
            )

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.ax.set_aspect('equal')
        self.ax.grid(True, ls=':', alpha=0.5)
        
        self.ax.set_xlim(-800, 800)  
        self.ax.set_ylim(-300, 1400)
        
        self.lines_robot = [self.ax.plot([], [], '-', lw=3, color='gray', alpha=0.4, solid_capstyle='round', label='Physics Belief' if i==0 else "")[0] for i in range(5)]
        self.lines_target = [self.ax.plot([], [], '--', lw=2, color=SEG_COLORS[i], alpha=0.8, label='Desired Shape' if i==0 else "")[0] for i in range(5)]
        self.joints_robot, = self.ax.plot([], [], 'ko', ms=6, zorder=20, label='Vision GT')
        self.pt_selected, = self.ax.plot([], [], 'ro', markersize=12, mec='k', mew=2, zorder=25, label='Target Node')
        
        self.ax.legend(loc='upper right') 
        
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', self.on_close) 
        
        self.timer = self.fig.canvas.new_timer(interval=20)
        self.timer.add_callback(self.control_loop)
        self.timer.start()
        
        print("💡 UI Controls: 'N'=Neural Mode | 'R'=Toggle REAL Robot | 1-5=Select Segment | Close window to exit safely.")
        plt.show()

    def dynamic_target_cb(self, msg):
        if len(msg.data) == 10:
            # 更新本地 q_target，控制循环会自动跟上这个新目标
            self.q_target = np.array(msg.data).reshape(5, 2)

    def load_resources(self, m_path, d_path):
        data = torch.load(d_path, map_location=self.device)
        self.scalers = data['scalers']
        self.model = TopoImpulseNet(self.scalers).to(self.device)
        checkpoint = torch.load(m_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        self.model.eval()
        print(f"✅ Model Loaded Successfully from {m_path}")

    # === 核心修改区 2：新增数据落盘函数 ===
    def save_experiment_data(self):
        """将完整实验数据打包为 .npz 文件"""
        if len(self.history_dq) == 0:
            return
            
        os.makedirs("experiment_data", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode_str = self.CONTROL_MODES[self.control_mode_idx]
        # 设置绝对路径
        save_dir = Path("/home/brandon/brandon/hyrd_robot/src/experiment_data")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode_str = self.CONTROL_MODES[self.control_mode_idx]
        
        # 构造完整的文件绝对路径
        filename = save_dir / f"exp_{mode_str}_{timestamp}.npz"
        
        np.savez(filename,
                 mode=mode_str,
                 q_target=self.rec_q_target,
                 target_poses=self.rec_target_poses,
                 start_ee_pos=self.start_ee_pos,
                 target_ee_pos=self.target_ee_pos,
                 history_q=np.array(self.history_q),
                 history_dq=np.array(self.history_dq),
                 history_err=np.array(self.history_err),
                 history_all_poses=np.array(self.history_all_poses),
                 history_betas=np.array(self.history_betas))
                 
        print(f"💾 Comprehensive experiment data saved to: {filename}")
    # ==================================

    def on_close(self, event):
        self.is_shutting_down = True

    def shutdown_sequence(self):
        print("\n🛑 Initiating Safe Homing Sequence...")
        if not self.use_real_robot or not ROS_ENABLED:
            return
        
        target_pos = np.ones(10) * 10.0
        start_time = time.time()
        
        while time.time() - start_time < 10.0:
            if self.real_q_robot is None: break
            
            curr_pos = self.real_q_robot.flatten()
            err = target_pos - curr_pos
            max_err = np.max(np.abs(err))
            
            if max_err < 1.0: 
                print("✅ Homing Complete.")
                break
                
            rpm = np.clip(err * 3.0, -50.0, 50.0)
            self.ros_node.send_rpm_command(rpm)
            time.sleep(0.05) 
            
        print("💤 Shutting down motors...")
        self.ros_node.send_rpm_command(np.zeros(10)) 
        time.sleep(0.1)

    def on_key(self, event):
        if self.is_shutting_down: return

        if event.key in ['e', 'E']:
            self.q_target = self.PRESET_TARGETS['easy'].copy()
            print("🎯 Target Switched to: EASY")
        elif event.key in ['m', 'M']:
            self.q_target = self.PRESET_TARGETS['medium'].copy()
            print("🎯 Target Switched to: MEDIUM")
        elif event.key in ['x', 'X']:
            self.q_target = self.PRESET_TARGETS['extreme'].copy()
            print("🎯 Target Switched to: EXTREME")
        elif event.key in ['i', 'I']:
            self.q_target = np.ones((5, 2)) * 70.0
            print("🎯 Target Reset to: ALL 70 (Identity Pose)")
        elif event.key in ['b', 'B']:
            self.q_target = np.ones((5, 2)) * 10.0
            print("🎯 Target Reset to: ALL 10 (Base Pose)")

        # =========================================================
        # 🔥 新增：打印当前 q_target (按 '?')
        # =========================================================
        elif event.key in ['?', '/']:
            # 格式化成直接可以复制粘贴为 Python 代码的 numpy 数组格式
            formatted_list = "np.array([\n"
            for i in range(5):
                formatted_list += f"    [{self.q_target[i,0]:.2f}, {self.q_target[i,1]:.2f}],\n"
            formatted_list += "])"
            print(f"\n📋 [COPY] Current q_target:\n{formatted_list}")

        # =========================================================
        # 🔥 新增：切换到序列中的下一个目标 (按 '>')
        # =========================================================
        elif event.key in ['.', '>']:
            if self.seq_idx < len(self.TARGET_SEQUENCE) - 1:
                self.seq_idx += 1
                self.q_target = self.TARGET_SEQUENCE[self.seq_idx].copy()
                print(f"⏭️ Target advanced to sequence index {self.seq_idx}/{len(self.TARGET_SEQUENCE)-1}")
            else:
                self.q_target = self.TARGET_SEQUENCE[-1].copy()
                print(f"⚠️ Already at the end of the sequence (Index {self.seq_idx}).")

        if event.key == ' ':
            self.control_active = not self.control_active
            if self.control_active:
                print(f"\n▶️ [{self.CONTROL_MODES[self.control_mode_idx]}] Control STARTED! Recording metrics...")
                
                # === 清空所有历史序列 ===
                self.history_dq.clear()
                self.history_err.clear()
                self.history_all_poses.clear()
                self.history_q.clear()
                self.history_betas.clear()
                
                poses_rob, _ = forward_kinematics_full_chain(self.q_robot)
                poses_targ, _ = forward_kinematics_full_chain(self.q_target)
                
                # 记录起始目标快照
                self.rec_q_target = self.q_target.copy()
                self.rec_target_poses = poses_targ.copy()
                
                if self.use_real_robot and self.real_vision_poses is not None:
                    self.start_ee_pos = self.real_vision_poses[-1, :2].copy()
                else:
                    self.start_ee_pos = poses_rob[-1, :2].copy()
                    
                self.target_ee_pos = poses_targ[-1, :2].copy()
            else:
                print("\n⏸️ Control STOPPED! Calculating metrics and saving data...")
                self.calculate_and_plot_metrics()
                # === 控制停止时自动保存数据 ===
                self.save_experiment_data()
        
        if event.key in ['1', '2', '3', '4', '5']: 
            self.selected_idx = int(event.key) - 1
            print(f"🎯 Selected segment: {self.selected_idx + 1}")
            
        if event.key in ['n', 'N']: 
            self.control_mode_idx = (self.control_mode_idx + 1) % 3
            print(f"🔄 Switched Control Mode to: {self.CONTROL_MODES[self.control_mode_idx]}")
            
        if event.key in ['r', 'R'] and ROS_ENABLED: 
            self.use_real_robot = not self.use_real_robot
            print(f"🚀 REAL ROBOT OVERRIDE: {'ON' if self.use_real_robot else 'OFF'}")
            
        step_size = 5.0 
        poses, _ = forward_kinematics_full_chain(self.q_target)
        curr_x, curr_y = poses[self.selected_idx][:2]
        
        if event.key in ['up', 'w', 'W']:
            self._move_target_to_absolute(curr_x, curr_y + step_size)
        elif event.key in ['down', 's', 'S']:
            self._move_target_to_absolute(curr_x, curr_y - step_size)
        elif event.key in ['left', 'a', 'A']:
            self._move_target_to_absolute(curr_x - step_size, curr_y)
        elif event.key in ['right', 'd', 'D']:
            self._move_target_to_absolute(curr_x + step_size, curr_y)

    def on_move(self, event):
        if self.is_shutting_down: return
        if event.inaxes != self.ax or event.button != 1: return
        self._move_target_to_absolute(event.xdata, event.ydata)

    def _move_target_to_absolute(self, target_global_x, target_global_y):
        poses, _ = forward_kinematics_full_chain(self.q_target)
        prev_p = poses[self.selected_idx-1][:3] if self.selected_idx > 0 else [0,0,0]
        
        dx = target_global_x - prev_p[0]
        dy = target_global_y - prev_p[1]
        c, s = np.cos(prev_p[2]), np.sin(prev_p[2])
        local_x, local_y = c*dx + s*dy, -s*dx + c*dy
        
        q = self.q_target[self.selected_idx].copy()
        
        SAFE_MARGIN = 3.0 
        LIMIT_MIN_MM = 10.0 + SAFE_MARGIN  
        LIMIT_MAX_MM = 160.0 - SAFE_MARGIN 
        
        CPL_OFFSET, CPL_COEFF, CPL_EXP = 10.0, 0.17, 1.1
        CPL_INV_EXP = 1.0 / 1.1
        def safe_pow(base, exp): return max(base, 0.0) ** exp
        
        MAX_BEND_DEG = 50.0 
        max_delta_q = np.radians(MAX_BEND_DEG) * C_LIST[self.selected_idx]
        
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
            
            ql, qr = q[0], q[1]
            
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
            
            q = np.array([ql, qr])
            
        self.q_target[self.selected_idx] = q

    def calculate_and_plot_metrics(self):
        if len(self.history_dq) < 10:
            print("⚠️ Not enough data points to calculate metrics. Run the control longer.")
            return

        dq_arr = np.array(self.history_dq)      
        err_arr = np.array(self.history_err)    
        all_poses_arr = np.array(self.history_all_poses) 
        ee_arr = all_poses_arr[:, -1, :]

        delta_dq = np.abs(np.diff(dq_arr, axis=0)) 
        mean_chatter_per_motor = np.mean(delta_dq, axis=0)
        chatter_mean = np.mean(mean_chatter_per_motor)
        chatter_max = np.max(mean_chatter_per_motor)

        actual_path_len = np.sum(np.linalg.norm(np.diff(ee_arr, axis=0), axis=1))
        ideal_dist = np.linalg.norm(self.target_ee_pos - self.start_ee_pos)
        efficiency = actual_path_len / ideal_dist if ideal_dist > 1.0 else 1.0

        tail_len = min(50, max(5, int(len(err_arr) * 0.2)))
        steady_state_errs = err_arr[-tail_len:]
        ss_var = np.var(steady_state_errs)
        ss_mean = np.mean(steady_state_errs)

        print("-" * 50)
        print(f"📊 METRICS REPORT ({self.CONTROL_MODES[self.control_mode_idx]})")
        print(f"1. Smoothness (Mean/Max Δdq): {chatter_mean:.4f} / {chatter_max:.4f}")
        print(f"2. Path Efficiency Ratio:     {efficiency:.2f} (Ideal is 1.0)")
        print(f"3. Steady-State Error (±Var): {ss_mean:.2f} ± {ss_var:.4f}")
        print("-" * 50)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.canvas.manager.set_window_title(f"Metrics Report: {self.CONTROL_MODES[self.control_mode_idx]}")

        axs[0].plot(err_arr, label='Avg Node Error', color='#1f77b4', lw=2)
        axs[0].axvspan(len(err_arr) - tail_len, len(err_arr), color='red', alpha=0.1, label='Steady State Window')
        axs[0].set_title(f"Convergence (SS Var: {ss_var:.4f})")
        axs[0].set_xlabel("Control Steps")
        axs[0].set_ylabel("Error (mm)")
        axs[0].grid(True, ls=':')
        axs[0].legend()

        axs[1].plot(np.linalg.norm(delta_dq, axis=1), label='Norm(||Δdq||)', color='#ff7f0e', lw=1.5)
        axs[1].set_title(f"Action Chattering (Mean: {chatter_mean:.4f})")
        axs[1].set_xlabel("Control Steps")
        axs[1].set_ylabel("Command Jump Magnitude")
        axs[1].grid(True, ls=':')
        axs[1].legend()

        all_poses_arr = np.array(self.history_all_poses) 
        
        for i in range(5):
            axs[2].plot(all_poses_arr[:, i, 0], all_poses_arr[:, i, 1], 
                        color=SEG_COLORS[i], lw=1.5, alpha=0.7, label=f'Node {i+1}')
            axs[2].scatter(all_poses_arr[0, i, 0], all_poses_arr[0, i, 1], color=SEG_COLORS[i], s=20)
            axs[2].scatter(all_poses_arr[-1, i, 0], all_poses_arr[-1, i, 1], color=SEG_COLORS[i], marker='X', s=40)

        axs[2].set_title(f"Shape Trajectory (Efficiency: {efficiency:.2f})")
        axs[2].set_xlabel("X (mm)")
        axs[2].set_ylabel("Y (mm)")
        axs[2].axis('equal')
        axs[2].grid(True, ls=':')

        plt.tight_layout()
        plt.show(block=False) 

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
        if self.is_shutting_down:
            self.timer.stop() 
            return
            
        fk_poses, rob_segs = forward_kinematics_full_chain(self.q_robot, return_arc_points=True)
        vision_targ_poses, targ_segs = forward_kinematics_full_chain(self.q_target, return_arc_points=True)
        
        if self.use_real_robot and self.real_vision_poses is not None and self.real_q_robot is not None:
            vision_rob_poses = self.real_vision_poses.copy()
            self.q_robot = self.real_q_robot.copy()
        else:
            vision_rob_poses = fk_poses
            
        base_xy = vision_rob_poses[:, :2].flatten()
        error = vision_targ_poses[:, :2].flatten() - base_xy

        if not self.control_active:
            if self.use_real_robot and ROS_ENABLED:
                self.ros_node.send_rpm_command(np.zeros(10)) 
                
            for i in range(5):
                self.lines_robot[i].set_data(rob_segs[i][:, 0], rob_segs[i][:, 1])
                self.lines_target[i].set_data(targ_segs[i][:, 0], targ_segs[i][:, 1])
            j_r = np.vstack((np.array([[0,0]]), vision_rob_poses[:, :2]))
            self.joints_robot.set_data(j_r[:, 0], j_r[:, 1])
            self.pt_selected.set_data([vision_targ_poses[self.selected_idx, 0]], [vision_targ_poses[self.selected_idx, 1]])
            
            status_str = "REAL" if self.use_real_robot else "SIM"
            mode_str = self.CONTROL_MODES[self.control_mode_idx]
            self.ax.set_title(f"[PAUSED] {status_str} {mode_str} | Press SPACE to Start")
            self.fig.canvas.draw()
            return
        
        eps = 1e-3; J_phy = np.zeros((10, 10))
        fk_base_poses, _ = forward_kinematics_full_chain(self.q_robot)
        fk_base_xy = fk_base_poses[:, :2].flatten() 
        
        for i in range(10):
            q_f = self.q_robot.flatten(); q_f[i] += eps
            new_p, _ = forward_kinematics_full_chain(q_f.reshape(5, 2))
            J_phy[:, i] = (new_p[:, :2].flatten() - fk_base_xy) / eps
        
        curr_J = J_phy
        beta_val = 1.0
        
        # === 核心修改区 3：规范化 10 维 β 记录 ===
        beta_vec = np.ones(10) # 默认为 PHY 模式下的全信任向量

        if self.control_mode_idx > 0: 
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
            
            x_in_batch = torch.cat([q_t.repeat(batch_size,1,1), dq_h.repeat(batch_size,1,1), 
                                    p_l_t.repeat(batch_size,1,1), dq_cmd_batch, j_f_t.repeat(batch_size,1,1)], dim=2)
            
            with torch.no_grad():
                pred_local_batch, _, beta_batch = self.model(x_in_batch, j_f_t.repeat(batch_size,1,1), dq_cmd_batch, return_internals=True)
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
            
            if self.control_mode_idx == 1:
                betas_xy = beta_batch[0, :, :2].cpu().numpy().flatten()
                beta_vec = betas_xy  # (10,)
                curr_J = beta_vec.reshape(10, 1) * J_phy + (1.0 - beta_vec.reshape(10, 1)) * J_net_global
                beta_val = np.mean(betas_xy[-2:])
                
            elif self.control_mode_idx == 2:
                beta_val = 0.0
                beta_vec = np.zeros(10) # 纯神经网络的 beta 为 0
                curr_J = J_net_global
        # ==========================================

        latency_ms = 70.0
        loop_ms = 20.0
        steps_delayed = latency_ms / loop_ms
        dq_velocity = (self.q_robot - self.q_prev).flatten()
        delay_compensation_xy = curr_J @ (dq_velocity * steps_delayed)
        predicted_base_xy = base_xy + delay_compensation_xy
        
        error = vision_targ_poses[:, :2].flatten() - predicted_base_xy
        
        node_errors = np.linalg.norm(error.reshape(5, 2), axis=1)
        convergence_threshold = 10.0 
        target_mu = 4.0 
        for i in range(5):
            if node_errors[i] > convergence_threshold:
                target_mu = float(i)
                break
                
        self.peak_mu = 0.85 * self.peak_mu + 0.15 * target_mu
        
        amp_root = 8.0  
        amp_tip = 4.0    
        max_mu = 4.0     
        peak_amplitude = amp_root - (amp_root - amp_tip) * (self.peak_mu / max_mu)
        
        floor_weight = 1.0 
        sigma = 0.8 
        gaussian_weights = floor_weight + (peak_amplitude - floor_weight) * np.exp(-((np.arange(5) - self.peak_mu)**2) / (2 * sigma**2))

        gaussian_weights = np.array([5.0, 3.0, 2.0, 5.0, 8.0])
        
        weight_array = np.repeat(gaussian_weights, 2)
        W = np.diag(weight_array)

        self.q_prev = self.q_robot.copy()

        avg_err = np.mean(node_errors)
        gain_val = 0.5 if self.control_mode_idx == 0 else 0.3
        current_gain = gain_val
        if avg_err < 2.0:
            ratio = np.clip((avg_err - 1.0) / (5.0 - 1.0), 0.1, 1.0)
            current_gain *= ratio
            
        step_error = error * current_gain
        
        max_cartesian_step = 10.0 
        step_norm = np.linalg.norm(step_error)
        if step_norm > max_cartesian_step:
            step_error = (step_error / step_norm) * max_cartesian_step

        try:
            lambda_val_adj = 0.15 if self.control_mode_idx == 0 else 0.4
            if avg_err < 1.0: lambda_val_adj *= 2.0
            dq = np.linalg.inv(curr_J.T @ W @ curr_J + lambda_val_adj * np.eye(10)) @ curr_J.T @ W @ step_error
        except: 
            dq = np.zeros(10)

        # === 核心修改区 4：追加到历史记录中 ===
        self.history_dq.append(dq.flatten().copy())
        self.history_err.append(avg_err)
        self.history_all_poses.append(vision_rob_poses[:, :2].copy())
        self.history_q.append(self.q_robot.copy())          
        self.history_betas.append(beta_vec.copy())          
        # ====================================
        
        if not self.use_real_robot:
            self.q_robot += dq.reshape(5, 2)
            self.q_robot = np.clip(self.q_robot, 0, 160)
        else:
            if avg_err < 0.5:
                rpm_cmd = np.zeros(10)
            else:
                rpm_cmd = dq.flatten() * 10.0 
                
            rpm_cmd = np.clip(rpm_cmd, -40.0, 40.0) 
            self.ros_node.send_rpm_command(rpm_cmd)
        
        for i in range(5):
            self.lines_robot[i].set_data(rob_segs[i][:, 0], rob_segs[i][:, 1])
            self.lines_target[i].set_data(targ_segs[i][:, 0], targ_segs[i][:, 1])
            
        j_r = np.vstack((np.array([[0,0]]), vision_rob_poses[:, :2]))
        self.joints_robot.set_data(j_r[:, 0], j_r[:, 1])
        self.pt_selected.set_data([vision_targ_poses[self.selected_idx, 0]], [vision_targ_poses[self.selected_idx, 1]])
        
        status_str = "REAL" if self.use_real_robot else "SIM"
        mode_str = self.CONTROL_MODES[self.control_mode_idx]
        self.ax.set_title(f"[{status_str}] {mode_str} | Peak: {self.peak_mu:.1f} | Beta: {beta_val:.2f}")
        self.fig.canvas.draw()

def main(args=None):
    m_p = "/home/brandon/brandon/hyrd_robot/lifelong_data/experiments/001/best_model.pth"
    d_p = "/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_smooth_strided.pt"
    
    sim_app = HybridPhantomSimV11(m_p, d_p)
    sim_app.shutdown_sequence()
    
    if ROS_ENABLED:
        try: rclpy.shutdown()
        except Exception: pass
    print("Hybrid Phantom Sim exited cleanly.")

if __name__ == "__main__":
    main()