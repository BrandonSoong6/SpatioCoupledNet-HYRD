#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
from collections import deque
from std_msgs.msg import Float32MultiArray
from robot_interfaces.msg import MotorCommand, MotorState, VisionState

# =================================================================================
# 1. 严谨的常量定义 & 物理参数
# =================================================================================
# 几何参数
C_LIST = np.array([92.0, 108.0, 123.5, 140.0, 156.0], dtype=np.float64)
N_VAL = 22.0
H0_VAL = 52.0
M_VAL = H0_VAL - 2 * N_VAL

# 硬件映射 [8,9]->Node0 ... [0,1]->Node4
# ⚠️ 调试重点：如果物理模式下动错了，检查这里是不是反了
MOT_REMAP_INDICES = [8, 9, 6, 7, 4, 5, 2, 3, 0, 1] 

# 控制参数
CONTROL_FREQ = 100.0
DT = 1.0 / CONTROL_FREQ
LATENCY_COMP_DT = 0.075 
MAX_RPM = 50.0
STS_MM_SEC_TO_RPM = 1.201 
MPPI_STRIDE_TIME = 0.05 

# Cost 权重
W_POS = 1.0
W_ANG = 20.0 

# =================================================================================
# 2. 核心数学与物理函数
# =================================================================================
def get_local_pose_from_q(q_pair, c_val, n_val, m_val):
    q_l, q_r = q_pair
    delta_q = q_l - q_r
    sum_q = q_l + q_r
    theta = delta_q / c_val 
    L_c = m_val + sum_q / 2.0
    
    if abs(theta) < 1e-6:
        return np.array([0.0, 2*n_val + L_c, 0.0])
    else:
        rho = L_c / theta
        lx = rho * (1.0 - np.cos(theta))
        ly = rho * np.sin(theta)
        c, s = np.cos(-theta), np.sin(-theta)
        x_local = -s * n_val + lx
        y_local = c * n_val + ly + n_val
        return np.array([x_local, y_local, -theta])

def compute_local_jacobian(q_pair, c_val, n_val, m_val):
    eps = 1e-4 
    J = np.zeros((3, 2), dtype=np.float64)
    # 简单的数值微分
    pl = get_local_pose_from_q([q_pair[0]+eps, q_pair[1]], c_val, n_val, m_val)
    ml = get_local_pose_from_q([q_pair[0]-eps, q_pair[1]], c_val, n_val, m_val)
    J[:, 0] = (pl - ml) / (2 * eps)
    
    pr = get_local_pose_from_q([q_pair[0], q_pair[1]+eps], c_val, n_val, m_val)
    mr = get_local_pose_from_q([q_pair[0], q_pair[1]-eps], c_val, n_val, m_val)
    J[:, 1] = (pr - mr) / (2 * eps)
    return J.flatten()

def global_to_local(parent_pose, current_pose):
    x_p, y_p, th_p = parent_pose
    x_c, y_c, th_c = current_pose
    th_local = th_c - th_p
    dx, dy = x_c - x_p, y_c - y_p
    cos_t, sin_t = np.cos(th_p), np.sin(th_p)
    return np.array([dx*cos_t + dy*sin_t, -dx*sin_t + dy*cos_t, th_local])

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# =================================================================================
# 3. 网络架构 (修复版：正确的参数注册)
# =================================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.gelu = nn.GELU()
    def forward(self, x): return self.gelu(x + self.net(x))

class TopoImpulseNet(nn.Module):
    def __init__(self, scalers, input_dim=15, feature_dim=128, gru_dim=128, dropout=0.0):
        super().__init__()
        
        # 1. 注册 Checkpoint 中存在的 Buffer (必须与训练代码命名完全一致)
        self.register_buffer('jac_mean', scalers['jacobian_mean'].float())
        self.register_buffer('jac_std', scalers['jacobian_std'].float())
        self.register_buffer('cmd_mean', scalers['dq_cmd_mean'].float())
        self.register_buffer('cmd_std', scalers['dq_cmd_std'].float())

        # 2. 注册推理需要的其他 Scaler (persistent=False)
        self.register_buffer('q_curr_mean', scalers['q_curr_mean'].float(), persistent=False)
        self.register_buffer('q_curr_std', scalers['q_curr_std'].float(), persistent=False)
        self.register_buffer('dq_hist_mean', scalers['dq_hist_mean'].float(), persistent=False)
        self.register_buffer('dq_hist_std', scalers['dq_hist_std'].float(), persistent=False)
        self.register_buffer('pose_loc_mean', scalers['pose_loc_mean'].float(), persistent=False)
        self.register_buffer('pose_loc_std', scalers['pose_loc_std'].float(), persistent=False)
        
        # 兼容性别名
        self.jacobian_mean = self.jac_mean
        self.jacobian_std = self.jac_std
        self.dq_cmd_mean = self.cmd_mean
        self.dq_cmd_std = self.cmd_std

        self.num_nodes = 5
        self.stage1_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, feature_dim), nn.LayerNorm(feature_dim), nn.GELU(), nn.Dropout(dropout),
                ResidualBlock(feature_dim, dropout), ResidualBlock(feature_dim, dropout) 
            ) for _ in range(self.num_nodes)
        ])
        self.stage2_gru = nn.GRU(feature_dim, gru_dim, 2, batch_first=True, bidirectional=True, dropout=dropout)
        
        self.net_expert_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim + gru_dim*2 + 6, 128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3) 
            ) for _ in range(self.num_nodes)
        ])
        self.confidence_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim + gru_dim*2 + 6, 64), nn.Tanh(),
                nn.Linear(64, 3), nn.Sigmoid() 
            ) for _ in range(self.num_nodes)
        ])

    def compute_nominal_delta(self, jac_norm, dq_cmd_norm):
        """ 纯物理预测: Delta = J * dq """
        B, N, _ = jac_norm.shape
        # 使用注册好的 buffer 反归一化
        jac_real = jac_norm * self.jac_std + self.jac_mean
        dq_real = dq_cmd_norm * self.cmd_std + self.cmd_mean
        
        J_matrix = jac_real.view(B, N, 3, 2)
        dq_vec = dq_real.unsqueeze(-1)
        # 矩阵乘法: (3x2) * (2x1) -> (3x1)
        nominal_delta = torch.matmul(J_matrix, dq_vec).squeeze(-1)
        return nominal_delta

    def forward(self, x_inputs, jacobian_norm, dq_cmd_norm):
        local_features = [self.stage1_experts[i](x_inputs[:, i, :]) for i in range(self.num_nodes)]
        gru_out, _ = self.stage2_gru(torch.stack(local_features, dim=1))
        nominal_delta = self.compute_nominal_delta(jacobian_norm, dq_cmd_norm)
        
        outputs = []
        for i in range(self.num_nodes):
            feature = torch.cat([local_features[i], gru_out[:, i, :], jacobian_norm[:, i, :]], dim=1)
            net_pred = self.net_expert_head[i](feature)
            beta = self.confidence_gate[i](feature)
            outputs.append(beta * nominal_delta[:, i, :] + (1.0 - beta) * net_pred)
        return torch.stack(outputs, dim=1)

# FK 模块
class DifferentiableFK(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, current_local_pose, pred_local_delta):
        batch_size = current_local_pose.shape[0]
        next_state = current_local_pose + pred_local_delta
        x, y, theta = next_state[:,:,0], next_state[:,:,1], next_state[:,:,2]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        
        T_curr = torch.eye(3, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        global_states = []
        for i in range(5):
            zeros = torch.zeros_like(x[:,i])
            row0 = torch.stack([cos_t[:,i], -sin_t[:,i], x[:,i]], dim=1)
            row1 = torch.stack([sin_t[:,i],  cos_t[:,i], y[:,i]], dim=1)
            row2 = torch.tensor([0,0,1], device=x.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, 3)
            T_local = torch.stack([row0, row1, row2], dim=1)
            T_curr = torch.bmm(T_curr, T_local)
            
            gx = T_curr[:, 0, 2]
            gy = T_curr[:, 1, 2]
            gth = torch.atan2(T_curr[:, 1, 0], T_curr[:, 0, 0])
            global_states.append(torch.stack([gx, gy, gth], dim=1))
            
        return torch.stack(global_states, dim=1) 

# =================================================================================
# 4. 控制器节点 (带 PHYSICS_ONLY 调试开关)
# =================================================================================
class HyRDController(Node):
    def __init__(self):
        super().__init__('hyrd_controller_node')

        # 🔥🔥🔥 DEBUG SWITCH 🔥🔥🔥
        # True: 忽略神经网络权重，只使用 J * dq 预测
        # False: 正常使用训练好的网络
        self.PHYSICS_ONLY = False 

        if self.PHYSICS_ONLY:
            self.get_logger().warn("⚠️⚠️⚠️ PHYSICS ONLY MODE ENABLED ⚠️⚠️⚠️")
            self.get_logger().warn("Neural Network is BYPASSED. Using Analytic Jacobian for Control.")
        else:
            self.get_logger().info("🔥 Initializing HyRD Controller (Neural Mode)...")

        self.MODEL_PATH = "/home/brandon/brandon/hyrd_robot/lifelong_data/experiments/001/best_model.pth"
        self.SCALER_PATH = "/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_smooth_strided.pt"
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_network()
        self.fk_layer = DifferentiableFK().to(self.DEVICE)
        
        self.current_q_raw = np.zeros(10)
        self.current_vision_raw = np.zeros(15) 
        self.target_15d = None 
        
        self.last_q_net = np.zeros((5, 2))
        self.last_dq_net = np.zeros((5, 2))
        
        self.sub_motor = self.create_subscription(MotorState, '/motor_state', self.motor_cb, 10)
        self.sub_vision = self.create_subscription(VisionState, '/vision/state', self.vision_cb, 10)
        self.sub_target = self.create_subscription(Float32MultiArray, '/robot/target_pose_15d', self.target_cb, 10)
        self.pub_cmd = self.create_publisher(MotorCommand, '/motor_cmd', 10)
        
        self.timer = self.create_timer(DT, self.control_loop)
        self.get_logger().info("✅ Controller Ready.")

    def _load_network(self):
        data = torch.load(self.SCALER_PATH, map_location=self.DEVICE)
        scalers = data['scalers']
        self.model = TopoImpulseNet(scalers, dropout=0.0).to(self.DEVICE)
        
        ckpt = torch.load(self.MODEL_PATH, map_location=self.DEVICE)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)
            
        self.model.eval()
        self.get_logger().info(f"✅ Model Loaded.")

    def motor_cb(self, msg):
        if len(msg.positions) == 10:
            self.current_q_raw = np.array(msg.positions)

    def vision_cb(self, msg):
        temp_dict = {}
        for i, vid in enumerate(msg.ids):
            temp_dict[vid] = [msg.x_local[i], msg.y_local[i], np.radians(msg.theta[i])] 
        feat = []
        for vid in range(1, 6):
            if vid in temp_dict: feat.extend(temp_dict[vid])
            else: feat.extend([0,0,0])
        self.current_vision_raw = np.array(feat)

    def target_cb(self, msg):
        raw = np.array(msg.data)
        if len(raw) != 15: return
        self.target_15d = raw.copy()
        angle_indices = [2, 5, 8, 11, 14]
        self.target_15d[angle_indices] = np.radians(self.target_15d[angle_indices])

    def compensate_vision_latency(self, raw_vision_flat, dq_net, dt):
        obs_global = raw_vision_flat.reshape(5, 3)
        obs_local = []
        prev_p = np.array([0., 0., 0.])
        for i in range(5):
            curr_p = obs_global[i]
            if i == 0: loc = curr_p
            else: loc = global_to_local(prev_p, curr_p)
            loc[2] = wrap_angle(loc[2])
            obs_local.append(loc)
            prev_p = curr_p
        
        pred_local_delta = []
        for i in range(5):
            q_pair = self.last_q_net[i] 
            J = compute_local_jacobian(q_pair, C_LIST[i], N_VAL, M_VAL).reshape(3, 2)
            dq_step = dq_net[i] 
            dq_latency = dq_step * (dt / DT)
            d_loc = J @ dq_latency
            pred_local_delta.append(d_loc)
            
        obs_local_tensor = torch.tensor(np.array(obs_local), dtype=torch.float32).unsqueeze(0).to(self.DEVICE)
        delta_tensor = torch.tensor(np.array(pred_local_delta), dtype=torch.float32).unsqueeze(0).to(self.DEVICE)
        
        with torch.no_grad():
            est_local = obs_local_tensor + delta_tensor
        return est_local

    def normalize_input(self, q, dq, pose, cmd, jac):
        n_q = (q - self.model.q_curr_mean) / self.model.q_curr_std
        n_dq = (dq - self.model.dq_hist_mean) / self.model.dq_hist_std
        n_pose = (pose - self.model.pose_loc_mean) / self.model.pose_loc_std
        n_cmd = (cmd - self.model.dq_cmd_mean) / self.model.dq_cmd_std
        n_jac = (jac - self.model.jacobian_mean) / self.model.jacobian_std
        return n_q, n_dq, n_pose, n_cmd, n_jac

    def control_loop(self):
        if self.target_15d is None or np.all(self.current_q_raw == 0): return
        
        # 1. 基础状态提取 (保持 Rad/Step 单位，与 Generator 一致)
        q_net = self.current_q_raw[MOT_REMAP_INDICES].reshape(5, 2)
        dq_net = q_net - self.last_q_net
        self.last_dq_net = dq_net.copy()
        self.last_q_net = q_net.copy()
        
        # 2. 视觉延迟补偿
        est_local_tensor = self.compensate_vision_latency(self.current_vision_raw, dq_net, LATENCY_COMP_DT)
        
        # 3. 构造 Jacobian (必须包含训练时的 -theta 符号，以对齐网络预期)
        # 注意：如果你的 compute_local_jacobian 已经去掉了 -theta 负号，
        # 为了适配旧网络，你需要在这里手动把 Jacobian 的第 3 行 (index 2, 5) 取反。
        jac_list = []
        for i in range(5):
            j = compute_local_jacobian(q_net[i], C_LIST[i], N_VAL, M_VAL)
            # 严谨对齐：确保 Jacobian 的 theta 分量与 data_generator 线 44 一致
            # 如果 compute_local_jacobian 返回的是 [x, y, theta]，则执行下两行：
            # j[2] *= -1.0  # dTheta/dqL
            # j[5] *= -1.0  # dTheta/dqR
            jac_list.append(j)
        jac_net = np.stack(jac_list)
        
        # --- MPPI 采样 ---
        BATCH_SIZE = 512
        max_disp = MAX_RPM * MPPI_STRIDE_TIME / STS_MM_SEC_TO_RPM
        cmd_noise = torch.randn(BATCH_SIZE, 5, 2, device=self.DEVICE) * (max_disp / 2.0)
        cmd_candidates = torch.clamp(cmd_noise, -max_disp, max_disp)
        
        # --- 对齐指令效率 (Causality Alignment) ---
        # 训练使用的是“实际位移”，指令 1.0 对应的实际位移通常只有 ~0.85
        MOTOR_EFFICIENCY = 0.88 # 严谨估计值
        
        # 构造 Batch
        q_batch = torch.tensor(q_net, dtype=torch.float32).unsqueeze(0).repeat(BATCH_SIZE, 1, 1).to(self.DEVICE)
        dq_batch = torch.tensor(dq_net, dtype=torch.float32).unsqueeze(0).repeat(BATCH_SIZE, 1, 1).to(self.DEVICE)
        jac_batch = torch.tensor(jac_net, dtype=torch.float32).unsqueeze(0).repeat(BATCH_SIZE, 1, 1).to(self.DEVICE)
        pose_batch = est_local_tensor.repeat(BATCH_SIZE, 1, 1)
        
        # 输入给网络的 cmd 必须是“打折”后的，使其符合训练集的位移分布
        n_q, n_dq, n_pose, n_cmd, n_jac = self.normalize_input(
            q_batch, dq_batch, pose_batch, cmd_candidates * MOTOR_EFFICIENCY, jac_batch
        )
        full_input = torch.cat([n_q, n_dq, n_pose, n_cmd, n_jac], dim=2)
        
        # --- 推理 ---
        with torch.no_grad():
            if self.PHYSICS_ONLY:
                pred_delta = self.model.compute_nominal_delta(n_jac, n_cmd)
            else:
                # 网络预测结果 pred_delta 的 theta 分量已经是正向的了（因为它学到了抵消负号）
                pred_delta = self.model(full_input, n_jac, n_cmd)
            
            # FK 算出全局
            pred_global = self.fk_layer(pose_batch, pred_delta) 
            
            # Cost 计算 (末端执行器)
            pred_tip = pred_global[:, -1, :] 
            target_tip = torch.tensor(self.target_15d[-3:], dtype=torch.float32, device=self.DEVICE)
            
            pos_err = torch.norm(pred_tip[:, :2] - target_tip[:2], dim=1)
            # 角度误差计算，确保目标角度也是 CCW 正向
            ang_err = 1.0 - torch.cos(pred_tip[:, 2] - target_tip[2])
            total_cost = pos_err * W_POS + ang_err * W_ANG
            
            best_idx = torch.argmin(total_cost)
            best_cmd_50ms = cmd_candidates[best_idx].cpu().numpy()
            
        # --- 下发 ---
        GAIN = 0.85 # 适度闭环增益
        target_rpm_net = (best_cmd_50ms / MPPI_STRIDE_TIME) * STS_MM_SEC_TO_RPM * GAIN
        target_rpm_net = np.clip(target_rpm_net, -MAX_RPM, MAX_RPM)
        
        final_rpm_10 = np.zeros(10)
        final_rpm_10[MOT_REMAP_INDICES] = target_rpm_net.flatten()
        
        msg = MotorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ids = list(range(1, 11))
        msg.target_rpms = final_rpm_10.tolist()
        self.pub_cmd.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)
    node = HyRDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        stop_msg = MotorCommand()
        stop_msg.ids = [1,2,3,4,5,6,7,8,9,10]
        stop_msg.target_rpms = [0.0]*10
        node.pub_cmd.publish(stop_msg)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()