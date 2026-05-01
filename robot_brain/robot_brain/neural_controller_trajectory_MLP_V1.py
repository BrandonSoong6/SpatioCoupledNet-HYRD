#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import time
from collections import deque

from robot_interfaces.msg import MotorCommand, MotorState, VisionState

# ================= 🔧 IMPORT 部分 =================
from robot_brain.inference_MLP_V1 import (
    JacobianCore, 
    compute_jacobian_xy_single, 
    C_LIST_CONFIG as C_LIST, 
    N_VAL, 
    M_VAL
)
# =================================================

class JacobianServoController(Node):
    def __init__(self):
        super().__init__('jacobian_servo_controller')
        
        # ================= ⚙️ 核心配置 =================
        self.CONFIG = {
            'JACOBIAN_MODE': 0,       # 0=Hybrid(融合), 1=Physics(纯物理)
            'CONTROL_FREQ': 100.0,    # 100Hz 实时伺服
            
            # === 伺服增益与软着陆 ===
            'SERVO_GAIN': 0.8,        # [基础增益] 离目标远时的全速增益
            'ADAPTIVE_GAIN': True,    # [开关] 是否开启自适应减速
            'SLOW_DOWN_ZONE': 15.0,   # [减速区] 进入目标 15mm 范围内开始减速
            'MIN_GAIN_SCALE': 0.15,   # [最小比例] 贴脸时增益降为基础的 15% (约 0.12)
            
            'DAMPING': 0.05,          # DLS 阻尼
            
            # === 阈值与限位 ===
            'DONE_THRES_MM': 2.0,     # 到位阈值
            'MAX_RPM': 60.0,          # 限制最大转速
            
            # === 预测与安全 ===
            'VISION_LATENCY_MS': 120, 
            'PREDICT_GAIN': 1.0,
            'VISION_TIMEOUT': 1.0,
            
            'MOTOR_IDS': [2, 1, 4, 3, 6, 5, 8, 7, 10, 9], 
            'VISION_IDS': [1, 2, 3, 4, 5]
        }
        # ===============================================

        self.get_logger().info("🧠 Loading Jacobian Core...")
        try:
            self.brain = JacobianCore()
            self.get_logger().info("✅ Jacobian Core Ready.")
        except Exception as e:
            self.get_logger().warn(f"⚠️ Core Load Failed: {e}. Fallback to Raw Physics.")
            self.brain = None

        # 状态变量
        self.curr_q = np.zeros(10)
        self.curr_xy_raw = np.zeros(10)      
        self.curr_xy_est = np.zeros(10)      
        self.vision_updated_t = 0.0
        
        self.target_xy = None
        self.is_active = False
        
        # 调试用：当前实时增益
        self.current_effective_gain = self.CONFIG['SERVO_GAIN']
        
        self.cmd_history = deque(maxlen=100) 

        # ROS 接口
        self.pub_cmd = self.create_publisher(MotorCommand, 'motor_cmd', 10)
        self.sub_motor = self.create_subscription(MotorState, 'motor_state', self.motor_callback, 10)
        self.sub_vision = self.create_subscription(VisionState, 'vision/state', self.vision_callback, 10)
        self.sub_target = self.create_subscription(Float32MultiArray, '/robot/target_pose_10d', self.target_callback, 10)

        self.dt = 1.0 / self.CONFIG['CONTROL_FREQ']
        self.timer = self.create_timer(self.dt, self.servo_loop)
        
        self.create_timer(0.5, self.print_diagnostics)
        
        print("\n" + "="*60)
        print("🚀 JACOBIAN VISUAL SERVOING (ADAPTIVE GAIN)")
        print(f"👉 Mode: {self.CONFIG['JACOBIAN_MODE']} | Base Gain: {self.CONFIG['SERVO_GAIN']}")
        print(f"👉 Soft Landing: <{self.CONFIG['SLOW_DOWN_ZONE']}mm -> Scale down to {self.CONFIG['MIN_GAIN_SCALE']*100}%")
        print("="*60 + "\n")

    def target_callback(self, msg):
        new_target = np.array(msg.data)
        self.target_xy = new_target
        self.is_active = True

    def predict_current_pose(self, raw_xy, raw_time):
        if len(self.cmd_history) < 2: return raw_xy
        latency_sec = self.CONFIG['VISION_LATENCY_MS'] / 1000.0
        cutoff_time = self.get_clock().now().nanoseconds/1e9 - latency_sec
        
        q_dbl = self.curr_q.astype(np.float64)
        J_curr = compute_jacobian_xy_single(q_dbl, C_LIST, N_VAL, M_VAL)

        delta_xy_sum = np.zeros(10)
        for t_cmd, dq_cmd in self.cmd_history:
            if t_cmd > cutoff_time:
                delta_xy_sum += (J_curr @ dq_cmd * self.dt)
                
        return raw_xy + delta_xy_sum * self.CONFIG['PREDICT_GAIN']

    def servo_loop(self):
        """ 🔥 核心伺服循环 """
        now_time = self.get_clock().now().nanoseconds / 1e9
        
        if (now_time - self.vision_updated_t) > self.CONFIG['VISION_TIMEOUT']:
            self.stop_motors()
            return

        if not self.is_active or self.target_xy is None:
            return 

        # 1. 计算误差
        error_vec = self.target_xy - self.curr_xy_est
        max_err = np.max(np.abs(error_vec))
        
        # 2. 到位检测
        if max_err < self.CONFIG['DONE_THRES_MM']:
            self.stop_motors()
            return

        # 3. 计算雅可比
        if self.CONFIG['JACOBIAN_MODE'] == 0 and self.brain is not None:
            J = self.brain.get_hybrid_jacobian(self.curr_q, self.curr_xy_est)
        else:
            q_dbl = self.curr_q.astype(np.float64)
            J = compute_jacobian_xy_single(q_dbl, C_LIST, N_VAL, M_VAL)

        # 4. 求解 DLS
        lambda_mat = self.CONFIG['DAMPING'] * np.eye(10)
        dq = np.linalg.solve(J.T @ J + lambda_mat, J.T @ error_vec)
        
        # ========================================================
        # 🔥🔥🔥 自适应增益逻辑 (Soft Landing) 🔥🔥🔥
        # ========================================================
        effective_gain = self.CONFIG['SERVO_GAIN']
        
        if self.CONFIG['ADAPTIVE_GAIN'] and max_err < self.CONFIG['SLOW_DOWN_ZONE']:
            # 线性插值：从 100% 降到 MIN_SCALE
            # ratio = 1.0 (at boundary) -> 0.0 (at target)
            ratio = max_err / self.CONFIG['SLOW_DOWN_ZONE']
            ratio = np.clip(ratio, 0.0, 1.0)
            
            # Scale factor calculation
            min_scale = self.CONFIG['MIN_GAIN_SCALE']
            scale_factor = min_scale + (1.0 - min_scale) * ratio
            
            effective_gain *= scale_factor
            
        self.current_effective_gain = effective_gain # 用于显示
        
        # 5. 应用增益
        target_vel_rad = dq * effective_gain
        
        # 6. 下发
        target_rpm = target_vel_rad * 9.55
        self.publish_cmd_safe_rpm(target_rpm)
        self.cmd_history.append((now_time, target_vel_rad))

    def publish_cmd_safe_rpm(self, rpms):
        rpms = np.clip(rpms, -self.CONFIG['MAX_RPM'], self.CONFIG['MAX_RPM'])
        msg = MotorCommand()
        msg.ids = self.CONFIG['MOTOR_IDS']
        msg.target_rpms = [float(x) for x in rpms]
        self.pub_cmd.publish(msg)

    def stop_motors(self):
        self.publish_cmd_safe_rpm(np.zeros(10))

    def motor_callback(self, msg):
        try:
            new_q = np.zeros(10)
            for i, mid in enumerate(self.CONFIG['MOTOR_IDS']):
                if mid in msg.ids: 
                    idx = msg.ids.index(mid)
                    new_q[i] = msg.positions[idx]
            self.curr_q = new_q
        except: pass

    def vision_callback(self, msg):
        try:
            temp = np.zeros(10)
            cnt = 0
            for i, vid in enumerate(self.CONFIG['VISION_IDS']):
                if vid in msg.ids:
                    idx = msg.ids.index(vid)
                    temp[2*i] = -msg.x_local[idx] 
                    temp[2*i+1] = msg.y_local[idx]
                    cnt += 1
            if cnt == 5:
                self.curr_xy_raw = temp
                self.vision_updated_t = self.get_clock().now().nanoseconds / 1e9
                self.curr_xy_est = self.predict_current_pose(temp, self.vision_updated_t)
        except: pass

    def print_diagnostics(self):
        if not self.is_active or self.target_xy is None:
            return

        error_vec = self.target_xy - self.curr_xy_est
        max_err = np.max(np.abs(error_vec))
        
        sec_errs = []
        for i in range(5):
            e = np.linalg.norm(error_vec[2*i : 2*i+2])
            sec_errs.append(e)

        print("-" * 60)
        mode = "HYBRID" if (self.CONFIG['JACOBIAN_MODE']==0 and self.brain) else "PHYSICS"
        status = 'MOVING' if max_err > self.CONFIG['DONE_THRES_MM'] else 'HOLDING'
        
        # 打印当前增益，方便调试软着陆效果
        print(f"📡 MODE: {mode} | Max Err: {max_err:.2f}mm | Gain: {self.current_effective_gain:.2f} | {status}")
        
        print(f"{'SEC':<5} | {'TARGET (X,Y)':<18} | {'CURRENT (X,Y)':<18} | {'ERR (mm)':<10}")
        for i in range(5):
            tx, ty = self.target_xy[2*i], self.target_xy[2*i+1]
            cx, cy = self.curr_xy_est[2*i], self.curr_xy_est[2*i+1]
            print(f"#{i+1:<4} | ({tx:>6.1f}, {ty:>6.1f})   | ({cx:>6.1f}, {cy:>6.1f})   | {sec_errs[i]:<10.2f}")
        print("-" * 60)

def main(args=None):
    rclpy.init(args=args)
    node = JacobianServoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_motors()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()