#!/usr/bin/env python3
# File: robot_brain/core/babbling_node.py
import rclpy
from rclpy.node import Node
import numpy as np
import os
import sys
import time
import datetime
import argparse

# 消息接口
from robot_interfaces.msg import MotorCommand, MotorState, VisionState

# ==========================================
# 1. 全局配置
# ==========================================
CONTROL_FREQ = 100.0          
DT = 1.0 / CONTROL_FREQ       
POINTS_PER_SEC = 0.5         

# ID 定义
STS_IDS = [1, 2, 3, 4, 5, 6]      
HLS_IDS = [7, 8, 9, 10]           
ALL_IDS = STS_IDS + HLS_IDS       
TOTAL_JOINTS = len(ALL_IDS)

STS_IDX = slice(0, 6)   
HLS_IDX = slice(6, 10)  

# === 视觉配置 ===
EXPECTED_VISION_IDS = [1, 2, 3, 4, 5] 
VISION_FEAT_DIM = len(EXPECTED_VISION_IDS) * 3 

# ==========================================
# 2. 物理与控制参数
# ==========================================
STS_MM_SEC_TO_RPM = 1.201   
STS_KP, STS_KD, STS_K_FF = 5.0, 0.03, 0.5
STS_MAX_RPM = 100.0

HLS_MM_SEC_TO_RPM = 1.201   
HLS_KP, HLS_KD, HLS_K_FF = 5.0, 0.03, 0.5
HLS_MAX_RPM = 100.0

class RobustRecorder(Node):
    def __init__(self, traj_path, output_path):
        super().__init__('robust_data_recorder')
        
        self.traj_path = traj_path
        self.output_path = output_path
        
        self.get_logger().info(f"🚀 Initializing Babbling Node (Fatal Block Mode)")
        
        self._init_control_vectors()
        
        # 状态标志位
        self.traj_ready = False   
        self.motor_ready = False  
        self.vision_ready = False 
        
        # 🔥 [新增] 致命错误标志位
        self.fatal_error = False
        self.estop_triggered = False
        
        # 资源加载
        self.traj_data = None 
        self.total_points = 0
        if not self._init_data_resource():
            self.get_logger().error("❌ CRITICAL: Data load failed. Exiting.")
            sys.exit(1)
        else:
            self.traj_ready = True

        self.record_limit_idx = self.total_points
        self.current_traj_idx = 0 
        self.data_buffer = [] 

        # 通信接口
        self.pub_cmd = self.create_publisher(MotorCommand, '/motor_cmd', 10)
        self.sub_state = self.create_subscription(MotorState, '/motor_state', self.state_callback, 10)
        self.sub_vision = self.create_subscription(VisionState, '/vision/state', self.vision_callback, 10)
        
        # 运行时状态
        self.current_pos = np.zeros(TOTAL_JOINTS)
        self.current_angles = None 
        self.last_motor_stamp = 0.0
        
        self.aligned_vision_data = np.zeros(VISION_FEAT_DIM) 
        self.is_plane_locked = 0.0
        self.last_vision_stamp = 0.0
        self.current_vision_angles = {} 
        
        self.prev_err = np.zeros(TOTAL_JOINTS)
        self.start_time = None
        self.homing_wait_start = None 
        
        self.state = "WAITING_FOR_SYSTEM_READY" 
        self.loop_counter = 0
        self.stabilize_counter = 0
        self.check_tick = 0 
        
        # 归位配置
        self.HOME_POS = np.ones(TOTAL_JOINTS) * 10.0 
        self.HOME_THRES = 1.0 
        
        self.timer = self.create_timer(DT, self.control_loop)

    def _init_control_vectors(self):
        self.KP_VEC = np.zeros(TOTAL_JOINTS)
        self.KD_VEC = np.zeros(TOTAL_JOINTS)
        self.KFF_VEC = np.zeros(TOTAL_JOINTS)
        self.RATIO_VEC = np.zeros(TOTAL_JOINTS)
        self.MAX_RPM_VEC = np.zeros(TOTAL_JOINTS)

        self.KP_VEC[STS_IDX], self.KD_VEC[STS_IDX], self.KFF_VEC[STS_IDX] = STS_KP, STS_KD, STS_K_FF
        self.RATIO_VEC[STS_IDX], self.MAX_RPM_VEC[STS_IDX] = STS_MM_SEC_TO_RPM, STS_MAX_RPM

        self.KP_VEC[HLS_IDX], self.KD_VEC[HLS_IDX], self.KFF_VEC[HLS_IDX] = HLS_KP, HLS_KD, HLS_K_FF
        self.RATIO_VEC[HLS_IDX], self.MAX_RPM_VEC[HLS_IDX] = HLS_MM_SEC_TO_RPM, HLS_MAX_RPM

    def _init_data_resource(self):
        try:
            if not os.path.exists(self.traj_path):
                self.get_logger().error(f"❌ File missing: {self.traj_path}")
                return False
            raw_data = np.load(self.traj_path)
            self.traj_data = raw_data.reshape(raw_data.shape[0], -1)
            self.total_points = self.traj_data.shape[0]
            return True
        except Exception as e:
            self.get_logger().error(f"❌ Load Exception: {e}")
            return False

    def state_callback(self, msg):
        self.last_motor_stamp = self.get_clock().now().nanoseconds / 1e9
        
        temp_pos = {uid: pos for uid, pos in zip(msg.ids, msg.positions)}
        if all(uid in temp_pos for uid in ALL_IDS):
            self.current_pos = np.array([temp_pos[uid] for uid in ALL_IDS])
            
            if not self.motor_ready:
                self.motor_ready = True
                self.get_logger().info("✅ Motor Feedback Detected.")

    def vision_callback(self, msg):
        self.last_vision_stamp = self.get_clock().now().nanoseconds / 1e9
        if not self.vision_ready:
            self.vision_ready = True
            self.get_logger().info("✅ Vision Feedback Detected.")

        self.is_plane_locked = 1.0 if msg.is_plane_locked else 0.0
        aligned_feats = np.zeros(VISION_FEAT_DIM)
        
        self.current_vision_angles = {}

        if len(msg.ids) > 0:
            vision_dict = {}
            for i, vid in enumerate(msg.ids):
                vision_dict[vid] = (msg.x_local[i], msg.y_local[i], msg.theta[i])
                self.current_vision_angles[vid] = msg.theta[i]
            
            for i, target_id in enumerate(EXPECTED_VISION_IDS):
                if target_id in vision_dict:
                    idx_start = i * 3
                    x, y, th = vision_dict[target_id]
                    aligned_feats[idx_start]     = x
                    aligned_feats[idx_start + 1] = y
                    aligned_feats[idx_start + 2] = th
        
        self.aligned_vision_data = aligned_feats

    def perform_safety_check(self):
        if not self.current_vision_angles:
            self.get_logger().warn("⚠️ Safety Check Skipped: No markers detected yet.")
            return True 
        
        TARGET_ANGLE = 90.0
        TOLERANCE = 3.0 
        
        bad_ids = []
        current_vals = []
        
        for vid in EXPECTED_VISION_IDS:
            if vid in self.current_vision_angles:
                ang = self.current_vision_angles[vid]
                if abs(ang - TARGET_ANGLE) > TOLERANCE:
                    bad_ids.append(vid)
                    current_vals.append(ang)
        
        if len(bad_ids) > 0:
            self.get_logger().error(f"\n{'='*40}")
            self.get_logger().error(f"🛑 VISUAL INTEGRITY CHECK FAILED!")
            self.get_logger().error(f"   Target: {TARGET_ANGLE}° ± {TOLERANCE}°")
            self.get_logger().error(f"   Bad Markers: {bad_ids}")
            self.get_logger().error(f"   Values:      {[round(x,1) for x in current_vals]}")
            self.get_logger().error(f"{'='*40}")
            return False
        
        return True

    def get_trajectory_point(self, t):
        idx_float = t * POINTS_PER_SEC
        idx_curr = int(idx_float)
        self.current_traj_idx = idx_curr 
        idx_next = idx_curr + 1
        
        if idx_next >= self.total_points:
            return None, None
            
        alpha = idx_float - idx_curr
        p_curr = self.traj_data[idx_curr]
        p_next = self.traj_data[idx_next]
        pos_ref = p_curr * (1 - alpha) + p_next * alpha
        vel_ff = (p_next - p_curr) * POINTS_PER_SEC
        return pos_ref, vel_ff

    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        
        # --- 状态机 ---
        if self.state == "WAITING_FOR_SYSTEM_READY":
            all_systems_go = self.traj_ready and self.motor_ready and self.vision_ready
            if all_systems_go:
                self.get_logger().info("🌟 ALL SYSTEMS GO! Starting Homing Sequence...")
                self.state = "HOMING"
            else:
                self.check_tick += 1
                if self.check_tick % 100 == 0:
                    self.get_logger().info("⏳ Waiting for feedback...")
            return 

        # 🔥 安全看门狗 (触发 FATAL ERROR)
        if now - self.last_motor_stamp > 0.2:
            self.get_logger().warn("⚠️ Motor Signal Lost! Triggering Fatal E-Stop.")
            self.fatal_error = True # 标记为致命错误
            self.emergency_stop()   # 进入死锁流程
            return

        if self.state == "HOMING":
            target_pos = self.HOME_POS
            err = target_pos - self.current_pos
            rpm = np.clip(err * 3.0, -20.0, 20.0) 
            self.send_motor_cmd(rpm)
            
            max_err = np.max(np.abs(err))
            if max_err < self.HOME_THRES:
                self.get_logger().info(f"⚓ Homing Reached (Err: {max_err:.2f}mm). Waiting for stability...")
                self.homing_wait_start = now
                self.state = "HOMING_WAIT"

        elif self.state == "HOMING_WAIT":
            target_pos = self.HOME_POS
            err = target_pos - self.current_pos
            rpm = np.clip(err * 3.0, -20.0, 20.0) 
            self.send_motor_cmd(rpm)
            
            if (now - self.homing_wait_start) > 1.0:
                self.get_logger().info("🔍 Performing Safety Check...")
                is_healthy = self.perform_safety_check()
                
                # 🔥 检查不通过 -> 致命错误
                if not is_healthy:
                    self.fatal_error = True
                    self.emergency_stop() 
                    return
                
                self.state = "STABILIZING"
                self.get_logger().info("✅ Mechanism OK. Stabilizing...")

        elif self.state == "STABILIZING":
            err = self.traj_data[0] - self.current_pos
            rpm = err * self.KP_VEC
            self.send_motor_cmd(rpm)
            
            self.stabilize_counter += 1
            if self.stabilize_counter > 50: 
                self.start_time = now
                self.state = "RECORDING"
                self.get_logger().info("🎥 RECORDING STARTED!")

        elif self.state == "RECORDING":
            t_elapsed = now - self.start_time
            pos_ref, vel_ff = self.get_trajectory_point(t_elapsed)

            if self.loop_counter % 10 == 0:
                progress = (self.current_traj_idx / self.record_limit_idx) * 100.0
                sys.stdout.write(f"\r🎥 Recording: {progress:6.2f}%")
                sys.stdout.flush()

            if pos_ref is None:
                self.get_logger().info("\n🎬 Trajectory Finished. Returning Home...")
                self.state = "RETURN_HOME"
                return

            err = pos_ref - self.current_pos
            d_err = (err - self.prev_err) / DT
            rpm_raw = (vel_ff * self.KFF_VEC + err * self.KP_VEC + d_err * self.KD_VEC) * self.RATIO_VEC
            rpm_cmd = np.maximum(np.minimum(rpm_raw, self.MAX_RPM_VEC), -self.MAX_RPM_VEC)
            
            self.prev_err = err
            self.send_motor_cmd(rpm_cmd)

            snapshot = np.concatenate([
                [now],                   
                [self.last_motor_stamp], 
                [self.last_vision_stamp],
                pos_ref,                 
                self.current_pos,        
                rpm_cmd,                 
                vel_ff,                  
                [self.is_plane_locked],  
                self.aligned_vision_data 
            ])
            self.data_buffer.append(snapshot)
            self.loop_counter += 1
            
        elif self.state == "RETURN_HOME":
            target_pos = self.HOME_POS
            err = target_pos - self.current_pos
            
            rpm = np.clip(err * 3.0, -20.0, 20.0)
            self.send_motor_cmd(rpm)
            
            max_err = np.max(np.abs(err))
            if max_err < self.HOME_THRES:
                self.get_logger().info(f"🛑 Home Reached (Err: {max_err:.2f}mm). Saving & Exiting.")
                self.finish_recording()

    def finish_recording(self):
        self.state = "FINISHED"
        self.save_dataset()
        self.send_motor_cmd(np.zeros(TOTAL_JOINTS))
        sys.exit(0) # 正常退出

    def send_motor_cmd(self, rpms):
        msg = MotorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.ids = ALL_IDS
        msg.target_rpms = rpms.tolist()
        self.pub_cmd.publish(msg)

    # =================================================================
    # 🔥🔥🔥 核心修改: 致命错误死锁逻辑 🔥🔥🔥
    # =================================================================
    def emergency_stop(self):
        """
        处理逻辑：
        1. 如果是 FATAL ERROR: 全力逃逸到 160.0，然后死循环卡住进程。
        2. 如果是 KeyboardInterrupt: 发送 0 速，安全退出。
        """
        if self.estop_triggered: return
        self.estop_triggered = True

        if self.fatal_error:
            self.get_logger().error(f"\n{'!'*50}")
            self.get_logger().error("🔥 FATAL MECHANISM FAILURE DETECTED! 🔥")
            self.get_logger().error(">>> INITIATING PANIC RETREAT TO 160.0 (SAFE POSE)")
            self.get_logger().error(f"{'!'*50}\n")
            
            # 1. 逃逸目标：最大行程 160.0
            target_escape = np.ones(TOTAL_JOINTS) * 160.0
            
            # 2. 开环逃逸 3.0 秒 (使用 time.sleep 阻塞一切)
            DURATION = 3.0
            STEPS = int(DURATION * 20) # 20Hz 发送
            
            for i in range(STEPS):
                # 基于最后一次已知的 current_pos 计算方向
                # 如果回调已经挂了，这个值虽然旧，但方向是对的
                err = target_escape - self.current_pos
                
                # 强力 P 控制，限速 80 RPM 保证能动
                escape_rpm = np.clip(err * 6.0, -80.0, 80.0)
                
                self.send_motor_cmd(escape_rpm)
                time.sleep(0.05) 
            
            self.get_logger().error("🛑 RETREAT COMPLETE. HOLDING POSITION.")
            self.get_logger().error("⛔ PROCESS IS NOW PERMANENTLY BLOCKED TO STOP ORCHESTRATOR.")
            
            # 3. 死锁：发送 0 速并无限循环
            # 这会让 Orchestrator 认为进程还在运行，从而卡在这一步
            while True:
                self.send_motor_cmd(np.zeros(TOTAL_JOINTS))
                time.sleep(1.0)
                self.get_logger().error("⛔ [BLOCKED] PLEASE CHECK ROBOT HARDWARE. CTRL+C TO KILL.", throttle_duration_sec=5)
        
        else:
            # 正常的手动中断 (Ctrl+C)
            self.get_logger().warn("🛑 User Interrupt (Safe Stop).")
            stop_cmd = [0.0] * TOTAL_JOINTS
            for _ in range(5):
                self.send_motor_cmd(np.array(stop_cmd))
                time.sleep(0.05)

    def save_dataset(self):
        if not self.data_buffer:
            self.get_logger().warn("⚠️ Buffer empty.")
            return

        self.get_logger().info("💾 Saving data...")
        final_data = np.array(self.data_buffer)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        np.save(self.output_path, final_data)
        self.get_logger().info(f"✅ Saved to {self.output_path}")

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_traj', type=str, required=True)
    parser.add_argument('--output_raw', type=str, required=True)
    args, unknown = parser.parse_known_args()
    
    node = RobustRecorder(traj_path=args.input_traj, output_path=args.output_raw)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        node.save_dataset()
    finally:
        # 这里会根据 self.fatal_error 决定是死锁还是退出
        node.emergency_stop()
        
        # 如果是致命错误，上面会死循环，永远走不到下面
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()