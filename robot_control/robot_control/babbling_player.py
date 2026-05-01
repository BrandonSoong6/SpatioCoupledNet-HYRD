#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import numpy as np
import os
import sys
import time
import datetime

# === 消息接口 ===
from robot_interfaces.msg import MotorCommand, MotorState, VisionState

# ==========================================
# 1. 全局配置
# ==========================================
CONTROL_FREQ = 100.0          
DT = 1.0 / CONTROL_FREQ       
POINTS_PER_SEC = 0.5         

# ID 定义 (电机)
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
    def __init__(self):
        super().__init__('robust_data_recorder')
        
        self.declare_parameter('test_mode', False)
        self.is_test_mode = self.get_parameter('test_mode').value
        
        mode_str = "🧪 TEST MODE (5% Only)" if self.is_test_mode else "💿 FULL RECORD MODE"
        self.get_logger().info(f"🚀 Initializing Recorder | {mode_str}")

        self._init_control_vectors()
        
        # === 关键修改 1: 就绪状态标志位 ===
        self.traj_ready = False   # 轨迹文件是否加载
        self.motor_ready = False  # 电机是否有心跳
        self.vision_ready = False # 视觉是否有心跳
        
        # === 2. 资源加载 (严格检查) ===
        self.traj_data = None 
        self.total_points = 0
        if not self._init_data_resource():
            self.get_logger().error("❌ CRITICAL: Data load failed. Exiting.")
            sys.exit(1)
        else:
            self.traj_ready = True

        self.record_limit_idx = int(self.total_points * 0.05) if self.is_test_mode else self.total_points
        self.current_traj_idx = 0 
        self.data_buffer = [] 

        # === 3. 通信接口 ===
        self.pub_cmd = self.create_publisher(MotorCommand, '/motor_cmd', 10)
        self.sub_state = self.create_subscription(MotorState, '/motor_state', self.state_callback, 10)
        self.sub_vision = self.create_subscription(VisionState, '/vision/state', self.vision_callback, 10)
        
        # === 4. 运行时状态 ===
        self.current_pos = np.zeros(TOTAL_JOINTS)
        self.last_motor_stamp = 0.0
        
        self.aligned_vision_data = np.zeros(VISION_FEAT_DIM) 
        self.is_plane_locked = 0.0
        self.last_vision_stamp = 0.0
        
        self.prev_err = np.zeros(TOTAL_JOINTS)
        self.start_time = None
        
        self.state = "WAITING_FOR_SYSTEM_READY" # 初始状态变更
        self.loop_counter = 0
        self.stabilize_counter = 0
        self.check_tick = 0 # 用于低频打印等待信息
        
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
            package_share_dir = get_package_share_directory('robot_control')
            data_path = os.path.join(package_share_dir, 'data', 'loom_pro_10d.npy')
            if not os.path.exists(data_path):
                self.get_logger().error(f"❌ File missing: {data_path}")
                return False
            raw_data = np.load(data_path)
            self.traj_data = raw_data.reshape(raw_data.shape[0], -1)
            self.total_points = self.traj_data.shape[0]
            self.get_logger().info(f"📂 Loaded Trajectory: {self.total_points} points")
            return True
        except Exception as e:
            self.get_logger().error(f"❌ Load Exception: {e}")
            return False

    def state_callback(self, msg):
        self.last_motor_stamp = self.get_clock().now().nanoseconds / 1e9
        
        # 收到第一帧且 ID 完整时，标记就绪
        temp_pos = {uid: pos for uid, pos in zip(msg.ids, msg.positions)}
        if all(uid in temp_pos for uid in ALL_IDS):
            self.current_pos = np.array([temp_pos[uid] for uid in ALL_IDS])
            if not self.motor_ready:
                self.motor_ready = True
                self.get_logger().info("✅ Motor Feedback Detected.")

    def vision_callback(self, msg):
        self.last_vision_stamp = self.get_clock().now().nanoseconds / 1e9
        
        # 收到第一帧时，标记就绪
        if not self.vision_ready:
            self.vision_ready = True
            self.get_logger().info("✅ Vision Feedback Detected.")

        self.is_plane_locked = 1.0 if msg.is_plane_locked else 0.0
        aligned_feats = np.zeros(VISION_FEAT_DIM)
        
        if len(msg.ids) > 0:
            vision_dict = {}
            for i, vid in enumerate(msg.ids):
                vision_dict[vid] = (msg.x_local[i], msg.y_local[i], msg.theta[i])
            
            for i, target_id in enumerate(EXPECTED_VISION_IDS):
                if target_id in vision_dict:
                    idx_start = i * 3
                    x, y, th = vision_dict[target_id]
                    aligned_feats[idx_start]     = x
                    aligned_feats[idx_start + 1] = y
                    aligned_feats[idx_start + 2] = th
        
        self.aligned_vision_data = aligned_feats

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
            # === 关键修改 2: 全系统检查逻辑 ===
            all_systems_go = self.traj_ready and self.motor_ready and self.vision_ready
            
            if all_systems_go:
                self.get_logger().info("🌟 ALL SYSTEMS GO! Starting Homing Sequence...")
                self.state = "HOMING"
            else:
                # 每100次循环(1秒)打印一次等待状态
                self.check_tick += 1
                if self.check_tick % 100 == 0:
                    missing = []
                    if not self.traj_ready: missing.append("Trajectory Data")
                    if not self.motor_ready: missing.append("Motor Feedback")
                    if not self.vision_ready: missing.append("Vision Feedback")
                    self.get_logger().warn(f"⏳ Waiting for: {', '.join(missing)}...")
            return # 只要不全，就直接跳出，不执行后续逻辑

        # --- 安全看门狗 (只有开始 HOMING 后才生效) ---
        # 此时我们期望所有信号都已经持续在线
        if now - self.last_motor_stamp > 0.2:
            self.get_logger().warn("⚠️ Motor Signal Lost! E-Stop.")
            self.emergency_stop()
            return
            
        # (可选) 如果视觉也丢了，是否要急停？通常视觉丢了不至于急停，但可以打印警告
        if now - self.last_vision_stamp > 0.5: 
             # 这里不急停，因为视觉可能被遮挡，但在录制中这是坏数据
             # 你可以决定是否要 log warn
             pass

        if self.state == "HOMING":
            target_pos = self.traj_data[0]
            err = target_pos - self.current_pos
            rpm = np.clip(err * 3.0, -15.0, 15.0) 
            self.send_motor_cmd(rpm)
            if np.all(np.abs(err) < 2.0):
                self.state = "STABILIZING"
                self.get_logger().info("⚓ Stabilizing...")

        elif self.state == "STABILIZING":
            err = self.traj_data[0] - self.current_pos
            rpm = err * self.KP_VEC
            self.send_motor_cmd(rpm)
            
            self.stabilize_counter += 1
            if self.stabilize_counter > 100: 
                self.start_time = now
                self.state = "RECORDING"
                self.get_logger().info("🎥 RECORDING STARTED!")

        elif self.state == "RECORDING":
            if self.is_test_mode and self.current_traj_idx >= self.record_limit_idx:
                self.get_logger().info(f"🧪 Test Limit Reached. Stopping.")
                self.finish_recording()
                return
            
            t_elapsed = now - self.start_time
            pos_ref, vel_ff = self.get_trajectory_point(t_elapsed)

            # --- 进度与时间预估 ---
            # 1. 计算基本进度
            progress = (self.current_traj_idx / self.record_limit_idx) * 100.0
            progress = min(progress, 100.0)

            # 2. 计算剩余时间 (ETA)
            # 这里的计算逻辑是：(已用时间 / 当前索引) * 剩余索引
            if self.current_traj_idx > 0:
                time_elapsed = t_elapsed
                total_est_time = (time_elapsed / self.current_traj_idx) * self.record_limit_idx
                time_remaining = total_est_time - time_elapsed
            else:
                time_elapsed = 0.0
                time_remaining = 0.0

            # 3. 每 10 帧更新一次终端
            if self.loop_counter % 10 == 0:
                bar_length = 30
                filled_length = int(bar_length * progress / 100)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                
                # 格式化时间为 MM:SS
                def fmt_time(s):
                    return f"{int(s//60):02d}:{int(s%60):02d}"

                sys.stdout.write(
                    f"\r🎥 Recording: [{bar}] {progress:6.2f}% | "
                    f"Time: {fmt_time(time_elapsed)}<{fmt_time(time_remaining)} | "
                    f"Points: {self.current_traj_idx}/{self.record_limit_idx}"
                )
                sys.stdout.flush()

            if pos_ref is None:
                self.get_logger().info("🎬 Trajectory Finished.")
                self.finish_recording()
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

    def finish_recording(self):
        """统一结束逻辑"""
        sys.stdout.write("\n")
        sys.stdout.flush()
        
        self.state = "FINISHED"
        self.save_dataset()
        self.send_motor_cmd(np.zeros(TOTAL_JOINTS))
        raise SystemExit

    def send_motor_cmd(self, rpms):
        msg = MotorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.ids = ALL_IDS
        msg.target_rpms = rpms.tolist()
        self.pub_cmd.publish(msg)

    def emergency_stop(self):
        stop_cmd = [0.0] * TOTAL_JOINTS
        for _ in range(5):
            self.send_motor_cmd(np.array(stop_cmd))
        self.get_logger().warn("🛑 E-STOP Triggered.")

    def save_dataset(self):
        if not self.data_buffer:
            self.get_logger().warn("⚠️ Buffer empty.")
            return

        self.get_logger().info("💾 Saving data...")
        final_data = np.array(self.data_buffer)
        
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "TEST" if self.is_test_mode else "FULL"
        save_dir = os.path.join(os.getcwd(), 'training_data')
        os.makedirs(save_dir, exist_ok=True)
        
        npy_path = os.path.join(save_dir, f"dataset_{tag}_{timestamp_str}.npy")
        np.save(npy_path, final_data)
        
        csv_path = os.path.join(save_dir, f"dataset_{tag}_{timestamp_str}.csv")
        
        vis_headers = []
        for vid in EXPECTED_VISION_IDS:
            vis_headers.extend([f"vis_x_id{vid}", f"vis_y_id{vid}", f"vis_th_id{vid}"])
            
        header_list = ["t_record", "t_motor", "t_vision"] + \
                      [f"tgt_q{i}" for i in range(1,11)] + \
                      [f"act_q{i}" for i in range(1,11)] + \
                      [f"cmd_rpm{i}" for i in range(1,11)] + \
                      [f"ff_vel{i}" for i in range(1,11)] + \
                      ["is_locked"] + vis_headers
                      
        header_str = ",".join(header_list)
        save_rows = final_data if len(final_data) < 5000 else final_data[:5000]
        np.savetxt(csv_path, save_rows, delimiter=",", header=header_str, comments='')
        self.get_logger().info(f"✅ Saved to {npy_path}")

def main(args=None):
    rclpy.init(args=args)
    node = RobustRecorder()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        node.get_logger().warn("⚠️ Interrupted!")
        node.save_dataset()
    except Exception as e:
        node.get_logger().error(f"❌ Error: {e}")
    finally:
        node.emergency_stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()