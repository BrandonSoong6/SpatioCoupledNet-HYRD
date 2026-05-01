#!/usr/bin/env python3
# File: ~/brandon/hyrd_robot/src/robot_brain/robot_brain/core/latency_calib.py
import rclpy
from rclpy.node import Node
import numpy as np
import time
import sys
import os
import json
import argparse
from scipy import stats 

from robot_interfaces.msg import MotorCommand, MotorState, VisionState

class RobustLatencyCalibrator(Node):
    def __init__(self, save_path=None):
        super().__init__('robust_latency_calibrator')
        self.save_path = save_path
        
        # ==========================================
        # ⚙️ 标定配置 (大幅度版)
        # ==========================================
        self.CONFIG = {
            'VISION_ID': 5,          # 盯着末端看
            'MOTOR_IDS': [1, 2],     # 动根部关节
            
            # 🔥 加大幅度：从 30mm 跑到 100mm，信噪比极高
            'POS_A_MM': 30.0,
            'POS_B_MM': 100.0,
            
            'TEST_CYCLES': 12,       # 跑 12 个来回足够了
            'HOLD_TIME': 0.8,        # 稍微停久一点让波形稳住
            
            'KP': 8.0,               # P增益
            'RPM_RATIO': 1.201,
            'MAX_RPM': 95.0,         # 允许快一点
            'CONTROL_FREQ': 100.0,
            'VISION_TIMEOUT': 1.0
        }

        # 状态变量
        self.curr_motor_mm = None 
        self.curr_vision = {'x': 0.0, 'y': 0.0, 't_update': time.time()}
        self.target_pos = self.CONFIG['POS_A_MM']
        self.history_data = [] 
        
        self.state = "INIT"
        self.state_start_time = 0.0
        self.cycle_count = 0
        self.current_step_target = "A" 

        # 通信
        self.pub_cmd = self.create_publisher(MotorCommand, 'motor_cmd', 10)
        self.sub_motor = self.create_subscription(MotorState, 'motor_state', self.motor_callback, 10)
        self.sub_vision = self.create_subscription(VisionState, 'vision/state', self.vision_callback, 10)
        
        self.dt = 1.0 / self.CONFIG['CONTROL_FREQ']
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info(f"🚀 Large-Amplitude Calibrator Started. Cycles: {self.CONFIG['TEST_CYCLES']}")

    def motor_callback(self, msg):
        found = []
        try:
            for tid in self.CONFIG['MOTOR_IDS']:
                if tid in msg.ids:
                    found.append(msg.positions[msg.ids.index(tid)])
            if found: self.curr_motor_mm = sum(found) / len(found)
        except: pass

    def vision_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        try:
            if self.CONFIG['VISION_ID'] in msg.ids:
                idx = msg.ids.index(self.CONFIG['VISION_ID'])
                self.curr_vision['x'] = msg.x_local[idx]
                self.curr_vision['y'] = msg.y_local[idx]
                self.curr_vision['t_update'] = now
        except: pass

    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        
        if self.curr_motor_mm is None: return
        if (now - self.curr_vision['t_update']) > self.CONFIG['VISION_TIMEOUT']:
            if self.state not in ["INIT", "ANALYZING"]:
                self.get_logger().error("Vision Lost! E-Stop.")
                self.emergency_stop()
                sys.exit(1)

        # 状态机
        if self.state == "INIT":
            self.target_pos = self.CONFIG['POS_A_MM']
            self.state = "HOMING"
            self.state_start_time = now

        elif self.state == "HOMING":
            # 慢慢走到起点
            if abs(self.curr_motor_mm - self.CONFIG['POS_A_MM']) < 2.0 and (now - self.state_start_time > 2.0):
                self.state = "CYCLING"
                self.state_start_time = now
                self.cycle_count = 0
                self.current_step_target = "B"
                self.target_pos = self.CONFIG['POS_B_MM']

        elif self.state == "CYCLING":
            self.record_data(now)
            
            # 阶跃切换
            if now - self.state_start_time > self.CONFIG['HOLD_TIME']:
                if self.current_step_target == "A":
                    self.target_pos = self.CONFIG['POS_B_MM']
                    self.current_step_target = "B"
                else:
                    self.target_pos = self.CONFIG['POS_A_MM']
                    self.current_step_target = "A"
                    self.cycle_count += 1
                    if self.cycle_count % 2 == 0:
                        self.get_logger().info(f"Cycle {self.cycle_count}/{self.CONFIG['TEST_CYCLES']}")

                self.state_start_time = now
                
                # 结束判定
                if self.cycle_count >= self.CONFIG['TEST_CYCLES']:
                    self.state = "ANALYZING"
                    self.emergency_stop()
                    self.analyze_and_save()
                    raise SystemExit 

        if self.state != "ANALYZING":
            self.run_pid()

    def run_pid(self):
        err = self.target_pos - self.curr_motor_mm
        # 简单的 P 控制
        rpm = np.clip(err * self.CONFIG['KP'] * self.CONFIG['RPM_RATIO'], -self.CONFIG['MAX_RPM'], self.CONFIG['MAX_RPM'])
        msg = MotorCommand()
        msg.ids = self.CONFIG['MOTOR_IDS']
        msg.target_rpms = [float(rpm)] * len(self.CONFIG['MOTOR_IDS'])
        self.pub_cmd.publish(msg)

    def record_data(self, t):
        self.history_data.append([t, self.target_pos, self.curr_motor_mm, self.curr_vision['x'], self.curr_vision['y']])

    def emergency_stop(self):
        msg = MotorCommand()
        msg.ids = self.CONFIG['MOTOR_IDS']
        msg.target_rpms = [0.0] * len(self.CONFIG['MOTOR_IDS'])
        self.pub_cmd.publish(msg)

    # =================================================================
    # 🔥🔥🔥 核心：双假设竞争分析逻辑 (Dual Hypothesis) 🔥🔥🔥
    # =================================================================
    def analyze_and_save(self):
        self.get_logger().info("🔍 Analyzing Data (Smart Polarity Mode)...")
        data = np.array(self.history_data)
        
        if len(data) < 10: return

        # 提取数据
        t = data[:, 0]
        mot = data[:, 2]
        vis_x = data[:, 3]
        vis_y = data[:, 4]

        # 1. 自动选轴 (选动得多的那个)
        range_mot = np.ptp(mot)
        range_vx = np.ptp(vis_x)
        range_vy = np.ptp(vis_y)
        vis = vis_y if range_vy > range_vx else vis_x
        range_vis = np.ptp(vis)

        print(f"\n📊 STATS: MotRange={range_mot:.1f}mm | VisRange={range_vis:.1f}mm")

        # 2. 准备两套数据：正相 vs 反相
        n_mot = (mot - mot.min()) / range_mot
        n_vis_normal = (vis - vis.min()) / range_vis       
        n_vis_inverted = 1.0 - n_vis_normal                

        # 3. 边缘提取函数 (阈值 0.5)
        def get_edges(time_arr, signal_arr):
            binary = (signal_arr > 0.5).astype(int)
            diff = np.diff(binary)
            indices = np.where(diff != 0)[0]
            edges = []
            for idx in indices:
                t0, t1 = time_arr[idx], time_arr[idx+1]
                y0, y1 = signal_arr[idx], signal_arr[idx+1]
                if abs(y1 - y0) > 1e-6:
                    t_cross = t0 + (0.5 - y0) * (t1 - t0) / (y1 - y0)
                    type_edge = 1 if y1 > y0 else -1
                    edges.append((t_cross, type_edge))
            return edges

        # 4. 延迟评估函数
        def evaluate_latency(mot_edges, vis_sig_arr):
            vis_edges = get_edges(t, vis_sig_arr)
            delays = []
            valid_matches = 0
            
            for m_t, m_type in mot_edges:
                # 寻找未来 0.0s ~ 0.8s 内的同向边缘
                candidates = [v for v in vis_edges if (v[0] - m_t) > 0.0 and (v[0] - m_t) < 0.8 and v[1] == m_type]
                if candidates:
                    # 取最近的一个
                    best = min(candidates, key=lambda x: x[0] - m_t)
                    delays.append((best[0] - m_t) * 1000.0)
                    valid_matches += 1
            
            if not delays: return -999.0, 0, 999.0 
            return np.median(delays), valid_matches, np.std(delays)

        # 5. 提取电机边缘
        mot_edges = get_edges(t, n_mot)
        if not mot_edges:
            self.get_logger().error("❌ No Motor Edges! Check Motion.")
            self._save_result(0.15, True) 
            return

        # 6. 🔥 竞争：看谁匹配得更好 🔥
        lat_norm, count_norm, std_norm = evaluate_latency(mot_edges, n_vis_normal)
        lat_inv, count_inv, std_inv = evaluate_latency(mot_edges, n_vis_inverted)
        
        print(f"\n🧪 HYPOTHESIS TEST:")
        print(f"   Option A (Normal):   Lat={lat_norm:.2f}ms | Matches={count_norm}/{len(mot_edges)} | Std={std_norm:.2f}")
        print(f"   Option B (Inverted): Lat={lat_inv:.2f}ms | Matches={count_inv}/{len(mot_edges)}  | Std={std_inv:.2f}")

        # 7. 决策逻辑
        final_latency = 0.150 # Default Fallback
        
        # 谁匹配到的点多，谁赢；如果一样多，谁方差小（更稳），谁赢。
        if count_inv > count_norm and lat_inv > 0:
            print("✅ Winner: INVERTED (More matches)")
            final_latency = lat_inv / 1000.0
        elif count_norm > count_inv and lat_norm > 0:
            print("✅ Winner: NORMAL (More matches)")
            final_latency = lat_norm / 1000.0
        else:
            # 数量一样，拼稳定性
            if lat_inv > 0 and (lat_norm < 0 or std_inv < std_norm):
                print("✅ Winner: INVERTED (Better stability)")
                final_latency = lat_inv / 1000.0
            elif lat_norm > 0:
                print("✅ Winner: NORMAL (Better stability)")
                final_latency = lat_norm / 1000.0
            else:
                print("⚠️ No valid positive latency found. Using default.")

        # 8. 保存
        self._save_result(final_latency, True)

    def _save_result(self, latency_s, success):
        result = {
            "recommended_latency_s": latency_s,
            "std_ms": 0.0, # 简化
            "timestamp": time.time()
        }
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as f:
                json.dump(result, f, indent=4)
            self.get_logger().info(f"💾 Saved {latency_s*1000:.2f}ms to {self.save_path}")

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=None)
    args, unknown = parser.parse_known_args()
    
    node = RobustLatencyCalibrator(save_path=args.save_path)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.emergency_stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()