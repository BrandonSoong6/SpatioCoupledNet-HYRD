#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
import sys

# === 引入你的自定义消息 ===
from robot_interfaces.msg import MotorCommand, MotorState, VisionState

# ==========================================
# 高精度标定配置
# ==========================================
CONFIG = {
    # --- ID 配置 ---
    'VISION_ID': 5,           
    'MOTOR_IDS': [1, 2],      
    
    # --- 循环测试参数 ---
    'POS_A_MM': 60.0,         # 位置 A (低位)
    'POS_B_MM': 90.0,        # 位置 B (高位)
    'TEST_CYCLES': 10,        # 循环次数 (总共 20*2 = 40 次阶跃)
    'HOLD_TIME': 0.5,         # 每次到位后保持时间(秒)，需足够让视觉稳定
    
    # --- PID & 运动参数 ---
    'KP': 10.0,               
    'RPM_RATIO': 1.201,       
    'MAX_RPM': 90.0,          # 稍微提高限速以确保快速响应
    
    # --- 系统参数 ---
    'CONTROL_FREQ': 100.0,    # 100Hz 采样
    'VISION_TIMEOUT': 1.0     
}

class RobustLatencyCalibrator(Node):
    def __init__(self):
        super().__init__('robust_latency_calibrator')

        # 1. 状态变量
        self.curr_motor_mm = None 
        self.curr_vision = {'x': 0.0, 'y': 0.0, 't_update': self.get_clock().now().nanoseconds / 1e9}

        self.target_pos = CONFIG['POS_A_MM']
        self.history_data = [] # [t, target, motor, vis_x, vis_y]
        
        # 状态机
        self.state = "INIT"
        self.state_start_time = 0.0
        self.cycle_count = 0
        self.current_step_target = "A" # A or B
        
        # 2. 通信
        self.pub_cmd = self.create_publisher(MotorCommand, 'motor_cmd', 10)
        self.sub_motor = self.create_subscription(MotorState, 'motor_state', self.motor_callback, 10)
        self.sub_vision = self.create_subscription(VisionState, 'vision/state', self.vision_callback, 10)

        # 3. 定时器
        self.dt = 1.0 / CONFIG['CONTROL_FREQ']
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(f"🚀 Robust Calibrator Started.")
        self.get_logger().info(f"   Target: {CONFIG['POS_A_MM']}mm <-> {CONFIG['POS_B_MM']}mm")
        self.get_logger().info(f"   Cycles: {CONFIG['TEST_CYCLES']} (Total Steps: {CONFIG['TEST_CYCLES']*2})")

    # ==========================================
    # 回调 (保持不变)
    # ==========================================
    def motor_callback(self, msg):
        found = []
        try:
            for tid in CONFIG['MOTOR_IDS']:
                if tid in msg.ids:
                    found.append(msg.positions[msg.ids.index(tid)])
            if found: self.curr_motor_mm = sum(found) / len(found)
        except: pass

    def vision_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9
        try:
            if CONFIG['VISION_ID'] in msg.ids:
                idx = msg.ids.index(CONFIG['VISION_ID'])
                self.curr_vision['x'] = msg.x_local[idx]
                self.curr_vision['y'] = msg.y_local[idx]
                self.curr_vision['t_update'] = now
        except: pass

    # ==========================================
    # 核心控制循环
    # ==========================================
    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        
        # 安全检查
        if self.curr_motor_mm is None: return
        if (now - self.curr_vision['t_update']) > CONFIG['VISION_TIMEOUT']:
            if self.state not in ["INIT", "ANALYZING"]:
                self.get_logger().error("Vision Lost! Emergency Stop.")
                self.emergency_stop()
                sys.exit(1)

        # === 状态机逻辑 ===
        if self.state == "INIT":
            self.get_logger().info("Sensors OK. Homing to Pos A...")
            self.target_pos = CONFIG['POS_A_MM']
            self.state = "HOMING"
            self.state_start_time = now

        elif self.state == "HOMING":
            # 等待归位并稳定
            if abs(self.curr_motor_mm - CONFIG['POS_A_MM']) < 2.0:
                if now - self.state_start_time > 2.0:
                    self.get_logger().info("✅ Homed. Starting Cycles...")
                    self.state = "CYCLING"
                    self.state_start_time = now
                    self.cycle_count = 0
                    self.current_step_target = "B" # 第一次动作去 B
                    self.target_pos = CONFIG['POS_B_MM']
            else:
                self.state_start_time = now

        elif self.state == "CYCLING":
            self.record_data(now)
            
            # 判断是否该切换方向
            if now - self.state_start_time > CONFIG['HOLD_TIME']:
                # 切换目标
                if self.current_step_target == "A":
                    self.target_pos = CONFIG['POS_B_MM']
                    self.current_step_target = "B"
                else:
                    self.target_pos = CONFIG['POS_A_MM']
                    self.current_step_target = "A"
                    self.cycle_count += 1 # 完成一个 A-B-A 周期算一次
                    self.get_logger().info(f"Cycle {self.cycle_count}/{CONFIG['TEST_CYCLES']} Complete")

                self.state_start_time = now
                
                # 检查是否结束
                if self.cycle_count >= CONFIG['TEST_CYCLES']:
                    self.state = "ANALYZING"
                    self.emergency_stop()
                    self.analyze_robust()
                    self.destroy_node()
                    sys.exit(0)

        # PID 控制
        if self.state != "ANALYZING":
            self.run_pid()

    def run_pid(self):
        err = self.target_pos - self.curr_motor_mm
        rpm = np.clip(err * CONFIG['KP'] * CONFIG['RPM_RATIO'], -CONFIG['MAX_RPM'], CONFIG['MAX_RPM'])
        msg = MotorCommand()
        msg.ids = CONFIG['MOTOR_IDS']
        msg.target_rpms = [float(rpm)] * len(CONFIG['MOTOR_IDS'])
        self.pub_cmd.publish(msg)

    def record_data(self, t):
        self.history_data.append([t, self.target_pos, self.curr_motor_mm, self.curr_vision['x'], self.curr_vision['y']])

    def emergency_stop(self):
        msg = MotorCommand()
        msg.ids = CONFIG['MOTOR_IDS']
        msg.target_rpms = [0.0] * len(CONFIG['MOTOR_IDS'])
        self.pub_cmd.publish(msg)

    # ==========================================
    # 鲁棒数据分析 (Robust Analysis)
    # ==========================================
    def analyze_robust(self):
        self.get_logger().info("Processing Data...")
        data = np.array(self.history_data)
        t = data[:, 0] - data[0, 0]
        tgt = data[:, 1]
        mot = data[:, 2]
        # 自动选轴
        range_x = np.ptp(data[:, 3])
        range_y = np.ptp(data[:, 4])
        raw_vis = data[:, 4] if range_y > range_x else data[:, 3]
        
        # 1. 归一化处理
        def normalize(arr):
            return (arr - np.min(arr)) / np.ptp(arr)
        
        n_mot = normalize(mot)
        # 自动反相检测
        if np.corrcoef(n_mot, normalize(raw_vis))[0,1] < 0:
            raw_vis = -raw_vis
        n_vis = normalize(raw_vis)

        # 2. 提取所有穿越 50% 的时刻
        def get_edges(time_arr, signal_arr, threshold=0.5):
            # 使用 diff 找到符号变化点 (0 -> 1 或 1 -> 0)
            binary = (signal_arr > threshold).astype(int)
            diff = np.diff(binary)
            
            rising_indices = np.where(diff == 1)[0]
            falling_indices = np.where(diff == -1)[0]
            
            # 使用线性插值获取精确的 crossing time，而不是直接取 index 的时间
            edges = []
            for idx in np.concatenate([rising_indices, falling_indices]):
                # y = y0 + (y1-y0)/(t1-t0)*(t-t0) -> find t where y=0.5
                t0, t1 = time_arr[idx], time_arr[idx+1]
                y0, y1 = signal_arr[idx], signal_arr[idx+1]
                if abs(y1 - y0) > 1e-6:
                    t_cross = t0 + (threshold - y0) * (t1 - t0) / (y1 - y0)
                    # 标记是上升(1)还是下降(-1)
                    type_edge = 1 if y1 > y0 else -1
                    edges.append((t_cross, type_edge))
            
            # 按时间排序
            edges.sort(key=lambda x: x[0])
            return edges

        mot_edges = get_edges(t, n_mot)
        vis_edges = get_edges(t, n_vis)

        # 3. 边缘配对 (Pairing)
        delays = []
        delays_rising = []
        delays_falling = []

        for m_t, m_type in mot_edges:
            # 寻找该电机动作后，最近的一个同向视觉动作
            # 限制：必须在电机动作后 0s ~ 0.5s 内 (防止匹配错位)
            candidates = [v for v in vis_edges if v[0] > m_t and v[0] < m_t + 0.5 and v[1] == m_type]
            
            if candidates:
                # 取第一个匹配到的
                v_t = candidates[0][0]
                dt_ms = (v_t - m_t) * 1000.0
                delays.append(dt_ms)
                if m_type == 1:
                    delays_rising.append(dt_ms)
                else:
                    delays_falling.append(dt_ms)

        # 4. 统计学滤波 (Outlier Rejection)
        def robust_stats(data_list, name="All"):
            if len(data_list) < 3: return 0.0, 0.0
            arr = np.array(data_list)
            
            # IQR 过滤: 剔除超过 Q3 + 1.5*IQR 或 Q1 - 1.5*IQR 的值
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            clean_arr = arr[(arr >= q1 - 1.5*iqr) & (arr <= q3 + 1.5*iqr)]
            
            # 如果过滤后数据太少，回退
            if len(clean_arr) < 2: clean_arr = arr

            # 掐头去尾 (Trimmed Mean): 去掉两端各 10%
            final_mean = stats.trim_mean(clean_arr, 0.1)
            final_std = np.std(clean_arr)
            
            print(f"[{name}] N={len(arr)}->{len(clean_arr)} | Mean: {final_mean:.2f}ms | Std: {final_std:.2f}ms")
            return final_mean, clean_arr

        print("\n" + "="*40)
        print("ROBUST STATISTICAL ANALYSIS")
        print("="*40)
        
        mean_all, clean_all = robust_stats(delays, "Total")
        mean_rise, _ = robust_stats(delays_rising, "Rising (Low->High)")
        mean_fall, _ = robust_stats(delays_falling, "Falling (High->Low)")

        print(f"\n🏆 RECOMMENDED LATENCY: {mean_all:.2f} ms")
        print("="*40)

        # 5. 绘图
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)

        # 子图1: 时域波形 (前3个周期放大)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_title("Waveform Analysis (First 15s)")
        limit_idx = np.searchsorted(t, 15.0)
        ax1.plot(t[:limit_idx], n_mot[:limit_idx], 'b', label='Motor', alpha=0.6)
        ax1.plot(t[:limit_idx], n_vis[:limit_idx], 'r', label='Vision', alpha=0.6)
        ax1.set_ylabel("Normalized Position")
        ax1.legend()

        # 子图2: 延迟分布直方图
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_title(f"Latency Histogram (N={len(clean_all)})")
        ax2.hist(clean_all, bins=10, color='gray', alpha=0.7, rwidth=0.85)
        ax2.axvline(mean_all, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_all:.1f}ms')
        ax2.set_xlabel("Latency (ms)")
        ax2.legend()

        # 子图3: 散点图 (看稳定性)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_title("Latency Stability over Time")
        ax3.plot(delays, 'o-', alpha=0.6)
        ax3.axhline(mean_all, color='r', linestyle='--')
        ax3.set_xlabel("Sample Index")
        ax3.set_ylabel("Latency (ms)")

        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = RobustLatencyCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        node.emergency_stop()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


# [INFO] [1769660475.679025856] [robust_latency_calibrator]: 🚀 Robust Calibrator Started.
# [INFO] [1769660475.679565841] [robust_latency_calibrator]:    Target: 60.0mm <-> 90.0mm
# [INFO] [1769660475.679982191] [robust_latency_calibrator]:    Cycles: 40 (Total Steps: 80)
# [INFO] [1769660475.687786295] [robust_latency_calibrator]: Sensors OK. Homing to Pos A...
# [INFO] [1769660478.437802169] [robust_latency_calibrator]: ✅ Homed. Starting Cycles...
# [INFO] [1769660478.938125015] [robust_latency_calibrator]: Cycle 1/40 Complete
# [INFO] [1769660479.957688873] [robust_latency_calibrator]: Cycle 2/40 Complete
# [INFO] [1769660480.967858076] [robust_latency_calibrator]: Cycle 3/40 Complete
# [INFO] [1769660481.978625049] [robust_latency_calibrator]: Cycle 4/40 Complete
# [INFO] [1769660482.988130660] [robust_latency_calibrator]: Cycle 5/40 Complete
# [INFO] [1769660484.007686689] [robust_latency_calibrator]: Cycle 6/40 Complete
# [INFO] [1769660485.017863438] [robust_latency_calibrator]: Cycle 7/40 Complete
# [INFO] [1769660486.037807644] [robust_latency_calibrator]: Cycle 8/40 Complete
# [INFO] [1769660487.057643707] [robust_latency_calibrator]: Cycle 9/40 Complete
# [INFO] [1769660488.067829666] [robust_latency_calibrator]: Cycle 10/40 Complete
# [INFO] [1769660489.067894457] [robust_latency_calibrator]: Cycle 11/40 Complete
# [INFO] [1769660490.087871851] [robust_latency_calibrator]: Cycle 12/40 Complete
# [INFO] [1769660491.098298245] [robust_latency_calibrator]: Cycle 13/40 Complete
# [INFO] [1769660492.117812408] [robust_latency_calibrator]: Cycle 14/40 Complete
# [INFO] [1769660493.127806046] [robust_latency_calibrator]: Cycle 15/40 Complete
# [INFO] [1769660494.147791219] [robust_latency_calibrator]: Cycle 16/40 Complete
# [INFO] [1769660495.157798790] [robust_latency_calibrator]: Cycle 17/40 Complete
# [INFO] [1769660496.158092683] [robust_latency_calibrator]: Cycle 18/40 Complete
# [INFO] [1769660497.177786703] [robust_latency_calibrator]: Cycle 19/40 Complete
# [INFO] [1769660498.178039889] [robust_latency_calibrator]: Cycle 20/40 Complete
# [INFO] [1769660499.187801202] [robust_latency_calibrator]: Cycle 21/40 Complete
# [INFO] [1769660500.207628769] [robust_latency_calibrator]: Cycle 22/40 Complete
# [INFO] [1769660501.217873742] [robust_latency_calibrator]: Cycle 23/40 Complete
# [INFO] [1769660502.227987674] [robust_latency_calibrator]: Cycle 24/40 Complete
# [INFO] [1769660503.237782279] [robust_latency_calibrator]: Cycle 25/40 Complete
# [INFO] [1769660504.247758814] [robust_latency_calibrator]: Cycle 26/40 Complete
# [INFO] [1769660505.257911180] [robust_latency_calibrator]: Cycle 27/40 Complete
# [INFO] [1769660506.267861538] [robust_latency_calibrator]: Cycle 28/40 Complete
# [INFO] [1769660507.287737910] [robust_latency_calibrator]: Cycle 29/40 Complete
# [INFO] [1769660508.297777063] [robust_latency_calibrator]: Cycle 30/40 Complete
# [INFO] [1769660509.308366458] [robust_latency_calibrator]: Cycle 31/40 Complete
# [INFO] [1769660510.327654829] [robust_latency_calibrator]: Cycle 32/40 Complete
# [INFO] [1769660511.327844952] [robust_latency_calibrator]: Cycle 33/40 Complete
# [INFO] [1769660512.348291617] [robust_latency_calibrator]: Cycle 34/40 Complete
# [INFO] [1769660513.367979796] [robust_latency_calibrator]: Cycle 35/40 Complete
# [INFO] [1769660514.387956405] [robust_latency_calibrator]: Cycle 36/40 Complete
# [INFO] [1769660515.398048827] [robust_latency_calibrator]: Cycle 37/40 Complete
# [INFO] [1769660516.407840437] [robust_latency_calibrator]: Cycle 38/40 Complete
# [INFO] [1769660517.417743510] [robust_latency_calibrator]: Cycle 39/40 Complete
# [INFO] [1769660518.427836375] [robust_latency_calibrator]: Cycle 40/40 Complete
# [INFO] [1769660518.428180882] [robust_latency_calibrator]: Processing Data...

# ========================================
# ROBUST STATISTICAL ANALYSIS
# ========================================
# [Total] N=79->79 | Mean: 70.46ms | Std: 9.80ms
# [Rising (Low->High)] N=40->40 | Mean: 74.98ms | Std: 9.10ms
# [Falling (High->Low)] N=39->39 | Mean: 65.93ms | Std: 8.28ms

# 🏆 RECOMMENDED LATENCY: 70.46 ms
# ========================================

# $$SE = \frac{Std}{\sqrt{N}} = \frac{14.5}{\sqrt{60}} \approx 1.87\text{ms}$$

