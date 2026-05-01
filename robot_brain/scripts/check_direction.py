#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import time
from robot_interfaces.msg import MotorCommand, MotorState, VisionState
from robot_brain.inference import compute_jacobian_xy_single, C_LIST, N_VAL, M_VAL

class DirectionChecker(Node):
    def __init__(self):
        super().__init__('direction_checker')
        self.pub = self.create_publisher(MotorCommand, 'motor_cmd', 10)
        self.sub_vis = self.create_subscription(VisionState, 'vision/state', self.vis_cb, 10)
        self.sub_mot = self.create_subscription(MotorState, 'motor_state', self.mot_cb, 10)
        
        self.curr_q = None
        self.curr_xy = None
        
        # === 🔧 修改这里：测试基座 (Base) ===
        # q9, q10 是基座电机 (索引 8, 9)
        self.TEST_MOTOR_IDS = [9, 10] 
        # Vision ID 1 是第一节 (Base Section)
        self.TEST_VIS_ID = 1     
        
        self.get_logger().info("🔍 Direction Checker Started (Testing Base: q9/10 vs Sec1)...")

    def mot_cb(self, msg):
        # 存完整的 q (10维)
        if self.curr_q is None: self.curr_q = np.zeros(10)
        for i, mid in enumerate(range(1, 11)): # 1..10
            if mid in msg.ids:
                self.curr_q[i] = msg.positions[msg.ids.index(mid)]

    def vis_cb(self, msg):
        if self.TEST_VIS_ID in msg.ids:
            idx = msg.ids.index(self.TEST_VIS_ID)
            self.curr_xy = np.array([msg.x_local[idx], msg.y_local[idx]])

    def run_test(self):
        while self.curr_q is None or self.curr_xy is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        print("\n=== 🛑 保持静止，记录初始状态 ===")
        start_q = self.curr_q.copy()
        start_xy = self.curr_xy.copy()
        print(f"Initial Q (Full): {np.round(start_q, 2)}")
        print(f"Initial XY (Sec1): {np.round(start_xy, 2)}")
        
        # 1. 计算物理预测
        # q9 是倒数第二个 (索引 8)
        # xy_sec1 是前两个 (索引 0, 1)
        J_phys = compute_jacobian_xy_single(start_q.astype(np.float64), C_LIST, N_VAL, M_VAL)
        
        # 关注: d(Sec1_X) / d(q9)
        j_x_q9 = J_phys[0, 8]
        # 关注: d(Sec1_Y) / d(q9)
        j_y_q9 = J_phys[1, 8]
        
        print("\n------------------------------------------------")
        print(f"📚 物理模型预测 (Base Sensitivity):")
        print(f"  d(X_sec1) / d(q9) = {j_x_q9:.4f}")
        print(f"  d(Y_sec1) / d(q9) = {j_y_q9:.4f}")
        print("------------------------------------------------\n")

        # 2. 动一下 q9
        MOVE_DQ = 0.5 # 动幅度大一点 (0.5 rad) 确保看清
        print(f"👉 正在执行动作: q9 (Base Left) 增加 {MOVE_DQ} rad ...")
        
        target_rpms = [0.0]*10
        # q9 对应索引 8，电机ID 9
        target_rpms[8] = 30.0 # 让 q9 动
        
        msg = MotorCommand()
        msg.ids = list(range(1, 11))
        msg.target_rpms = target_rpms
        
        t0 = time.time()
        while time.time() - t0 < 0.8: # 动 0.8秒
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)
            
        # 停
        msg.target_rpms = [0.0]*10
        self.pub.publish(msg)
        time.sleep(1.0) # 等稳
        
        # 3. 再次读取
        rclpy.spin_once(self, timeout_sec=0.1) # 刷新一下
        end_xy = self.curr_xy.copy()
        real_delta_xy = end_xy - start_xy
        
        print(f"\n👀 视觉观测到的变化: {real_delta_xy}")
        
        # 4. 判定
        print("\n=== 🏁 结论分析 ===")
        
        # X 轴判定
        if abs(real_delta_xy[0]) < 0.5:
            print("⚠️ X轴变化太小，跳过判定")
        else:
            if np.sign(j_x_q9) == np.sign(real_delta_xy[0]):
                print("✅ X轴方向: 正常 (MATCH)")
            else:
                print("❌ X轴方向: 反了 (REVERSED) -> 需要在 vision_cb 取反 X")
                
        # Y 轴判定
        if abs(real_delta_xy[1]) < 0.5:
            print("⚠️ Y轴变化太小，跳过判定")
        else:
            if np.sign(j_y_q9) == np.sign(real_delta_xy[1]):
                print("✅ Y轴方向: 正常 (MATCH)")
            else:
                print("❌ Y轴方向: 反了 (REVERSED) -> 需要在 vision_cb 取反 Y")

def main():
    rclpy.init()
    node = DirectionChecker()
    try:
        node.run_test()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()