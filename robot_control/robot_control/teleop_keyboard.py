import rclpy
from rclpy.node import Node
from robot_interfaces.msg import MotorCommand
import cv2
import numpy as np
import threading
# 引入 pynput 实现真·多键无冲监听
from pynput import keyboard 

# === 1. 参数配置 ===
TARGET_RPM = 50.0  # 目标速度

# === 2. 键位映射 ===
# ID: (正向键, 反向键)
SERVO_MAP = {
    10: ('w', 's'),
    9:  ('e', 'd'),
    8:  ('r', 'f'),
    7:  ('t', 'g'),
    6:  ('y', 'h'),
    5:  ('u', 'j'),
    4:  ('i', 'k'),
    3:  ('o', 'l'),
    2:  ('p', ';'),
    1:  ('[', "'") 
}

# 颜色定义
COLOR_BG = (30, 30, 30)
COLOR_BTN_IDLE = (50, 50, 50)
COLOR_BTN_POS  = (0, 180, 0)    # 绿
COLOR_BTN_NEG  = (0, 0, 180)    # 红
COLOR_TEXT_IDLE = (180, 180, 180)
COLOR_TEXT_ACTIVE = (255, 255, 255)

class SmoothTeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')
        self.pub = self.create_publisher(MotorCommand, 'motor_cmd', 10)
        
        # 记录当前按下的所有键 (线程安全集合)
        self.active_keys = set()
        self.lock = threading.Lock()
        
        # 记录每个电机上一次发送的指令，防止重复发送刷屏，但保证逻辑闭环
        # 格式: {id: last_rpm}
        self.last_motor_states = {i: 0.0 for i in range(1, 11)}

        self.get_logger().info("Smooth Multi-Key Teleop Started.")
        self.get_logger().info("Please keep the OpenCV window focused!")

        # 启动键盘监听器 (非阻塞)
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        """键盘按下回调"""
        try:
            char_key = key.char  # 普通字母键
        except AttributeError:
            char_key = None      # 特殊键

        if char_key:
            with self.lock:
                self.active_keys.add(char_key)

    def on_release(self, key):
        """键盘抬起回调"""
        try:
            char_key = key.char
        except AttributeError:
            char_key = None
            
        if char_key:
            with self.lock:
                if char_key in self.active_keys:
                    self.active_keys.remove(char_key)
        
        # 处理 ESC 退出
        if key == keyboard.Key.esc:
            return False

    def process_logic(self):
        """核心逻辑：根据当前按下的键集合，计算电机指令"""
        current_commands = {} # {id: target_rpm}

        with self.lock:
            keys_snapshot = self.active_keys.copy()

        # 1. 遍历映射表，确定每个电机的目标速度
        for servo_id, (pos_key, neg_key) in SERVO_MAP.items():
            target = 0.0
            
            # 逻辑：如果同时按下了正反键，则为0（互斥）
            is_pos = pos_key in keys_snapshot
            is_neg = neg_key in keys_snapshot
            
            if is_pos and not is_neg:
                target = TARGET_RPM
            elif not is_pos and is_neg:
                target = -TARGET_RPM
            
            current_commands[servo_id] = target

        # 2. 批量发送指令 (仅发送变化的，或者为了心跳包可以定期发)
        # 这里为了配合 C++ 的 Latch 逻辑，只要状态变了就必须发
        ids_to_send = []
        rpms_to_send = []
        
        for sid, target in current_commands.items():
            if self.last_motor_states[sid] != target:
                ids_to_send.append(sid)
                rpms_to_send.append(float(target))
                self.last_motor_states[sid] = target
        
        if ids_to_send:
            msg = MotorCommand()
            msg.ids = ids_to_send
            msg.target_rpms = rpms_to_send
            self.pub.publish(msg)
            # self.get_logger().info(f"Update: {dict(zip(ids_to_send, rpms_to_send))}")

        return current_commands # 返回给UI用于绘制

def draw_button(img, x, y, w, h, text, state):
    """绘制按钮 UI"""
    color = COLOR_BTN_IDLE
    text_color = COLOR_TEXT_IDLE
    thickness = 1
    
    if state == 1:   # 正向激活
        color = COLOR_BTN_POS
        text_color = COLOR_TEXT_ACTIVE
        thickness = -1
    elif state == -1: # 反向激活
        color = COLOR_BTN_NEG
        text_color = COLOR_TEXT_ACTIVE
        thickness = -1
        
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if thickness == 1:
        cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2 - 2
    cv2.putText(img, text, (tx, ty), font, font_scale, text_color, 1, cv2.LINE_AA)

def main(args=None):
    rclpy.init(args=args)
    node = SmoothTeleopNode()
    
    # GUI 参数
    W, H = 500, 650
    btn_w, btn_h = 100, 40
    col1_x = 120
    col2_x = 280
    start_y = 80
    gap_y = 55
    
    try:
        while rclpy.ok():
            # 1. 处理逻辑并获取当前状态
            current_cmds = node.process_logic()
            
            # 2. 绘制 UI
            canvas = np.full((H, W, 3), COLOR_BG, dtype=np.uint8)
            cv2.putText(canvas, "MULTI-KEY TELEOP", (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 绘制每一行
            for servo_id, (pos_key, neg_key) in SERVO_MAP.items():
                row_idx = 10 - servo_id
                y = start_y + row_idx * gap_y
                
                rpm = current_cmds.get(servo_id, 0.0)
                
                # 左侧按钮状态
                st_l = 1 if rpm > 0.1 else 0
                draw_button(canvas, col1_x, y, btn_w, btn_h, pos_key.upper(), st_l)
                
                # 中间 ID
                cv2.putText(canvas, f"ID {servo_id}", (col1_x + 115, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                
                # 右侧按钮状态
                st_r = -1 if rpm < -0.1 else 0
                draw_button(canvas, col2_x, y, btn_w, btn_h, neg_key.upper(), st_r)

            cv2.putText(canvas, "Press 'Q' to Quit", (20, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
            
            # 3. 显示
            cv2.imshow("Robot Teleop", canvas)
            
            # 4. 刷新 (这里的 waitKey 仅用于刷新 GUI，不负责输入逻辑)
            # 30Hz 刷新率
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break
            
            rclpy.spin_once(node, timeout_sec=0)

    except KeyboardInterrupt:
        pass
    finally:
        # 安全停止
        stop_msg = MotorCommand()
        stop_msg.ids = list(range(1, 11))
        stop_msg.target_rpms = [0.0] * 10
        node.pub.publish(stop_msg)
        
        node.listener.stop()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()