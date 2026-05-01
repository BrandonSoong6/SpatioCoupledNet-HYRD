import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from robot_interfaces.msg import MotorCommand  

class XboxTeleopNode(Node):
    def __init__(self):
        super().__init__('xbox_teleop_node')
        
        # 创建发布者，发给 C++ 驱动节点
        self.cmd_pub = self.create_publisher(MotorCommand, 'motor_cmd', 10)
        
        # 监听 Xbox 手柄消息
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        # --- 控制参数 ---
        self.max_rpm = 60.0   # 摇杆推满时，前进/后退的最大转速
        self.turn_rpm = 30.0  # 摇杆拨满时，左右差分产生的最大转速
        
        # 分组定义
        self.front_ids = [7, 8, 9, 10]          # 前两节 (现由右摇杆控制)
        self.back_ids = [1, 2, 3, 4, 5, 6]      # 后三节 (现由左摇杆控制)
        
        self.get_logger().info('Xbox Teleop Node Started. Listening to /joy ...')

    def joy_callback(self, msg: Joy):
        # 提取手柄轴数据
        left_x = msg.axes[0]
        left_y = msg.axes[1]
        right_x = msg.axes[3]
        right_y = msg.axes[4]
        
        cmd_msg = MotorCommand()
        cmd_msg.ids = []
        cmd_msg.target_rpms = []

        # ==========================================
        # 1. 处理前两节 (右摇杆控制)
        # ==========================================
        front_base_spd = right_y * self.max_rpm
        front_diff_spd = right_x * self.turn_rpm
        
        # 转向反转：差分速度的符号互换
        front_odd_rpm  = front_base_spd + front_diff_spd
        front_even_rpm = front_base_spd - front_diff_spd
        
        for fid in self.front_ids:
            cmd_msg.ids.append(fid)
            if fid % 2 != 0:
                cmd_msg.target_rpms.append(float(front_odd_rpm))
            else:
                cmd_msg.target_rpms.append(float(front_even_rpm))

        # ==========================================
        # 2. 处理后三节 (左摇杆控制)
        # ==========================================
        back_base_spd = left_y * self.max_rpm
        back_diff_spd = left_x * self.turn_rpm
        
        # 转向反转：差分速度的符号互换
        back_odd_rpm  = back_base_spd + back_diff_spd
        back_even_rpm = back_base_spd - back_diff_spd
        
        for bid in self.back_ids:
            cmd_msg.ids.append(bid)
            if bid % 2 != 0:
                cmd_msg.target_rpms.append(float(back_odd_rpm))
            else:
                cmd_msg.target_rpms.append(float(back_even_rpm))

        # ==========================================
        # 3. 发布最终指令
        # ==========================================
        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = XboxTeleopNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()