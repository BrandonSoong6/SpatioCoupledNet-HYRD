from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. C++ 驱动节点 - 使用 Ubuntu 原生终端 (gnome-terminal)
        Node(
            package='robot_driver',
            executable='driver_node',
            name='driver_node',
            output='screen',
            parameters=[{'serial_port': '/dev/sts_servo'}],
            # 使用 gnome-terminal，并强制设置窗口大小为 130x30
            prefix="gnome-terminal --geometry=130x30 -- " 
        ),

        # 2. 视觉节点 - 依然在启动 launch 的主终端打印日志
        Node(
            package='robot_control',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=[{'show_local_window': True}]
        ),

        # 3. 键盘控制节点 - 后台静默运行，不弹出单独窗口
        Node(
            package='robot_control',
            executable='teleop_keyboard',
            name='teleop_keyboard',
            output='screen'
            # 删除了 prefix，因为它现在用 pynput 全局监听，不需要弹窗
        )
    ])