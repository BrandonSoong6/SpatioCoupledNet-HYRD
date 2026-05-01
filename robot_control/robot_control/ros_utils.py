import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from robot_interfaces.msg import VisionState
from cv_bridge import CvBridge

class VisionPublisher:
    def __init__(self, node: Node):
        self.node = node
        self.bridge = CvBridge()
        
        # 1. 视觉状态发布者 (自定义消息)
        self.state_pub = node.create_publisher(VisionState, 'vision/state', 10)
        
        # 2. 图像发布者 (用于 RViz)
        self.image_pub = node.create_publisher(Image, 'vision/image_debug', 10)
        
    def publish_status(self, ids, x_list, y_list, theta_list, is_locked):
        """
        打包并发送 VisionState 消息
        """
        msg = VisionState()
        msg.header = Header()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link" # 方便在 RViz 中显示
        
        msg.ids = [int(i) for i in ids]
        msg.x_local = [float(v) for v in x_list]
        msg.y_local = [float(v) for v in y_list]
        msg.theta = [float(v) for v in theta_list]
        msg.is_plane_locked = bool(is_locked)
        
        self.state_pub.publish(msg)

    def publish_debug_image(self, frame_cv):
        """
        发送图像到 RViz
        """
        try:
            # encoding="bgr8" 是 OpenCV 默认格式
            msg = self.bridge.cv2_to_imgmsg(frame_cv, encoding="bgr8")
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            self.image_pub.publish(msg)
        except Exception as e:
            self.node.get_logger().error(f"Image publish failed: {e}")