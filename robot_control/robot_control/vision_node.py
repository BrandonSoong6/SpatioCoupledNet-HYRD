import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import cv2
import numpy as np
import collections
import time
import os
import threading
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

# === 导入依赖 ===
import robot_control.config as cfg
from robot_control.drivers.camera import Camera
from robot_control.algorithms.tracker_2d import MarkerTracker
from robot_control.algorithms.pose_3d import GlobalPlaneEstimator
from robot_control.algorithms.math_utils import pixel_to_world_on_plane, get_projected_circle_pts
from robot_control.ros_utils import VisionPublisher

# ==========================================
# 辅助绘图函数 (4合1 高清调试图)
# ==========================================

def draw_debug_grid(debug_dict):
    """生成 4合1 (2x2) 调试拼图"""
    if not debug_dict: return np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    keys = ["2. Global Yellow", "3. Merged Mask", "4. Final Candidates", "5. Valid Ellipses"]
    target_w = 960; target_h = 540
    small_imgs = []
    
    for k in keys:
        img = debug_dict.get(k)
        if img is None: img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else: img = cv2.resize(img, (target_w, target_h))
            
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        cv2.rectangle(img, (0,0), (target_w, 40), (0,0,0), -1)
        cv2.putText(img, k, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        small_imgs.append(img)
        
    while len(small_imgs) < 4: small_imgs.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    top_row = np.hstack(small_imgs[:2])
    bot_row = np.hstack(small_imgs[2:4])
    return np.vstack([top_row, bot_row])

def draw_status_overlay(img, is_calibrating, init_progress):
    """仅在左上角显示简单的系统状态"""
    if is_calibrating:
        txt = f"SYSTEM: INITIALIZING {int(init_progress*100)}%"
        col = (0, 255, 255) # 黄
    else:
        txt = "SYSTEM: LOCKED"
        col = (0, 0, 255) # 红
    
    #加个黑色背景让字更清楚
    cv2.rectangle(img, (10, 10), (350, 40), (0,0,0), -1)
    cv2.putText(img, txt, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

# ==========================================
# 辅助计算类
# ==========================================

class PoseStabilizer:
    def __init__(self, history=15):
        self.history = history
        self.origins = collections.deque(maxlen=history)
        self.y_vecs = collections.deque(maxlen=history)
        self.last_R = None
        self.last_origin = None

    def update(self, p1_3d, p5_3d):
        p1 = np.array(p1_3d[:2])
        p5 = np.array(p5_3d[:2])
        vec_y = p5 - p1
        norm = np.linalg.norm(vec_y)
        if norm < 1e-3: return None, None
        unit_y = vec_y / norm
        origin = p1 - (unit_y * 52.0)
        self.origins.append(origin)
        self.y_vecs.append(unit_y) 
        avg_origin = np.mean(self.origins, axis=0)
        avg_unit_y = np.mean(self.y_vecs, axis=0)
        norm_avg = np.linalg.norm(avg_unit_y)
        if norm_avg < 1e-6: return None, None
        final_unit_y = avg_unit_y / norm_avg
        final_unit_x = np.array([final_unit_y[1], -final_unit_y[0]])
        R = np.array([final_unit_x, final_unit_y])
        self.last_R = R
        self.last_origin = avg_origin
        return R, avg_origin

    def get_current_transform(self):
        return self.last_R, self.last_origin

    def reset(self):
        self.origins.clear()
        self.y_vecs.clear()
        self.last_R = None
        self.last_origin = None

def apply_local_transform(point_3d, R, origin):
    p_xy = np.array(point_3d[:2])
    p_local = np.dot(R, p_xy - origin)
    return p_local

def calculate_local_orientation(dots_local):
    if len(dots_local) != 3: return 0.0
    p0, p1, p2 = dots_local[0], dots_local[1], dots_local[2]
    d_sq_01 = np.sum((p0 - p1)**2)
    d_sq_12 = np.sum((p1 - p2)**2)
    d_sq_20 = np.sum((p2 - p0)**2)
    if d_sq_01 >= d_sq_12 and d_sq_01 >= d_sq_20:
        p_free = p2; p_base_start = p0; p_base_end = p1
    elif d_sq_12 >= d_sq_01 and d_sq_12 >= d_sq_20:
        p_free = p0; p_base_start = p1; p_base_end = p2
    else:
        p_free = p1; p_base_start = p0; p_base_end = p2
    vec_base = p_base_end - p_base_start
    vec_start_to_free = p_free - p_base_start
    base_len_sq = np.sum(vec_base**2)
    if base_len_sq < 1e-6: return 0.0
    t = np.dot(vec_start_to_free, vec_base) / base_len_sq
    p_proj = p_base_start + t * vec_base
    vec_direction = p_proj - p_free
    return np.degrees(np.arctan2(vec_direction[1], vec_direction[0]))

# ==========================================
# ROS 2 主节点
# ==========================================

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        test_ = 0
        
        # 1. 参数声明
        self.declare_parameter('enable_image_stream', (True if test_ == 1 else False))
        self.declare_parameter('show_local_window', (True if test_ == 1 else False))
        self.declare_parameter('device_id', cfg.DEV_PATH)
        
        self.enable_stream = self.get_parameter('enable_image_stream').get_parameter_value().bool_value
        self.show_window = self.get_parameter('show_local_window').get_parameter_value().bool_value
        dev_path = self.get_parameter('device_id').get_parameter_value().string_value
        
        self.get_logger().info(f"Vision Node Started. Stream: {self.enable_stream}")

        # 2. 初始化 ROS 发布器
        self.ros_pub = VisionPublisher(self)
        self.obs_pub = self.create_publisher(Point, 'vision/obstacle_pos', 10)
        self.bridge = CvBridge()
        self.roi_img_pub = self.create_publisher(Image, 'vision/roi_image', 10)
        self.homography_pub = self.create_publisher(Float32MultiArray, 'vision/homography', 10)
        self.cached_H = None  # 用于缓存计算好的单应性矩阵

        # 3. 初始化视觉组件
        try:
            self.cam = Camera() 
        except Exception as e:
            self.get_logger().fatal(f"Camera Init Error: {e}")
            raise e

        self.pose_estimator = GlobalPlaneEstimator(self.cam.new_mtx, None)
        self.color_tracker = MarkerTracker()
        self.stabilizer = PoseStabilizer(history=10)
        
        # --- 相机初始化 ---
        self.cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.HEIGHT)
        # 强制 MJPG 以防止 USB 带宽瓶颈
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        if not self.cap.isOpened():
            self.get_logger().fatal(f"Could not open device: {dev_path}")
            return

        # =========================================================
        # 🔥🔥🔥 核心修改：多线程帧缓冲区管理 🔥🔥🔥
        # =========================================================
        self.frame_lock = threading.Lock()  # 线程锁
        self.latest_frame = None            # 最新的那一帧
        self.thread_running = True          # 线程控制标志
        
        # 启动独立线程：专门负责从 USB 读图，不进行任何处理
        self.read_thread = threading.Thread(target=self._camera_reader_loop, daemon=True)
        self.read_thread.start()
        
        self.get_logger().info("🚀 Camera Reading Thread Started (Low Latency Mode)")
        # =========================================================

        # 4. 状态变量
        self.global_rvec = None
        self.global_tvec = None
        self.is_calibrating = True       
        self.calibration_start_time = None
        self.CALIBRATION_DURATION = 3.0    
        self.REF_ID = 1
        self.TARGET_ID = 5
        self.last_img_pub_time = 0.0
        self.IMG_PUB_INTERVAL = 0.1 

        # ==========================================
        # 新增：障碍物专属检测配置
        # ==========================================
        self.OBS_MARKER_ID = 1  # 打印的障碍物 Marker ID
        self.obs_aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
        # 兼容不同版本的 OpenCV
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.obs_aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self.obs_aruco_params = cv2.aruco.DetectorParameters()

        # 5. 主定时器
        # 改为 0.01s (100Hz)，只要有新图就立刻处理
        self.timer = self.create_timer(0.01, self.timer_callback)

    def _camera_reader_loop(self):
        """
        [独立线程] 
        唯一任务：死循环清空缓冲区，永远只保留最后一张图。
        这保证了 Linux 内核缓冲区永远不会积压。
        """
        while self.thread_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame # 直接覆盖，丢弃旧的
            else:
                # 如果读失败，稍微睡一下避免死循环炸 CPU
                time.sleep(0.01)

    def timer_callback(self):
        # 1. 从线程缓冲区取图
        frame_raw = None
        
        with self.frame_lock:
            if self.latest_frame is not None:
                frame_raw = self.latest_frame.copy() # 拷贝出来处理
                self.latest_frame = None             # 清空，防止重复处理同一帧
        
        # 如果没有新图（处理速度 > 相机速度），直接返回，不浪费 CPU
        if frame_raw is None:
            return

        # --- 这里的代码每秒只会跑大约 30 次 ---
        curr_time = self.get_clock().now().nanoseconds / 1e9 
        
        # 处理逻辑...
        frame_rect = self.cam.rectify(frame_raw)

        # ============================================================
        # 全局椭圆 ROI 实现 (Global Elliptical ROI)
        # ============================================================
        # h, w = frame_rect.shape[:2]
        # center = ((w * 93) // 200, (h * 95) // 200)
        # axes_len = (int(0.6 * h), h // 2) 
        # roi_mask = np.zeros((h, w), dtype=np.uint8)
        # cv2.ellipse(roi_mask, center, axes_len, 0.0, 0.0, 360.0, 255, -1)
        # frame_rect = cv2.bitwise_and(frame_rect, frame_rect, mask=roi_mask)
        # # ============================================================

        # frame_vis = frame_rect.copy()

        # # --------------- 计算椭圆的外接矩形并裁剪为处理 ROI ---------------
        # coords = cv2.findNonZero(roi_mask)
        # if coords is not None:
        #     rx, ry, rw, rh = cv2.boundingRect(coords)
        #     if rw <= 0 or rh <= 0:
        #         rx, ry, rw, rh = 0, 0, w, h
        # else:
        #     rx, ry, rw, rh = 0, 0, w, h

        # try:
        #     proc_frame = frame_rect[ry:ry+rh, rx:rx+rw].copy()
        #     if proc_frame.size == 0:
        #         proc_frame = frame_rect.copy(); rx, ry = 0, 0; rw, rh = w, h
        # except Exception:
        #     proc_frame = frame_rect.copy(); rx, ry = 0, 0; rw, rh = w, h

        # hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
        
        # ============================================================
        # 矩形 ROI 替换 (原椭圆的最小外接矩形)
        # ============================================================
        h, w = frame_rect.shape[:2]
        
        # 1. 纯数学计算原椭圆参数，求得安全的最小外接矩形边界
        cx, cy = (w * 93) // 200, (h * 95) // 200
        ax, ay = int(0.6 * h), h // 2 
        
        x_min = max(0, cx - ax)
        y_min = max(0, cy - ay)
        x_max = min(w, cx + ax)
        y_max = min(h, cy + ay)
        
        rx, ry = x_min, y_min
        rw, rh = x_max - x_min, y_max - y_min

        # 2. 直接截取矩形处理区域，抛弃黑色遮罩逻辑
        proc_frame = frame_rect[ry:ry+rh, rx:rx+rw].copy()
        
        # ============================================================
        # 新增：如果平面已固定，发布纯净图像与单应性矩阵
        # ============================================================
        if not self.is_calibrating and self.cached_H is not None:
            # 发布去畸变、裁剪且无UI的画面
            if self.roi_img_pub.get_subscription_count() > 0:
                img_msg = self.bridge.cv2_to_imgmsg(proc_frame, encoding="bgr8")
                self.roi_img_pub.publish(img_msg)
            
            # 发布 3x3 单应性矩阵 (拍平成长度为9的数组)
            if self.homography_pub.get_subscription_count() > 0:
                h_msg = Float32MultiArray()
                h_msg.data = self.cached_H.flatten().tolist()
                self.homography_pub.publish(h_msg)
        # ============================================================

        # 3. 这里的 frame_vis 必须保留全图视角！
        frame_vis = frame_rect.copy()

        # ============================================================

        hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)

        # === 1. 初始化与全局 Marker 检测逻辑 ===
        init_progress = 0.0
        
        # 🚀【核心修改 1】将 Marker 识别移出 is_calibrating 块，保证全局每帧都能抓取到 Marker
        has_pose, rvec, tvec, corners, ids = self.pose_estimator.get_plane_pose(frame_rect)
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame_vis, corners, ids, borderColor=(60, 60, 60))

        if self.is_calibrating:
            if has_pose:
                if self.calibration_start_time is None:
                    self.calibration_start_time = curr_time
                
                elapsed = curr_time - self.calibration_start_time
                init_progress = min(elapsed / self.CALIBRATION_DURATION, 1.0)
                self.global_rvec, self.global_tvec = rvec, tvec
                
                if elapsed >= self.CALIBRATION_DURATION:
                    # 获取当前标定好的局部坐标系转换关系
                    local_R, local_origin = self.stabilizer.get_current_transform()
                    
                    if local_R is not None and local_origin is not None:
                        self.is_calibrating = False
                        self.get_logger().info("Global Plane Locked!")
                        
                        # === 修改点：计算直接从“机器人局部平面”到“裁剪图像”的单应性矩阵 ===
                        # 1. 在机器人局部坐标系 (Local Frame) 中取一个 100x100 的参考正方形
                        # 此时 (0,0) 就是机器人的物理基座，Y轴是脊柱方向
                        pts_local = np.array([
                            [0.0, 0.0],
                            [100.0, 0.0],
                            [100.0, 100.0],
                            [0.0, 100.0]
                        ], dtype=np.float32)
                        
                        # 2. 将局部坐标反算回全局 3D 坐标
                        # 公式推导: p_local = R @ (p_global - origin)  =>  p_global = R^T @ p_local + origin
                        pts_3d_global = []
                        for p_loc in pts_local:
                            p_glob_xy = local_R.T @ p_loc + local_origin
                            # 加上平面的虚拟 Z 轴高度
                            pts_3d_global.append([p_glob_xy[0], p_glob_xy[1], cfg.VIRTUAL_Z_OFFSET])
                        
                        pts_3d_global = np.array(pts_3d_global, dtype=np.float32)
                        
                        # 3. 将这四个点投影到全图相机的 2D 像素坐标
                        pts_2d_full, _ = cv2.projectPoints(
                            pts_3d_global, self.global_rvec, self.global_tvec, self.cam.new_mtx, None
                        )
                        
                        # 4. 减去剪切偏移 (rx, ry)，得到在纯净裁剪图 (roi_image) 上的坐标
                        pts_2d_cropped = pts_2d_full.reshape(-1, 2) - np.array([rx, ry])
                        
                        # 5. 直接建立 Local 物理平面 (X,Y) -> 裁剪图象像素 (u,v) 的单应性矩阵
                        self.cached_H, _ = cv2.findHomography(pts_local, pts_2d_cropped)
                        self.get_logger().info("Homography Matrix Initialized in LOCAL Frame!")
                        # ==========================================================
            else:
                self.calibration_start_time = None
                init_progress = 0.0

        # === 2. 颜色追踪 ===
        tracker_status, objects_2d, debug_dict = self.color_tracker.process(
            hsv, proc_frame, return_debug=True
        )

        # === 3. 核心计算 ===
        p1_mid_3d = None
        p5_mid_3d = None
        raw_3d_data = {}
        log_messages = [] 
        
        pub_ids = []
        pub_x = []
        pub_y = []
        pub_theta = []

        pixel_world_cache = {}

        if self.global_rvec is not None and objects_2d:
            # 3.1 像素 -> 3D
            for obj_id, data in objects_2d.items():
                mid_uv = data.get('midpoint_uv')
                dots = data.get('dots', [])
                color = cfg.ID_COLORS.get(obj_id, (255, 255, 255))
                
                if mid_uv is None: continue

                mid_uv_full = (int(mid_uv[0] + rx), int(mid_uv[1] + ry))
                cv2.circle(frame_vis, tuple(map(int, mid_uv_full)), 6, color, -1)
                cv2.circle(frame_vis, tuple(map(int, mid_uv_full)), 8, (255, 255, 255), 1)

                if len(dots) >= 3:
                    pts_full = [(int(p[0] + rx), int(p[1] + ry)) for p in dots]
                    pts_np = np.array(pts_full, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame_vis, [pts_np], True, color, 1, cv2.LINE_AA)

                mid_uv_global = (mid_uv[0] + rx, mid_uv[1] + ry)
                mid_key = (int(round(mid_uv_global[0])), int(round(mid_uv_global[1])))
                p_mid_3d = pixel_world_cache.get(mid_key)
                if p_mid_3d is None:
                    p_mid_3d = pixel_to_world_on_plane(mid_uv_global, self.cam.new_mtx, self.global_rvec, self.global_tvec, cfg.VIRTUAL_Z_OFFSET)
                    pixel_world_cache[mid_key] = p_mid_3d

                if p_mid_3d is not None:
                    raw_3d_data[obj_id] = {'midpoint_3d': p_mid_3d, 'dots': dots}
                    if obj_id == self.REF_ID: p1_mid_3d = p_mid_3d
                    if obj_id == self.TARGET_ID: p5_mid_3d = p_mid_3d

            # 3.2 坐标系更新
            local_R, local_origin = None, None
            if self.is_calibrating and p1_mid_3d is not None and p5_mid_3d is not None:
                local_R, local_origin = self.stabilizer.update(p1_mid_3d, p5_mid_3d)
            else:
                local_R, local_origin = self.stabilizer.get_current_transform()

            # 🚀【修改后】独立扫描并计算 25h9 障碍物 (脱离原有标定 Marker 的字典)
            if not self.is_calibrating and local_R is not None and local_origin is not None:
                # 使用专属的 25h9 字典独立寻找障碍物
                obs_corners, obs_ids, _ = cv2.aruco.detectMarkers(
                    frame_rect, self.obs_aruco_dict, parameters=self.obs_aruco_params
                )
                
                if obs_ids is not None and len(obs_ids) > 0:
                    target_indices = np.where(obs_ids.flatten() == self.OBS_MARKER_ID)[0]
                    
                    if len(target_indices) > 0:
                        idx = target_indices[0]
                        marker_corners = obs_corners[idx][0]
                        
                        # 物理尺寸不影响投影，直接提取像素角点求几何中心
                        obs_uv = np.mean(marker_corners, axis=0) 
                        obs_u, obs_v = int(obs_uv[0]), int(obs_uv[1])
                        
                        # 可视化标注中心点
                        cv2.circle(frame_vis, (obs_u, obs_v), 12, (0, 0, 255), -1)
                        cv2.circle(frame_vis, (obs_u, obs_v), 16, (255, 255, 255), 2)
                        cv2.putText(frame_vis, f"OBS(ID:{self.OBS_MARKER_ID})", (obs_u + 20, obs_v - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        # 像素转 3D (corners 坐标是基于全图 frame_rect，无需补偿 rx/ry)
                        obs_3d = pixel_to_world_on_plane(
                            (obs_uv[0], obs_uv[1]), self.cam.new_mtx, self.global_rvec, self.global_tvec, cfg.VIRTUAL_Z_OFFSET
                        )
                        
                        if obs_3d is not None:
                            # 将全局 3D 坐标转换到机器人的本地坐标系
                            obs_x, obs_y = apply_local_transform(obs_3d, local_R, local_origin)
                            
                            # 发送 ROS 消息
                            try:
                                msg = Point()
                                msg.x = float(obs_x)
                                msg.y = float(obs_y)
                                msg.z = 0.0 
                                self.obs_pub.publish(msg)
                            except AttributeError:
                                self.get_logger().warn("self.obs_pub is missing! Check __init__", once=True)
                            
                            # 将坐标追加到底部日志中
                            log_messages.append(f"OBS: xy({obs_x:>+7.1f}, {obs_y:>+7.1f})")

            # 3.3 结果输出 (机器人的各个关节状态)
            if local_R is not None and local_origin is not None:
                # 绘制原点十字
                z_h = p1_mid_3d[2] if p1_mid_3d is not None else 0.0
                origin_3d = np.array([[local_origin[0], local_origin[1], z_h]], dtype=np.float32)
                p_org_2d, _ = cv2.projectPoints(origin_3d, self.global_rvec, self.global_tvec, self.cam.new_mtx, None)
                if p_org_2d is not None:
                    p_org = tuple(np.int32(p_org_2d.flatten()))
                    cv2.drawMarker(frame_vis, p_org, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

                # 遍历物体
                for obj_id in sorted(raw_3d_data.keys()):
                    raw = raw_3d_data[obj_id]
                    mid_3d = raw['midpoint_3d']
                    dots = raw['dots']

                    final_x, final_y = apply_local_transform(mid_3d, local_R, local_origin)
                    
                    theta = 0.0
                    if len(dots) == 3:
                        dots_local = []
                        for dot_uv in dots:
                            dot_global = (dot_uv[0] + rx, dot_uv[1] + ry)
                            dot_key = (int(round(dot_global[0])), int(round(dot_global[1])))
                            d_world = pixel_world_cache.get(dot_key)
                            if d_world is None:
                                d_world = pixel_to_world_on_plane(dot_global, self.cam.new_mtx, self.global_rvec, self.global_tvec, cfg.VIRTUAL_Z_OFFSET)
                                pixel_world_cache[dot_key] = d_world
                            dx, dy = apply_local_transform(d_world, local_R, local_origin)
                            dots_local.append(np.array([dx, dy]))
                        theta = calculate_local_orientation(dots_local)

                    pub_ids.append(obj_id)
                    pub_x.append(final_x)
                    pub_y.append(final_y)
                    pub_theta.append(theta)

                    log_str = f"ID{obj_id}: xy({final_x:>+7.1f}, {final_y:>+7.1f}) th:{theta:>+6.1f}"
                    log_messages.append(log_str)

        frame_vis = frame_vis[ry:ry+rh, rx:rx+rw]

        # === UI 绘制 ===
        draw_status_overlay(frame_vis, self.is_calibrating, init_progress)
        
        if log_messages:
            full_log = " | ".join(log_messages)
            h, w = frame_vis.shape[:2]
            bar_height = 35 
            bottom_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
            cv2.putText(bottom_bar, full_log, (15, 22), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            frame_vis = np.vstack([frame_vis, bottom_bar])
            
        # === 4. 发布消息 ===
        self.ros_pub.publish_status(pub_ids, pub_x, pub_y, pub_theta, not self.is_calibrating)
        
        if self.enable_stream:
            if curr_time - self.last_img_pub_time > self.IMG_PUB_INTERVAL:
                grid_img = draw_debug_grid(debug_dict)
                self.ros_pub.publish_debug_image(grid_img)
                self.last_img_pub_time = curr_time

        # === 5. 本地窗口 ===
        if self.show_window:
            cv2.imshow("Vision Local", frame_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.color_tracker.reset(); self.stabilizer.reset()
                self.is_calibrating = True 
                self.calibration_start_time = None
                self.get_logger().info("System Reset.")
    
    def __del__(self):
        # 保留空实现以防对象被回收时调用，但主要使用 shutdown()
        try:
            self.shutdown()
        except Exception:
            pass

    def shutdown(self):
        if hasattr(self, 'cap') and getattr(self, 'cap') is not None and self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.shutdown()
        except Exception:
            pass
        node.destroy_node()

if __name__ == '__main__':
    main()