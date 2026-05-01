import cv2
import numpy as np
import os
import sys
from ament_index_python.packages import get_package_share_directory

# 使用相对导入，假设 camera.py 位于 robot_control.drivers 下
try:
    from .. import config as cfg
except (ImportError, ValueError):
    # 备用导入逻辑
    import robot_control.config as cfg

class Camera:
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.new_mtx = None
        self.map1 = None
        self.map2 = None
        
        # 获取包的安装路径
        self.package_share_dir = get_package_share_directory('robot_control')
        
        self._load_params()
        self._init_undistort_maps()

    def _load_params(self):
        # 修正：从 share/robot_control/data/ 目录下读取
        param_path = os.path.join(self.package_share_dir, 'data', 'jerry_cam_params_1080p_2.npz')
        
        if os.path.exists(param_path):
            try:
                data = np.load(param_path)
                self.mtx = data['mtx'].astype(np.float32)
                self.dist = data['dist'].astype(np.float32)
                # 移除 print，改用 ROS 2 日志（如果需要可传入 node 对象）
                print(f"✅ Camera Params Loaded from: {param_path}")
            except Exception as e:
                print(f"❌ Error loading param file: {e}")
                sys.exit(1)
        else:
            print(f"❌ Error: {param_path} missing! Check if 'data' is in setup.py")
            sys.exit(1)

    def _init_undistort_maps(self):
        # 假设 cfg 中定义了分辨率，否则可从 self.mtx 尺寸推导
        self.new_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, 
            (cfg.WIDTH, cfg.HEIGHT), 
            0, 
            (cfg.WIDTH, cfg.HEIGHT)
        )
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.mtx, self.dist, 
            None, 
            self.new_mtx, 
            (cfg.WIDTH, cfg.HEIGHT), 
            cv2.CV_32FC1
        )

    def rectify(self, frame):
        """对图像进行去畸变校正"""
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)