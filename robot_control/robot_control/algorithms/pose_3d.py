import cv2
import numpy as np
import robot_control.config as cfg # 确保路径引用正确

class GlobalPlaneEstimator:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist # 传入 None 即可，如果图像已去畸变
        
        # === 兼容性处理开始 ===
        try:
            # 尝试新版 API (OpenCV 4.7.0+)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
            self.aruco_params = cv2.aruco.DetectorParameters()
            # 新版需要构造检测器对象
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.is_new_api = True
        except (AttributeError, TypeError):
            # 回退到旧版 API (OpenCV < 4.7.0)
            # 注意：旧版部分版本可能需要 Dictionary_get
            try:
                self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_16h5)
            except AttributeError:
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
                
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.is_new_api = False
        # === 兼容性处理结束 ===

        self.board_model = self._generate_board_model()

    def _generate_board_model(self):
        obj_points = {}
        for tag_id in range(cfg.TOTAL_TAGS):
            r = tag_id // cfg.GRID_COLS
            c = tag_id % cfg.GRID_COLS
            base_x = cfg.OFFSET_X + c * cfg.PITCH_X
            base_y = cfg.OFFSET_Y + r * cfg.PITCH_Y
            
            # Z=0 (Apriltag 平面)
            p_tl = [base_x,            base_y,            0.0]
            p_tr = [base_x + cfg.TAG_SIZE, base_y,            0.0]
            p_br = [base_x + cfg.TAG_SIZE, base_y + cfg.TAG_SIZE, 0.0]
            p_bl = [base_x,            base_y + cfg.TAG_SIZE, 0.0]
            
            obj_points[tag_id] = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float32)
        return obj_points

    def get_plane_pose(self, rect_frame):
        """
        返回: (success, rvec, tvec, corners, ids)
        """
        # === 根据 API 版本选择检测方式 ===
        if self.is_new_api:
            corners, ids, _ = self.detector.detectMarkers(rect_frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                rect_frame, self.aruco_dict, parameters=self.aruco_params
            )
        
        if ids is None:
            return False, None, None, None, None

        ids = ids.flatten()
        all_3d = []
        all_2d = []
        
        for i, tag_id in enumerate(ids):
            if tag_id in self.board_model:
                all_3d.append(self.board_model[tag_id])
                all_2d.append(corners[i][0])

        if len(all_3d) < 1: 
            return False, None, None, corners, ids 
            
        object_pts_flat = np.vstack(all_3d)
        image_pts_flat = np.vstack(all_2d)

        # 这里假设输入的是去畸变后的图，所以 dist=None
        success, rvec, tvec = cv2.solvePnP(
            object_pts_flat, image_pts_flat, self.mtx, None, flags=cv2.SOLVEPNP_SQPNP
        )
        
        return success, rvec, tvec, corners, ids