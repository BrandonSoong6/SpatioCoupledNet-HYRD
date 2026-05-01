import cv2
import numpy as np
import robot_control.config as cfg

class RedTracker:
    def __init__(self):
        # 预创建调试窗口，设置一个合理的初始大小
        self.debug_window_name = "Black Tracker Debug (L:Mask, R:Result)"
        cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.debug_window_name, 960, 360) # 宽度设大点容纳两张图

    def process(self, hsv_frame, bgr_frame):
        """
        输入: hsv_frame (裁剪后的区域HSV), bgr_frame (裁剪后的区域BGR原图)
        输出: 最佳中心点坐标 (相对于裁剪区域)
        功能: 在一个独立窗口中同时显示掩膜和识别结果
        """
        # ===========================
        # 1. 图像处理核心逻辑
        # ===========================
        # 使用配置的黑色阈值提取掩膜
        # ⚠️注意：如果画面全黑，尝试调大 cfg.BLACK_UPPER 的第三个值(V)到 60 或 80
        black_mask = cv2.inRange(hsv_frame, cfg.BLACK_LOWER, cfg.BLACK_UPPER)

        # 形态学处理：开运算去噪，闭运算填洞
        kernel = np.ones((5, 5), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

        # 寻找轮廓
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_center = None
        max_area = 0
        result_frame = bgr_frame.copy() # 拷贝一份用于画图，不影响原数据

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > cfg.RED_MIN_AREA:
                # 在结果图上画出所有符合面积要求的轮廓（绿色）
                cv2.drawContours(result_frame, [cnt], -1, (0, 255, 0), 1)
                
                if area > max_area:
                    max_area = area
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        best_center = (cx, cy)

        # 如果找到了最佳目标，画出中心点和文字（红色）
        if best_center is not None:
            cv2.circle(result_frame, best_center, 8, (0, 0, 255), -1)
            # 画个十字准星更清晰
            cv2.drawMarker(result_frame, best_center, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(result_frame, f"Target: {best_center}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ===========================
        # 2. 可视化拼接逻辑
        # ===========================
        # 将单通道掩膜转为三通道 BGR，以便拼接
        mask_bgr = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR)
        
        # 在掩膜图上加个标题
        cv2.putText(mask_bgr, "Segmentation Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # 水平拼接 (Horizontal Stack): 左边是掩膜，右边是结果
        combined_view = np.hstack([mask_bgr, result_frame])
        
        # 显示拼好的图
        cv2.imshow(self.debug_window_name, combined_view)
        
        # 🔥 关键：因为不能改 Vision Node，必须在这里加 waitKey 才能让这个窗口刷新
        cv2.waitKey(1) 

        return best_center