import numpy as np
import os
# ==========================================
# 1. 硬件与路径 (Hardware & Paths)
# ==========================================
DEV_PATH = "/dev/jerry_cam"
PARAM_FILE = "/home/brandon/brandon/hyrd_robot/src/robot_control/robot_control/jerry_cam_params_1080p_2.npz"
WIDTH = 1920
HEIGHT = 1080
# 获取当前文件的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 定义标定文件路径
CALIB_FILE = os.path.join(BASE_DIR, "jerry_cam_params_1080p_2.npz")

# ==========================================
# 2. 颜色阈值 (Color Thresholds)
# ==========================================
# 青色 ROI (Cyan)
CYAN_LOWER = np.array([70, 40, 160])   
CYAN_UPPER = np.array([120, 255, 255]) 

# 黄色点 (Yellow Dots)
YELLOW_LOWER = np.array([15, 30, 150]) 
YELLOW_UPPER = np.array([45, 255, 255])

# 红色目标 (Red Target - 需要两段 HSV)
# RED_LOWER_1 = np.array([0, 100, 100])
# RED_UPPER_1 = np.array([10, 255, 255])
# RED_LOWER_2 = np.array([160, 100, 100])
# RED_UPPER_2 = np.array([180, 255, 255])

# 第一段：色调 0-90，亮度低于 60 (根据环境光线，如果太暗识别不到，可调大到 80)
RED_LOWER_1 = np.array([0, 0, 0])
RED_UPPER_1 = np.array([99, 255, 60]) 

# 第二段：色调 91-180，亮度保持一致
RED_LOWER_2 = np.array([100, 0, 0])
RED_UPPER_2 = np.array([255, 255, 60])

# 红色椭圆面积最小阈值 (根据你说的"大的红色椭圆"调整)
RED_MIN_AREA = 100

BLACK_LOWER = np.array([0, 0, 0])
BLACK_UPPER = np.array([179, 255, 40]) 

# 既然你要看大椭圆，面积阈值建议设大
RED_MIN_AREA = 800

# ==========================================
# 3. 2D 识别参数 (Tracker Params)
# ==========================================
ELLIPSE_MIN_AREA = 100     
ELLIPSE_MAX_AREA = 700   
ELLIPSE_MIN_RATIO = 0.4  

ROI_MIN_AREA = 700        
ROI_MAX_AREA = 2000      
ROI_MIN_RATIO = 2.5       
ROI_MAX_RATIO = 5.0       
ROI_MIN_SOLIDITY = 0.45    

EXPAND_LONG_RATIO = 1.7  
EXPAND_SHORT_RATIO = 1.2 
ROI_EXPAND_FIXED = 10.0   

MAX_LOST_FRAMES = 5
MAX_MERGE_DIST = 50

# ==========================================
# 4. 3D 几何与平面参数 (Geometry)
# ==========================================
TAG_SIZE = 45.0
GAP_X = 15.0
GAP_Y = 15.0
PITCH_X = TAG_SIZE + GAP_X
PITCH_Y = TAG_SIZE + GAP_Y

GRID_COLS = 3
GRID_ROWS = 4
TOTAL_TAGS = GRID_COLS * GRID_ROWS

OFFSET_X = 522.5
OFFSET_Y = 36.0

VIRTUAL_Z_OFFSET = -54.2

# ==========================================
# 5. 可视化配置 (Visualization)
# ==========================================
# BGR 格式：避免使用 Cyan (255, 255, 0) 和 Yellow (0, 255, 255)
ID_COLORS = {
    1: (0, 0, 255),      # 红 Red
    2: (0, 255, 0),      # 绿 Green
    3: (255, 0, 0),      # 蓝 Blue
    4: (255, 0, 255),    # 紫 Magenta
    5: (0, 140, 255)     # 橙 Orange
}

# 接触点样式
CONTACT_CIRCLE_RADIUS_MM = 8.0  # 在 3D 平面上画一个半径 8mm 的圆
CONTACT_COLOR = (10, 20, 0)  

# 图注 (Legend)
LEGEND_X = 1650   # 右上角起始 X
LEGEND_Y = 50     # 起始 Y
LEGEND_GAP = 30   # 行间距