# File: robot_brain/core/config.py
import torch
import os

class BrainConfig:
    # ================= 📂 路径配置 =================
    # 根目录 (自动定位到 user home 下的 lifelong_data)
    DATA_ROOT = os.path.expanduser("~/brandon/hyrd_robot/lifelong_data")
    MODEL_DIR = os.path.expanduser("~/brandon/hyrd_robot/lifelong_data/models")
    
    # 模型文件名
    MODEL_LATEST_NAME = "gnk_physics_latest.pth"
    MODEL_BEST_NAME = "gnk_physics_best_deploy.pth" # 仅权重，用于推理

    # ================= 🤖 物理执行配置 (新增) =================
    # 单次 Loop 采样的轨迹点数
    # 减少此数值可以加快 Loop 循环速度，增加模型更新频率
    TRAJECTORY_POINTS = 120 # 建议范围: 50 ~ 500
    
    # ================= 🧠 训练超参数 (基准值) =================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 经验池
    MEMORY_WINDOW_SIZE = 10     # 建议调大一点，配合自动调参，让模型看多一点历史
    
    # --- 自动调参的“锚点” (当数据量很大时的理想参数) ---
    BATCH_SIZE = 4096           # 最大 Batch Size (上限)
    MAX_EPOCHS_FINE_TUNE = 100   
    MAX_EPOCHS_INIT = 100       
    
    LR_MAX = 1e-3               # 基准学习率 (对应 Batch=4096 时)
    LR_MIN = 1e-6

    WARMUP_EPOCHS = 5          # 学习率预热周期
    
    # 正则化基准 (对应大数据量时，约 1e-4 ~ 1e-3)
    # 初期我们会自动把它放大 10~100 倍
    WEIGHT_DECAY = 1e-4         
    
    PATIENCE = 10               # 早停耐心值
    
    # ================= 🛑 智能跳过 (Pre-Inference) =================
    # 如果新数据的 Zero-shot Loss 低于此值 (单位: mm 归一化后的值)，跳过训练
    # 建议值: 0.05 ~ 0.1 (取决于你的归一化尺度)
    PRE_INF_LOSS_THRESHOLD = 0.02 
    
    # 物理门控预设
    # Alpha > 0.8 表示物理模型很准，此时如果 Loss 也很低，更应该跳过
    PRE_INF_ALPHA_THRESHOLD = 0.85

    @classmethod
    def get_model_path(cls):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        return os.path.join(cls.MODEL_DIR, cls.MODEL_LATEST_NAME)