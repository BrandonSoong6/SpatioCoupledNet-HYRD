# File: robot_brain/core/BigTrain_config.py
import torch
import os

class BigTrainConfig:
    # 路径配置
    DATA_ROOT = os.path.expanduser("~/brandon/hyrd_robot/lifelong_data")
    MODEL_DIR = os.path.expanduser("~/brandon/hyrd_robot/lifelong_data/models")
    MODEL_NAME = "gnk_offline_big.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 验证集比例
    VAL_SPLIT = 0.2
    
    # === 🔧 抗过拟合核弹级配置 ===
    BATCH_SIZE = 4096           
    MAX_EPOCHS = 2000           # 跑久一点，因为学得慢了
    WARMUP_EPOCHS = 50          # 预热时间拉长
    
    # 1. 学习率：降维打击，不求快，求稳
    LR_MAX = 1e-4               # (原 3e-4 -> 1e-4)
    LR_MIN = 1e-7
    
    # 2. 正则化：强力惩罚，迫使网络只学通用规律
    WEIGHT_DECAY = 0.02         # (原 1e-3 -> 放大 20 倍)
    
    PATIENCE = 100              # 给它更多耐心去磨
    
    @classmethod
    def get_model_path(cls):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        return os.path.join(cls.MODEL_DIR, cls.MODEL_NAME)