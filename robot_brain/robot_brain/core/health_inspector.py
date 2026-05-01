# File: robot_brain/core/health_inspector.py
import torch
import torch.nn as nn
import numpy as np
import os
from colorama import Fore, Style

from robot_brain.core.config import BrainConfig as CFG
from robot_brain.core.trainer import PhysicsGatedGNK

class HealthInspector:
    def __init__(self, reference_model_path, device=None):
        """
        初始化健康监测器
        :param reference_model_path: 离线训练好的"金标准"模型路径
        """
        self.device = device if device else CFG.DEVICE
        self.model_path = reference_model_path
        self.model = None
        self.offline_scalers = None
        
        # === 🚨 阈值配置 ===
        self.THRESHOLD_CRITICAL = 15.0  # mm, 超过此值视为严重损坏
        self.THRESHOLD_WARNING = 8.0    # mm, 超过此值视为潜在风险
        
        self._load_reference_model()

    def _load_reference_model(self):
        if not os.path.exists(self.model_path):
            print(f"{Fore.RED}❌ [Health] Reference model not found: {self.model_path}{Style.RESET_ALL}")
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 1. 必须加载离线模型的 Scalers
            if 'scalers' not in checkpoint:
                raise ValueError("Checkpoint missing 'scalers'.")
            self.offline_scalers = checkpoint['scalers']

            # 2. 初始化模型 (XY-Only: 20 -> 10)
            self.model = PhysicsGatedGNK(
                state_dim=20,
                action_dim=10,
                output_dim=10,
                scalers=self.offline_scalers
            ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"{Fore.CYAN}🩺 [Health] Inspector Ready. Baseline: {os.path.basename(self.model_path)}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ [Health] Init failed: {e}{Style.RESET_ALL}")
            self.model = None

    def analyze_batch(self, batch_path):
        """
        对单个 Batch 进行健康诊断
        :return: dict 包含诊断结果 (如果失败返回 None)
        """
        if self.model is None:
            return None

        pt_path = os.path.join(batch_path, "train_data.pt")
        if not os.path.exists(pt_path):
            return {"status": "SKIPPED", "reason": "No data"}

        try:
            # 加载新数据
            data = torch.load(pt_path, map_location=self.device)
            
            s_n = data['inputs_state'].to(self.device)
            a_n = data['inputs_action'].to(self.device)
            a_p = data['inputs_action_phys'].to(self.device)
            j_p = data['j_phys'].to(self.device)
            tgt_norm = data['targets'].to(self.device)

            criterion = nn.MSELoss()

            with torch.no_grad():
                # 预测
                pred_norm, alpha = self.model(s_n, a_n, a_p, j_p)
                
                # 1. 计算 MSE
                loss_mse = criterion(pred_norm, tgt_norm).item()
                
                # 2. 计算物理误差 (mm)
                tgt_mean = self.offline_scalers['target_mean'].to(self.device)
                tgt_std = self.offline_scalers['target_std'].to(self.device)

                pred_real = pred_norm * tgt_std + tgt_mean
                tgt_real = tgt_norm * tgt_std + tgt_mean
                
                # 计算末端误差 (最后两维 XY)
                diff = pred_real - tgt_real
                tip_diff = diff[:, -2:] 
                tip_error_mm = torch.norm(tip_diff, dim=1).mean().item()
                avg_alpha = alpha.mean().item()

            # === 判定状态 ===
            status_code = "HEALTHY"
            is_healthy = True
            
            if tip_error_mm > self.THRESHOLD_CRITICAL:
                status_code = "CRITICAL"
                is_healthy = False
            elif tip_error_mm > self.THRESHOLD_WARNING:
                status_code = "WARNING"
                is_healthy = True # 警告但不中断

            return {
                "status": status_code,
                "is_healthy": is_healthy,
                "tip_error_mm": tip_error_mm,
                "mse": loss_mse,
                "alpha": avg_alpha,
                "samples": s_n.shape[0]
            }

        except Exception as e:
            print(f"{Fore.RED}❌ [Health] Analysis failed for {os.path.basename(batch_path)}: {e}{Style.RESET_ALL}")
            return None

# 简单的测试入口
if __name__ == "__main__":
    # 测试代码
    path = "/home/brandon/brandon/hyrd_robot/lifelong_data/models/gnk_offline_big_best_001.pth"
    inspector = HealthInspector(path)
    # 模拟测试
    # res = inspector.analyze_batch("...")
    # print(res)