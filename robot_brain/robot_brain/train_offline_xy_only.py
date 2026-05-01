#!/usr/bin/env python3
# File: scripts/train_offline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import csv  # <--- 新增: 用于保存日志
from tqdm import tqdm
from colorama import Fore, Style, init

from robot_brain.core.BigTrain_config import BigTrainConfig as CFG
from robot_brain.core.trainer import PhysicsGatedGNK

init(autoreset=True)

def load_mega_dataset():
    """直接加载聚合好的 Mega Dataset"""
    mega_path = os.path.join(CFG.DATA_ROOT, "mega_xy_dataset.pt")
    print(f"{Fore.CYAN}🔍 Loading Mega Dataset from: {mega_path}{Style.RESET_ALL}")
    
    if not os.path.exists(mega_path):
        print(f"{Fore.RED}❌ Dataset not found! Run generate_xy_dataset.py first.{Style.RESET_ALL}")
        return None, None
        
    data = torch.load(mega_path, map_location='cpu')
    
    if 'meta' in data:
        print(f"   ℹ️  Info: {data['meta']}")
    
    ds = TensorDataset(
        data['inputs_state'], 
        data['inputs_action'], 
        data['inputs_action_phys'], 
        data['j_phys'], 
        data['targets']
    )
    
    return ds, data['scalers']

def train_offline():
    print(f"{Fore.MAGENTA}{Style.BRIGHT}=== 🚀 BigTrain: XY-Only Mode (with CSV Logging) ==={Style.RESET_ALL}")
    
    # 1. 加载数据
    full_dataset, scalers = load_mega_dataset()
    if full_dataset is None: return
    
    total_samples = len(full_dataset)
    print(f"{Fore.GREEN}📚 Loaded {total_samples} samples.{Style.RESET_ALL}")

    # 2. 划分训练/验证集
    val_size = int(total_samples * CFG.VAL_SPLIT)
    train_size = total_samples - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    
    print(f"   Train Samples: {train_size}")
    print(f"   Val Samples:   {val_size}")
    
    # 3. Loader
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. 模型初始化 (XY-Only Dimensions)
    device = CFG.DEVICE
    model = PhysicsGatedGNK(
        state_dim=20,    # 10q + 10xy
        action_dim=10,   # 10dq
        output_dim=10,   # 10dx (XY Only)
        scalers=scalers
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR_MAX, weight_decay=CFG.WEIGHT_DECAY)
    
    # 5. 辅助变量
    tgt_mean = scalers['target_mean'].to(device)
    tgt_std = scalers['target_std'].to(device)
    
    # 自动侦测单位
    avg_std = tgt_std.mean().item()
    unit_scale = 1.0
    unit_name = "mm"
    if avg_std < 0.1: 
        unit_scale = 1000.0
        unit_name = "mm"
        print(f"   👉 Detected Unit: Meters. Will display in mm (x1000).")
    
    # === 📝 CSV 日志初始化 ===
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)
    log_path = os.path.join(CFG.MODEL_DIR, "train_log.csv")
    print(f"{Fore.YELLOW}📝 Logging training history to: {log_path}{Style.RESET_ALL}")
    
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    # 写入表头
    log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'tip_x_mm', 'tip_y_mm', 'alpha', 'lr'])
    # ========================

    # 6. 训练循环
    warmup_ep = CFG.WARMUP_EPOCHS
    total_ep = CFG.MAX_EPOCHS
    scheduler1 = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_ep)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=total_ep - warmup_ep, eta_min=CFG.LR_MIN)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_ep])
    
    criterion = nn.MSELoss()
    best_loss = float('inf')
    patience_counter = 0
    save_path = CFG.get_model_path()
    
    pbar = tqdm(range(total_ep), desc="🔥 Running", unit="ep")
    
    print(f"{'Ep':<4} | {'Train':<7} | {'Val':<7} | {'Tip-X':<9} | {'Tip-Y':<9} | {'Alpha':<5} | {'LR':<8} | {'Status'}")
    print("-" * 90)

    try:
        for epoch in pbar:
            # --- Train ---
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                s_n, a_n, a_p, j_p, tgt = [b.to(device) for b in batch]
                optimizer.zero_grad()
                pred, alpha = model(s_n, a_n, a_p, j_p)
                loss = criterion(pred, tgt)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train = train_loss / len(train_loader)
            
            # --- Val ---
            model.eval()
            val_loss = 0.0
            alpha_sum = 0.0
            tip_x_err_sum = 0.0
            tip_y_err_sum = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    s_n, a_n, a_p, j_p, tgt = [b.to(device) for b in batch]
                    pred, alpha = model(s_n, a_n, a_p, j_p)
                    val_loss += criterion(pred, tgt).item()
                    alpha_sum += alpha.mean().item()
                    
                    # 物理误差计算
                    pred_real = pred * tgt_std + tgt_mean
                    tgt_real = tgt * tgt_std + tgt_mean
                    abs_diff = torch.abs(pred_real - tgt_real)
                    
                    batch_tip_x_err = abs_diff[:, -2].mean().item() * unit_scale
                    batch_tip_y_err = abs_diff[:, -1].mean().item() * unit_scale
                    
                    tip_x_err_sum += batch_tip_x_err
                    tip_y_err_sum += batch_tip_y_err
            
            avg_val = val_loss / len(val_loader)
            avg_alpha = alpha_sum / len(val_loader)
            avg_tip_x = tip_x_err_sum / len(val_loader)
            avg_tip_y = tip_y_err_sum / len(val_loader)
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # --- 保存最佳模型 ---
            is_best = False
            if avg_val < best_loss:
                best_loss = avg_val
                patience_counter = 0
                is_best = True
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scalers': scalers,
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'config': {'batch_size': CFG.BATCH_SIZE, 'lr': CFG.LR_MAX, 'xy_only': True}
                }, save_path)
            else:
                patience_counter += 1

            status_str = f"{Fore.GREEN}▼ BEST{Style.RESET_ALL}" if is_best else f"{Fore.YELLOW}▲ +1{Style.RESET_ALL}"
            patience_str = f"{patience_counter}/{CFG.PATIENCE}"
            if patience_counter > CFG.PATIENCE * 0.8: patience_str = f"{Fore.RED}{patience_str}{Style.RESET_ALL}"
            
            # === 📝 写入 CSV ===
            log_writer.writerow([epoch+1, avg_train, avg_val, avg_tip_x, avg_tip_y, avg_alpha, current_lr])
            log_file.flush() # 强制刷新，防止中断丢失数据
            # ==================

            log_msg = (f"{epoch+1:<4} | "
                       f"{avg_train:.5f} | "
                       f"{avg_val:.5f} | "
                       f"{avg_tip_x:.2f}{unit_name:<3}  | "
                       f"{avg_tip_y:.2f}{unit_name:<3}  | "
                       f"{avg_alpha:.3f} | "
                       f"{current_lr:.1e} | "
                       f"{status_str}")
            
            pbar.write(log_msg)
                
            if patience_counter >= CFG.PATIENCE:
                pbar.write(f"\n{Fore.RED}🛑 Early stopping triggered.{Style.RESET_ALL}")
                break

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        log_file.close() # 确保文件关闭
        print(f"📝 Log saved to: {log_path}")

    print(f"\n{Fore.GREEN}✅ XY-Only Training Complete!{Style.RESET_ALL}")
    print(f"🏆 Best Val Loss: {best_loss:.6f}")
    print(f"🎯 Final Tip X Error: {avg_tip_x:.3f} {unit_name}")
    print(f"🎯 Final Tip Y Error: {avg_tip_y:.3f} {unit_name}")

if __name__ == "__main__":
    train_offline()