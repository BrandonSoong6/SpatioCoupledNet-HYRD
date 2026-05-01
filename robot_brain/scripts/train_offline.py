#!/usr/bin/env python3
# File: scripts/train_offline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import numpy as np
import os
import glob
import random
from tqdm import tqdm
from colorama import Fore, Style, init

from robot_brain.core.BigTrain_config import BigTrainConfig as CFG
from robot_brain.core.trainer import PhysicsGatedGNK

init(autoreset=True)

def load_dataset_from_dirs(dir_list, desc="Loading"):
    datasets = []
    for b_dir in tqdm(dir_list, desc=desc, leave=False):
        pt_path = os.path.join(b_dir, "train_data.pt")
        if os.path.exists(pt_path):
            try:
                d = torch.load(pt_path, map_location='cpu')
                ds = TensorDataset(
                    d['inputs_state'], d['inputs_action'], 
                    d['inputs_action_phys'], d['j_phys'], d['targets']
                )
                datasets.append(ds)
            except Exception as e:
                print(f"{Fore.RED}⚠️ Error loading {b_dir}: {e}{Style.RESET_ALL}")
    
    if not datasets: return None, 0
    full_ds = ConcatDataset(datasets)
    return full_ds, len(full_ds)

def prepare_data():
    print(f"{Fore.CYAN}🔍 Scanning for batch data in {CFG.DATA_ROOT}...{Style.RESET_ALL}")
    all_batch_dirs = sorted(glob.glob(os.path.join(CFG.DATA_ROOT, "batch_*")))
    
    if not all_batch_dirs:
        print(f"{Fore.RED}❌ No data found!{Style.RESET_ALL}")
        return None, None, None

    latest_dir = all_batch_dirs[-1]
    try:
        latest_pt = torch.load(os.path.join(latest_dir, "train_data.pt"), map_location='cpu')
        scalers = latest_pt['scalers']
    except:
        print(f"{Fore.RED}❌ Failed to load scalers from {latest_dir}{Style.RESET_ALL}")
        return None, None, None

    random.seed(42) 
    shuffled_dirs = all_batch_dirs.copy()
    random.shuffle(shuffled_dirs)
    
    split_idx = int(len(shuffled_dirs) * (1 - CFG.VAL_SPLIT))
    train_dirs = shuffled_dirs[:split_idx]
    val_dirs = shuffled_dirs[split_idx:]
    
    print(f"📂 Found {len(all_batch_dirs)} batches.")
    print(f"   Train Batches: {len(train_dirs)}")
    print(f"   Val Batches:   {len(val_dirs)}")

    train_ds, n_train = load_dataset_from_dirs(train_dirs, desc="Loading Train Set")
    val_ds, n_val = load_dataset_from_dirs(val_dirs, desc="Loading Val Set")
    
    print(f"{Fore.GREEN}📚 Data Ready. Train Samples: {n_train} | Val Samples: {n_val}{Style.RESET_ALL}")
    return train_ds, val_ds, scalers

def train_offline():
    print(f"{Fore.MAGENTA}{Style.BRIGHT}=== 🚀 BigTrain: Anti-Overfit Mode ==={Style.RESET_ALL}")
    
    # === 🛡️ 配置自检：确认猛药是否生效 ===
    print(f"{Fore.YELLOW}🔧 Active Config:{Style.RESET_ALL}")
    print(f"   • LR Max:       {CFG.LR_MAX}")
    print(f"   • Weight Decay: {CFG.WEIGHT_DECAY} (High Regularization)")
    print(f"   • Batch Size:   {CFG.BATCH_SIZE}")
    print(f"   • Dropout:      20% (Hardcoded in Trainer)")
    print("-" * 50)
    
    train_ds, val_ds, scalers = prepare_data()
    if train_ds is None or val_ds is None: return

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    device = CFG.DEVICE
    model = PhysicsGatedGNK(scalers=scalers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR_MAX, weight_decay=CFG.WEIGHT_DECAY)
    
    tgt_mean = scalers['target_mean'].to(device)
    tgt_std = scalers['target_std'].to(device)

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
    
    print(f"{'Ep':<4} | {'Train':<7} | {'Val':<7} | {'TipXY(mm)':<9} | {'TipSC':<7} | {'Alpha':<5} | {'LR':<8} | {'Pat':<6} | {'Status'}")
    print("-" * 95)

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
        tip_xy_error_sum = 0.0
        tip_sc_error_sum = 0.0
        
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
                
                batch_tip_xy_err = abs_diff[:, -4:-2].mean().item()
                batch_tip_sc_err = abs_diff[:, -2:].mean().item()
                
                tip_xy_error_sum += batch_tip_xy_err
                tip_sc_error_sum += batch_tip_sc_err
        
        avg_val = val_loss / len(val_loader)
        avg_alpha = alpha_sum / len(val_loader)
        avg_tip_xy = tip_xy_error_sum / len(val_loader)
        avg_tip_sc = tip_sc_error_sum / len(val_loader)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
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
                'config': {'batch_size': CFG.BATCH_SIZE, 'lr': CFG.LR_MAX}
            }, save_path)
        else:
            patience_counter += 1

        status_str = f"{Fore.GREEN}▼ BEST{Style.RESET_ALL}" if is_best else f"{Fore.YELLOW}▲ +1{Style.RESET_ALL}"
        patience_str = f"{patience_counter}/{CFG.PATIENCE}"
        if patience_counter > CFG.PATIENCE * 0.8: patience_str = f"{Fore.RED}{patience_str}{Style.RESET_ALL}"

        log_msg = (f"{epoch+1:<4} | "
                   f"{avg_train:.5f} | "
                   f"{avg_val:.5f} | "
                   f"{Fore.CYAN}{avg_tip_xy:.2f}mm{Style.RESET_ALL}   | "
                   f"{avg_tip_sc:.3f}   | "
                   f"{avg_alpha:.3f} | "
                   f"{current_lr:.1e} | "
                   f"{patience_str:<6} | "
                   f"{status_str}")
        
        pbar.write(log_msg)
            
        if patience_counter >= CFG.PATIENCE:
            pbar.write(f"\n{Fore.RED}🛑 Early stopping triggered.{Style.RESET_ALL}")
            break
            
    print(f"\n{Fore.GREEN}✅ Training Complete!{Style.RESET_ALL}")
    print(f"🏆 Best Val Loss: {best_loss:.6f}")
    print(f"🎯 Final Tip XY Error: {avg_tip_xy:.4f} mm")
    print(f"💾 Model Saved to: {save_path}")

if __name__ == "__main__":
    try:
        train_offline()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted.")