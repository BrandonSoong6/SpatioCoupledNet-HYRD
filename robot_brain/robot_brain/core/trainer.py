# File: robot_brain/core/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
# [新增] 引入调度器组合工具
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import numpy as np
import os
import math 
from tqdm import tqdm
from colorama import Fore, Style, init

from robot_brain.core.config import BrainConfig as CFG

init(autoreset=True)

# ... (ResidualBlock 和 PhysicsGatedGNK 保持不变) ...
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
        )
    def forward(self, x): return x + self.block(x)

class PhysicsGatedGNK(nn.Module):
    def __init__(self, state_dim=30, action_dim=10, output_dim=20, hidden_dim=512, scalers=None):
        super().__init__()
        if scalers is None: raise ValueError("❌ Scalers required!")
        
        self.register_buffer('target_mean', scalers['target_mean'])
        self.register_buffer('target_std', scalers['target_std'])
        
        # === 🔧 关键修改: 手动指定 Dropout=0.2 (20%) ===
        # 这会随机切断 20% 的神经连接，强迫网络生成鲁棒特征
        self.residual_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.SiLU(),
            ResidualBlock(hidden_dim, dropout=0.2), # 🛡️ Anti-Overfit
            ResidualBlock(hidden_dim, dropout=0.2), # 🛡️ Anti-Overfit
            ResidualBlock(hidden_dim, dropout=0.2), # 🛡️ Anti-Overfit
            ResidualBlock(hidden_dim, dropout=0.2), # 🛡️ Anti-Overfit
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 1), nn.Sigmoid() 
        )
        nn.init.constant_(self.gate_net[-2].bias, 2.0) 
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

    def forward(self, state_norm, action_norm, action_phys, j_phys):
        dx_phys_raw = torch.bmm(j_phys, action_phys.unsqueeze(-1)).squeeze(-1)
        dx_phys_norm = (dx_phys_raw - self.target_mean) / (self.target_std + 1e-7)
        net_in = torch.cat([state_norm, action_norm], dim=1)
        dx_residual_norm = self.residual_net(net_in)
        alpha = self.gate_net(state_norm)
        dx_final = dx_phys_norm + (1.0 - alpha) * dx_residual_norm
        return dx_final, alpha

# ... (LifelongTrainer 类) ...
class LifelongTrainer:
    def __init__(self):
        self.device = CFG.DEVICE
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
    def _load_pt_file(self, path):
        if not os.path.exists(path): raise FileNotFoundError(f"Data not found: {path}")
        d = torch.load(path, map_location='cpu') 
        dataset = TensorDataset(d['inputs_state'], d['inputs_action'], d['inputs_action_phys'], d['j_phys'], d['targets'])
        return dataset, d['scalers']

    def _init_or_load_model(self, scalers, force_init=False):
        self.model = PhysicsGatedGNK(scalers=scalers).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=CFG.LR_MAX, weight_decay=CFG.WEIGHT_DECAY)
        
        ckpt_path = CFG.get_model_path()
        if not force_init and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.target_mean.data = scalers['target_mean'].to(self.device)
            self.model.target_std.data = scalers['target_std'].to(self.device)
            return True 
        else:
            print(f"{Fore.GREEN}✨ [Trainer] Initializing NEW model.{Style.RESET_ALL}")
            return False 

    def _get_dynamic_hyperparams(self, total_samples):
        """Scale-Aware Auto-Tuning (V2)"""
        target_bs = int(total_samples / 20) 
        active_bs = np.clip(target_bs, 128, CFG.BATCH_SIZE)
        
        ref_samples = 50000 
        decay_factor = ref_samples / (total_samples + 1e-5)
        active_wd = CFG.WEIGHT_DECAY * decay_factor 
        active_wd = np.clip(active_wd, 1e-4, 0.05)
        
        bs_ratio = active_bs / CFG.BATCH_SIZE
        active_lr = CFG.LR_MAX * math.sqrt(bs_ratio)
        active_lr = max(1e-5, active_lr)
        
        return int(active_bs), active_wd, active_lr

    def train_online(self, new_batch_path, memory_paths):
        print(f"\n{Fore.CYAN}=== 🧠 Online Training Cycle (Auto-Tuned + Warmup) ==={Style.RESET_ALL}")
        
        # 1. 加载验证集
        val_ds, new_scalers = self._load_pt_file(new_batch_path)
        val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False)
        
        # 2. 初始化
        is_resume = self._init_or_load_model(new_scalers)
        
        # 3. Pre-Inference
        if is_resume:
            print(f"🔍 Running Pre-Inference Check...")
            initial_val_loss, avg_alpha = self._validate(val_loader)
            print(f"   Pre-Check Loss: {initial_val_loss:.5f} | Alpha: {avg_alpha:.3f}")
            if initial_val_loss < CFG.PRE_INF_LOSS_THRESHOLD:
                print(f"{Fore.GREEN}✅ Skipped (Loss < {CFG.PRE_INF_LOSS_THRESHOLD}).{Style.RESET_ALL}")
                return {"status": "skipped", "final_loss": initial_val_loss, "alpha": avg_alpha}
        
        # 4. 构建训练集
        train_datasets = []
        for p in memory_paths:
            try:
                ds, _ = self._load_pt_file(p)
                train_datasets.append(ds)
            except: pass
        if not train_datasets: return {"status": "failed"}
        
        full_train_ds = ConcatDataset(train_datasets)
        total_samples = len(full_train_ds)
        
        # === 🧪 自动调参 ===
        act_bs, act_wd, act_lr = self._get_dynamic_hyperparams(total_samples)
        
        print(f"{Fore.YELLOW}📊 Auto-Tuning: N={total_samples}")
        print(f"   BS: {act_bs:<4} (Config Max: {CFG.BATCH_SIZE})")
        print(f"   WD: {act_wd:.2e} (Config Base: {CFG.WEIGHT_DECAY})")
        print(f"   LR: {act_lr:.2e} (Warmup -> Target){Style.RESET_ALL}")
        
        train_loader = DataLoader(full_train_ds, batch_size=act_bs, shuffle=True, num_workers=0)
        
        # === ⚡ 重置优化器 ===
        self.optimizer = optim.AdamW(self.model.parameters(), lr=act_lr, weight_decay=act_wd)
        
        # === 🔥 [新增] Warmup + Cosine 组合调度器 ===
        epochs = CFG.MAX_EPOCHS_FINE_TUNE if is_resume else CFG.MAX_EPOCHS_INIT
        warmup_ep = CFG.WARMUP_EPOCHS
        
        if epochs > warmup_ep:
            # 阶段1: 线性预热 (从 10% LR 爬升到 100% LR)
            scheduler1 = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_ep)
            # 阶段2: 余弦衰减 (从 100% LR 降到 Min LR)
            scheduler2 = CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_ep, eta_min=CFG.LR_MIN)
            # 串联
            self.scheduler = SequentialLR(self.optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_ep])
        else:
            # 如果总轮数太少，直接用余弦
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=CFG.LR_MIN)
        
        # 5. 训练循环
        pbar = tqdm(range(epochs), desc=f"{Fore.MAGENTA}🔥 Tuning{Style.RESET_ALL}", unit="ep", 
                    bar_format="{l_bar}{bar:20}{r_bar}", leave=False)

        best_loss = float('inf')
        patience_counter = 0
        avg_alpha = 0.0
        
        for epoch in pbar:
            self.model.train()
            train_loss_sum = 0
            for batch in train_loader:
                s_n, a_n, a_p, j_p, tgt = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                pred, alpha = self.model(s_n, a_n, a_p, j_p)
                loss = self.criterion(pred, tgt)
                loss.backward()
                self.optimizer.step()
                train_loss_sum += loss.item()
            
            avg_train = train_loss_sum / len(train_loader)
            avg_val, avg_alpha = self._validate(val_loader)
            self.scheduler.step()
            
            # 显示当前 LR 方便调试
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({"Tr": f"{avg_train:.4f}", "Val": f"{avg_val:.4f}", "α": f"{avg_alpha:.2f}", "LR": f"{current_lr:.1e}"})
            
            if avg_val < best_loss:
                best_loss = avg_val
                patience_counter = 0
                self._save_checkpoint(best_loss)
            else:
                patience_counter += 1
                
            if patience_counter >= CFG.PATIENCE:
                pbar.write(f"{Fore.YELLOW}🛑 Early stop at ep {epoch+1}{Style.RESET_ALL}")
                break
        
        pbar.close()
        print(f"{Fore.GREEN}✅ Done. Best Loss: {best_loss:.5f}{Style.RESET_ALL}")
        return {"status": "trained", "final_loss": best_loss, "alpha": avg_alpha}

    def _validate(self, loader):
        self.model.eval()
        val_loss = 0.0
        alpha_sum = 0.0
        steps = 0
        with torch.no_grad():
            for batch in loader:
                s_n, a_n, a_p, j_p, tgt = [b.to(self.device) for b in batch]
                pred, alpha = self.model(s_n, a_n, a_p, j_p)
                val_loss += self.criterion(pred, tgt).item()
                alpha_sum += alpha.mean().item()
                steps += 1
        return val_loss / steps, alpha_sum / steps

    def _save_checkpoint(self, loss):
        path = CFG.get_model_path()
        torch.save({'model_state_dict': self.model.state_dict(), 'best_loss': loss}, path)
        torch.save(self.model.state_dict(), path.replace("latest", "best_deploy"))

if __name__ == "__main__":
    print("Testing Trainer Class...")