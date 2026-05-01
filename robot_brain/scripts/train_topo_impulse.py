import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import argparse
import time
import random
from tqdm import tqdm
from datetime import datetime

# ==========================================
# 🔧 全局配置中心 (Config V8)
# ==========================================
class Config:
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "/home/brandon/brandon/hyrd_robot/lifelong_data/models"
    DATA_PATH = "/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert.pt"
    
    NUM_WORKERS = 8          
    PIN_MEMORY = True         
    
    EPOCHS = 1000             
    BATCH_SIZE = 2048
    SEQ_LEN = 50
    
    BASE_BATCH_SIZE = 256     
    BASE_LR = 5e-5         
    
    # 保持 1e-5，这个学习率目前看来是能收敛到 0.89mm 的，没问题
    LR = 1e-5     
    
    WARMUP_EPOCHS = 20        
    
    # [修改 1] 忍耐度从 30 降为 10。一旦 10 个 Epoch 没进步，立刻把 LR 砍半。
    LR_PATIENCE = 10          
    LR_FACTOR = 0.5           
    MIN_LR = 1e-7  # 允许降得更低
    
    # [修改 2] 权重衰减从 1e-4 增加到 1e-3，强力抑制过拟合
    WEIGHT_DECAY = 1e-3
    
    # [修改 3] Dropout 从 0.1 增加到 0.25，增加模型鲁棒性
    DROPOUT = 0.25             
    
    LAMBDA_LOCAL = 10.0
    LAMBDA_SHAPE = 1.0
    
    VAL_RATIO = 0.15
    TEST_RATIO = 0.05
    
    # [修改 4] 早停忍耐度也配合缩短，避免无效训练
    PATIENCE = 25             
    DELTA = 1e-6
    
    PRINT_MM = True

os.makedirs(Config.SAVE_DIR, exist_ok=True)

# ================= 1. 工具类 =================
class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

    def load_state(self, state_dict):
        self.best_score = state_dict['best_score']
        self.counter = state_dict['counter']
        self.early_stop = state_dict['early_stop']

    def state_dict(self):
        return {
            'best_score': self.best_score,
            'counter': self.counter,
            'early_stop': self.early_stop
        }

# ================= 2. 数学层 =================
class DifferentiableFK(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, current_local_pose, pred_local_delta):
        batch_size = current_local_pose.shape[0]
        next_state = current_local_pose + pred_local_delta
        x, y, theta = next_state[:,:,0], next_state[:,:,1], next_state[:,:,2]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        zeros, ones = torch.zeros_like(x), torch.ones_like(x)
        r0 = torch.stack([cos_t, -sin_t, x], dim=2)
        r1 = torch.stack([sin_t, cos_t, y], dim=2)
        r2 = torch.stack([zeros, zeros, ones], dim=2)
        T_local = torch.stack([r0, r1, r2], dim=2)
        T_curr = torch.eye(3, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        global_points_list = []
        for i in range(5):
            T_curr = torch.bmm(T_curr, T_local[:, i, :, :])
            global_points_list.append(T_curr[:, :2, 2])
        return torch.stack(global_points_list, dim=1)

# ================= 3. 网络架构 =================
class TopoImpulseNet(nn.Module):
    def __init__(self, input_dim=18, feature_dim=64, gru_dim=64, dropout=0.1):
        super().__init__()
        self.num_nodes = 5
        self.stage1_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(self.num_nodes)
        ])
        self.stage2_gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=gru_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.gru_dropout = nn.Dropout(dropout)
        head_input_dim = feature_dim + (gru_dim * 2) + 6 
        self.stage3_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 3) 
            ) for _ in range(self.num_nodes)
        ])
        self.stage3_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 3), 
                nn.Sigmoid()
            ) for _ in range(self.num_nodes)
        ])

    def forward(self, x_inputs, jacobian_raw):
        local_features = []
        for i in range(self.num_nodes):
            feat = self.stage1_experts[i](x_inputs[:, i, :]) 
            local_features.append(feat)
        local_features_tensor = torch.stack(local_features, dim=1) 
        gru_out, _ = self.stage2_gru(local_features_tensor) 
        gru_out = self.gru_dropout(gru_out)
        outputs = []
        for i in range(self.num_nodes):
            head_in = torch.cat([local_features[i], gru_out[:, i, :], jacobian_raw[:, i, :]], dim=1) 
            raw_delta = self.stage3_heads[i](head_in)
            gate = self.stage3_gates[i](head_in)
            outputs.append(raw_delta * gate)
        return torch.stack(outputs, dim=1)

# ================= 4. 数据集 =================
class RoboSeqDataset(Dataset):
    def __init__(self, pt_path, seq_len=50):
        data = torch.load(pt_path)
        self.tensors = data['data']
        self.scalers = data['scalers']
        self.seq_len = seq_len
        self.total_frames = self.tensors['q_curr'].shape[0]
        self.num_seqs = self.total_frames // seq_len
        
    def __len__(self): return self.num_seqs
    def _norm(self, tensor, key):
        mean = self.scalers[f'{key}_mean']
        std = self.scalers[f'{key}_std']
        return (tensor - mean) / std
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        q_feat = self._norm(self.tensors['q_curr'][start:end], 'q_curr')
        dq_feat = self._norm(self.tensors['dq_hist'][start:end], 'dq_hist')
        pose_feat = self.tensors['pose_loc'][start:end]
        pose_norm = self._norm(pose_feat, 'pose_loc')
        cmd_feat = self._norm(self.tensors['dq_cmd'][start:end], 'dq_cmd')
        jac_feat = self.tensors['jacobian'][start:end]
        jac_norm = self._norm(jac_feat, 'jacobian')
        tgt_delta = self.tensors['tgt_delta'][start:end]
        tgt_global = self.tensors['tgt_global'][start:end]
        return {
            'q': q_feat.float(), 'dq': dq_feat.float(),
            'pose': pose_norm.float(), 'pose_raw': pose_feat.float(),
            'cmd': cmd_feat.float(), 'jac': jac_norm.float(),
            'tgt_delta': tgt_delta.float(), 'tgt_global': tgt_global.float()
        }

# ================= 5. 数据划分 & Epoch =================
def split_continuous_blocks(total_seqs, val_ratio, test_ratio, seed):
    if seed is None: seed = int(time.time())
    random.seed(seed)
    n_val = int(total_seqs * val_ratio)
    n_test = int(total_seqs * test_ratio)
    val_start = random.randint(0, total_seqs - n_val)
    val_indices = list(range(val_start, val_start + n_val))
    possible_starts = []
    if val_start - n_test >= 0: possible_starts.extend(range(0, val_start - n_test + 1))
    if total_seqs - n_test >= val_start + n_val: possible_starts.extend(range(val_start + n_val, total_seqs - n_test + 1))
    test_start = random.choice(possible_starts)
    test_indices = list(range(test_start, test_start + n_test))
    reserved = set(val_indices + test_indices)
    train_indices = [i for i in range(total_seqs) if i not in reserved]
    return train_indices, val_indices, test_indices

def run_epoch(model, dataloader, math_layer, device, optimizer=None, is_train=True):
    if is_train: model.train()
    else: model.eval()
    
    # 累加器
    total_loss = 0      # 用于反向传播的 Composite Loss
    total_mse = 0       # 用户想看的纯 MSE
    total_node_errors = np.zeros(5)
    max_error_mm_epoch = 0.0
    
    pbar = tqdm(dataloader, desc="Step", leave=False, dynamic_ncols=True)
    
    for batch in pbar:
        # b_size, seq_len = batch['q'].shape[0], batch['q'].shape[1]
        # [修改点1] 删除了 epsilon 的初始化
        
        loss_seq = 0
        mse_seq = 0
        batch_node_errors = np.zeros(5)
        batch_max_error = 0.0
        
        with torch.set_grad_enabled(is_train):
            for t in range(Config.SEQ_LEN): # 这里用Config.SEQ_LEN更保险，或者之前的 batch['q'].shape[1]
                # 1. 载入数据
                q_t = batch['q'][:, t].to(device)
                dq_t = batch['dq'][:, t].to(device)
                pose_t = batch['pose'][:, t].to(device)
                cmd_t = batch['cmd'][:, t].to(device)
                jac_t = batch['jac'][:, t].to(device)
                pose_raw_t = batch['pose_raw'][:, t].to(device)
                gt_delta = batch['tgt_delta'][:, t].to(device)
                gt_global = batch['tgt_global'][:, t].to(device)
                
                # 2. 前向传播
                # [修改点2] 删除了 epsilon，并调整 input_vec
                input_vec = torch.cat([q_t, dq_t, pose_t, cmd_t, jac_t], dim=2)
                pred_delta = model(input_vec, jac_t)
                pred_global = math_layer(pose_raw_t, pred_delta)
                
                # 3. 计算 Loss (用于优化)
                l_local = nn.HuberLoss(delta=1.0)(pred_delta, gt_delta)
                l_shape = nn.HuberLoss(delta=1.0)(pred_global, gt_global)
                loss_step = l_shape + Config.LAMBDA_LOCAL * l_local
                loss_seq += loss_step
                
                # 4. 计算纯 MSE (用于展示)
                mse_step = nn.MSELoss()(pred_global, gt_global)
                mse_seq += mse_step
                
                # [修改点3] 删除了 epsilon 更新代码 (闭环切断)
                
                # 6. 计算物理误差 (mm) - 仅可视化
                with torch.no_grad():
                    dist = torch.norm(pred_global - gt_global, dim=2) # [B, 5]
                    batch_node_errors += dist.mean(dim=0).cpu().numpy() 
                    current_max = dist.max().item()
                    if current_max > batch_max_error: 
                        batch_max_error = current_max
            
            # Sequence Loop 结束
            
            # 反向传播 (仅 Train)
            loss_final = loss_seq / Config.SEQ_LEN
            if is_train:
                optimizer.zero_grad()
                loss_final.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # 记录数据
            total_loss += loss_final.item()
            total_mse += (mse_seq / Config.SEQ_LEN).item()
            total_node_errors += (batch_node_errors / Config.SEQ_LEN)
            if batch_max_error > max_error_mm_epoch: 
                max_error_mm_epoch = batch_max_error
            
            pbar.set_postfix({'MSE': f"{(mse_seq/Config.SEQ_LEN).item():.2f}"})
            
    # Epoch 结束
    return {
        'loss': total_loss / len(dataloader),  # 优化器用的 Loss
        'mse': total_mse / len(dataloader),    # 纯 MSE
        'node_errs': total_node_errors / len(dataloader),
        'max_err': max_error_mm_epoch
    }

# ================= 6. 辅助：状态保存函数 =================
def save_full_state(path, epoch, model, optimizer, scheduler, early_stopping):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'es_state_dict': early_stopping.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    torch.save(state, path)

# ================= 7. 训练主循环 =================
def train(args):    
    print(f"🚀 Topo-Impulse V8 (Fix & Clean) | Device: {Config.DEVICE}")
    print(f"📦 BS={Config.BATCH_SIZE} | Workers={Config.NUM_WORKERS} | Target LR={Config.LR:.2e}")
    
    torch.backends.cudnn.benchmark = True

    # 1. 数据
    full_dataset = RoboSeqDataset(os.path.expanduser(Config.DATA_PATH), seq_len=Config.SEQ_LEN)
    train_idx, val_idx, test_idx = split_continuous_blocks(
        len(full_dataset), Config.VAL_RATIO, Config.TEST_RATIO, Config.SEED
    )
    
    kwargs = {
        'num_workers': Config.NUM_WORKERS, 
        'pin_memory': Config.PIN_MEMORY,
        'persistent_workers': True
    }
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=Config.BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=Config.BATCH_SIZE, shuffle=False, **kwargs)
    
    # 2. 模型
    # [修改点4] input_dim 从 18 改为 15 (因为删除了 epsilon 的 3 维)
    model = TopoImpulseNet(input_dim=15, dropout=Config.DROPOUT).to(Config.DEVICE)
    math_layer = DifferentiableFK().to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # ✅ 修复：移除了 verbose=True
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=Config.LR_FACTOR, patience=Config.LR_PATIENCE,
        min_lr=Config.MIN_LR
    )
    
    # 路径
    best_model_path = os.path.join(Config.SAVE_DIR, 'topo_best_model.pth')
    resume_path = os.path.join(Config.SAVE_DIR, 'topo_latest_checkpoint.pth')
    interrupt_path = os.path.join(Config.SAVE_DIR, 'topo_interrupted.pth')
    
    early_stopping = EarlyStopping(patience=Config.PATIENCE, delta=Config.DELTA, path=best_model_path)

    # 📊 飞行记录仪
    history = {
        'epoch': [],
        'train_loss': [], 'val_loss': [],
        'train_mse': [],  'val_mse': [],
        'val_max_err': [],
        'lr': []
    }
    log_path = os.path.join(Config.SAVE_DIR, 'training_log.json')
    
    start_epoch = 0

    # 3. Resume
    if args.resume:
        load_path = args.resume if args.resume != 'latest' else resume_path
        if os.path.exists(load_path):
            print(f"🔄 Resuming from checkpoint: {load_path}")
            checkpoint = torch.load(load_path, map_location=Config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                plateau_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'es_state_dict' in checkpoint:
                early_stopping.load_state(checkpoint['es_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"⏩ Jumping to Epoch {start_epoch+1}")
        else:
            print(f"⚠️ Checkpoint {load_path} not found! Starting from scratch.")

    # 4. Loop
    epoch_pbar = tqdm(range(start_epoch, Config.EPOCHS), desc="Progress", unit="ep", initial=start_epoch, total=Config.EPOCHS)
    
    try:
        for epoch in epoch_pbar:
            if epoch < Config.WARMUP_EPOCHS:
                warmup_lr = Config.LR * (epoch + 1) / Config.WARMUP_EPOCHS
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                lr_status = f"Warmup {warmup_lr:.2e}"
            else:
                lr_status = f"LR {optimizer.param_groups[0]['lr']:.2e}"
                
            train_metrics = run_epoch(model, train_loader, math_layer, Config.DEVICE, optimizer, is_train=True)
            val_metrics = run_epoch(model, val_loader, math_layer, Config.DEVICE, is_train=False)
            
            # 使用 MSE 来做 Scheduler 的指标，而不是 loss
            if epoch >= Config.WARMUP_EPOCHS:
                plateau_scheduler.step(val_metrics['mse'])
            
            early_stopping(val_metrics['mse'], model)

            # ✅ 计算 RMSE (mm)
            train_rmse = train_metrics['mse'] ** 0.5
            val_rmse = val_metrics['mse'] ** 0.5

            # ✅ 格式优化：打印 RMSE 而不是 MSE
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            node_err_str = "/".join([f"{e:.2f}" for e in val_metrics['node_errs']])
            es_status = f"{early_stopping.counter}/{early_stopping.patience}"

            tqdm.write(
                f"\n[{current_time}] Ep {epoch+1} | {lr_status} | "
                f"Train RMSE: {train_rmse:.2f}mm | Val RMSE: {val_rmse:.2f}mm | "  # <--- 修改这里
                f"MaxErr: {val_metrics['max_err']:.1f}mm | Nodes: {node_err_str} | "
                f"ES: {es_status}"
            )

            # 📝 记录数据 (建议把 RMSE 也存进去，或者直接替换 MSE)
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_mse'].append(train_metrics['mse']) # 保留原始 MSE 以备不时之需
            history['val_mse'].append(val_metrics['mse'])
            # 新增 RMSE 记录 (可选，如果以后画图想直接画 RMSE)
            if 'train_rmse' not in history: history['train_rmse'] = []
            if 'val_rmse' not in history: history['val_rmse'] = []
            history['train_rmse'].append(train_rmse)
            history['val_rmse'].append(val_rmse)
            
            history['val_max_err'].append(val_metrics['max_err'])
            history['lr'].append(optimizer.param_groups[0]['lr'])

            # 💾 实时保存 (JSON 格式通用性最好)
            import json
            with open(log_path, 'w') as f:
                json.dump(history, f, indent=4)
            
            save_full_state(resume_path, epoch, model, optimizer, plateau_scheduler, early_stopping)
            
            if early_stopping.early_stop:
                print("🛑 Early stopping triggered!")
                break
                
    except KeyboardInterrupt:
        print("\n\n⚠️  MANUAL INTERRUPTION DETECTED (Ctrl+C) ⚠️")
        save_full_state(interrupt_path, epoch, model, optimizer, plateau_scheduler, early_stopping)
        print(f"✅ Emergency state saved to: {interrupt_path}")
        return

    # 5. Final Test
    print("\n🏁 Training Finished. Running Final Test with Best Model...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    test_metrics = run_epoch(model, test_loader, math_layer, Config.DEVICE, is_train=False)
    print(f"🧪 Final Test MSE: {test_metrics['mse']:.2f} | Max Err: {test_metrics['max_err']:.1f}mm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=Config.DATA_PATH)
    parser.add_argument('--resume', nargs='?', const='latest', help="Path to checkpoint or 'latest'/'interrupted'")
    args = parser.parse_args()
    
    if args.resume == 'interrupted':
        args.resume = os.path.join(Config.SAVE_DIR, 'topo_interrupted.pth')
    elif args.resume == 'latest':
        args.resume = os.path.join(Config.SAVE_DIR, 'topo_latest_checkpoint.pth')
        
    Config.DATA_PATH = os.path.expanduser(args.data_path)
    train(args)