import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import argparse
import time
import random
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 🔧 Config V11.4 - R2 is Back
# ==========================================
class Config:
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # [Path Settings]
    DATA_PATH = "/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_smooth_strided.pt"
    EXPERIMENT_ROOT = "/home/brandon/brandon/hyrd_robot/lifelong_data/experiments"
    
    SAVE_DIR = "" 
    PLOT_DIR = ""
    
    NUM_WORKERS = 6          
    PIN_MEMORY = True         
    
    EPOCHS = 3000             
    BATCH_SIZE = 512
    SEQ_LEN = 50
    
    # [Hyperparameters]
    LR = 5e-5     
    WARMUP_EPOCHS = 5        
    LR_PATIENCE = 15          
    LR_FACTOR = 0.5           
    MIN_LR = 1e-7  
    
    WEIGHT_DECAY = 5e-5 
    DROPOUT = 0.2             
    
    LAMBDA_LOCAL = 5.0
    LAMBDA_SHAPE = 1.0
    
    # Weights Analysis:
    # High Theta weight delays Theta learning but ensures safety.
    W_X = 1.1      
    W_Y = 1.0      
    W_TH = 6900.0  
    
    VAL_RATIO = 0.15
    TEST_RATIO = 0.05
    PATIENCE = 60             
    DELTA = 1e-6
    
    PLOT_INTERVAL = 1

# ================= 1. Experiment Manager =================
def setup_experiment_dir(resume_id=None):
    os.makedirs(Config.EXPERIMENT_ROOT, exist_ok=True)
    existing_dirs = [d for d in os.listdir(Config.EXPERIMENT_ROOT) 
                     if os.path.isdir(os.path.join(Config.EXPERIMENT_ROOT, d)) and d.isdigit()]
    existing_ids = sorted([int(d) for d in existing_dirs])
    
    target_id = None
    start_fresh = True

    if resume_id is None:
        next_id = existing_ids[-1] + 1 if existing_ids else 1
        target_id = next_id
        start_fresh = True
        print(f"🆕 Starting NEW experiment run: {target_id:03d}")
    else:
        start_fresh = False
        if resume_id == 'latest':
            if not existing_ids:
                raise FileNotFoundError(f"❌ No experiments found.")
            target_id = existing_ids[-1]
            print(f"🔄 Resuming LATEST experiment run: {target_id:03d}")
        else:
            target_id = int(resume_id)
            if target_id not in existing_ids:
                raise FileNotFoundError(f"Experiment {target_id:03d} not found.")
            print(f"🔄 Resuming SPECIFIC experiment run: {target_id:03d}")

    exp_dir = os.path.join(Config.EXPERIMENT_ROOT, f"{target_id:03d}")
    plot_dir = os.path.join(exp_dir, "plots")
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    Config.SAVE_DIR = exp_dir
    Config.PLOT_DIR = plot_dir
    
    return exp_dir, start_fresh

# ================= 2. Utilities =================
class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric, model):
        score = -val_metric
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

    def load_state(self, state_dict):
        self.best_score = state_dict['best_score']
        self.counter = state_dict['counter']
        self.early_stop = state_dict['early_stop']

    def state_dict(self):
        return {'best_score': self.best_score, 'counter': self.counter, 'early_stop': self.early_stop}

def calculate_r2(pred, target):
    """
    R2 Calculation for Delta (Micro)
    """
    target_mean = torch.mean(target, dim=0, keepdim=True)
    ss_tot = torch.sum((target - target_mean) ** 2, dim=0)
    ss_res = torch.sum((target - pred) ** 2, dim=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return torch.mean(r2).item()

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

# ================= 3. Math Layer =================
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

# ================= 4. Network Architecture =================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.net(x))

class TopoImpulseNet(nn.Module):
    def __init__(self, scalers, input_dim=18, feature_dim=128, gru_dim=128, dropout=0.2):
        super().__init__()
        self.register_buffer('jac_mean', scalers['jacobian_mean'])
        self.register_buffer('jac_std', scalers['jacobian_std'])
        self.register_buffer('cmd_mean', scalers['dq_cmd_mean'])
        self.register_buffer('cmd_std', scalers['dq_cmd_std'])
        self.num_nodes = 5
        
        self.stage1_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(feature_dim, dropout),
                ResidualBlock(feature_dim, dropout) 
            ) for _ in range(self.num_nodes)
        ])
        
        self.stage2_gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=gru_dim,
            num_layers=2, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.gru_dropout = nn.Dropout(dropout)
        
        head_input_dim = feature_dim + (gru_dim * 2) + 6 
        
        self.net_expert_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64), 
                nn.GELU(),
                nn.Linear(64, 3) 
            ) for _ in range(self.num_nodes)
        ])
        
        self.confidence_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 3), 
                nn.Sigmoid() 
            ) for _ in range(self.num_nodes)
        ])
        
        self._init_weights()

    def _init_weights(self):
        for gate in self.confidence_gate:
            nn.init.constant_(gate[-2].bias, 5.0)

    def compute_nominal_delta(self, jac_norm, dq_cmd_norm):
        B, N, _ = jac_norm.shape
        jac_real = jac_norm * self.jac_std + self.jac_mean
        dq_real = dq_cmd_norm * self.cmd_std + self.cmd_mean
        J_matrix = jac_real.view(B, N, 3, 2)
        dq_vec = dq_real.unsqueeze(-1)
        nominal_delta = torch.matmul(J_matrix, dq_vec).squeeze(-1)
        return nominal_delta

    def forward(self, x_inputs, jacobian_norm, dq_cmd_norm, return_internals=False):
        local_features = []
        for i in range(self.num_nodes):
            feat = self.stage1_experts[i](x_inputs[:, i, :]) 
            local_features.append(feat)
        local_features_tensor = torch.stack(local_features, dim=1) 
        gru_out, _ = self.stage2_gru(local_features_tensor) 
        gru_out = self.gru_dropout(gru_out)
        nominal_delta = self.compute_nominal_delta(jacobian_norm, dq_cmd_norm)
        
        outputs = []
        betas = []
        net_preds = []
        
        for i in range(self.num_nodes):
            feature = torch.cat([local_features[i], gru_out[:, i, :], jacobian_norm[:, i, :]], dim=1)
            net_prediction = self.net_expert_head[i](feature) 
            beta = self.confidence_gate[i](feature)
            final_pred = beta * nominal_delta[:, i, :] + (1.0 - beta) * net_prediction
            outputs.append(final_pred)
            if return_internals:
                betas.append(beta)
                net_preds.append(net_prediction)
            
        final_stack = torch.stack(outputs, dim=1)
        if return_internals:
            return final_stack, nominal_delta, torch.stack(betas, dim=1), torch.stack(net_preds, dim=1)
        else:
            return final_stack

# ================= 5. Dataset =================
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

# ================= 6. Visualization =================
def visualize_physics_alignment(model, dataset, device, epoch, save_path, math_layer, total_points=1000):
    model.eval()
    
    # 计算需要拼接多少个连续的 seq (每个 seq 50 点)
    num_seqs = total_points // Config.SEQ_LEN
    # 随机选一个能放得下连续 num_seqs 的起点
    start_idx = random.randint(0, len(dataset) - num_seqs - 1)
    
    all_gt, all_phy, all_net, all_betas = [], [], [], []
    
    with torch.no_grad():
        for i in range(num_seqs):
            seq = dataset[start_idx + i]
            
            # 准备输入
            q_t = seq['q'].to(device)
            dq_t = seq['dq'].to(device)
            pose_t = seq['pose'].to(device)
            cmd_t = seq['cmd'].to(device)
            jac_t = seq['jac'].to(device)
            gt_delta = seq['tgt_delta'].to(device)
            
            input_vec = torch.cat([q_t, dq_t, pose_t, cmd_t, jac_t], dim=2)
            # 推理
            pred_delta, nom_delta, betas, _ = model(input_vec, jac_t, cmd_t, return_internals=True)
            
            # 存入列表用于拼接
            all_gt.append(gt_delta.cpu())
            all_phy.append(nom_delta.cpu())
            all_net.append(pred_delta.cpu())
            all_betas.append(betas.cpu())

    # 在时间轴上拼接数据 (dim=0 是时间维度)
    gt_full = torch.cat(all_gt, dim=0).numpy()
    phy_full = torch.cat(all_phy, dim=0).numpy()
    net_full = torch.cat(all_net, dim=0).numpy()
    betas_full = torch.cat(all_betas, dim=0).numpy()

    node_idx = 4 # 末端执行器
    t_axis = np.arange(gt_full.shape[0])
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    plt.suptitle(f"Epoch {epoch} - Long Sequence Analysis ({gt_full.shape[0]} Points)", fontsize=16)
    
    comp_names = ['Delta X (mm)', 'Delta Y (mm)', 'Delta Theta (rad)']
    
    for i in range(3):
        # 1. 轨迹对比
        ax_wave = axes[0, i]
        ax_wave.plot(t_axis, gt_full[:, node_idx, i], 'k-', alpha=0.4, label='GT')
        ax_wave.plot(t_axis, phy_full[:, node_idx, i], 'r--', alpha=0.7, label='Phy')
        ax_wave.plot(t_axis, net_full[:, node_idx, i], 'g-', alpha=0.8, label='Net')
        ax_wave.set_title(f"{comp_names[i]} Sequence")
        if i == 0: ax_wave.legend()
        
        # 2. 误差累计分析 (重点看长序列下是否发散)
        ax_err = axes[1, i]
        err_phy = np.abs(phy_full[:, node_idx, i] - gt_full[:, node_idx, i])
        err_net = np.abs(net_full[:, node_idx, i] - gt_full[:, node_idx, i])
        ax_err.fill_between(t_axis, err_phy, color='red', alpha=0.2, label='Phy Err')
        ax_err.fill_between(t_axis, err_net, color='green', alpha=0.4, label='Net Err')
        ax_err.set_title(f"Error Magnitude")
        
        # 3. Gate 行为分析
        ax_gate = axes[2, i]
        ax_gate.plot(t_axis, betas_full[:, node_idx, i], color='blue')
        ax_gate.set_ylim(-0.1, 1.1)
        ax_gate.set_title(f"Gate Behavior (0=Net, 1=Phy)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"long_seq_epoch_{epoch:03d}.png"))
    plt.clf()
    plt.close()

# ================= 7. Training Logic =================
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
    
    total_loss = 0      
    total_mse = 0        
    total_phy_mse = 0   
    total_r2 = 0        # [Added Back]
    total_node_mses = np.zeros(5) 
    
    pbar = tqdm(dataloader, desc="Step", leave=False, dynamic_ncols=True)
    
    for batch in pbar:
        loss_seq = 0
        mse_seq = 0
        phy_mse_seq = 0
        r2_seq = 0      # [Added Back]
        batch_node_mses = np.zeros(5)
        
        with torch.set_grad_enabled(is_train):
            for t in range(Config.SEQ_LEN): 
                q_t = batch['q'][:, t].to(device)
                dq_t = batch['dq'][:, t].to(device)
                pose_t = batch['pose'][:, t].to(device)
                cmd_t = batch['cmd'][:, t].to(device) 
                jac_t = batch['jac'][:, t].to(device) 
                pose_raw_t = batch['pose_raw'][:, t].to(device)
                gt_delta = batch['tgt_delta'][:, t].to(device)
                gt_global = batch['tgt_global'][:, t].to(device)
                
                input_vec = torch.cat([q_t, dq_t, pose_t, cmd_t, jac_t], dim=2)
                
                pred_delta, nom_delta, _, _ = model(input_vec, jac_t, cmd_t, return_internals=True)
                
                # Global FK for Metrics
                pred_global = math_layer(pose_raw_t, pred_delta)
                phy_global = math_layer(pose_raw_t, nom_delta) 
                
                # Loss Calculation
                pred_dx, pred_dy, pred_dth = pred_delta[:,:,0], pred_delta[:,:,1], pred_delta[:,:,2]
                gt_dx, gt_dy, gt_dth = gt_delta[:,:,0], gt_delta[:,:,1], gt_delta[:,:,2]

                huber = nn.HuberLoss(delta=1.0)
                l_x = huber(pred_dx, gt_dx)
                l_y = huber(pred_dy, gt_dy)
                l_th = huber(pred_dth, gt_dth)

                l_local = (Config.W_X * l_x) + (Config.W_Y * l_y) + (Config.W_TH * l_th)
                l_shape = nn.HuberLoss(delta=1.0)(pred_global, gt_global)
                
                loss_step = l_shape + Config.LAMBDA_LOCAL * l_local
                loss_seq += loss_step
                
                # Metrics
                mse_val = nn.MSELoss()(pred_global, gt_global)
                mse_seq += mse_val
                
                phy_mse_val = nn.MSELoss()(phy_global, gt_global)
                phy_mse_seq += phy_mse_val
                
                # R2 Calculation [Added Back]
                r2_step = calculate_r2(pred_delta.reshape(-1, 3), gt_delta.reshape(-1, 3))
                r2_seq += r2_step
                
                with torch.no_grad():
                    dist_sq = torch.sum((pred_global - gt_global)**2, dim=2) 
                    batch_node_mses += dist_sq.mean(dim=0).cpu().numpy() 
            
            loss_final = loss_seq / Config.SEQ_LEN
            if is_train:
                optimizer.zero_grad()
                loss_final.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss_final.item()
            total_mse += (mse_seq / Config.SEQ_LEN).item()
            total_phy_mse += (phy_mse_seq / Config.SEQ_LEN).item()
            total_r2 += (r2_seq / Config.SEQ_LEN) # [Added Back]
            total_node_mses += (batch_node_mses / Config.SEQ_LEN)
            
    n_batches = len(dataloader)
    node_rmses = np.sqrt(total_node_mses / n_batches)
    
    return {
        'loss': total_loss / n_batches,                   
        'global_rmse': np.sqrt(total_mse / n_batches),    
        'phy_rmse': np.sqrt(total_phy_mse / n_batches),   
        'r2': total_r2 / n_batches,                       # [Added Back]
        'node_rmses': node_rmses
    }

def train(args):    
    exp_dir, start_fresh = setup_experiment_dir(args.resume)
    best_model_path = os.path.join(exp_dir, 'best_model.pth')
    checkpoint_path = os.path.join(exp_dir, 'checkpoint.pth')
    log_path = os.path.join(exp_dir, 'log.json')
    
    print(f"🚀 Topo-Impulse V11.4 (R2 Returns) | Device: {Config.DEVICE}")
    print(f"📂 Output Dir: {exp_dir}")
    
    torch.backends.cudnn.benchmark = True

    full_dataset = RoboSeqDataset(os.path.expanduser(Config.DATA_PATH), seq_len=Config.SEQ_LEN)
    train_idx, val_idx, test_idx = split_continuous_blocks(
        len(full_dataset), Config.VAL_RATIO, Config.TEST_RATIO, Config.SEED
    )
    
    val_dataset = Subset(full_dataset, val_idx)
    kwargs = {'num_workers': Config.NUM_WORKERS, 'pin_memory': Config.PIN_MEMORY, 'persistent_workers': True}
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=Config.BATCH_SIZE, shuffle=False, **kwargs)
    
    model = TopoImpulseNet(
        scalers=full_dataset.scalers,
        input_dim=15, feature_dim=128, gru_dim=128, dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    math_layer = DifferentiableFK().to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=Config.LR_FACTOR, patience=Config.LR_PATIENCE, min_lr=Config.MIN_LR)
    
    early_stopping = EarlyStopping(patience=Config.PATIENCE, delta=Config.DELTA, path=best_model_path)

    history = {'epoch': [], 'train_rmse': [], 'val_rmse': [], 'phy_rmse': [], 'val_loss': [], 'val_r2': [], 'lr': []}
    
    start_epoch = 0
    if not start_fresh and os.path.exists(checkpoint_path):
        print(f"📥 Loading Checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint: plateau_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'es_state_dict' in checkpoint: early_stopping.load_state(checkpoint['es_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if os.path.exists(log_path):
                with open(log_path, 'r') as f: history = json.load(f)
        except Exception as e:
            print(f"⚠️ Resume failed: {e}. Starting fresh.")
            
    epoch_pbar = tqdm(range(start_epoch, Config.EPOCHS), desc="Progress", unit="ep", initial=start_epoch, total=Config.EPOCHS)
    
    try:
        for epoch in epoch_pbar:
            if epoch < Config.WARMUP_EPOCHS and start_fresh:
                warmup_lr = Config.LR * (epoch + 1) / Config.WARMUP_EPOCHS
                for param_group in optimizer.param_groups: param_group['lr'] = warmup_lr
                lr_status = f"{warmup_lr:.2e}"
            else:
                lr_status = f"{optimizer.param_groups[0]['lr']:.2e}"
                
            train_metrics = run_epoch(model, train_loader, math_layer, Config.DEVICE, optimizer, is_train=True)
            val_metrics = run_epoch(model, val_loader, math_layer, Config.DEVICE, is_train=False)
            
            if (epoch + 1) % Config.PLOT_INTERVAL == 0 or epoch == 0:
                visualize_physics_alignment(model, val_dataset, Config.DEVICE, epoch+1, Config.PLOT_DIR, math_layer)
            
            plateau_scheduler.step(val_metrics['global_rmse'])
            early_stopping(val_metrics['global_rmse'], model)

            phy_rmse = val_metrics['phy_rmse']
            val_net_rmse = val_metrics['global_rmse']
            train_net_rmse = train_metrics['global_rmse']
            hybrid_loss = val_metrics['loss']
            val_r2 = val_metrics['r2'] # [Added Back]
            
            node_rmse_str = "/".join([f"{e:.5f}" for e in val_metrics['node_rmses']])
            
            tqdm.write(
                f"\n[Ep {epoch+1:03d}] LR:{lr_status} | "
                f"Train_G: {train_net_rmse:.5f} | "
                f"Val_G: {val_net_rmse:.5f} (Phy: {phy_rmse:.5f}) | "
                f"R2: {val_r2:.4f} | "  # [Added Back]
                f"Nodes: {node_rmse_str}"
            )

            history['epoch'].append(epoch + 1)
            history['train_rmse'].append(train_net_rmse)
            history['val_rmse'].append(val_net_rmse)
            history['phy_rmse'].append(phy_rmse)
            history['val_r2'].append(val_r2) # [Added Back]
            history['val_loss'].append(hybrid_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            with open(log_path, 'w') as f: json.dump(history, f, indent=4)
            save_full_state(checkpoint_path, epoch, model, optimizer, plateau_scheduler, early_stopping)
            
            if early_stopping.early_stop:
                print(f"🛑 Early stopping triggered at Global RMSE: {val_net_rmse:.5f}!")
                break
                
    except KeyboardInterrupt:
        print(f"\n\n⚠️  MANUAL INTERRUPTION. Saving to {checkpoint_path}")
        save_full_state(checkpoint_path, epoch, model, optimizer, plateau_scheduler, early_stopping)
        return

    print("\n🏁 Final Test...")
    if os.path.exists(best_model_path): model.load_state_dict(torch.load(best_model_path))
    test_metrics = run_epoch(model, test_loader, math_layer, Config.DEVICE, is_train=False)
    
    phy = test_metrics['phy_rmse']
    net = test_metrics['global_rmse']
    gain = (phy - net) / phy * 100
    node_rmse_str = "/".join([f"{e:.5f}" for e in test_metrics['node_rmses']])
    
    print(f"🧪 Final Result: Phy {phy:.5f} -> Net {net:.5f} ({gain:+.2f}%) | R2: {test_metrics['r2']:.4f}")
    print(f"   Node RMSEs: {node_rmse_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', nargs='?', const='latest', help="Resume experiment ID")
    args = parser.parse_args()
    
    train(args)