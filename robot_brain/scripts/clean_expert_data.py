import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def clean_and_save_dataset(input_path, output_path):
    print(f"📂 加载原始数据: {input_path}")
    if not os.path.exists(input_path):
        print("❌ 文件不存在")
        return

    data_dict = torch.load(input_path, map_location='cpu')
    full_data = data_dict['data']
    tgt_delta = full_data['tgt_delta'].numpy() # [N, 5, 3]
    
    num_samples = tgt_delta.shape[0]
    print(f"📊 原始样本数: {num_samples}")
    
    # 初始化一个全 True 的掩码 (Mask)
    # 只要某一个采样点在任意 Segment 或任意维度上不合格，这个位置就变成 False
    keep_mask = np.ones(num_samples, dtype=bool)
    
    dims_name = ['dx', 'dy', 'dtheta']
    
    # ================= 核心清洗逻辑 =================
    # 系数设定：5.0 倍标准差
    # 对于正态分布，5sigma 覆盖 99.9999%，所以任何落在 5sigma 之外的必定是传感器故障
    SIGMA_THRESHOLD = 5.0 
    
    print("\nStarting Per-Segment Cleaning Analysis:")
    print("-" * 60)
    
    # 遍历每个 Segment (0-4) 和每个 Dimension (0-2)
    for seg_idx in range(5):
        for dim_idx in range(3):
            # 提取当前通道的数据
            raw_col = tgt_delta[:, seg_idx, dim_idx]
            
            # 计算该通道独立的统计量
            mu = np.mean(raw_col)
            std = np.std(raw_col)
            
            # 设定阈值
            lower = mu - SIGMA_THRESHOLD * std
            upper = mu + SIGMA_THRESHOLD * std
            
            # 找到坏点 (Outliers)
            # 逻辑：小于下界 或 大于上界
            outliers = (raw_col < lower) | (raw_col > upper)
            num_outliers = np.sum(outliers)
            
            # 更新全局掩码：如果当前点是 outlier，则对应位置设为 False (剔除)
            # 使用逻辑与操作：keep_mask = keep_mask AND (NOT outliers)
            keep_mask = keep_mask & (~outliers)
            
            if num_outliers > 0:
                print(f"Seg {seg_idx+1} | {dims_name[dim_idx]:<6} : Mean={mu:+.4f}, Std={std:.4f} | Range [{lower:.3f}, {upper:.3f}] | Found {num_outliers} outliers")

    # ================= 统计结果 =================
    num_kept = np.sum(keep_mask)
    num_removed = num_samples - num_kept
    percent_kept = 100 * num_kept / num_samples
    
    print("-" * 60)
    print(f"✅ 清洗完成 Summary:")
    print(f"   原始数量: {num_samples}")
    print(f"   保留数量: {num_kept} ({percent_kept:.2f}%)")
    print(f"   剔除数量: {num_removed}")
    
    if num_kept == 0:
        print("❌ 错误：所有数据都被剔除了！请检查阈值设置。")
        return

    # ================= 可视化对比 (重点看 Segment 5) =================
    print("\n📊 正在绘制 Segment 5 清洗前后对比图...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    # Col 0: Raw, Col 1: Cleaned
    
    # 只展示 Segment 5 (Index 4)，因为它是最乱的
    seg_idx_vis = 4 
    clean_delta = tgt_delta[keep_mask] # 应用掩码后的数据
    
    for i in range(3):
        # 原始数据直方图
        ax_raw = axes[i, 0]
        data_raw = tgt_delta[:, seg_idx_vis, i]
        ax_raw.hist(data_raw, bins=100, color='gray', log=True, alpha=0.5)
        ax_raw.set_title(f"Raw Seg 5 - {dims_name[i]} (N={num_samples})")
        
        # 清洗后数据直方图
        ax_clean = axes[i, 1]
        data_clean = clean_delta[:, seg_idx_vis, i]
        ax_clean.hist(data_clean, bins=100, color='#1f77b4', log=True, alpha=0.8)
        ax_clean.set_title(f"Cleaned Seg 5 - {dims_name[i]} (N={num_kept})")
        
        # 统一坐标轴范围以便对比
        d_min = min(np.min(data_raw), np.min(data_clean))
        d_max = max(np.max(data_raw), np.max(data_clean))
        # 稍微缩进一点，不要让极端的 outlier 把图撑得太扁
        # 使用 99.9% 分位数来定坐标轴视觉范围
        vis_min = np.percentile(data_clean, 0.1) * 2.0
        vis_max = np.percentile(data_clean, 99.9) * 2.0
        
        ax_raw.set_xlim(vis_min, vis_max)
        ax_clean.set_xlim(vis_min, vis_max)

    plt.suptitle(f"Segment 5 Before/After Cleaning (Threshold: {SIGMA_THRESHOLD}$\sigma$)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("cleaning_result_comparison.png")
    print("   对比图已保存至 cleaning_result_comparison.png")
    # plt.show() # 如果在远程服务器，可以注释掉这行

    # ================= 保存新数据集 =================
    print(f"\n💾 正在保存清洗后的数据到: {output_path}")
    
    new_data = {}
    
    # 遍历字典中所有的 key，应用同样的 mask
    # 注意：这里假设所有张量的第一维度都是 N
    for k, v in full_data.items():
        if isinstance(v, torch.Tensor):
            new_data[k] = v[keep_mask]
        elif isinstance(v, np.ndarray):
            new_data[k] = v[keep_mask]
        else:
            print(f"⚠️ Warning: Key '{k}' type {type(v)} skipped (not tensor/array).")
            
    # 重新计算 Scalers (均值/方差)，因为数据分布变了
    print("🔄 重新计算 Scalers...")
    new_scalers = {}
    keys_to_norm = ["q_curr", "dq_hist", "pose_loc", "dq_cmd", "jacobian", "tgt_delta"]
    
    for k in keys_to_norm:
        if k in new_data:
            data_tensor = new_data[k]
            # Flatten to [Total_Elements, Feature_Dim] for calculating stats
            # e.g., [N, 5, 3] -> [N*5, 3] mean across axis 0
            flat_data = data_tensor.reshape(-1, data_tensor.shape[-1])
            
            mean = torch.mean(flat_data, dim=0)
            std = torch.std(flat_data, dim=0) + 1e-6
            
            new_scalers[f"{k}_mean"] = mean
            new_scalers[f"{k}_std"] = std
    
    save_dict = {
        "data": new_data,
        "scalers": new_scalers
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(save_dict, output_path)
    print("✅ 完成! Clean dataset saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 输入文件
    parser.add_argument('--in_path', type=str, 
                        default='/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_big.pt')
    # 输出文件
    parser.add_argument('--out_path', type=str, 
                        default='/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_big_clean.pt')
    args = parser.parse_args()
    
    clean_and_save_dataset(args.in_path, args.out_path)