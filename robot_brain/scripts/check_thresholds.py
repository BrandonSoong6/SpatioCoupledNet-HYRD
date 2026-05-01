import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_raw_stats(file_path):
    print(f"📂 Reading file: {file_path}")
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    # 1. Load Data
    data_dict = torch.load(file_path, map_location='cpu')
    tgt_delta = data_dict['data']['tgt_delta'].numpy() # Shape: [N, 5, 3]
    
    num_samples = tgt_delta.shape[0]
    print(f"✅ Loaded {num_samples} samples.")
    
    dims_name = ['dx (mm)', 'dy (mm)', 'dtheta (rad)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

    # 2. Setup Plot: 3 Rows (Dims) x 5 Columns (Segments)
    fig, axes = plt.subplots(3, 5, figsize=(24, 12), sharex='row') # Share X per row to compare ranges visually!
    
    print("\n📊 Computing Statistics per Segment...")
    print(f"{'Seg':<5} | {'Dim':<10} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Range':<10}")
    print("-" * 80)

    for dim in range(3):
        for seg in range(5):
            ax = axes[dim, seg]
            data = tgt_delta[:, seg, dim]
            
            # Statistics
            mu = np.mean(data)
            std = np.std(data)
            d_min = np.min(data)
            d_max = np.max(data)
            d_range = d_max - d_min
            
            # Print to console
            print(f"{seg+1:<5} | {dims_name[dim]:<10} | {mu:+.2e} | {std:.2e}   | {d_min:+.2f}   | {d_max:+.2f}   | {d_range:.2f}")

            # Plot Histogram (Log Scale to see outliers)
            # bins=100 ensures we see the shape detail
            ax.hist(data, bins=100, color=colors[dim], alpha=0.7, log=True)
            
            # Annotation
            stats_text = (f"$\mu$={mu:.2e}\n"
                          f"$\sigma$={std:.2e}\n"
                          f"Rg=[{d_min:.1f}, {d_max:.1f}]")
            
            # Place text on plot
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right', 
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Styling
            if dim == 0:
                ax.set_title(f"Segment {seg+1}", fontsize=12, fontweight='bold')
            if seg == 0:
                ax.set_ylabel(f"{dims_name[dim]}\nCount (Log)", fontsize=11)
            
            ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    plt.suptitle(f"Raw Data Distribution (No Cleaning) - N={num_samples}\n(X-axis shared per row to compare width)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = "raw_distribution_check.png"
    plt.savefig(save_path)
    print(f"\n💾 Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, 
                        default='/home/brandon/brandon/hyrd_robot/lifelong_data/mega_expert_big.pt')
    args = parser.parse_args()
    
    plot_raw_stats(args.path)