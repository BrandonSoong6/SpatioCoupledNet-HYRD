#!/usr/bin/env python3
import os
import sys
import json
import shutil
import argparse
from colorama import Fore, Style, init

init(autoreset=True)

# 指向你的数据目录
DATA_ROOT = os.path.expanduser("~/brandon/hyrd_robot/lifelong_data")
INDEX_FILE = os.path.join(DATA_ROOT, "memory_index.json")

def load_index():
    if not os.path.exists(INDEX_FILE):
        print(f"{Fore.RED}❌ Index file not found at {INDEX_FILE}{Style.RESET_ALL}")
        return None
    with open(INDEX_FILE, 'r') as f:
        return json.load(f)

def save_index(data):
    with open(INDEX_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"{Fore.GREEN}✅ Index updated successfully.{Style.RESET_ALL}")

def delete_batch(batch_id):
    """删除指定的 Batch (文件 + 索引)"""
    data = load_index()
    if data is None: return

    # 1. 从 JSON 索引中移除
    removed_h = False
    removed_e = False
    
    if batch_id in data['history']:
        data['history'].remove(batch_id)
        removed_h = True
    
    if batch_id in data['elites']:
        data['elites'].remove(batch_id)
        removed_e = True

    if not (removed_h or removed_e):
        print(f"{Fore.YELLOW}⚠️ Batch {batch_id} not found in Index.{Style.RESET_ALL}")
    else:
        save_index(data)
        print(f"🗑️ Removed {batch_id} from Memory Index.")

    # 2. 删除物理文件夹
    batch_dir = os.path.join(DATA_ROOT, batch_id)
    if os.path.exists(batch_dir):
        try:
            shutil.rmtree(batch_dir)
            print(f"🗑️ Deleted folder: {batch_dir}")
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to delete folder: {e}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️ Folder {batch_dir} does not exist.{Style.RESET_ALL}")

def reset_context():
    """换环境模式：清除短期记忆，保留模型和精英"""
    data = load_index()
    if data is None: return
    
    # 这里的策略是：
    # 1. 物理文件不删（留着以后还能分析）
    # 2. 只是把它们从 'history' 列表里踢出去，这样训练器就不会再读取它们
    # 3. 'elites' (精英/困难样本) 建议保留，因为那是物理特性的边界，通常换了环境也没变
    
    old_len = len(data['history'])
    data['history'] = [] # 清空滑动窗口
    
    save_index(data)
    print(f"{Fore.CYAN}🔄 Context Reset: Forgot {old_len} recent batches from active memory.")
    print(f"   (Physical files are NOT deleted, just ignored for future training){Style.RESET_ALL}")

def factory_reset():
    """删库跑路模式：清除所有数据和模型"""
    confirm = input(f"{Fore.RED}☢️ WARNING: This will DELETE ALL DATA and MODELS. Type 'yes' to confirm: {Style.RESET_ALL}")
    if confirm != "yes":
        print("Cancelled.")
        return

    if os.path.exists(DATA_ROOT):
        shutil.rmtree(DATA_ROOT)
        print(f"{Fore.GREEN}💥 All data wiped. System is tabula rasa.{Style.RESET_ALL}")
    else:
        print("Nothing to delete.")

def main():
    parser = argparse.ArgumentParser(description="Robot Brain Memory Management Tool")
    parser.add_argument('--del_batch', type=str, help="Delete a specific batch (e.g., batch_012)")
    parser.add_argument('--new_place', action='store_true', help="Clear short-term memory for new location")
    parser.add_argument('--nuke', action='store_true', help="Factory reset (Delete EVERYTHING)")
    parser.add_argument('--status', action='store_true', help="Show current memory status")
    
    args = parser.parse_args()

    if args.del_batch:
        delete_batch(args.del_batch)
    elif args.new_place:
        reset_context()
    elif args.nuke:
        factory_reset()
    elif args.status:
        data = load_index()
        if data:
            print(f"🧠 Memory Status:")
            print(f"   - History Window: {len(data['history'])} batches")
            print(f"   - Elite Samples:  {len(data['elites'])} batches")
            print(f"   - Current Avg Loss: {data.get('avg_loss', 'N/A')}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()