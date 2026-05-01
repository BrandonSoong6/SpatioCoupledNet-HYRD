# File: robot_brain/orchestrator.py
#!/usr/bin/env python3
import os
import sys
import time
import zlib
import json
import shutil
import subprocess
from datetime import datetime
from colorama import Fore, Style, init

# 引入核心配置
from robot_brain.core.config import BrainConfig as CFG
from robot_brain.core.trainer import LifelongTrainer

init(autoreset=True)

class MemoryManager:
    """
    🧠 记忆海马体: 管理短期滑动窗口、长期精英记忆以及性能统计
    """
    def __init__(self, index_path):
        self.index_path = index_path
        self.window_size = CFG.MEMORY_WINDOW_SIZE
        self.elite_threshold_multiplier = 1.5 
        
        self.state = {
            "history": [],    
            "elites": [],     
            "avg_loss": 0.05,
            "timings": {} 
        }
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                loaded = json.load(f)
                if "timings" not in loaded: loaded["timings"] = {}
                self.state = loaded

    def save_index(self):
        with open(self.index_path, 'w') as f:
            json.dump(self.state, f, indent=4)

    def register_batch(self, batch_id):
        if batch_id not in self.state['history']:
            self.state['history'].append(batch_id)
        self.save_index()

    def record_timing(self, task_name, duration, quantity=1):
        if quantity <= 0: quantity = 1
        rate = duration / quantity 
        if task_name not in self.state['timings']:
            self.state['timings'][task_name] = []
        self.state['timings'][task_name].append(rate)
        if len(self.state['timings'][task_name]) > 20:
            self.state['timings'][task_name].pop(0)
        self.save_index()

    def get_expected_duration(self, task_name, quantity=1):
        if task_name not in self.state['timings'] or not self.state['timings'][task_name]:
            return None
        rates = self.state['timings'][task_name]
        avg_rate = sum(rates) / len(rates)
        return avg_rate * quantity

    def update_elite_status(self, batch_id, loss):
        self.state['avg_loss'] = 0.9 * self.state['avg_loss'] + 0.1 * loss
        threshold = self.state['avg_loss'] * self.elite_threshold_multiplier
        if loss > threshold:
            if batch_id not in self.state['elites']:
                self.state['elites'].append(batch_id)
                self.save_index()
                return True, threshold
        return False, threshold

    def get_training_pool(self, current_batch_id):
        candidates = set()
        recent_history = [bid for bid in self.state['history'] if bid != current_batch_id]
        candidates.update(recent_history[-self.window_size:])
        elites = [bid for bid in self.state['elites'] if bid != current_batch_id]
        candidates.update(elites)
        
        pt_paths = []
        for bid in candidates:
            path = os.path.join(CFG.DATA_ROOT, bid, "train_data.pt")
            if os.path.exists(path):
                pt_paths.append(path)
        return pt_paths

class RobotBrain:
    TASK_MAP = {
        "Planning": "TRAJECTORY_POINTS",
        "Calibration": "Fixed",
        "Execution": "TRAJECTORY_POINTS",
        "Cleaning": "Fixed",
        "FeatureEng": "Fixed",
        "Training": "MAX_EPOCHS_FINE_TUNE" 
    }
    TASK_SEQUENCE = ["Planning", "Calibration", "Execution", "Cleaning", "FeatureEng", "Training"]

    def __init__(self):
        self.setup_workspace()
        self.memory = MemoryManager(os.path.join(CFG.DATA_ROOT, "memory_index.json"))
        self.trainer = LifelongTrainer()
        self.script_path = os.path.dirname(os.path.abspath(__file__))

    def setup_workspace(self):
        os.makedirs(CFG.DATA_ROOT, exist_ok=True)
        os.makedirs(CFG.MODEL_DIR, exist_ok=True)

    def get_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def format_eta(self, seconds):
        if seconds < 60: return f"{seconds:.1f}s"
        return f"{int(seconds//60)}m {int(seconds%60)}s"

    def get_task_quantity(self, task_name):
        source = self.TASK_MAP.get(task_name, "Fixed")
        if source == "Fixed":
            return 1
        elif source == "TRAJECTORY_POINTS":
            return CFG.TRAJECTORY_POINTS
        elif source == "MAX_EPOCHS_FINE_TUNE":
            return CFG.MAX_EPOCHS_FINE_TUNE
        return 1

    def get_time_estimates(self, current_task_name):
        curr_qty = self.get_task_quantity(current_task_name)
        step_est = self.memory.get_expected_duration(current_task_name, curr_qty)
        batch_rem = 0.0
        found = False
        has_history = False
        for task in self.TASK_SEQUENCE:
            if task == current_task_name: found = True
            if found:
                qty = self.get_task_quantity(task)
                t = self.memory.get_expected_duration(task, qty)
                if t is not None:
                    batch_rem += t
                    has_history = True
        msg_parts = []
        if step_est: msg_parts.append(f"Step Est: {self.format_eta(step_est)}")
        if has_history: msg_parts.append(f"Batch Rem: {self.format_eta(batch_rem)}")
        if msg_parts: return f" ({' | '.join(msg_parts)})"
        return ""

    def log(self, message, level="INFO", task_name=None):
        ts = self.get_timestamp()
        eta_suffix = ""
        if task_name and level == "STEP":
            eta_suffix = self.get_time_estimates(task_name)
        if level == "INFO": print(f"{Fore.WHITE}[{ts}] ℹ️  {message}{Style.RESET_ALL}")
        elif level == "STEP": print(f"{Fore.CYAN}[{ts}] 🔹 {Style.BRIGHT}{message}{eta_suffix}{Style.RESET_ALL}")
        elif level == "SUCCESS": print(f"{Fore.GREEN}[{ts}] ✅ {message}{Style.RESET_ALL}")
        elif level == "WARN": print(f"{Fore.YELLOW}[{ts}] ⚠️  {message}{Style.RESET_ALL}")
        elif level == "ERROR": print(f"{Fore.RED}[{ts}] ❌ {message}{Style.RESET_ALL}")
        elif level == "HEADER":
            print(f"\n{Fore.MAGENTA}{Style.BRIGHT}={'='*60}")
            print(f"[{ts}] 🚀 {message}")
            print(f"={'='*60}{Style.RESET_ALL}")

    def run_command(self, cmd_args, task_name, quantity=1):
        start_t = time.time()
        env = os.environ.copy()
        env["NUMBA_DISABLE_COVERAGE"] = "1"
        try:
            subprocess.run(cmd_args, check=True, env=env)
            duration = time.time() - start_t
            self.memory.record_timing(task_name, duration, quantity)
            self.log(f"{task_name} Completed (Duration: {duration:.2f}s)", "SUCCESS")
        except subprocess.CalledProcessError as e:
            self.log(f"{task_name} Failed! Error: {e}", "ERROR")
            raise e

    def loop(self):
        self.log("Robot Brain: Lifelong Learning Engine Started.", "HEADER")
        
        # === 🔄 严格断点逻辑 ===
        existing_batches = [d for d in os.listdir(CFG.DATA_ROOT) if d.startswith("batch_")]
        if existing_batches:
            last_batch_idx = max([int(d.split('_')[1]) for d in existing_batches])
            last_batch_dir = os.path.join(CFG.DATA_ROOT, f"batch_{last_batch_idx:03d}")
            if os.path.exists(os.path.join(last_batch_dir, "train_data.pt")):
                loop_idx = last_batch_idx + 1
                self.log(f"Previous batch {last_batch_idx} verified complete. Starting Loop {loop_idx}.", "SUCCESS")
            else:
                loop_idx = last_batch_idx
                self.log(f"⚠️ Previous batch {last_batch_idx} incomplete. Deleting to restart fresh...", "WARN")
                try:
                    shutil.rmtree(last_batch_dir)
                    self.log(f"🗑️ Deleted incomplete folder: {last_batch_dir}", "SUCCESS")
                except Exception as e:
                    self.log(f"❌ Failed to delete folder: {e}", "ERROR")
        else:
            loop_idx = 0

        while True:
            batch_id = f"batch_{loop_idx:03d}"
            batch_dir = os.path.join(CFG.DATA_ROOT, batch_id)
            os.makedirs(batch_dir, exist_ok=True)
            
            mem_info = f"History: {len(self.memory.state['history'])} | Elites: {len(self.memory.state['elites'])} | Avg Loss: {self.memory.state['avg_loss']:.4f}"
            self.log(f"STARTING LOOP {loop_idx} (ID: {batch_id})", "HEADER")
            self.log(f"System State: {mem_info}", "INFO")
            
            loop_start_time = time.time()

            try:
                # ================= Step 1: Planning =================
                task_name = "Planning"
                qty = CFG.TRAJECTORY_POINTS
                self.log("[1/5] Trajectory Planning (Loom)...", "STEP", task_name)
                traj_file = os.path.join(batch_dir, "targets.npy")
                if not os.path.exists(traj_file):
                    t0 = time.time()
                    from robot_brain.core.loom_planner import LoomPlanner
                    planner = LoomPlanner()
                    planner.generate_trajectory(n_points=qty, seed=zlib.adler32(batch_id.encode()), save_path=traj_file)
                    duration = time.time() - t0
                    self.memory.record_timing(task_name, duration, qty)
                    self.log(f"Generated {qty} targets. Duration: {duration:.2f}s", "SUCCESS")
                else: self.log(f"Targets exist, skipping.", "WARN")

                # ================= Step 2: Calibration =================
                task_name = "Calibration"
                self.log("[2/5] Dynamic Latency Calibration...", "STEP", task_name)
                latency_file = os.path.join(batch_dir, "latency.json")
                if not os.path.exists(latency_file):
                    self.run_command(["python3", os.path.join(self.script_path, "core/latency_calib.py"), "--save_path", latency_file], task_name)
                else: self.log("Latency config exists, skipping.", "WARN")
                
                # ================= Step 3: Execution =================
                task_name = "Execution"
                qty = CFG.TRAJECTORY_POINTS
                self.log("[3/5] Physical Execution (Motor Babbling)...", "STEP", task_name)
                raw_file = os.path.join(batch_dir, "raw_data.npy")
                if not os.path.exists(raw_file):
                    self.run_command(["python3", os.path.join(self.script_path, "core/babbling_node.py"), "--input_traj", traj_file, "--output_raw", raw_file], task_name, quantity=qty)
                else: self.log("Raw data exists, skipping.", "WARN")

                # ================= Step 4: ETL =================
                clean_file = os.path.join(batch_dir, "clean_data.npy")
                seg_file = os.path.join(batch_dir, "segments.npy")
                pt_file = os.path.join(batch_dir, "train_data.pt")
                if not os.path.exists(pt_file):
                    self.log("[4/5] Data Cleaning...", "STEP", "Cleaning")
                    self.run_command(["python3", os.path.join(self.script_path, "core/data_cleaner.py"), "--raw", raw_file, "--latency", latency_file, "--out_clean", clean_file, "--out_seg", seg_file], "Cleaning")
                    self.log("[4/5+] Physics Feature Engineering...", "STEP", "FeatureEng")
                    self.run_command(["python3", os.path.join(self.script_path, "core/feature_eng.py"), "--clean", clean_file, "--seg", seg_file, "--out_pt", pt_file], "FeatureEng")
                else: self.log("[4/5] ETL & Features exist, skipping.", "WARN")

                self.memory.register_batch(batch_id)

                # ================= Step 5: Online Training =================
                task_name = "Training"
                qty = CFG.MAX_EPOCHS_FINE_TUNE
                self.log("[5/5] Online Continuous Learning...", "STEP", task_name)
                
                training_pool = self.memory.get_training_pool(batch_id)
                t0 = time.time()
                
                if len(training_pool) == 0:
                    self.log("⚠️ [Cold Start] Accumulating memory...", "WARN")
                    self.memory.record_timing(task_name, time.time() - t0, qty)
                else:
                    self.log(f"Training on {len(training_pool)} historical batches + 1 new val batch", "INFO")
                    result = self.trainer.train_online(
                        new_batch_path=pt_file,
                        memory_paths=training_pool
                    )
                    self.memory.record_timing(task_name, time.time() - t0, qty)
                    
                    if result['status'] in ['trained', 'skipped']:
                        loss = result.get('final_loss', 999.0)
                        alpha = result.get('alpha', 0.0)
                        
                        # === 💾 关键修改: 保存训练日志和模型备份 ===
                        # 1. 保存 Logs
                        log_path = os.path.join(batch_dir, "train_log.json")
                        with open(log_path, 'w') as f:
                            json.dump(result, f, indent=4)
                        
                        # 2. 备份 Model
                        # 无论这次训练结果如何，都把当前生效的 global latest 模型复制一份到当前 batch 文件夹
                        latest_model_path = CFG.get_model_path() # .../models/gnk_physics_latest.pth
                        backup_model_path = os.path.join(batch_dir, "model_ckpt.pth")
                        if os.path.exists(latest_model_path):
                            shutil.copy2(latest_model_path, backup_model_path)
                            self.log(f"💾 Archived Model & Logs to {batch_id}/", "SUCCESS")
                        # ============================================

                        is_elite, threshold = self.memory.update_elite_status(batch_id, loss)
                        status_msg = f"Final Loss: {loss:.4f} | Alpha: {alpha:.3f}"
                        if is_elite:
                            self.log(f"🏆 {status_msg} -> Marked as ELITE (Loss > {threshold:.4f})", "WARN")
                        else:
                            self.log(f"✅ {status_msg} -> Normal Sample", "SUCCESS")

                total_loop_time = time.time() - loop_start_time
                self.log(f"LOOP {loop_idx} COMPLETE in {total_loop_time:.1f}s. Resting 2s...", "HEADER")
                time.sleep(2)
                loop_idx += 1

            except KeyboardInterrupt:
                print(f"\n{Fore.RED}🛑 User Interrupted.{Style.RESET_ALL}")
                break
            except Exception as e:
                self.log(f"Critical Error in Loop {loop_idx}: {e}", "ERROR")
                self.log("Waiting 10s before retry...", "WARN")
                time.sleep(10)
                continue

def main(args=None):
    try:
        brain = RobotBrain()
        brain.loop()
    except KeyboardInterrupt:
        print("\n👋 Robot Brain Shutting Down.")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")

if __name__ == "__main__":
    main()