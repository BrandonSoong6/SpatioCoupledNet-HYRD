#!/usr/bin/env python3
# File: robot_brain/orchestrator.py
import os
import sys
import time
import zlib
import json
import shutil
import csv
import subprocess
from datetime import datetime, timedelta
from collections import deque
from colorama import Fore, Style, init

# 引入核心配置
from robot_brain.core.config import BrainConfig as CFG
try:
    from robot_brain.core.health_inspector import HealthInspector
except ImportError:
    print(f"{Fore.RED}❌ Critical: Could not import HealthInspector. Check filename!{Style.RESET_ALL}")
    sys.exit(1)

init(autoreset=True)

class TimeEstimator:
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)
    def update(self, duration):
        self.history.append(duration)
    def get_avg(self):
        if not self.history: return None
        return sum(self.history) / len(self.history)
    def get_eta_str(self):
        avg = self.get_avg()
        if avg is None: return "Calculating..."
        return f"~{avg:.1f}s"

class DataFactory:
    """
    🏭 纯数据生产工厂 (Data Factory Mode)
    """
    
    # ================= [新增] 熔断阈值 (基于 MSE Loss) =================
    # 注意: 这是归一化后的 MSE Loss。
    # 正常训练好的模型 Loss 约为 0.001 ~ 0.01。
    # 如果机构损坏，Loss 可能会飙升到 0.1 以上。
    # 您设定的 2.0 是一个非常宽松的“完全崩坏”阈值。
    KILL_LOSS_THRESHOLD = 2.0  
    KILL_TIMES_THRESHOLD = 3   # 连续不合格次数
    # =================================================================

    STAGES = [
        ("Planning",     "📐 Generating Trajectory"),
        ("Calibration",  "⏱️  Calibrating Latency"),
        ("Execution",    "🦾 Physical Execution"),
        ("ETL",          "🧹 Cleaning & Features"),
        ("HealthCheck",  "🩺 Diagnostics")
    ]

    def __init__(self):
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.setup_workspace()
        
        self.offline_model_path = os.path.join(CFG.DATA_ROOT, "models", "gnk_offline_big_best_001.pth")
        
        if os.path.exists(self.offline_model_path):
            self.inspector = HealthInspector(self.offline_model_path)
            self.has_inspector = True
        else:
            self.log(f"Warning: Offline model not found. Health check will be skipped.", "WARN")
            self.has_inspector = False
            self.inspector = None

        # 连续故障计数器
        self.consecutive_kill_count = 0 

        self.stage_timers = {k: TimeEstimator() for k, _ in self.STAGES}
        self.loop_timer = TimeEstimator(window_size=10)
        
        self.audit_csv = os.path.join(CFG.DATA_ROOT, "audit_log.csv")
        self._init_audit_log()

    def setup_workspace(self):
        os.makedirs(CFG.DATA_ROOT, exist_ok=True)
        os.makedirs(os.path.join(CFG.DATA_ROOT, "models"), exist_ok=True)

    def _init_audit_log(self):
        if not os.path.exists(self.audit_csv):
            with open(self.audit_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'batch_id', 'status', 
                    'samples', 'tip_error_mm', 'alpha', 
                    'mse_loss', 'loop_duration_s', 'latency_ms'
                ])

    def log(self, message, level="INFO", stage=None):
        ts = datetime.now().strftime("%H:%M:%S")
        eta_suffix = ""
        if stage and level == "STEP":
            est = self.stage_timers[stage].get_eta_str()
            eta_suffix = f" {Fore.BLACK}{Style.BRIGHT}(Est: {est}){Style.RESET_ALL}"

        if level == "INFO": print(f"{Fore.WHITE}[{ts}] ℹ️  {message}{Style.RESET_ALL}")
        elif level == "STEP": print(f"{Fore.CYAN}[{ts}] 🔹 {Style.BRIGHT}{message}{eta_suffix}{Style.RESET_ALL}")
        elif level == "SUCCESS": print(f"{Fore.GREEN}[{ts}] ✅ {message}{Style.RESET_ALL}")
        elif level == "WARN": print(f"{Fore.YELLOW}[{ts}] ⚠️  {message}{Style.RESET_ALL}")
        elif level == "ERROR": print(f"{Fore.RED}[{ts}] ❌ {message}{Style.RESET_ALL}")
        elif level == "CRITICAL":
            print(f"\n{Fore.RED}{Style.BRIGHT}{'='*60}")
            print(f"[{ts}] 💀 CRITICAL: {message}")
            print(f"{'='*60}{Style.RESET_ALL}\n")
        elif level == "HEADER":
            print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'-'*60}")
            print(f"[{ts}] 🏭 {message}")
            print(f"{'-'*60}{Style.RESET_ALL}")

    def run_subproc(self, cmd_args, stage_name):
        start_t = time.time()
        env = os.environ.copy()
        env["NUMBA_DISABLE_COVERAGE"] = "1"
        try:
            subprocess.run(cmd_args, check=True, env=env)
            duration = time.time() - start_t
            self.stage_timers[stage_name].update(duration)
            return duration
        except subprocess.CalledProcessError as e:
            self.log(f"{stage_name} Failed! Exit Code: {e.returncode}", "ERROR")
            raise e

    def append_audit(self, data):
        with open(self.audit_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.get('batch_id', 'N/A'),
                data.get('status', 'UNKNOWN'),
                data.get('samples', 0),
                f"{data.get('tip_error_mm', 0):.2f}",
                f"{data.get('alpha', 0):.3f}",
                f"{data.get('mse', 0):.6f}",
                f"{data.get('duration', 0):.1f}",
                f"{data.get('latency_ms', 0):.1f}"
            ])

    def find_next_loop_idx(self):
        existing_dirs = sorted([d for d in os.listdir(CFG.DATA_ROOT) if d.startswith("batch_")])
        if not existing_dirs: return 0
        last_dir = existing_dirs[-1]
        last_idx = int(last_dir.split('_')[1])
        last_path = os.path.join(CFG.DATA_ROOT, last_dir)
        if os.path.exists(os.path.join(last_path, "done.txt")):
            self.log(f"Resuming from Batch {last_idx}. Next is {last_idx + 1}.", "SUCCESS")
            return last_idx + 1
        else:
            self.log(f"Found incomplete Batch {last_idx}. Cleaning up...", "WARN")
            try: shutil.rmtree(last_path)
            except: pass
            return last_idx

    def loop(self):
        self.log("Starting Data Factory Engine...", "HEADER")
        if self.has_inspector:
            self.log(f"Health Inspector Active. Kill Threshold: MSE > {self.KILL_LOSS_THRESHOLD} ({self.KILL_TIMES_THRESHOLD} strikes)", "SUCCESS")

        loop_idx = self.find_next_loop_idx()

        while True:
            batch_id = f"batch_{loop_idx:03d}"
            batch_dir = os.path.join(CFG.DATA_ROOT, batch_id)
            os.makedirs(batch_dir, exist_ok=True)

            loop_start = time.time()
            eta_msg = f" | Avg Loop: {self.loop_timer.get_avg():.1f}s" if self.loop_timer.get_avg() else ""
            self.log(f"PROCESSING BATCH {batch_id}{eta_msg}", "HEADER")
            current_latency_ms = 0.0

            try:
                # 1. Planning
                self.log(self.STAGES[0][1], "STEP", "Planning")
                traj_file = os.path.join(batch_dir, "targets.npy")
                from robot_brain.core.loom_planner import LoomPlanner 
                seed = zlib.adler32(f"{batch_id}_{time.time()}".encode()) 
                LoomPlanner().generate_trajectory(n_points=CFG.TRAJECTORY_POINTS, seed=seed, save_path=traj_file)
                
                # 2. Calibration
                self.log(self.STAGES[1][1], "STEP", "Calibration")
                latency_file = os.path.join(batch_dir, "latency.json")
                self.run_subproc(["python3", os.path.join(self.script_path, "core/latency_calib.py"), "--save_path", latency_file], "Calibration")
                with open(latency_file, 'r') as f: current_latency_ms = json.load(f).get('recommended_latency_s', 0) * 1000.0
                
                # 3. Execution
                self.log(self.STAGES[2][1], "STEP", "Execution")
                raw_file = os.path.join(batch_dir, "raw_data.npy")
                self.run_subproc(["python3", os.path.join(self.script_path, "core/babbling_node.py"), "--input_traj", traj_file, "--output_raw", raw_file], "Execution")

                # 4. ETL
                self.log(self.STAGES[3][1], "STEP", "ETL")
                clean_file = os.path.join(batch_dir, "clean_data.npy")
                seg_file = os.path.join(batch_dir, "segments.npy")
                pt_file = os.path.join(batch_dir, "train_data.pt")
                self.run_subproc(["python3", os.path.join(self.script_path, "core/data_cleaner.py"), "--raw", raw_file, "--latency", latency_file, "--out_clean", clean_file, "--out_seg", seg_file], "ETL")
                self.run_subproc(["python3", os.path.join(self.script_path, "core/feature_eng.py"), "--clean", clean_file, "--seg", seg_file, "--out_pt", pt_file], "ETL")

                # 5. Health Check & Kill Switch
                self.log(self.STAGES[4][1], "STEP", "HealthCheck")
                health_data = {}
                
                if self.has_inspector:
                    res = self.inspector.analyze_batch(batch_dir)
                    if res:
                        health_data = res
                        err_mm = res['tip_error_mm']
                        val_loss = res['mse'] # [修改] 获取 MSE Loss
                        
                        # === [修改] 智能熔断逻辑 (基于 MSE) ===
                        if val_loss > self.KILL_LOSS_THRESHOLD:
                            self.consecutive_kill_count += 1
                            self.log(f"⚠️ High Loss: {val_loss:.4f} (Threshold: {self.KILL_LOSS_THRESHOLD}). Strike {self.consecutive_kill_count}/{self.KILL_TIMES_THRESHOLD}", "WARN")
                            
                            if self.consecutive_kill_count >= self.KILL_TIMES_THRESHOLD:
                                self.log(f"Mechanism likely DEAD (Loss Explosion). Stopping Factory.", "CRITICAL")
                                self.append_audit({'batch_id': batch_id, 'status': 'KILL_STOP', 'tip_error_mm': err_mm, 'mse': val_loss})
                                sys.exit(1) # 熔断退出
                        else:
                            if self.consecutive_kill_count > 0:
                                self.log(f"✅ Loss normalized ({val_loss:.4f}). Counter reset.", "SUCCESS")
                            self.consecutive_kill_count = 0
                            self.log(f"Health OK | MSE: {val_loss:.5f} | Tip Err: {err_mm:.2f}mm | Alpha: {res['alpha']:.2f}", "SUCCESS")
                        # ====================================
                
                # Finish
                with open(os.path.join(batch_dir, "done.txt"), "w") as f: f.write(datetime.now().isoformat())
                loop_duration = time.time() - loop_start
                self.loop_timer.update(loop_duration)
                
                self.append_audit({
                    'batch_id': batch_id, 'status': 'OK', 'samples': health_data.get('samples', 0),
                    'tip_error_mm': health_data.get('tip_error_mm', 0), 'alpha': health_data.get('alpha', 0),
                    'mse': health_data.get('mse', 0), 'duration': loop_duration, 'latency_ms': current_latency_ms
                })
                self.log(f"Batch {batch_id} Done in {loop_duration:.1f}s. Cooldown 2s...", "SUCCESS")
                time.sleep(2)
                loop_idx += 1

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}🛑 User stopped the factory.{Style.RESET_ALL}"); break
            except Exception as e:
                self.log(f"Factory Halted: {e}", "ERROR"); time.sleep(5); continue

def main(args=None):
    """
    ROS2 控制台脚本的标准入口点
    """
    try:
        factory = DataFactory()
        factory.loop()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}👋 Data Factory Shutting Down.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Fatal Error in main: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    # 如果直接 python3 运行也能生效
    main()