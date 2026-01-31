import re
import time
import sys
import subprocess
import os
import signal
import shutil
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Model Series Definitions
MODEL_SERIES = {
    "llama": ("llama-68m", "tiny-llama-1.1b", "llama-2-13b"),
    "vicuna": ("vicuna-68m", "tiny-vicuna-1b", "vicuna-13b-v1.5"),
    "qwen": ("qwen/Qwen3-0.6B", "qwen/Qwen3-1.7B", "qwen/Qwen3-14B")
}

class TrainingManager:
    def __init__(self, model_series_name="llama", start_script="cmds/train_rl_mixed.sh", log_file=None):
        self.model_series_name = model_series_name
        self.models = MODEL_SERIES.get(model_series_name)
        if not self.models:
            raise ValueError(f"Unknown model series: {model_series_name}")
            
        self.start_script = start_script
        self.log_file = log_file or f"train_rl_{model_series_name}.log"
        self.process = None
        
        # Convergence criteria
        self.window_size = 8             # Increased for multi-agent stability
        self.stagnation_threshold = 0.005 # 0.5% change
        self.check_interval = 10
        self.min_training_steps = 30     # More records before checking convergence

        # Patterns
        self.tps_pattern = re.compile(r"Average Generation Speed: ([\d\.]+) tokens/s")
        # 支持区分 main 和 little agent
        self.loss_pattern_main = re.compile(r"\[rl_adapter_main\] Step: \d+, Loss: ([\d\.]+)")
        self.loss_pattern_little = re.compile(r"\[rl_adapter_little\] Step: \d+, Loss: ([\d\.]+)")
        self.reward_pattern_main = re.compile(r"\[rl_adapter_main\] Step: \d+, .*Reward: ([\d\.]+)")
        self.reward_pattern_little = re.compile(r"\[rl_adapter_little\] Step: \d+, .*Reward: ([\d\.]+)")
        
        self.tps_history = []
        self.loss_history_main = []
        self.loss_history_little = []
        self.reward_history_main = []
        self.reward_history_little = []
        self.best_tps = 0.0

        # Suffix the model series name to the checkpoint directory
        self.checkpoint_dir = Path(f"checkpoints_{model_series_name}")
        self.status_file = self.checkpoint_dir / "training_status.json"
        self.best_checkpoints_dir = self.checkpoint_dir / "best"
        
        self.top_checkpoints = []
        self._load_existing_top_checkpoints()

    def _load_existing_top_checkpoints(self):
        """扫描 best 目录，填充 top_checkpoints 列表。"""
        if not self.best_checkpoints_dir.exists():
            return
            
        for folder in self.best_checkpoints_dir.iterdir():
            if folder.is_dir() and folder.name.startswith("tps_"):
                try:
                    # 文件夹格式: tps_X.XXX_MMDD_HHMMSS
                    parts = folder.name.split('_')
                    tps = float(parts[1])
                    self.top_checkpoints.append((tps, folder))
                except (ValueError, IndexError):
                    continue
        
        self.top_checkpoints.sort(key=lambda x: x[0], reverse=True)
        # 确保只保留前三个（以防万一）
        while len(self.top_checkpoints) > 3:
            _, path = self.top_checkpoints.pop()
            if path.exists():
                shutil.rmtree(path)
        
        if self.top_checkpoints:
            self.best_tps = self.top_checkpoints[0][0]

    def save_best_checkpoint(self, tps_val):
        """如果当前 TPS 是前三名之一，则保存检查点和训练状态。"""
        if not self.best_checkpoints_dir.exists():
            self.best_checkpoints_dir.mkdir(parents=True, exist_ok=True)

        is_top = False
        if len(self.top_checkpoints) < 3:
            is_top = True
        elif tps_val > min(self.top_checkpoints, key=lambda x: x[0])[0]:
            is_top = True
            
        if is_top:
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            folder_name = f"tps_{tps_val:.3f}_{timestamp}"
            save_path = self.best_checkpoints_dir / folder_name
            save_path.mkdir(parents=True, exist_ok=True)
            
            print(f"[{datetime.now()}] 发现新的更优性能 (TPS: {tps_val:.3f})，正在保存到 {save_path}...")
            
            checkpoint_dir = self.checkpoint_dir
            files_to_copy = [
                "rl_adapter_main.pth", "rl_adapter_main.pth.buffer",
                "rl_adapter_little.pth", "rl_adapter_little.pth.buffer",
                "training_status.json"
            ]
            
            # 同步最新的训练状态
            self.save_training_status()

            for f_name in files_to_copy:
                src = checkpoint_dir / f_name
                if src.exists():
                    shutil.copy(src, save_path / f_name)
            
            self.top_checkpoints.append((tps_val, save_path))
            self.top_checkpoints.sort(key=lambda x: x[0], reverse=True)
            
            if len(self.top_checkpoints) > 3:
                worst_tps, worst_path = self.top_checkpoints.pop()
                print(f"[{datetime.now()}] 淘汰旧的检查点: {worst_path} (TPS: {worst_tps:.3f})")
                if worst_path.exists():
                    shutil.rmtree(worst_path)

    def save_training_status(self):
        """Saves TPS and Loss history to a JSON file for plotting."""
        status = {
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_tps": self.best_tps,
            "tps_history": self.tps_history,
            "loss_history_main": self.loss_history_main,
            "loss_history_little": self.loss_history_little,
            "reward_history_main": self.reward_history_main,
            "reward_history_little": self.reward_history_little
        }
        try:
            with open(self.status_file, "w") as f:
                json.dump(status, f, indent=4)
        except Exception as e:
            print(f"[{datetime.now()}] Failed to save status: {e}")

    def prepare_checkpoints(self):
        """Migrate old single checkpoint to new dual checkpoint structure or verify existing ones."""
        checkpoint_dir = self.checkpoint_dir
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        old_pth = checkpoint_dir / "rl_adapter.pth"
        old_buffer = checkpoint_dir / "rl_adapter.pth.buffer"
        
        targets = [
            ("rl_adapter_main.pth", "rl_adapter_main.pth.buffer"),
            ("rl_adapter_little.pth", "rl_adapter_little.pth.buffer")
        ]
        
        # Check if new checkpoints already exist
        existing_new = [t for t, _ in targets if (checkpoint_dir / t).exists()]
        
        if len(existing_new) == len(targets):
            print(f"[{datetime.now()}] 检测到现有的双 Agent 检查点: {', '.join(existing_new)}。将直接加载继续训练。")
            return

        if old_pth.exists():
            print(f"[{datetime.now()}] 检测到旧的单 Agent 检查点 {old_pth}，正在迁移到双 Agent 结构...")
            for model_file, buffer_file in targets:
                target_pth = checkpoint_dir / model_file
                target_buf = checkpoint_dir / buffer_file
                
                if not target_pth.exists():
                    shutil.copy(old_pth, target_pth)
                    print(f"   -> 已创建 {target_pth}")
                
                if old_buffer.exists() and not target_buf.exists():
                    shutil.copy(old_buffer, target_buf)
                    print(f"   -> 已同步 Replay Buffer: {target_buf}")
        else:
            if existing_new:
                print(f"[{datetime.now()}] 部分双 Agent 检查点已存在: {', '.join(existing_new)}。缺失的部分将重新开始。")
            else:
                print(f"[{datetime.now()}] 未发现任何历史检查点 (old or new)。Agent 将从头开始训练。")

    def get_best_gpu(self):
        """Finds the GPU with the most free memory."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            
            gpu_memory = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    idx, mem_used = map(int, line.split(','))
                    gpu_memory.append((idx, mem_used))
            
            if not gpu_memory:
                return "0"
            gpu_memory.sort(key=lambda x: x[1])
            return str(gpu_memory[0][0])
        except Exception as e:
            return "0"

    def start_training(self):
        """Starts the training process."""
        self.prepare_checkpoints()
        
        # Clear old log
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        # Select GPU
        gpu_id = self.get_best_gpu()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        # Pass model series to training script
        env["LITTLE_MODEL"] = self.models[0]
        env["DRAFT_MODEL"] = self.models[1]
        env["TARGET_MODEL"] = self.models[2]
            
        print(f"[{datetime.now()}] Starting training series {self.model_series_name}: {self.start_script} (Env: 34.6Mbps / 0ms)")
        
        with open(self.log_file, "w") as out:
            self.process = subprocess.Popen(
                ["bash", self.start_script],
                stdout=out,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                env=env
            )
        
        print(f"[{datetime.now()}] Training started with PID: {self.process.pid} on GPU {gpu_id}")

    def stop_training(self):
        """Stops the training process."""
        if self.process:
            print(f"[{datetime.now()}] Stopping training process group {self.process.pid}...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
                print("Training stopped successfully.")
            except:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass

    def check_convergence(self):
        """Checks convergence based on TPS."""
        if len(self.tps_history) < self.min_training_steps:
            return False

        current_window = self.tps_history[-self.window_size:]
        prev_window = self.tps_history[-self.window_size*2 : -self.window_size]
        
        curr_avg = np.mean(current_window)
        prev_avg = np.mean(prev_window)
        
        relative_change = abs(curr_avg - prev_avg) / (prev_avg + 1e-9)
        
        if len(self.tps_history) % 5 == 0:
            print(f"   [Monitor] Curr TPS: {curr_avg:.2f} | Change: {relative_change:.2%}")
        
        return relative_change < self.stagnation_threshold

    def clean_stale_processes(self):
        """Cleanup old processes."""
        print(f"[{datetime.now()}] Cleaning up environment...")
        try:
            subprocess.run("pkill -9 -f 'eval_mt_bench|eval_gsm8k|eval_cnndm|eval_xsum|eval_humaneval|accelerate'", shell=True)
            time.sleep(2)
        except:
            pass

    def run_manager(self):
        self.clean_stale_processes()
        self.start_training()
        
        last_pos = 0
        try:
            while True:
                if self.process.poll() is not None:
                    print(f"[{datetime.now()}] Training process exited (Code: {self.process.returncode})")
                    break
                
                try:
                    with open(self.log_file, 'r') as f:
                        f.seek(last_pos)
                        new_data = f.read()
                        last_pos = f.tell()
                        
                    if new_data:
                        tps_matches = self.tps_pattern.findall(new_data)
                        loss_main = self.loss_pattern_main.findall(new_data)
                        loss_little = self.loss_pattern_little.findall(new_data)
                        reward_main = self.reward_pattern_main.findall(new_data)
                        reward_little = self.reward_pattern_little.findall(new_data)
                        
                        for val in loss_main:
                            self.loss_history_main.append(float(val))
                        for val in loss_little:
                            self.loss_history_little.append(float(val))
                        for val in reward_main:
                            self.reward_history_main.append(float(val))
                        for val in reward_little:
                            self.reward_history_little.append(float(val))

                        for val in tps_matches:
                            tps_val = float(val)
                            self.tps_history.append(tps_val)
                            if tps_val > self.best_tps:
                                self.best_tps = tps_val
                            # 保存最优检查点逻辑
                            self.save_best_checkpoint(tps_val)
                        
                        if tps_matches or loss_main or loss_little or reward_main or reward_little:
                            self.save_training_status()
                            
                        for val in tps_matches:
                            if self.check_convergence():
                                print(f"\n[{datetime.now()}] *** CONVERGENCE REACHED ***")
                                self.stop_training()
                                return
                except FileNotFoundError:
                    pass
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            self.stop_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Training Manager")
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "vicuna", "qwen"],
                        help="Model series to train (llama, vicuna, qwen)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    manager = TrainingManager(model_series_name=args.model, start_script="cmds/train_rl_mixed.sh")
    manager.run_manager()

