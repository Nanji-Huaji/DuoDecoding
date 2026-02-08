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
import re
import subprocess
import os
import signal
import time
import shutil
import json
import sys
import numpy as np
import argparse

# Model Series Definitions
MODEL_SERIES = {
    "llama": ("llama-68m", "tiny-llama-1.1b", "llama-2-13b"),
    "vicuna": ("vicuna-68m", "tiny-vicuna-1b", "vicuna-13b-v1.5"),
    "qwen": ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-14B"),
    "qwen-32b": ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"),
    "qwen15": ("Qwen/Qwen3-0.6B", "Qwen/Qwen1.5-1.8B-Chat", "Qwen/Qwen1.5-7B-Chat"),
}

MODEL_ACC_HEAD_MAP = {
    "llama-68m": "src/SpecDec_pp/checkpoints/llama-1.1b/exp-weight6-layer3",  # Fallback
    "tiny-llama-1.1b": "src/SpecDec_pp/checkpoints/llama-1.1b/exp-weight6-layer3",
    "llama-2-13b": "src/SpecDec_pp/checkpoints/llama-13b/exp-weight6-layer3",
    "vicuna-68m": "src/SpecDec_pp/checkpoints/tiny-vicuna-1b/exp-weight6-layer3",  # Fallback
    "tiny-vicuna-1b": "src/SpecDec_pp/checkpoints/tiny-vicuna-1b/exp-weight6-layer3",
    "vicuna-13b-v1.5": "src/SpecDec_pp/checkpoints/vicuna-v1.5-13b/exp-weight6-layer3",
    "Qwen/Qwen3-0.6B": "src/SpecDec_pp/checkpoints/qwen-3-1.7b/exp-weight6-layer3",  # Fallback
    "Qwen/Qwen3-1.7B": "src/SpecDec_pp/checkpoints/qwen-3-1.7b/exp-weight6-layer3",
    "Qwen/Qwen3-14B": "src/SpecDec_pp/checkpoints/qwen-3-14b/exp-weight6-layer3",
    "Qwen/Qwen3-32B": "src/SpecDec_pp/checkpoints/qwen-3-32b/exp-weight6-layer3",
    "Qwen/Qwen1.5-1.8B-Chat": "src/SpecDec_pp/checkpoints/qwen1.5-1.8b/exp-weight-layer3",
    "Qwen/Qwen1.5-7B-Chat": "src/SpecDec_pp/checkpoints/qwen1.5-7b/exp-weight6-layer3",
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
        # 支持区分 main 和 little agent，增加对路径形式名称的兼容性
        self.loss_pattern_main = re.compile(r"\[.*rl_adapter_main.*\] Step: \d+, Loss: ([\d\.]+)")
        self.loss_pattern_little = re.compile(r"\[.*rl_adapter_little.*\] Step: \d+, Loss: ([\d\.]+)")
        self.reward_pattern_main = re.compile(r"\[.*rl_adapter_main.*\] Step: \d+, .*Reward: ([\d\.]+)")
        self.reward_pattern_little = re.compile(r"\[.*rl_adapter_little.*\] Step: \d+, .*Reward: ([\d\.]+)")
        
        self.tps_history = []
        self.loss_history_main = []
        self.loss_history_little = []
        self.reward_history_main = []
        self.reward_history_little = []
        self.best_tps = 0.0

        # 使用统一的 checkpoints/ 目录，按系列名称组织子目录
        self.checkpoint_dir = Path(f"checkpoints/{model_series_name}")
        self.status_file = self.checkpoint_dir / "training_status.json"
        self.best_checkpoints_dir = self.checkpoint_dir / "best"
        
        self.top_checkpoints = []
        
        # 加载之前的训练状态 (如果有)
        if self.status_file.exists():
            try:
                with open(self.status_file, "r") as f:
                    status = json.load(f)
                    
                    # 严格校验模型系列
                    saved_series = status.get("model_series")
                    if saved_series and saved_series != self.model_series_name:
                        print(f"[{datetime.now()}] 错误: 发现不匹配的训练状态!")
                        print(f"[{datetime.now()}] 路径 {self.status_file} 中的模型为 {saved_series}，但当前请求为 {self.model_series_name}。")
                        print(f"[{datetime.now()}] 请手动清理或移动 {self.checkpoint_dir} 目录。")
                        sys.exit(1)
                        
                    self.best_tps = status.get("best_tps", 0.0)
                    self.tps_history = status.get("tps_history", [])
                    self.loss_history_main = status.get("loss_history_main", [])
                    self.loss_history_little = status.get("loss_history_little", [])
                    self.reward_history_main = status.get("reward_history_main", [])
                    self.reward_history_little = status.get("reward_history_little", [])
                    print(f"[{datetime.now()}] 加载了现有的训练状态。当前最佳 TPS: {self.best_tps:.3f}")
            except SystemExit:
                sys.exit(1)
            except Exception as e:
                print(f"[{datetime.now()}] 加载训练状态失败: {e}")

        self._load_existing_top_checkpoints()

    def _load_existing_top_checkpoints(self):
        """扫描 best 目录，填充 top_checkpoints 列表。"""
        if not self.best_checkpoints_dir.exists():
            return
            
        for folder in self.best_checkpoints_dir.iterdir():
            if folder.is_dir() and folder.name.startswith("tps_"):
                # 1. 首先通过文件夹名后缀进行初步筛选
                if not folder.name.endswith(f"_{self.model_series_name}"):
                    continue
                
                # 2. 进一步检查文件夹内的 training_status.json (如果有的话)
                status_file = folder / "training_status.json"
                if status_file.exists():
                    try:
                        with open(status_file, "r") as f:
                            status = json.load(f)
                        # 如果记录了模型系列，则必须匹配
                        if "model_series" in status and status["model_series"] != self.model_series_name:
                            print(f"[{datetime.now()}] 警告: 检查点 {folder.name} 的 model_series ({status['model_series']}) 与当前设置 ({self.model_series_name}) 不符，已忽略。")
                            continue
                    except Exception as e:
                        print(f"[{datetime.now()}] 读取 {status_file} 失败: {e}")
                        # 如果读取失败但文件夹名对得上，暂时保留还是跳过？
                        # 为了“严格”，我们在这里保持谨慎，如果不确定就不加载
                
                try:
                    # 文件夹格式: tps_X.XXX_MMDD_HHMMSS_modelname
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
    def extract_model_size(self, model_name):
        """从模型名称中提取参数大小（单位：B）。
        
        Examples:
            'llama-2-13b' -> 13
            'Qwen/Qwen3-34B' -> 34
            'tiny-llama-1.1b' -> 1.1
            'vicuna-68m' -> 0.068
        """
        # 匹配各种参数大小格式：68m, 1.1b, 13b, 34B 等
        pattern = r'(\d+\.?\d*)([mMbB])'
        match = re.search(pattern, model_name)
        
        if not match:
            return 0.0
        
        size = float(match.group(1))
        unit = match.group(2).lower()
        
        # 转换为 B (十亿参数)
        if unit == 'm':
            size = size / 1000.0  # M to B
        # unit == 'b' 已经是 B
        
        return size
    
    def get_required_gpu_count(self):
        """根据模型系列中所有模型的总参数量决定需要的GPU数量。
        
        注意：训练时会同时加载 little + draft + target 三个模型，
        因此需要计算总显存需求。
        """
        total_size = 0.0
        model_sizes = []
        for model_name in self.models:
            size = self.extract_model_size(model_name)
            total_size += size
            model_sizes.append(f"{size:.2f}B")
        
        print(f"[{datetime.now()}] 模型参数量: {' + '.join(model_sizes)} = {total_size:.2f}B (total)")
        
        # 根据总参数大小分配GPU数量
        # 考虑到额外的显存开销（激活、梯度、优化器状态等），使用保守估计
        # 一般来说，加载模型需要约 2x 参数量的显存（FP16/BF16）+ 训练开销
        if total_size >= 70:  # 70B+需要4个GPU (例如 1.7B + 14B + 70B = 85.7B)
            return 4
        elif total_size >= 35:  # 35B+需要3个GPU (例如 1.7B + 14B + 32B = 47.7B)
            return 3
        elif total_size >= 15:  # 15B+需要2个GPU (例如 0.6B + 1.7B + 14B = 16.3B)
            return 2
        else:
            return 1  # 默认单GPU
    
    def get_best_gpus(self, count=1):
        """找到具有最多空闲显存的count个GPU。
        
        Args:
            count: 需要的GPU数量
            
        Returns:
            str: GPU ID列表，用逗号分隔，如 "0" 或 "0,1"
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            
            gpu_memory = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    gpu_id = int(parts[0].strip())
                    free_mem = int(parts[1].strip())
                    gpu_memory.append((gpu_id, free_mem))
            
            if not gpu_memory:
                return "0" if count == 1 else ",".join(str(i) for i in range(count))
            
            # 按空闲内存降序排序
            gpu_memory.sort(key=lambda x: x[1], reverse=True)
            
            # 选择前count个GPU
            selected_gpus = [str(gpu_memory[i][0]) for i in range(min(count, len(gpu_memory)))]
            
            return ",".join(selected_gpus)
        except Exception as e:
            print(f"[{datetime.now()}] GPU检测失败: {e}，使用默认GPU配置")
            return "0" if count == 1 else ",".join(str(i) for i in range(count))
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
            folder_name = f"tps_{tps_val:.3f}_{timestamp}_{self.model_series_name}"
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
            "model_series": self.model_series_name,
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

    def start_training(self):
        """Starts the training process."""
        self.prepare_checkpoints()
        
        # Clear old log
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        # 根据模型大小自动选择GPU数量
        required_gpu_count = self.get_required_gpu_count()
        gpu_ids = self.get_best_gpus(required_gpu_count)
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        env["PYTHONUNBUFFERED"] = "1"
        
        print(f"[{datetime.now()}] 模型系列 {self.model_series_name} 需要 {required_gpu_count} 个GPU")
        print(f"[{datetime.now()}] 已分配 GPU: {gpu_ids}")
        
        # Pass model series to training script
        env["MODEL_SERIES_NAME"] = self.model_series_name
        env["LITTLE_MODEL"] = self.models[0]
        env["DRAFT_MODEL"] = self.models[1]
        env["TARGET_MODEL"] = self.models[2]

        env["MAIN_RL_PATH"] = str(self.checkpoint_dir / "rl_adapter_main.pth")
        env["LITTLE_RL_PATH"] = str(self.checkpoint_dir / "rl_adapter_little.pth")

        env["ACC_HEAD_PATH"] = MODEL_ACC_HEAD_MAP.get(self.models[2], "")
        env["SMALL_DRAFT_ACC_HEAD_PATH"] = MODEL_ACC_HEAD_MAP.get(self.models[1], "")
        env["DRAFT_TARGET_ACC_HEAD_PATH"] = MODEL_ACC_HEAD_MAP.get(self.models[2], "")
            
        print(f"[{datetime.now()}] Starting training series {self.model_series_name}: {self.start_script} (Env: mmWave Trace / 10ms NTT)")
        
        with open(self.log_file, "w") as out:
            self.process = subprocess.Popen(
                ["bash", self.start_script],
                stdout=out,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                env=env
            )
        
        print(f"[{datetime.now()}] Training started with PID: {self.process.pid} on GPU {gpu_ids}")

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
        """Cleanup old processes related to THIS model series only."""
        print(f"[{datetime.now()}] Cleaning up existing processes for {self.model_series_name}...")
        try:
            # 只针对当前模型系列相关的标记进行清理，避免干扰正在运行的其他模型训练
            # 我们通过匹配命令行中的模型参数来定位进程
            patterns = []
            if self.model_series_name == "llama":
                patterns = ["llama-68m", "tiny-llama-1.1b", "Llama-2-13b"]
            elif self.model_series_name == "vicuna":
                patterns = ["vicuna-68m", "tiny-vicuna-1b", "vicuna-13b-v1.5"]
            elif self.model_series_name == "qwen":
                patterns = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-14B"]
            elif self.model_series_name == "qwen-32b":
                patterns = ["Qwen3-1.7B", "Qwen3-14B", "Qwen3-32B"]
            elif self.model_series_name == "qwen15":
                patterns = ["Qwen3-0.6B", "Qwen1.5-1.8B", "Qwen1.5-7B"]

            for p in patterns:
                # 使用 pkill -f 匹配包含特定模型路径的进程
                # 这样可以精准杀掉当前系列的训练进程，而不会误杀其他系列的进程
                subprocess.run(f"pkill -9 -f '{p}'", shell=True)
            
            # 同时也清理属于当前系列的评估脚本（如果有的话）
            # 虽然 eval_mixed 会随机采样，但它启动时参数里会有对应的模型名
            time.sleep(1)
        except:
            pass

    def run_manager(self):
        self.clean_stale_processes()
        self.start_training()
        
        last_pos = 0
        try:
            while True:
                poll_result = self.process.poll()
                if poll_result is not None:
                    print(f"[{datetime.now()}] Training process exited (Code: {poll_result})")
                    # 重要：即使进程退出了，也要运行一次停止逻辑来清理可能残留的子进程
                    self.stop_training()
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
                            # 定期将最新状态打印到终端，让用户看到进度
                            if reward_main:
                                print(f"   [Progress] Step: {len(self.reward_history_main)} | Main Reward: {self.reward_history_main[-1]:.4f} | Little Reward: {self.reward_history_little[-1] if self.reward_history_little else 0:.4f}")
                            elif tps_matches:
                                print(f"   [Progress] Step: {len(self.tps_history)} | Latest TPS: {tps_matches[-1]}")
                            
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
        finally:
            # 最终安全保障
            if self.process and self.process.poll() is None:
                self.stop_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Training Manager")
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "vicuna", "qwen", "qwen-32b", "qwen15"],
                        help="Model series to train (llama, vicuna, qwen, qwen-32b, qwen15)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    manager = TrainingManager(model_series_name=args.model, start_script="cmds/train_rl_mixed.sh")
    manager.run_manager()
