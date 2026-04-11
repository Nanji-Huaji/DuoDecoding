import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.acc_head_registry import resolve_acc_head_path
from src.rl_agent_registry import ROLE_MAIN, get_rl_agent_spec

# Model Series Definitions
MODEL_SERIES = {
    "llama": ("llama-68m", "tiny-llama-1.1b", "llama-2-13b"),
    "llama-70b": ("llama-68m", "llama-2-7b-chat", "meta-llama/Llama-2-70b-chat-hf"),
    "vicuna": ("vicuna-68m", "tiny-vicuna-1b", "vicuna-13b-v1.5"),
    "qwen": ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-14B"),
    "qwen-32b": ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"),
    "qwen15": (
        "Qwen/Qwen1.5-0.5B-Chat",
        "Qwen/Qwen1.5-1.8B-Chat",
        "Qwen/Qwen1.5-7B-Chat",
    ),
}

RL_AGENT_ROOT = Path("checkpoints/rl_agents")
ADAPTIVE_METHOD = "adaptive_decoding"


class TrainingManager:
    def __init__(
        self,
        model_series_name="llama",
        start_script="cmds/train_rl_mixed_adaptive.sh",
        log_file=None,
    ):
        self.model_series_name = model_series_name
        self.models = MODEL_SERIES.get(model_series_name)
        if not self.models:
            raise ValueError(f"Unknown model series: {model_series_name}")

        self.start_script = start_script
        self.log_file = log_file or f"train_rl_adaptive_{model_series_name}.log"
        self.process = None

        self.window_size = 8
        self.stagnation_threshold = 0.005
        self.check_interval = 10
        self.min_training_steps = 30

        self.tps_pattern = re.compile(r"Average Generation Speed: ([\d\.]+) tokens/s")
        self.loss_pattern_main = re.compile(
            r"\[.*rl_adapter_main.*\] Step: \d+, Loss: ([\d\.]+)"
        )
        self.reward_pattern_main = re.compile(
            r"\[.*rl_adapter_main.*\] Step: \d+, .*Reward: ([\d\.]+)"
        )

        self.tps_history = []
        self.loss_history_main = []
        self.reward_history_main = []
        self.best_tps = 0.0

        self.checkpoint_dir = RL_AGENT_ROOT / ADAPTIVE_METHOD / self.model_series_name
        self.status_file = self.checkpoint_dir / "training_status.json"
        self.best_checkpoints_dir = self.checkpoint_dir / "best"
        self.legacy_checkpoint_dir = Path("checkpoints") / self.model_series_name
        self.main_rl_spec = get_rl_agent_spec(
            ADAPTIVE_METHOD,
            ROLE_MAIN,
            little_model=None,
            draft_model=self.models[1],
            target_model=self.models[2],
            checkpoint_root=RL_AGENT_ROOT,
        )

        self.top_checkpoints = []

        if self.status_file.exists():
            try:
                with open(self.status_file, "r") as f:
                    status = json.load(f)

                saved_series = status.get("model_series")
                if saved_series and saved_series != self.model_series_name:
                    print(f"[{datetime.now()}] 错误: 发现不匹配的训练状态!")
                    print(
                        f"[{datetime.now()}] 路径 {self.status_file} 中的模型为 {saved_series}，但当前请求为 {self.model_series_name}。"
                    )
                    print(
                        f"[{datetime.now()}] 请手动清理或移动 {self.checkpoint_dir} 目录。"
                    )
                    sys.exit(1)

                self.best_tps = status.get("best_tps", 0.0)
                self.tps_history = status.get("tps_history", [])
                self.loss_history_main = status.get("loss_history_main", [])
                self.reward_history_main = status.get("reward_history_main", [])
                print(
                    f"[{datetime.now()}] 加载了现有的训练状态。当前最佳 TPS: {self.best_tps:.3f}"
                )
            except SystemExit:
                raise
            except Exception as e:
                print(f"[{datetime.now()}] 加载训练状态失败: {e}")

        self._load_existing_top_checkpoints()

    def _load_existing_top_checkpoints(self):
        if not self.best_checkpoints_dir.exists():
            return

        for folder in self.best_checkpoints_dir.iterdir():
            if not folder.is_dir() or not folder.name.startswith("tps_"):
                continue
            if not folder.name.endswith(f"_{self.model_series_name}"):
                continue

            status_file = folder / "training_status.json"
            if status_file.exists():
                try:
                    with open(status_file, "r") as f:
                        status = json.load(f)
                    if (
                        "model_series" in status
                        and status["model_series"] != self.model_series_name
                    ):
                        print(
                            f"[{datetime.now()}] 警告: 检查点 {folder.name} 的 model_series ({status['model_series']}) 与当前设置 ({self.model_series_name}) 不符，已忽略。"
                        )
                        continue
                except Exception as e:
                    print(f"[{datetime.now()}] 读取 {status_file} 失败: {e}")
                    continue

            try:
                tps = float(folder.name.split("_")[1])
                self.top_checkpoints.append((tps, folder))
            except (ValueError, IndexError):
                continue

        self.top_checkpoints.sort(key=lambda x: x[0], reverse=True)
        while len(self.top_checkpoints) > 3:
            _, path = self.top_checkpoints.pop()
            if path.exists():
                shutil.rmtree(path)

        if self.top_checkpoints:
            self.best_tps = self.top_checkpoints[0][0]

    def extract_model_size(self, model_name):
        pattern = r"(\d+\.?\d*)([mMbB])"
        match = re.search(pattern, model_name)
        if not match:
            return 0.0

        size = float(match.group(1))
        if match.group(2).lower() == "m":
            size /= 1000.0
        return size

    def get_required_gpu_count(self):
        total_size = 0.0
        model_sizes = []
        for model_name in self.models[1:]:
            size = self.extract_model_size(model_name)
            total_size += size
            model_sizes.append(f"{size:.2f}B")

        print(
            f"[{datetime.now()}] 模型参数量: {' + '.join(model_sizes)} = {total_size:.2f}B (total)"
        )

        if total_size >= 70:
            return 4
        if total_size >= 35:
            return 3
        if total_size >= 15:
            return 2
        return 1

    def get_best_gpus(self, count=1):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            gpu_memory = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                gpu_id_str, free_mem_str = line.split(",")
                gpu_memory.append((int(gpu_id_str.strip()), int(free_mem_str.strip())))

            if not gpu_memory:
                return "0" if count == 1 else ",".join(str(i) for i in range(count))

            gpu_memory.sort(key=lambda x: x[1], reverse=True)
            selected_gpus = [
                str(gpu_memory[i][0]) for i in range(min(count, len(gpu_memory)))
            ]
            return ",".join(selected_gpus)
        except Exception as e:
            print(f"[{datetime.now()}] GPU检测失败: {e}，使用默认GPU配置")
            return "0" if count == 1 else ",".join(str(i) for i in range(count))

    def save_best_checkpoint(self, tps_val):
        if not self.best_checkpoints_dir.exists():
            self.best_checkpoints_dir.mkdir(parents=True, exist_ok=True)

        is_top = len(self.top_checkpoints) < 3
        if not is_top and self.top_checkpoints:
            is_top = tps_val > min(self.top_checkpoints, key=lambda x: x[0])[0]

        if not is_top:
            return

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        folder_name = f"tps_{tps_val:.3f}_{timestamp}_{self.model_series_name}"
        save_path = self.best_checkpoints_dir / folder_name
        save_path.mkdir(parents=True, exist_ok=True)

        print(
            f"[{datetime.now()}] 发现新的更优性能 (TPS: {tps_val:.3f})，正在保存到 {save_path}..."
        )

        self.save_training_status()
        for src in (
            Path(self.main_rl_spec.latest_path),
            Path(self.main_rl_spec.latest_path + ".buffer"),
            self.status_file,
        ):
            if src.exists():
                shutil.copy(src, save_path / src.name)

        self.top_checkpoints.append((tps_val, save_path))
        self.top_checkpoints.sort(key=lambda x: x[0], reverse=True)

        if len(self.top_checkpoints) > 3:
            worst_tps, worst_path = self.top_checkpoints.pop()
            print(
                f"[{datetime.now()}] 淘汰旧的检查点: {worst_path} (TPS: {worst_tps:.3f})"
            )
            if worst_path.exists():
                shutil.rmtree(worst_path)

    def save_training_status(self):
        status = {
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_series": self.model_series_name,
            "best_tps": self.best_tps,
            "tps_history": self.tps_history,
            "loss_history_main": self.loss_history_main,
            "reward_history_main": self.reward_history_main,
        }
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.status_file, "w") as f:
                json.dump(status, f, indent=4)
        except Exception as e:
            print(f"[{datetime.now()}] Failed to save status: {e}")

    def prepare_checkpoints(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        main_path = Path(self.main_rl_spec.latest_path)
        main_buffer = Path(self.main_rl_spec.latest_path + ".buffer")
        main_path.parent.mkdir(parents=True, exist_ok=True)

        if main_path.exists():
            print(
                f"[{datetime.now()}] 检测到现有的 Adaptive Decoding Agent 检查点: {main_path}。将直接加载继续训练。"
            )
            return

        old_pth = self.legacy_checkpoint_dir / "rl_adapter.pth"
        old_buffer = self.legacy_checkpoint_dir / "rl_adapter.pth.buffer"
        if old_pth.exists():
            print(
                f"[{datetime.now()}] 检测到旧的单 Agent 检查点 {old_pth}，正在迁移到 adaptive_decoding 结构..."
            )
            shutil.copy(old_pth, main_path)
            print(f"   -> 已创建 {main_path}")
            if old_buffer.exists() and not main_buffer.exists():
                shutil.copy(old_buffer, main_buffer)
                print(f"   -> 已同步 Replay Buffer: {main_buffer}")
            return

        print(
            f"[{datetime.now()}] 未发现任何历史检查点。Adaptive Decoding Agent 将从头开始训练。"
        )

    def start_training(self):
        self.prepare_checkpoints()

        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        required_gpu_count = self.get_required_gpu_count()
        gpu_ids = self.get_best_gpus(required_gpu_count)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        env["PYTHONUNBUFFERED"] = "1"
        env["MODEL_SERIES_NAME"] = self.model_series_name
        env["DRAFT_MODEL"] = self.models[1]
        env["TARGET_MODEL"] = self.models[2]
        env["MAIN_RL_PATH"] = self.main_rl_spec.latest_path
        env["MAIN_RL_BEST_PATH"] = self.main_rl_spec.best_path
        env["ACC_HEAD_PATH"] = resolve_acc_head_path(self.models[1], self.models[2])

        print(
            f"[{datetime.now()}] 模型系列 {self.model_series_name} 需要 {required_gpu_count} 个GPU"
        )
        print(f"[{datetime.now()}] 已分配 GPU: {gpu_ids}")
        print(
            f"[{datetime.now()}] Starting adaptive decoding training series {self.model_series_name}: {self.start_script}"
        )

        with open(self.log_file, "w") as out:
            self.process = subprocess.Popen(
                ["bash", self.start_script],
                stdout=out,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                env=env,
            )

        print(
            f"[{datetime.now()}] Training started with PID: {self.process.pid} on GPU {gpu_ids}"
        )

    def stop_training(self):
        if self.process:
            print(
                f"[{datetime.now()}] Stopping training process group {self.process.pid}..."
            )
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
                print("Training stopped successfully.")
            except Exception:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass

    def check_convergence(self):
        if len(self.tps_history) < self.min_training_steps:
            return False

        current_window = self.tps_history[-self.window_size :]
        prev_window = self.tps_history[-self.window_size * 2 : -self.window_size]
        curr_avg = np.mean(current_window)
        prev_avg = np.mean(prev_window)
        relative_change = abs(curr_avg - prev_avg) / (prev_avg + 1e-9)

        if len(self.tps_history) % 5 == 0:
            print(
                f"   [Monitor] Curr TPS: {curr_avg:.2f} | Change: {relative_change:.2%}"
            )

        return relative_change < self.stagnation_threshold

    def clean_stale_processes(self):
        print(
            f"[{datetime.now()}] Cleaning up existing processes for {self.model_series_name}..."
        )
        try:
            patterns = []
            if self.model_series_name == "llama":
                patterns = ["tiny-llama-1.1b", "Llama-2-13b"]
            elif self.model_series_name == "llama-70b":
                patterns = ["Llama-2-7b-chat-hf", "Llama-2-70b-chat-hf"]
            elif self.model_series_name == "vicuna":
                patterns = ["tiny-vicuna-1b", "vicuna-13b-v1.5"]
            elif self.model_series_name == "qwen":
                patterns = ["Qwen3-1.7B", "Qwen3-14B"]
            elif self.model_series_name == "qwen-32b":
                patterns = ["Qwen3-14B", "Qwen3-32B"]
            elif self.model_series_name == "qwen15":
                patterns = ["Qwen1.5-1.8B-Chat", "Qwen1.5-7B-Chat"]

            for pattern in patterns:
                subprocess.run(f"pkill -9 -f '{pattern}'", shell=True)

            time.sleep(1)
        except Exception:
            pass

    def run_manager(self):
        self.clean_stale_processes()
        self.start_training()

        last_pos = 0
        try:
            while True:
                poll_result = self.process.poll()
                if poll_result is not None:
                    print(
                        f"[{datetime.now()}] Training process exited (Code: {poll_result})"
                    )
                    self.stop_training()
                    break

                try:
                    with open(self.log_file, "r") as f:
                        f.seek(last_pos)
                        new_data = f.read()
                        last_pos = f.tell()

                    if new_data:
                        tps_matches = self.tps_pattern.findall(new_data)
                        loss_main = self.loss_pattern_main.findall(new_data)
                        reward_main = self.reward_pattern_main.findall(new_data)

                        for val in loss_main:
                            self.loss_history_main.append(float(val))
                        for val in reward_main:
                            self.reward_history_main.append(float(val))

                        for val in tps_matches:
                            tps_val = float(val)
                            self.tps_history.append(tps_val)
                            if tps_val > self.best_tps:
                                self.best_tps = tps_val
                            self.save_best_checkpoint(tps_val)

                        if tps_matches or loss_main or reward_main:
                            self.save_training_status()
                            if reward_main:
                                print(
                                    f"   [Progress] Step: {len(self.reward_history_main)} | Main Reward: {self.reward_history_main[-1]:.4f}"
                                )
                            elif tps_matches:
                                print(
                                    f"   [Progress] Step: {len(self.tps_history)} | Latest TPS: {tps_matches[-1]}"
                                )

                        for _ in tps_matches:
                            if self.check_convergence():
                                print(
                                    f"\n[{datetime.now()}] *** CONVERGENCE REACHED ***"
                                )
                                self.stop_training()
                                return
                except FileNotFoundError:
                    pass
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            self.stop_training()
        finally:
            if self.process and self.process.poll() is None:
                self.stop_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto Training Manager for adaptive decoding"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        choices=["llama", "llama-70b", "vicuna", "qwen", "qwen-32b", "qwen15"],
        help="Model series to train (llama, llama-70b, vicuna, qwen, qwen-32b, qwen15)",
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    manager = TrainingManager(model_series_name=args.model)
    manager.run_manager()
