import re
import time
import sys
import subprocess
import os
import signal
import numpy as np
from pathlib import Path
from datetime import datetime

class TrainingManager:
    def __init__(self, start_script="cmds/train_rl.sh", log_file="train_rl.log"):
        self.start_script = start_script
        self.log_file = log_file
        self.process = None
        
        # Convergence criteria
        self.window_size = 5
        self.stagnation_threshold = 0.01  # 1% change
        self.check_interval = 10
        self.min_training_steps = 20     # Minimum TPS records before checking convergence

        # Patterns
        self.tps_pattern = re.compile(r"Average Generation Speed: ([\d\.]+) tokens/s")
        self.loss_pattern = re.compile(r"Loss: ([\d\.]+)")
        
        self.tps_history = []
        self.loss_history = []

    def get_best_gpu(self):
        """Finds the GPU with the most free memory."""
        try:
            # Run nvidia-smi to get memory usage of all GPUs
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
                print("No GPUs found via nvidia-smi. Defaulting to GPU 0.")
                return "0"
                
            # Sort by memory usage (ascending)
            # Find the GPU with minimum memory usage
            gpu_memory.sort(key=lambda x: x[1])
            
            best_gpu = gpu_memory[0][0]
            mem_used = gpu_memory[0][1]
            
            # If the best GPU uses more than 2GB, it might be busy, but we return it anyway 
            # (or you could implement a threshold to wait)
            print(f"[{datetime.now()}] Selected GPU {best_gpu} (Memory used: {mem_used} MiB)")
            return str(best_gpu)
            
        except Exception as e:
            print(f"Error checking GPU status: {e}. Defaulting to GPU 0.")
            return "0"

    def start_training(self):
        """Starts the training process in the background using nohup-like behavior."""
        print(f"[{datetime.now()}] Starting training using {self.start_script}...")
        
        # Clear old log
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        # Select GPU
        gpu_id = self.get_best_gpu()
        
        # Prepare environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
            
        with open(self.log_file, "w") as out:
            # Using setsid to create a new session, preventing signal propagation (like Ctrl+C)
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
            except Exception as e:
                print(f"Error stopping process: {e}")
                # Force kill if needed
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass

    def check_convergence(self):
        """Checks if training has converged based on TPS history."""
        if len(self.tps_history) < self.min_training_steps:
            return False

        current_window = self.tps_history[-self.window_size:]
        prev_window = self.tps_history[-self.window_size*2 : -self.window_size]
        
        if len(prev_window) < self.window_size:
            return False
            
        curr_avg = np.mean(current_window)
        prev_avg = np.mean(prev_window)
        
        relative_change = abs(curr_avg - prev_avg) / (prev_avg + 1e-9)
        
        print(f"   >>> TPS SMA Check: Curr={curr_avg:.2f}, Prev={prev_avg:.2f}, Change={relative_change:.2%}")
        
        return relative_change < self.stagnation_threshold

    def clean_stale_processes(self):
        print(f"[{datetime.now()}] Cleaning up any stale training processes...")
        try:
            # Kill any python processes running eval scripts
            subprocess.run("pkill -f 'eval_mt_bench|eval_gsm8k|eval_cnndm|eval_xsum|eval_humaneval'", shell=True)
            # Kill any accelerate processes
            subprocess.run("pkill -f 'accelerate launch'", shell=True)
        except Exception as e:
            print(f"Cleanup warning: {e}")


    def run_manager(self):
        self.clean_stale_processes()
        self.start_training()
        
        last_pos = 0
        try:
            while True:
                # Check if process is still alive
                if self.process.poll() is not None:
                    print(f"[{datetime.now()}] Training process exited unexpectedly with code {self.process.returncode}.")
                    # Determine if it finished all tasks or crashed
                    # We can check the log tail
                    break
                
                # Read new logs
                try:
                    with open(self.log_file, 'r') as f:
                        f.seek(last_pos)
                        new_data = f.read()
                        last_pos = f.tell()
                        
                    if new_data:
                        # Log parsing
                        tps_matches = self.tps_pattern.findall(new_data)
                        loss_matches = self.loss_pattern.findall(new_data)
                        
                        for val in tps_matches:
                            self.tps_history.append(float(val))
                            print(f"[{datetime.now()}] Monitor: New TPS recorded -> {val}")
                            
                            # Check convergence only when new data comes
                            if self.check_convergence():
                                print(f"\n[{datetime.now()}] !!! CONVERGENCE REACHED !!!")
                                print("TPS has stabilized. Stopping training to save resources.")
                                self.stop_training()
                                return

                        for val in loss_matches:
                            self.loss_history.append(float(val))
                            # Optional: Check for loss convergence too
                            # print(f"[{datetime.now()}] Monitor: New Loss recorded -> {val}")
                            
                except FileNotFoundError:
                    pass
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\nManager stopped by user.")
            self.stop_training()

if __name__ == "__main__":
    # Ensure current directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    manager = TrainingManager(start_script="cmds/train_rl.sh", log_file="train_rl.log")
    manager.run_manager()
