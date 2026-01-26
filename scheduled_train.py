import time
import subprocess
import os
import sys

def get_gpu_utilization():
    """Returns a list of GPU utilization percentages."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        return [int(x.strip()) for x in output.strip().split('\n') if x.strip()]
    except Exception as e:
        print(f"Error checking GPU: {e}")
        return [100] # Assume busy on error

def monitor_and_train():
    # 1. 等待一小时
    print(f"[{time.ctime()}] 脚本已启动，进入1小时等待期...")
    time.sleep(3600)
    
    # 2. 监测 GPU 空闲
    print(f"[{time.ctime()}] 开始检测 GPU 状态 (需空闲 5 分钟)...")
    idle_minutes = 0
    IDLE_THRESHOLD = 5 # GPU 使用率低于 5% 视为理想空闲
    REQUIRED_IDLE_TIME = 5 # 连续 5 分钟
    
    while True:
        utils = get_gpu_utilization()
        # 如果所有 GPU 的使用率都低于阈值
        if all(u < IDLE_THRESHOLD for u in utils):
            idle_minutes += 1
            print(f"[{time.ctime()}] GPU 已连续空闲 {idle_minutes} 分钟 (当前: {utils}%)")
        else:
            if idle_minutes > 0:
                print(f"[{time.ctime()}] 检测到 GPU 活动 ({utils}%)，空闲计时重置")
            idle_minutes = 0
            
        if idle_minutes >= REQUIRED_IDLE_TIME:
            print(f"[{time.ctime()}] GPU 空闲达标，启动 RL 训练...")
            break
            
        time.sleep(60) # 每分钟检查一次

    # 3. 启动训练
    # 使用当前 Python 解释器运行 auto_train_manager.py
    cmd = [sys.executable, "auto_train_manager.py"]
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 使用 subprocess.run 启动，训练过程的输出会打印到控制台
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n训练被用户中止")
    except Exception as e:
        print(f"训练启动失败: {e}")

if __name__ == "__main__":
    monitor_and_train()
