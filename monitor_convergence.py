import re
import time
import sys
import numpy as np
from pathlib import Path

def monitor_training(log_file_path, window_size=5, stagnation_threshold=0.01, check_interval=10):
    """
    Monitors the training log for convergence based on TPS (Tokens Per Second).
    
    Args:
        log_file_path (str): Path to the log file.
        window_size (int): Number of recent data points to consider for moving average.
        stagnation_threshold (float): Threshold for relative change to consider as stagnation (convergence).
        check_interval (int): Time in seconds between checks.
    """
    print(f"Monitoring {log_file_path}...")
    
    tps_pattern = re.compile(r"Average Generation Speed: ([\d\.]+) tokens/s")
    # Assuming Loss might be printed in future updates or if I add it. 
    # Currently based on user request, they want to read loss too.
    # If loss isn't in the log, we can't fully use it, but I will add the pattern.
    loss_pattern = re.compile(r"Loss: ([\d\.]+)") 
    
    tps_history = []
    loss_history = []
    
    last_pos = 0
    
    try:
        while True:
            if not Path(log_file_path).exists():
                print(f"Log file {log_file_path} not found. Waiting...")
                time.sleep(check_interval)
                continue

            with open(log_file_path, 'r') as f:
                f.seek(last_pos)
                new_data = f.read()
                last_pos = f.tell()
                
            if new_data:
                # Find all new TPS matches
                tps_matches = tps_pattern.findall(new_data)
                loss_matches = loss_pattern.findall(new_data)
                
                if tps_matches:
                    for tps in tps_matches:
                        val = float(tps)
                        tps_history.append(val)
                        print(f"[Monitor] New TPS recorded: {val}")

                if loss_matches:
                    for loss in loss_matches:
                        val = float(loss)
                        loss_history.append(val)
                        print(f"[Monitor] New Loss recorded: {val}")
                
                # Check for convergence
                if len(tps_history) >= window_size * 2:
                    recent_avg = np.mean(tps_history[-window_size:])
                    prev_avg = np.mean(tps_history[-window_size*2:-window_size])
                    
                    relative_change = abs(recent_avg - prev_avg) / (prev_avg + 1e-9)
                    
                    print(f"[Monitor] TPS SMA({window_size}): {recent_avg:.2f} (Prev: {prev_avg:.2f}, Change: {relative_change:.2%})")
                    
                    if relative_change < stagnation_threshold:
                        print(f"\n[Monitor] CONVERGENCE DETECTED!")
                        print(f"TPS has stabilized within {stagnation_threshold*100}% over the last {window_size} evaluations.")
                        print("Suggestion: You can stop the training now.")
                        # In a real automated pipeline, we might exit with a specific code 
                        # or trigger a stop command, but here we just advise.
                        # return True 
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("Monitoring stopped.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_convergence.py <log_file_path>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    monitor_training(log_file)
