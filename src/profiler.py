"""Performance profiling utilities for debugging inference speed."""

import time
import torch
from contextlib import contextmanager
from typing import Optional, Dict, List
import psutil
import os


class PerformanceProfiler:
    """Track GPU memory, time, and other metrics during inference."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.records: List[Dict] = []
        self.current_context: Optional[str] = None
        self.context_start_time: Optional[float] = None
        
    @contextmanager
    def profile(self, name: str, verbose: bool = True):
        """Context manager to profile a code block."""
        if not self.enabled:
            yield
            return
            
        # Record before
        torch.cuda.synchronize()
        start_time = time.time()
        start_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        start_max_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            # Record after
            torch.cuda.synchronize()
            end_time = time.time()
            end_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            end_max_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            elapsed = end_time - start_time
            mem_delta = end_mem - start_mem
            max_mem_delta = end_max_mem - start_max_mem
            
            record = {
                'name': name,
                'elapsed_time': elapsed,
                'start_mem_gb': start_mem,
                'end_mem_gb': end_mem,
                'mem_delta_gb': mem_delta,
                'max_mem_gb': end_max_mem,
                'max_mem_delta_gb': max_mem_delta,
            }
            self.records.append(record)
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"[Profile] {name}")
                print(f"  Time: {elapsed:.3f}s")
                print(f"  Memory: {start_mem:.2f}GB -> {end_mem:.2f}GB (Δ{mem_delta:+.2f}GB)")
                print(f"  Max Memory: {end_max_mem:.2f}GB (Δ{max_mem_delta:+.2f}GB)")
                print(f"{'='*60}\n")
    
    def print_summary(self):
        """Print summary of all profiled sections."""
        if not self.records:
            print("No profiling data recorded.")
            return
            
        print("\n" + "="*80)
        print("PERFORMANCE PROFILING SUMMARY")
        print("="*80)
        
        total_time = sum(r['elapsed_time'] for r in self.records)
        
        print(f"\n{'Section':<40} {'Time (s)':<12} {'Memory Δ (GB)':<15} {'Max Mem (GB)':<15}")
        print("-" * 80)
        
        for record in self.records:
            print(f"{record['name']:<40} "
                  f"{record['elapsed_time']:>10.3f}  "
                  f"{record['mem_delta_gb']:>+14.2f}  "
                  f"{record['max_mem_gb']:>14.2f}")
        
        print("-" * 80)
        print(f"{'TOTAL':<40} {total_time:>10.3f}s")
        print("="*80 + "\n")
    
    def reset(self):
        """Clear all profiling records."""
        self.records.clear()


def log_gpu_memory(stage: str = ""):
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        print(f"[{stage}] No GPU available")
        return
        
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"\n[GPU Memory - {stage}]")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    print(f"  Max Allocated: {max_allocated:.2f} GB")
    
    # Also log system memory
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / 1024**3
    print(f"  System RAM: {ram_gb:.2f} GB")


def check_model_device_map(model, name: str = "Model"):
    """Check and print device map of a model."""
    print(f"\n[{name} Device Map]")
    if hasattr(model, 'hf_device_map'):
        print(f"  HF Device Map: {model.hf_device_map}")
    else:
        print(f"  Device: {next(model.parameters()).device}")
    
    # Count parameters per device
    device_params = {}
    total_params = 0
    for name, param in model.named_parameters():
        device = str(param.device)
        if device not in device_params:
            device_params[device] = 0
        device_params[device] += param.numel()
        total_params += param.numel()
    
    print(f"  Total Parameters: {total_params / 1e9:.2f}B")
    for device, count in device_params.items():
        pct = (count / total_params) * 100
        print(f"    {device}: {count / 1e9:.2f}B ({pct:.1f}%)")
