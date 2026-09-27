#!/usr/bin/env python3
"""守门：等目标 GPU 真正空闲再放行实验。

动机
----
本机长期有第三方任务（`lqh` 的 TTS 抽取、`yhy` 的 ray dashboard）。同一份配置
在不同时段测出的吞吐可以差 ±44%（14.37 / 11.41 / 10.15 / 9.13 tok/s），所以
"跑之前先确认机器状态"必须变成脚本，而不是靠人记得看一眼 nvidia-smi。

判定标准（全部满足才算空闲）
---------------------------
  1. GPU 利用率连续 N 次采样都低于阈值；
  2. 已用显存低于阈值（第三方可能只占显存不算，或反过来）；
  3. **本脚本不检查 CPU load**，只报告 —— 因为 CPU 争抢会拖慢但不会算错，
     而且 CUDA Graph 已经把解码循环对 CPU 的敏感度降低了约 4 倍。

退出码：0 = 已空闲（放行）；1 = 超时（未放行）。

用法
----
  .venv/bin/python scripts/wait_gpu_idle.py --gpu 0 --timeout 7200
  .venv/bin/python scripts/wait_gpu_idle.py --gpu 0 --util-threshold 5 --samples 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _q(args: list[str]) -> str:
    return subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True, timeout=20
    ).stdout.strip()


def gpu_state(index: int) -> tuple[int, int, list[tuple[int, int]]]:
    """返回 (利用率 %, 已用显存 MiB, [(pid, MiB), ...] 该卡上的计算进程)。"""
    uuid = _q(["--query-gpu=uuid", "--format=csv,noheader", "-i", str(index)])
    util, mem = (
        _q(
            [
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(index),
            ]
        ).split(", ")[:2]
    )
    procs: list[tuple[int, int]] = []
    for line in _q(
        [
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0] == uuid:
            try:
                procs.append((int(parts[1]), int(parts[2])))
            except ValueError:
                pass
    return int(util), int(mem), procs


def load_avg() -> str:
    try:
        return f"{open('/proc/loadavg').read().split()[0]}"
    except Exception:  # noqa: BLE001
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description="等 GPU 空闲再放行")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--util-threshold", type=int, default=10, help="利用率上限 %")
    ap.add_argument("--mem-threshold", type=int, default=8000, help="已用显存上限 MiB")
    ap.add_argument("--samples", type=int, default=4, help="需要连续满足的采样次数")
    ap.add_argument("--interval", type=int, default=15, help="采样间隔秒")
    ap.add_argument("--timeout", type=int, default=7200, help="总超时秒")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    ok = 0
    while True:
        util, mem, procs = gpu_state(a.gpu)
        others = [(p, m) for p, m in procs if p != 0]
        idle = util <= a.util_threshold and mem <= a.mem_threshold
        ok = ok + 1 if idle else 0
        elapsed = int(time.time() - t0)
        if not a.quiet:
            who = ",".join(f"pid{p}({m}MiB)" for p, m in others) or "无"
            print(
                f"  [{time.strftime('%H:%M:%S')} +{elapsed}s] GPU{a.gpu} "
                f"util={util:>3}% mem={mem:>6}MiB load={load_avg():<5} "
                f"他人进程={who}  达标 {ok}/{a.samples}",
                flush=True,
            )
        if ok >= a.samples:
            print(
                f"### GPU{a.gpu} 已空闲（util<={a.util_threshold}% 且 "
                f"mem<={a.mem_threshold}MiB 连续 {a.samples} 次）—— 放行",
                flush=True,
            )
            return 0
        if elapsed >= a.timeout:
            print(
                f"### 等待 {a.timeout}s 仍未空闲（当前 util={util}% mem={mem}MiB）"
                f" —— 不放行",
                flush=True,
            )
            return 1
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
