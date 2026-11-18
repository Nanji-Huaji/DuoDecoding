"""Persistent eval.worker processes and exclusive leases on their device sets."""

import atexit
import json
import os
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path

from .config import ExpConfig


class _WorkerHandle:
    """One JSON request/response stream; only the child loads models."""

    def __init__(self, devices: str):
        env = os.environ.copy()
        root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
        env["CUDA_VISIBLE_DEVICES"] = devices
        env["PYTHONIOENCODING"] = "utf-8"
        self._lock = threading.Lock()
        self._broken = False
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "eval.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="utf-8",
            env=env,
            start_new_session=True,
        )

    @property
    def alive(self) -> bool:
        return not self._broken and self.proc.poll() is None

    def kill(self) -> None:
        # start_new_session makes the child the process group leader. Do not
        # take the I/O lock: a job may be blocked waiting for this very child.
        self._broken = True
        with suppress(OSError):
            os.killpg(self.proc.pid, signal.SIGKILL)

    def close(self) -> None:
        with self._lock, suppress(OSError, ValueError):
            self.proc.stdin.close()  # EOF lets eval.worker release its ModelStore.
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.kill()
            with suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=30)
        with suppress(OSError, ValueError):
            self.proc.stdout.close()

    def run(self, config: ExpConfig, log_dir: str) -> dict:
        name = config["exp_name"]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = Path(log_dir) / f"{name.replace('/', '_')}_{stamp}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"开始实验: {name}, GPU: {config['CUDA_VISIBLE_DEVICES']}")
        print(f"日志文件: {path}")
        job = json.dumps({"config": config, "log_file": str(path)}, ensure_ascii=False)
        with self._lock:
            try:
                if not self.alive:
                    raise BrokenPipeError("worker exited")
                self.proc.stdin.write(job + "\n")
                self.proc.stdin.flush()
                result = json.loads(self.proc.stdout.readline())
                if not isinstance(result, dict) or result.get("exp_name") != name:
                    raise ValueError("invalid response from eval.worker")
                if not {"status", "result", "log_file"} <= result.keys():
                    raise ValueError("incomplete response from eval.worker")
            except (OSError, ValueError):
                code = self.proc.poll()
                self.kill()  # EOF or a broken protocol: never reuse this stream.
                result = {
                    "exp_name": name,
                    "status": "failed",
                    "log_file": str(path) if path.exists() else "",
                    "result": {
                        "error": f"评测 worker 未返回有效结果（退出码 {code}）；"
                        "本配置未完成，可用 resume 重试"
                    },
                }
            except BaseException:
                self.kill()
                raise

        if result["status"] == "failed":
            error = result["result"]
            message = (
                error.get("error", "未知错误") if isinstance(error, dict) else error
            )
            print(f"实验 {name} 失败: {message}")
            if path.is_file():
                print(f"--- Error Log Content ({path}) ---")
                print(path.read_text(encoding="utf-8", errors="replace"))
                print("--- End Error Log ---")
        return result


def _device_ids(devices: str | int) -> tuple[str, ...]:
    # Keep CUDA device *order* for placement, but test overlap by membership.
    return tuple(part.strip() for part in str(devices).split(","))


class _WorkerPool:
    """Reuse resident workers on the same devices; never overlap active leases."""

    def __init__(self):
        self._condition = threading.Condition()
        self._workers: dict[tuple[str, ...], _WorkerHandle] = {}
        self._busy: set[str] = set()
        self.stopped = threading.Event()

    @contextmanager
    def acquire(self, config: ExpConfig):
        ids = _device_ids(config["CUDA_VISIBLE_DEVICES"])
        with self._condition:
            self._condition.wait_for(
                lambda: self.stopped.is_set() or self._busy.isdisjoint(ids)
            )
            if self.stopped.is_set():
                raise RuntimeError("worker pool stopped")
            for old_ids, worker in list(self._workers.items()):
                if not set(ids).isdisjoint(old_ids) and (
                    old_ids != ids or not worker.alive
                ):
                    self._workers.pop(old_ids).close()
            if ids not in self._workers:
                self._workers[ids] = _WorkerHandle(",".join(ids))
            worker = self._workers[ids]
            self._busy.update(ids)
        try:
            yield worker
        finally:
            with self._condition:
                self._busy.difference_update(ids)
                self._condition.notify_all()

    def retire(self, devices: str) -> None:
        """End a matrix group before its scheduler releases these GPUs."""
        ids = _device_ids(devices)
        with self._condition:
            worker = self._workers.get(ids)
        if worker is not None:
            # Remain visible to interrupt() while waiting for graceful exit.
            worker.close()
            with self._condition:
                del self._workers[ids]

    def interrupt(self) -> None:
        """Stop new jobs and unblock active ones, including their subprocess trees."""
        with self._condition:
            self.stopped.set()
            for worker in self._workers.values():
                worker.kill()
            self._condition.notify_all()

    def shutdown(self) -> None:
        with self._condition:
            self.stopped.set()
            handles = list(self._workers.values())
            self._workers.clear()
            self._condition.notify_all()
        for worker in handles:
            worker.close()


# Explicit-device callers (e.g. threshold scans) keep idle workers between calls.
_WORKER_POOL = _WorkerPool()
atexit.register(_WORKER_POOL.shutdown)


def run_exp(config: ExpConfig, log_dir: str = "logs") -> dict:
    """Run one explicitly placed job and return its own result, never old metrics."""
    with _WORKER_POOL.acquire(config) as worker:
        return worker.run(config, log_dir)
