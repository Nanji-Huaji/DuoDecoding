"""Run compatible configs consecutively on one exclusive allocation per group."""

import queue
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor

from .config import ExpConfig
from .summaries import save_summary
from .workers import _WorkerPool


def get_available_gpus() -> list[int]:
    """Probe idle GPUs only when the caller actually needs a device."""
    from src.nvml import get_available_gpus as probe

    return probe()


class GPUManager:
    """Track exclusive allocations from the GPUs idle when this matrix starts."""

    def __init__(self):
        self.all_gpu_ids = set(self.get_available_gpus())
        self.available_gpus = self.all_gpu_ids.copy()
        self.lock = threading.Lock()
        print(f"初始化GPU管理器，可用GPU: {sorted(self.available_gpus)}")

    def get_available_gpus(self) -> list[int]:
        return get_available_gpus()

    def acquire_gpu(self, count: int = 1) -> list[int] | None:
        with self.lock:
            if len(self.available_gpus) < count:
                return None
            selected = sorted(self.available_gpus)[:count]
            self.available_gpus.difference_update(selected)
            print(f"分配GPU {selected}")
            return selected

    def release_gpu(self, gpu_ids: int | list[int]) -> None:
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        with self.lock:
            self.available_gpus.update(gpu_ids)
            print(f"释放GPU {gpu_ids}")

    def has_available_gpu(self) -> bool:
        with self.lock:
            return bool(self.available_gpus)


def _group_configs_by_model(configs: list[ExpConfig]) -> dict:
    from eval.decoder import model_key

    groups = {}
    for config in configs:
        try:
            key = model_key(config)
        except (ValueError, KeyError):
            # Let eval.worker report the invalid config in its own job log.
            key = None
        groups.setdefault(key, []).append(config)
    return groups


def _needed_gpus(key) -> int:
    """Only paths for roles whose weights will actually be loaded count."""
    paths = " ".join(path.lower() for _, path, _ in key.weights) if key else ""
    if "70b" in paths:
        return 4
    return 2 if "27b" in paths or "32b" in paths else 1


def _run_group(group, needed, gpus, pool, log_dir, emit):
    capacity = len(gpus.all_gpu_ids)
    if needed > capacity:
        error = (
            f"跳过实验 {group[0]['exp_name']} 等 {len(group)} 个同组配置: "
            f"需要 {needed} 个GPU，但管理器只追踪了 {capacity} 个GPU"
        )
        print(error)
        for config in group:
            emit(
                {
                    "exp_name": config["exp_name"],
                    "result": {"error": error},
                    "log_file": "",
                    "status": "skipped_capacity_mismatch",
                    "config": config.copy(),
                }
            )
        return

    if pool.stopped.is_set():
        return
    ids = gpus.acquire_gpu(needed)
    if ids is None:
        print(f"等待可用GPU运行实验组: {group[0]['exp_name']} (需要 {needed} GPUs)")
    while ids is None:
        if pool.stopped.wait(1):
            return
        ids = gpus.acquire_gpu(needed)

    devices = ",".join(map(str, ids))
    try:
        for original in group:
            if pool.stopped.is_set():
                break
            config = original.copy()
            config["CUDA_VISIBLE_DEVICES"] = devices
            try:
                with pool.acquire(config) as worker:
                    result = worker.run(config, log_dir)
            except Exception as exc:
                if pool.stopped.is_set():
                    break
                traceback.print_exc()
                result = {
                    "exp_name": config["exp_name"],
                    "result": f"执行异常: {exc}",
                    "log_file": "",
                    "status": "exception",
                }
            result["config"] = config
            emit(result)
    finally:
        try:
            pool.retire(devices)
        finally:
            gpus.release_gpu(ids)


def run_experiments_parallel(
    configs: list[ExpConfig],
    max_workers: int = 2,
    log_dir: str = "logs",
    summary_file: str | None = None,
) -> list[dict]:
    """Keep one worker hot per group; checkpoint each result in completion order."""
    if not configs:
        return []
    gpus = GPUManager()
    if not gpus.has_available_gpu():
        print("错误: 没有可用的GPU")
        return []

    from tqdm import tqdm

    groups = _group_configs_by_model(configs)
    parallelism = min(max_workers, len(gpus.all_gpu_ids))
    print(
        f"将并行运行 {len(configs)} 个实验（{len(groups)} 个模型组），"
        f"最大并发数: {parallelism}"
    )

    pool = _WorkerPool()
    events = queue.SimpleQueue()
    results = []
    executor = ThreadPoolExecutor(max_workers=parallelism)
    try:
        for key, group in groups.items():
            future = executor.submit(
                _run_group, group, _needed_gpus(key), gpus, pool, log_dir, events.put
            )
            # One FIFO for per-config results and group completions. Unlike a
            # bare sentinel, a Future also reports unexpected scheduler errors.
            future.add_done_callback(events.put)

        remaining = len(groups)
        with tqdm(total=len(configs), desc="Running Experiments") as bar:
            while remaining:
                event = events.get()
                if isinstance(event, Future):
                    event.result()
                    remaining -= 1
                    continue
                results.append(event)
                bar.update(1)
                print(f"实验完成: {event['exp_name']}, 状态: {event['status']}")
                if summary_file:
                    save_summary(summary_file, results)
    except BaseException:
        pool.interrupt()
        raise
    finally:
        executor.shutdown(cancel_futures=True)
        pool.shutdown()
    return results
