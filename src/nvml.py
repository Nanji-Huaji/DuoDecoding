from __future__ import annotations

from typing import List

import pynvml


def get_available_gpus(
    max_used_memory_mb: float | None = None,
    max_gpu_utilization: float = 5,
) -> List[int]:
    """Return GPU ids that look idle enough for running a new experiment.

    ``max_used_memory_mb`` 缺省 1024，可被环境变量 GPU_MAX_USED_MEMORY_MB
    覆盖（GPU 上有他人常驻小服务时调高，如 8192）。
    """
    import os

    if max_used_memory_mb is None:
        max_used_memory_mb = float(
            os.environ.get("GPU_MAX_USED_MEMORY_MB", "1024")
        )
    pynvml.nvmlInit()
    try:
        available_gpus: List[int] = []
        device_count = pynvml.nvmlDeviceGetCount()

        for gpu_id in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = mem_info.used / 1024 / 1024
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util_info.gpu

            if (
                mem_used_mb < max_used_memory_mb
                and gpu_util < max_gpu_utilization
            ):
                available_gpus.append(gpu_id)
                print(
                    f"GPU {gpu_id}: 可用 (显存: {mem_used_mb:.1f}MB, 利用率: {gpu_util}%)"
                )
            else:
                print(
                    f"GPU {gpu_id}: 忙碌 (显存: {mem_used_mb:.1f}MB, 利用率: {gpu_util}%)"
                )

        return available_gpus
    finally:
        pynvml.nvmlShutdown()
