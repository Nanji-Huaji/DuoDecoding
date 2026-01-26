import subprocess
import os
import json
import threading
import time
from pathlib import Path

from typing import TypedDict

from typing import Literal

from typing import List

from typing import Dict

from datetime import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed


class ExpConfig(TypedDict):
    CUDA_VISIBLE_DEVICES: Literal["0", "1"]
    eval_mode: str
    edge_end_bandwidth: int | float
    edge_cloud_bandwidth: int | float
    cloud_end_bandwidth: int | float
    transfer_top_k: int
    exp_name: str
    use_precise: bool
    ntt_ms_edge_cloud: int | float
    ntt_ms_edge_end: int | float
    small_draft_threshold: float
    draft_target_threshold: float
    use_rl_adapter: bool
    disable_rl_update: bool


# Global Constants

NTT_MS_EDGE_CLOUD = 0
NTT_MS_EDGE_END = 0

cmd_temp = """
echo "Running experiment: {eval_mode}"
CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} accelerate launch \
    --num_processes 1 \
    --main_process_port 29051 \
    eval/eval_mt_bench.py \
    --eval_mode {eval_mode} \
    -e llama \
    --task_name mt_bench \
    --draft_model tiny-llama-1.1b \
    --target_model Llama-2-13b \
    --little_model llama-68m \
    --max_tokens 128 \
    --temp 0.0 \
    --gamma1 4 \
    --gamma2 26 \
    --edge_end_bandwidth {edge_end_bandwidth} \
    --edge_cloud_bandwidth {edge_cloud_bandwidth} \
    --cloud_end_bandwidth {cloud_end_bandwidth} \
    --transfer_top_k {transfer_top_k} \
    --exp_name {exp_name} \
    --ntt_ms_edge_cloud {ntt_ms_edge_cloud} \
    --ntt_ms_edge_end {ntt_ms_edge_end} \
    --small_draft_threshold {small_draft_threshold} \
    --draft_target_threshold {draft_target_threshold} \
"""


def add_args(
    base_cmd: str, extra_arg: str, value_of_extra_args: str | None = None
) -> str:
    # 修复语法错误
    return (
        base_cmd.rstrip()
        + f" \\\n    --{extra_arg} {value_of_extra_args if value_of_extra_args is not None else ''}"
    )


def get_file_path(exp_name: str) -> str:
    dir_path = f"exp/{exp_name}"
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith("_metrics.json"):  # 修复语法错误
                return os.path.join(root, file)
    print(f"File not found for {exp_name}")
    return ""


def run_exp(config: ExpConfig, log_dir: str = "logs") -> dict:
    """运行实验并重定向日志"""
    # 创建日志目录
    Path(log_dir).mkdir(exist_ok=True)

    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        log_dir, f"{config['exp_name'].replace('/', '_')}_{timestamp}.log"
    )

    cmd = cmd_temp.format(**config)
    if config.get("use_precise", False):
        cmd = add_args(cmd, "use_precise")
    if config.get("use_rl_adapter", False):
        cmd = add_args(cmd, "use_rl_adapter")
    if config.get("disable_rl_update", False):
        cmd = add_args(cmd, "disable_rl_update")

    print(
        f"开始实验: {config['exp_name']}, GPU: {config['CUDA_VISIBLE_DEVICES']}"
    )
    print(f"日志文件: {log_file}")

    try:
        # 重定向输出到日志文件
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(
                f"实验配置: {json.dumps(config, indent=2, ensure_ascii=False)}\n"
            )
            f.write(f"执行命令: {cmd}\n")
            f.write("=" * 80 + "\n")

            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        # 读取结果文件
        result_file = get_file_path(config["exp_name"])
        if result_file:
            with open(result_file, "r") as f:
                result_data = f.read()

            # 尝试解析JSON字符串为字典
            try:
                parsed_result = json.loads(result_data)
                return {
                    "exp_name": config["exp_name"],
                    "result": parsed_result,  # 现在是字典对象
                    "log_file": log_file,
                    "status": "success",
                }
            except json.JSONDecodeError as e:
                return {
                    "exp_name": config["exp_name"],
                    "result": {
                        "error": "JSON解析失败",
                        "raw_data": result_data,
                    },
                    "log_file": log_file,
                    "status": "json_error",
                }
        else:
            return {
                "exp_name": config["exp_name"],
                "result": {"error": "结果文件未找到"},
                "log_file": log_file,
                "status": "no_result",
            }

    except subprocess.CalledProcessError as e:
        error_msg = f"实验失败，错误代码: {e.returncode}"
        print(f"实验 {config['exp_name']} 失败: {error_msg}")

        if os.path.exists(log_file):
            print(f"--- Error Log Content ({log_file}) ---")
            with open(log_file, "r", encoding="utf-8") as f:
                print(f.read())
            print(f"--- End Error Log ---")

        return {
            "exp_name": config["exp_name"],
            "result": {"error": error_msg},
            "log_file": log_file,
            "status": "failed",
        }


class GPUManager:
    def __init__(self):
        self.available_gpus = set(self.get_available_gpus())
        self.lock = threading.Lock()
        print(f"初始化GPU管理器，可用GPU: {sorted(self.available_gpus)}")

    def get_available_gpus(self) -> List[int]:
        """检测完全空闲的GPU"""
        import pynvml

        pynvml.nvmlInit()
        available_gpus = []

        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # 检查显存使用情况
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = mem_info.used / 1024 / 1024

            # 检查GPU利用率
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util_info.gpu

            # 如果显存使用小于1024MB且GPU利用率小于5%，认为是空闲的
            if mem_used_mb < 1024 and gpu_util < 5:
                available_gpus.append(i)
                print(
                    f"GPU {i}: 可用 (显存: {mem_used_mb:.1f}MB, 利用率: {gpu_util}%)"
                )
            else:
                print(
                    f"GPU {i}: 忙碌 (显存: {mem_used_mb:.1f}MB, 利用率: {gpu_util}%)"
                )

        return available_gpus

    def acquire_gpu(self) -> int | None:
        """获取一个可用的GPU"""
        with self.lock:
            if self.available_gpus:
                gpu_id = self.available_gpus.pop()
                print(f"分配GPU {gpu_id}")
                return gpu_id
            return None

    def release_gpu(self, gpu_id: int):
        """释放GPU"""
        with self.lock:
            self.available_gpus.add(gpu_id)
            print(f"释放GPU {gpu_id}")

    def has_available_gpu(self) -> bool:
        """检查是否有可用GPU"""
        with self.lock:
            return len(self.available_gpus) > 0


def run_experiment_with_gpu(
    config: ExpConfig, gpu_manager: GPUManager, log_dir: str = "logs"
) -> dict:
    """在指定GPU上运行实验"""
    gpu_id = None
    try:
        # 等待可用GPU
        while gpu_id is None:
            gpu_id = gpu_manager.acquire_gpu()
            if gpu_id is None:
                print(f"等待可用GPU运行实验: {config['exp_name']}")
                time.sleep(5)

        # 更新配置中的GPU设备
        config["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # 运行实验
        result = run_exp(config, log_dir)
        # 添加配置参数到结果中
        result["config"] = config.copy()
        return result

    finally:
        # 释放GPU
        if gpu_id is not None:
            gpu_manager.release_gpu(gpu_id)


def run_experiments_parallel(
    configs: List[ExpConfig], max_workers: int = 2, log_dir: str = "logs"
) -> List[dict]:
    """并行运行多个实验"""
    gpu_manager = GPUManager()

    # 检查是否有足够的GPU
    if len(gpu_manager.available_gpus) == 0:
        print("错误: 没有可用的GPU")
        return []

    print(
        f"将并行运行 {len(configs)} 个实验，最大并发数: {min(max_workers, len(gpu_manager.available_gpus))}"
    )

    all_results = []
    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(gpu_manager.available_gpus))
    ) as executor:
        # 提交所有任务
        future_to_config = {
            executor.submit(
                run_experiment_with_gpu, config, gpu_manager, log_dir
            ): config
            for config in configs
        }

        # 收集结果
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                all_results.append(result)
                print(
                    f"实验完成: {config['exp_name']}, 状态: {result['status']}"
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                error_result = {
                    "exp_name": config["exp_name"],
                    "result": f"执行异常: {exc}",
                    "log_file": "",
                    "status": "exception",
                }
                all_results.append(error_result)
                print(f"实验异常: {config['exp_name']}, 错误: {exc}")

    return all_results


def get_available_gpus() -> List[int]:
    """检测完全空闲的GPU"""
    import pynvml

    pynvml.nvmlInit()
    available_gpus = []

    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        # 检查显存使用情况
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_mb = mem_info.used / 1024 / 1024  # type: ignore

        # 检查GPU利用率
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util_info.gpu

        # 如果显存使用小于1024MB且GPU利用率小于5%，认为是空闲的
        if mem_used_mb < 1024 and gpu_util < 5:  # type: ignore
            available_gpus.append(i)
            print(
                f"GPU {i}: 可用 (显存: {mem_used_mb:.1f}MB, 利用率: {gpu_util}%)"
            )
        else:
            print(
                f"GPU {i}: 忙碌 (显存: {mem_used_mb:.1f}MB, 利用率: {gpu_util}%)"
            )

    return available_gpus


def create_config(
    eval_mode: str,
    ntt_ms_edge_cloud: int | float = NTT_MS_EDGE_CLOUD,
    ntt_ms_edge_end: int | float = NTT_MS_EDGE_END,
    use_precise: bool = True,
    CUDA_VISIBLE_DEVICES: Literal["0", "1"] = "0",
    edge_end_bandwidth: int | float=100,
    edge_cloud_bandwidth: int | float=100,
    cloud_end_bandwidth: int | float=100,
    transfer_top_k: int = 300,
    small_draft_threshold: float = 0.8,
    draft_target_threshold: float = 0.8,
    use_rl_adapter: bool = False,
    disable_rl_update: bool = False,
) -> ExpConfig:
    # 使用包含微秒的时间戳以确保在循环中生成的配置具有唯一的 exp_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return ExpConfig(
        CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
        eval_mode=eval_mode,
        edge_end_bandwidth=edge_end_bandwidth,
        edge_cloud_bandwidth=edge_cloud_bandwidth,
        cloud_end_bandwidth=cloud_end_bandwidth,
        transfer_top_k=transfer_top_k,
        exp_name=f"{eval_mode}/{eval_mode}_{timestamp}",
        use_precise=use_precise,
        ntt_ms_edge_cloud=ntt_ms_edge_cloud,
        ntt_ms_edge_end=ntt_ms_edge_end,
        small_draft_threshold=small_draft_threshold,
        draft_target_threshold=draft_target_threshold,
        use_rl_adapter=use_rl_adapter,
        disable_rl_update=disable_rl_update,
    )


bandwidth_config = [
    (563, 34.6),
    # (350, 25.0),
    # (200, 15.0),
    # (100, 5.0),
    # (33.2, 0.14),
]

# 在此添加实验

config_to_run = []

edge_end_bw = 563
edge_cloud_bw = 34.6
transfer_top_k = 300

threshold_config = [
    (little_draft / 10, draft_target / 10) for little_draft in range(1, 8, 1) for draft_target in range(1, 8, 1)
]

for small_draft_threshold, draft_target_threshold in threshold_config:
    config_to_run.append(
        create_config(
            eval_mode="adaptive_tridecoding",
            ntt_ms_edge_cloud=NTT_MS_EDGE_CLOUD,
            ntt_ms_edge_end=NTT_MS_EDGE_END,
            use_precise=False,
            edge_end_bandwidth=edge_end_bw,
            edge_cloud_bandwidth=edge_cloud_bw,
            cloud_end_bandwidth=edge_cloud_bw,
            transfer_top_k=transfer_top_k,
        )
        )
    config_to_run[-1]["small_draft_threshold"] = small_draft_threshold
    config_to_run[-1]["draft_target_threshold"] = draft_target_threshold

if __name__ == "__main__":
    # 创建日志目录
    log_dir = "exp_logs"
    Path(log_dir).mkdir(exist_ok=True)

    # 并行运行实验
    all_results = run_experiments_parallel(
        config_to_run, max_workers=2, log_dir=log_dir
    )

    # 保存汇总结果
    summary_file = (
        f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印汇总报告
    print("\n" + "=" * 80)
    print("实验汇总报告:")
    print("=" * 80)

    successful = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] == "failed")
    no_result = sum(1 for r in all_results if r["status"] == "no_result")
    exception = sum(1 for r in all_results if r["status"] == "exception")

    print(f"总实验数: {len(all_results)}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    print(f"无结果: {no_result}")
    print(f"异常: {exception}")
    print(f"\n汇总结果已保存到: {summary_file}")

    for result in all_results:
        print(f"\n实验: {result['exp_name']}")
        print(f"状态: {result['status']}")
        if result.get("log_file"):
            print(f"日志: {result['log_file']}")
