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

from enum import Enum

from tqdm import tqdm


class EvalDataset(str, Enum):
    mt_bench = "eval/eval_mt_bench_noeval.py"
    humaneval = "eval/eval_humaneval.py"
    # cnndm = "eval/eval_cnndm.py"
    # xsum = "eval/eval_xsum.py"
    gsm8k = "eval/eval_gsm8k.py"


class EvalMode(str, Enum):
    autoregression = "large"
    dssd = "dist_split_spec"
    dsd = "dist_spec"
    cuhlm = "uncertainty_decoding"
    # tridecoding = "tridecoding" # ablation of adaptive tridecoding
    # # cee_sd_without_arp = "ceesd_without_arp"  # ours without arp
    ceesd = "adaptive_tridecoding"  # ours
    cee_cuhlm = "cee_cuhlm"


model_acc_head_map = {
    "llama-2-7b-chat": "src/SpecDec_pp/checkpoints/llama-2-chat-7b/exp-weight6-layer3",
    "tiny-llama-1.1b": "src/SpecDec_pp/checkpoints/llama-1.1b/exp-weight6-layer3",
    "llama-2-13b": "src/SpecDec_pp/checkpoints/llama-13b/exp-weight6-layer3",
    "vicuna-13b-v1.5": "src/SpecDec_pp/checkpoints/vicuna-v1.5-13b/exp-weight6-layer3",
    "tiny-vicuna-1b": "src/SpecDec_pp/checkpoints/tiny-vicuna-1b/exp-weight6-layer3",
    "qwen/Qwen3-14B": "src/SpecDec_pp/checkpoints/qwen-3-14b/exp-weight6-layer3",
    "qwen/Qwen3-1.7B": "src/SpecDec_pp/checkpoints/qwen-3-1.7b/exp-weight6-layer3",
    "llama-2-chat-70b": "src/SpecDec_pp/checkpoints/llama-2-chat-70b/exp-weight6-layer3",
}


class ExpConfig(TypedDict):
    CUDA_VISIBLE_DEVICES: str | int
    eval_mode: str | EvalMode
    edge_end_bandwidth: int | float
    edge_cloud_bandwidth: int | float
    cloud_end_bandwidth: int | float
    transfer_top_k: int
    num_shots: int
    exp_name: str
    use_precise: bool
    use_stochastic_comm: bool
    use_rl_adapter: bool
    disable_rl_update: bool
    small_draft_threshold: float
    draft_target_threshold: float
    ntt_ms_edge_cloud: int | float
    ntt_ms_edge_end: int | float
    eval_dataset: EvalDataset
    draft_model: str
    target_model: str
    little_model: str
    small_draft_acc_head_path: str
    draft_target_acc_head_path: str
    main_rl_path: str
    little_rl_path: str
    max_tokens: int
    use_early_stopping: bool


# Global Constants

NTT_MS_EDGE_CLOUD = 10
NTT_MS_EDGE_END = 0

cmd_temp = """
echo "Running experiment: {eval_mode}"
CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} /home/tiantianyi/code/DuoDecoding/.venv/bin/accelerate launch \\
    --num_processes 1 \
    --main_process_port 29051 \
    {eval_dataset} \
    --eval_mode {eval_mode} \
    -e llama \
    --draft_model {draft_model} \
    --target_model {target_model} \
    --little_model {little_model} \
    --max_tokens {max_tokens} \
    --temp 0.0 \
    --gamma1 5 \
    --gamma2 5 \
    --edge_end_bandwidth {edge_end_bandwidth} \
    --edge_cloud_bandwidth {edge_cloud_bandwidth} \
    --cloud_end_bandwidth {cloud_end_bandwidth} \
    --transfer_top_k {transfer_top_k} \
    --small_draft_threshold {small_draft_threshold} \
    --draft_target_threshold {draft_target_threshold} \
    --num_shots {num_shots} \
    --exp_name {exp_name} \
    --ntt_ms_edge_cloud {ntt_ms_edge_cloud} \
    --ntt_ms_edge_end {ntt_ms_edge_end} \
"""


def add_args(
    base_cmd: str, extra_arg: str, value_of_extra_args: str | None = None
) -> str:
    return (
        base_cmd.rstrip()
        + f" \\\n    --{extra_arg} {value_of_extra_args if value_of_extra_args is not None else ''}"
    )


def get_file_path(exp_name: str) -> str:
    # 原始逻辑：假设 exp_name 对应一个具体的文件夹
    target_dir = f"exp/{exp_name}"

    # 1. 如果 exp/{exp_name} 确实是一个文件夹，直接遍历
    if os.path.isdir(target_dir):
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith("_metrics.json"):
                    return os.path.join(root, file)

    # 2. 如果文件夹不存在，可能是 exp_name 包含了 "父目录/文件前缀" 的结构
    # 例如 exp_name="dssd/dssd_timestamp"，文件可能在 "exp/dssd/" 下
    if "/" in exp_name:
        parent_dir, file_prefix = os.path.split(exp_name)
        search_dir = os.path.join("exp", parent_dir)

        if os.path.isdir(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    # 检查文件名是否包含前缀 (即 timestamp 部分) 且以后缀结尾
                    if file_prefix in file and file.endswith("_metrics.json"):
                        return os.path.join(root, file)

    print(f"File not found for {exp_name}")
    return ""


def get_model_series(model_name: str) -> str:
    """根据模型名称识别所属系列"""
    name = model_name.lower()
    if "llama" in name:
        return "llama"
    elif "vicuna" in name:
        return "vicuna"
    elif "qwen" in name:
        return "qwen"
    return "unknown"


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
    if config.get("use_stochastic_comm", False):
        cmd = add_args(cmd, "use_stochastic_comm")
    if config.get("use_rl_adapter", False):
        cmd = add_args(cmd, "use_rl_adapter")
    if config.get("disable_rl_update", False):
        cmd = add_args(cmd, "disable_rl_update")
    if config.get("use_early_stopping", False):
        cmd = add_args(cmd, "use_early_stopping")
    if config.get("small_draft_acc_head_path"):
        cmd = add_args(
            cmd,
            "small_draft_acc_head_path",
            config["small_draft_acc_head_path"],
        )
    if config.get("draft_target_acc_head_path"):
        cmd = add_args(
            cmd,
            "draft_target_acc_head_path",
            config["draft_target_acc_head_path"],
        )
    if config.get("main_rl_path"):
        cmd = add_args(
            cmd,
            "main_rl_path",
            config["main_rl_path"],
        )
    if config.get("little_rl_path"):
        cmd = add_args(
            cmd,
            "little_rl_path",
            config["little_rl_path"],
        )

    # Derive task_name based on eval_dataset or manually
    script_path = str(config.get("eval_dataset", ""))
    task_name = "unknown"
    if "mt_bench" in script_path:
        task_name = "mt_bench"
    elif "humaneval" in script_path:
        task_name = "humaneval"
    elif "cnndm" in script_path:
        task_name = "cnndm"
    elif "xsum" in script_path:
        task_name = "xsum"
    elif "gsm8k" in script_path:
        task_name = "gsm8k"
    elif "specbench" in script_path:
        task_name = "specbench"
    elif "mixed" in script_path:
        task_name = "mixed"

    cmd = add_args(cmd, "task_name", task_name)

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

    def acquire_gpu(self, count: int = 1) -> List[int] | None:
        """获取 count 个可用的GPU"""
        with self.lock:
            if len(self.available_gpus) >= count:
                # 优先选择利用率低的（即ID较小的，因为get_available_gpus已经过滤了）
                selected = sorted(list(self.available_gpus))[:count]
                for gpu in selected:
                    self.available_gpus.remove(gpu)
                print(f"分配GPU {selected}")
                return selected
            return None

    def release_gpu(self, gpu_ids: int | List[int]):
        """释放GPU"""
        with self.lock:
            if isinstance(gpu_ids, int):
                gpu_ids = [gpu_ids]
            for gpu_id in gpu_ids:
                self.available_gpus.add(gpu_id)
            print(f"释放GPU {gpu_ids}")

    def has_available_gpu(self) -> bool:
        """检查是否有可用GPU"""
        with self.lock:
            return len(self.available_gpus) > 0


def run_experiment_with_gpu(
    config: ExpConfig, gpu_manager: GPUManager, log_dir: str = "logs"
) -> dict:
    """在指定GPU上运行实验"""
    gpu_ids = None

    # 检测是否为70b模型实验
    is_large_model = "70b" in str(config).lower()
    needed_gpus = 4 if is_large_model else 1

    try:
        if is_large_model:
            # 70b模型：如果资源不足直接跳过
            gpu_ids = gpu_manager.acquire_gpu(needed_gpus)
            if gpu_ids is None:
                msg = f"跳过实验 {config['exp_name']}: 70b模型需要{needed_gpus}个GPU，资源不足"
                print(msg)
                return {
                    "exp_name": config["exp_name"],
                    "result": {"error": msg},
                    "log_file": "",
                    "status": "skipped",
                    "config": config.copy(),
                }
        else:
            # 普通模型：等待直到有可用GPU
            while gpu_ids is None:
                gpu_ids = gpu_manager.acquire_gpu(needed_gpus)
                if gpu_ids is None:
                    print(f"等待可用GPU运行实验: {config['exp_name']}")
                    time.sleep(5)

        # 更新配置中的GPU设备
        config["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        # 运行实验
        result = run_exp(config, log_dir)
        # 添加配置参数到结果中
        result["config"] = config.copy()
        return result

    finally:
        # 释放GPU
        if gpu_ids is not None:
            gpu_manager.release_gpu(gpu_ids)


def run_experiments_parallel(
    configs: List[ExpConfig],
    max_workers: int = 2,
    log_dir: str = "logs",
    summary_file: str | None = None,
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
        for future in tqdm(
            as_completed(future_to_config),
            total=len(configs),
            desc="Running Experiments",
        ):
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

            # 中途保存结果
            if summary_file:
                with open(summary_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

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
    ntt_ms_edge_cloud: int | float = 0,
    ntt_ms_edge_end: int | float = 0,
    use_precise: bool = True,
    use_stochastic_comm: bool = False,
    CUDA_VISIBLE_DEVICES: str | int = "0",
    use_rl_adapter: bool = False,
    disable_rl_update: bool = False,
    edge_end_bandwidth: int | float = 100,
    edge_cloud_bandwidth: int | float = 100,
    cloud_end_bandwidth: int | float = 100,
    small_draft_threshold: float = 0.8,
    draft_target_threshold: float = 0.6,
    transfer_top_k: int = 300,
    num_shots: int = 3,
    max_tokens: int = 128,
    use_early_stopping: bool = False,
    eval_dataset: EvalDataset = EvalDataset.mt_bench,
    # 新添加的参数
    draft_model: str = "tiny-vicuna-1b",
    target_model: str = "vicuna-13b-v1.5",
    little_model: str = "vicuna-68m",
    small_draft_acc_head_path: str | None = None,
    draft_target_acc_head_path: str | None = None,
    main_rl_path: str | None = None,
    little_rl_path: str | None = None,
) -> ExpConfig:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ceesd, cee_cuhlm 需要用到 ARP 和 RL Adapter
    if eval_mode in [EvalMode.ceesd, EvalMode.cee_cuhlm]:
        if small_draft_acc_head_path is None:
            small_draft_acc_head_path = model_acc_head_map.get(draft_model, "")
        if draft_target_acc_head_path is None:
            draft_target_acc_head_path = model_acc_head_map.get(target_model, "")
        
        # 自动推导 RL Adapter 路径
        series = get_model_series(target_model)
        if main_rl_path is None:
            main_rl_path = f"checkpoints/{series}/rl_adapter_main.pth"
        if little_rl_path is None:
            little_rl_path = f"checkpoints/{series}/rl_adapter_little.pth"
            
    else:
        # 其他模式不需要
        small_draft_acc_head_path = ""
        draft_target_acc_head_path = ""
        main_rl_path = ""
        little_rl_path = ""

    return ExpConfig(
        eval_dataset=eval_dataset,
        CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
        eval_mode=eval_mode,
        edge_end_bandwidth=edge_end_bandwidth,
        edge_cloud_bandwidth=edge_cloud_bandwidth,
        cloud_end_bandwidth=cloud_end_bandwidth,
        transfer_top_k=transfer_top_k,
        num_shots=num_shots,
        max_tokens=max_tokens,
        small_draft_threshold=small_draft_threshold,
        draft_target_threshold=draft_target_threshold,
        exp_name=f"{eval_mode}/{eval_dataset.name}/{eval_mode}_{num_shots}shot_{timestamp}",
        use_precise=use_precise,
        use_stochastic_comm=use_stochastic_comm,
        ntt_ms_edge_cloud=ntt_ms_edge_cloud,
        ntt_ms_edge_end=ntt_ms_edge_end,
        use_rl_adapter=use_rl_adapter,
        disable_rl_update=disable_rl_update,
        draft_model=draft_model,
        target_model=target_model,
        little_model=little_model,
        use_early_stopping=use_early_stopping,
        small_draft_acc_head_path=small_draft_acc_head_path,
        draft_target_acc_head_path=draft_target_acc_head_path,
        main_rl_path=main_rl_path,
        little_rl_path=little_rl_path,
    )


config_to_run = []

llama_series = ("llama-68m", "tiny-llama-1.1b", "llama-2-13b")
vicuna_series = ("vicuna-68m", "tiny-vicuna-1b", "vicuna-13b-v1.5")
qwen_series = ("qwen/Qwen3-0.6B", "qwen/Qwen3-1.7B", "qwen/Qwen3-14B")
llama_chat_series = ("llama-68m", "llama-2-chat-7b", "llama-2-chat-70b")


# 根据 LaTeX 表格定义的模型组合 (Draft, Target)
# Row 1 (TinyLlama): Col 2 (TinyLlama-1B) = x, Col 3 (Llama-13B) = x
# Row 2 (Llama-68M): Col 3 (Llama-13B) = x
specified_pairs_llama = [
    ("llama-68m", "tiny-llama-1.1b"),
    ("tiny-llama-1.1b", "llama-2-13b"),
    ("llama-68m", "llama-2-13b"),
]

specified_pairs_vicuna = [
    ("tiny-vicuna-1b", "vicuna-13b-v1.5"),
    ("vicuna-68m", "vicuna-13b-v1.5"),
    ("vicuna-68m", "tiny-vicuna-1b"),
]

specified_pairs_qwen = [
    ("qwen/Qwen3-0.6B", "qwen/Qwen3-1.7B"),
    ("qwen/Qwen3-0.6B", "qwen/Qwen3-14B"),
    ("qwen/Qwen3-1.7B", "qwen/Qwen3-14B"),
]

edge_cloud_bandwidth = [
    # 5.2,  # 弱 4G / 强 3G 边缘
    # 7.8,  # 8M 宽带的典型实测值
    # 10.0,  # 标准 10M 宽带
    # 12.4,  # 繁忙时段的 4G LTE
    # 15.5,  # ADSL2+ 的理论极限附近
    # 18.2,  # 20M 宽带的各种损耗后速度
    # 20.0,  # 标准 20M 宽带
    23.6,  # 信号良好的 4G 平均值
    # 25.0,  # FCC 定义的宽带及格线
    # 28.9,  # 30M 宽带的一般表现
    # 32.4,  # Wi-Fi 穿墙后的衰减值
    # 38.7,  # 4G+ (载波聚合) 波动值
    # 42.1,  # 50M 宽带在高峰期的表现
    # 45.5,  # 50M 宽带 Wi-Fi 传输损耗值
    # 48.8,  # 50M 宽带非常接近满速的值
    # 50.0,  # 标准 50M 宽带满速
]

for little_model, draft_model, target_model in (
    llama_series,
    # vicuna_series,
    # qwen_series,
):
    for dataset in [EvalDataset.mt_bench,]:
        for mode in [EvalMode.cee_cuhlm,]:
            for edge_cloud_bw in edge_cloud_bandwidth:
                config = create_config(
                    eval_mode=mode,
                    ntt_ms_edge_cloud=NTT_MS_EDGE_CLOUD,
                    ntt_ms_edge_end=NTT_MS_EDGE_END,
                    use_precise=False,
                    use_stochastic_comm=True,
                    edge_end_bandwidth=563,
                    edge_cloud_bandwidth=edge_cloud_bw,
                    cloud_end_bandwidth=edge_cloud_bw,
                    small_draft_threshold=0.8,
                    draft_target_threshold=0.6,
                    transfer_top_k=1024,
                    max_tokens=128,
                    num_shots=8,
                    eval_dataset=dataset,
                    draft_model=draft_model if mode in [EvalMode.ceesd, EvalMode.cee_cuhlm] else little_model,
                    target_model=target_model,
                    little_model=little_model,
                    use_rl_adapter=True,
                    disable_rl_update=True,
                    use_early_stopping=False,
                )
                config_to_run.append(config)


if __name__ == "__main__":
    # 创建日志目录
    log_dir = "exp_logs"
    Path(log_dir).mkdir(exist_ok=True)

    # 提前确定汇总结果文件名
    summary_file = (
        f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # 并行运行实验
    all_results = run_experiments_parallel(
        config_to_run, max_workers=4, log_dir=log_dir, summary_file=summary_file
    )

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
