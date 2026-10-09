import argparse
import json
import os
import shutil
import subprocess
import signal
import contextlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, TypedDict

from tqdm import tqdm

from src.acc_head_registry import resolve_acc_head_path
from src.nvml import get_available_gpus as detect_available_gpus
from src.rl_agent_registry import (
    ROLE_LITTLE,
    ROLE_MAIN,
    get_rl_agent_spec,
    resolve_rl_agent_paths,
)


class EvalDataset(str, Enum):
    mt_bench = "eval/eval_mt_bench.py"
    humaneval = "eval/eval_humaneval.py"
    cnndm = "eval/eval_cnndm.py"
    xsum = "eval/eval_xsum.py"
    gsm8k = "eval/eval_gsm8k.py"
    mt_bench_noeval = "eval/eval_mt_bench_noeval.py"


class EvalMode(str, Enum):
    autoregression = "large"
    # sd = "sd"
    dssd = "dist_split_spec"
    dsd = "dist_spec"
    cuhlm = "uncertainty_decoding"
    # TK-SLT [WCSP'25 Zheng & Yang]：DSD + top-K 稀疏 logits 上行（基线）
    tk_slt = "tk_slt"
    # tridecoding = "tridecoding"  # ablation of adaptive tridecoding
    # # cee_sd_without_arp = "ceesd_without_arp"  # ours without arp
    ceesd = "adaptive_tridecoding"  # ours
    cee_cuhlm = "cee_cuhlm"
    cee_dssd = "cee_dssd"
    cee_dsd = "cee_dsd"
    adaptive_decoding = "adaptive_decoding"


#: eval_mode 的值 → 枚举名（如 "dist_spec" → "dsd"），供 --only-modes 接受两种写法
_MODE_NAME_BY_VALUE = {member.value: member.name for member in EvalMode}


class ExpConfig(TypedDict):
    CUDA_VISIBLE_DEVICES: str | int
    eval_mode: str | EvalMode
    edge_end_bandwidth: int | float
    edge_cloud_bandwidth: int | float
    cloud_end_bandwidth: int | float
    min_bandwidth_mbps: float
    comm_bw_model: str
    transfer_top_k: int
    num_shots: int
    num_samples_per_task: int
    eval_data_num: int | None
    exp_name: str
    use_precise: bool
    use_stochastic_comm: bool
    use_rl_adapter: bool
    disable_rl_update: bool
    small_draft_threshold: float
    draft_target_threshold: float
    uncertainty_threshold: float
    ntt_ms_edge_cloud: int | float
    ntt_ms_edge_end: int | float
    eval_dataset: EvalDataset
    run_full_dataset: bool
    random_sample: bool
    sample_seed: int
    draft_model: str
    target_model: str
    little_model: str
    acc_head_path: str
    small_draft_acc_head_path: str
    draft_target_acc_head_path: str
    main_rl_path: str
    little_rl_path: str
    main_rl_best_path: str
    little_rl_best_path: str
    max_tokens: int
    gamma: int
    gamma1: int
    gamma2: int
    use_early_stopping: bool
    dump_network_stats: bool
    batch_delay: int | float
    use_cuda_graph: bool


# Global Constants

# 论文主实验（Table V）的通信口径，冻结值见 docs/protocol.md §2：
# edge-cloud 取论文 §IV-A 建模的 50 ms 固定系统开销，edge-end 为 0.317 ms。
# Table II 的 76.36 ms / 941 Mbps 是实测值，按仓库决策只用于敏感性分析，
# 不再作为主表口径（docs/param_ledger.md §6 决策 1）。
NTT_MS_EDGE_CLOUD = 50
NTT_MS_EDGE_END = 0.317


def resolve_accelerate() -> str:
    """从当前解释器推导 accelerate 可执行文件，避免绑死某个绝对路径。

    优先取与 ``sys.executable`` 同目录下的 ``accelerate``（保证用与环境
    自洽的那一个）；不存在时回退到 ``PATH`` 查找；两者都失败则显式报错，
    而不是让 shell 报一句含义不明的 "accelerate: not found"。
    """
    interpreter_accelerate = Path(sys.executable).parent / "accelerate"
    if interpreter_accelerate.exists():
        return str(interpreter_accelerate)
    path_accelerate = shutil.which("accelerate")
    if path_accelerate is not None:
        return path_accelerate
    raise RuntimeError(
        f"未找到 accelerate：{interpreter_accelerate} 不存在，"
        f"PATH 中也没有 accelerate 可执行文件。"
        f"请安装 accelerate 或激活对应的虚拟环境。"
    )


cmd_temp = """
echo "Running experiment: {eval_mode}"
CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} {accelerate} launch \\
    --num_processes 1 \
    --main_process_port 29051 \
    {eval_dataset} \
    --eval_mode {eval_mode} \
    --protocol paper_table5 \
    --comm_accounting honest \
    --draft_model {draft_model} \
    --target_model {target_model} \
    --little_model {little_model} \
    --max_tokens {max_tokens} \
    --temp 0.0 \
    --gamma1 {gamma1} \
    --gamma2 {gamma2} \
    --gamma {gamma} \
    --edge_end_bandwidth {edge_end_bandwidth} \
    --edge_cloud_bandwidth {edge_cloud_bandwidth} \
    --cloud_end_bandwidth {cloud_end_bandwidth} \
    --min_bandwidth_mbps {min_bandwidth_mbps} \
    --comm_bw_model {comm_bw_model} \
    --transfer_top_k {transfer_top_k} \
    --num_samples_per_task {num_samples_per_task} \
    --eval_data_num {eval_data_num} \
    --sample_seed {sample_seed} \
    --small_draft_threshold {small_draft_threshold} \
    --draft_target_threshold {draft_target_threshold} \
    --uncertainty_threshold {uncertainty_threshold} \
    --num_shots {num_shots} \
    --exp_name {exp_name} \
    --ntt_ms_edge_cloud {ntt_ms_edge_cloud} \
    --ntt_ms_edge_end {ntt_ms_edge_end} \
    --batch_delay {batch_delay} \
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
    elif "qwen1.5" in name:
        return "qwen15"
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

    cmd = cmd_temp.format(accelerate=resolve_accelerate(), **config)
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
    if config.get("acc_head_path"):
        cmd = add_args(
            cmd,
            "acc_head_path",
            config["acc_head_path"],
        )
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
    if config.get("main_rl_best_path"):
        cmd = add_args(
            cmd,
            "main_rl_best_path",
            config["main_rl_best_path"],
        )
    if config.get("little_rl_best_path"):
        cmd = add_args(
            cmd,
            "little_rl_best_path",
            config["little_rl_best_path"],
        )
    if config.get("dump_network_stats", False):
        cmd = add_args(cmd, "dump_network_stats")
    if config.get("run_full_dataset", False):
        cmd = add_args(cmd, "run_full_dataset")
    if config.get("random_sample", False):
        cmd = add_args(cmd, "random_sample")
    if config.get("use_cuda_graph", False):
        # 定长 decode 前向捕获成 CUDA Graph（单步图 + (1,K) padding 验证图），
        # 绕开逐算子 ~70µs 的 launch 开销；KV 缓存跨样本复用（见 graph_decode.py）
        cmd = add_args(cmd, "use_cuda_graph")

    # --task_name 已删除（无消费者，见 docs/param_inventory.md §2）：
    # RL adapter 的 task one-hot 由各评测类硬编码 self.task 提供。

    print(f"开始实验: {config['exp_name']}, GPU: {config['CUDA_VISIBLE_DEVICES']}")
    print(f"日志文件: {log_file}")

    try:
        # 重定向输出到日志文件
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"实验配置: {json.dumps(config, indent=2, ensure_ascii=False)}\n")
            f.write(f"执行命令: {cmd}\n")
            f.write("=" * 80 + "\n")

            # start_new_session：子树独立进程组——父侧异常/Ctrl-C 时 killpg
            # 能整棵收掉 accelerate 子树，防止孤儿进程继续占卡（R4）
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                returncode = proc.wait()
            except BaseException:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                raise
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, cmd)

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
            except json.JSONDecodeError:
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
            print("--- End Error Log ---")

        return {
            "exp_name": config["exp_name"],
            "result": {"error": error_msg},
            "log_file": log_file,
            "status": "failed",
        }


class GPUManager:
    def __init__(self):
        # 初始检测一次，获取所有被管理器追踪的GPU ID（包括当前空闲的）
        # 这里假设启动时检测到的空闲GPU就是我们能管理的所有资源池
        # 如果需要更复杂的管理（例如动态发现），逻辑需要修改
        initial_gpus = self.get_available_gpus()
        self.available_gpus = set(initial_gpus)
        self.all_gpu_ids = set(initial_gpus)  # 记录总容量用于容量检查
        self.lock = threading.Lock()
        print(f"初始化GPU管理器，可用GPU: {sorted(self.available_gpus)}")

    def get_available_gpus(self) -> List[int]:
        """检测完全空闲的GPU"""
        return detect_available_gpus()

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
    is_middle_model = "27b" in str(config).lower() or "32b" in str(config).lower()
    needed_gpus = 4 if is_large_model else 2 if is_middle_model else 1

    # 死锁预防：检查系统总GPU数是否满足需求
    # 注意：这里检查的是GPU管理器初始识别到的总数（包括忙碌的），而不是当前空闲的
    total_tracked_gpus = len(gpu_manager.all_gpu_ids)
    if needed_gpus > total_tracked_gpus:
        msg = f"跳过实验 {config['exp_name']}: 需要 {needed_gpus} 个GPU，但管理器只追踪了 {total_tracked_gpus} 个GPU"
        print(msg)
        return {
            "exp_name": config["exp_name"],
            "result": {"error": msg},
            "log_file": "",
            "status": "skipped_capacity_mismatch",
        }

    try:
        # 等待直到有可用GPU
        waiting_printed = False
        while gpu_ids is None:
            gpu_ids = gpu_manager.acquire_gpu(needed_gpus)
            if gpu_ids is None:
                if not waiting_printed:
                    print(
                        f"等待可用GPU运行实验: {config['exp_name']} (需要 {needed_gpus} GPUs)"
                    )
                    waiting_printed = True
                time.sleep(10)

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
                print(f"实验完成: {config['exp_name']}, 状态: {result['status']}")
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

            # 中途保存结果（原子写：被中断也不会留下半个 JSON，
            # 下游 calculate_consistency.py 直接 json.load 才不会炸）
            if summary_file:
                tmp_summary = summary_file + ".tmp"
                with open(tmp_summary, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                os.replace(tmp_summary, summary_file)

    return all_results


def get_available_gpus() -> List[int]:
    """检测完全空闲的GPU"""
    return detect_available_gpus()


def create_config(
    eval_mode: str | EvalMode,
    ntt_ms_edge_cloud: int | float = 0,
    ntt_ms_edge_end: int | float = 0,
    batch_delay: int | float = 50e-3,
    use_precise: bool = True,
    use_stochastic_comm: bool = False,
    CUDA_VISIBLE_DEVICES: str | int = "0",
    use_rl_adapter: bool = False,
    disable_rl_update: bool = False,
    edge_end_bandwidth: int | float = 100,
    edge_cloud_bandwidth: int | float = 100,
    cloud_end_bandwidth: int | float = 100,
    min_bandwidth_mbps: float = 5.0,
    comm_bw_model: str = "instant",
    small_draft_threshold: float = 0.3,
    draft_target_threshold: float = 0.9,
    uncertainty_threshold: float = 0.8,
    transfer_top_k: int = 300,
    gamma: int = 5,
    gamma1: int = 5,
    gamma2: int = 5,
    num_shots: int = 3,
    num_samples_per_task: int = 1,
    eval_data_num: int | None = 80,
    max_tokens: int = 128,
    use_early_stopping: bool = False,
    eval_dataset: str | EvalDataset = EvalDataset.mt_bench,
    run_full_dataset: bool = False,
    random_sample: bool = False,
    sample_seed: int = 1234,
    # 新添加的参数
    draft_model: str = "tiny-vicuna-1b",
    target_model: str = "vicuna-13b-v1.5",
    little_model: str = "vicuna-68m",
    acc_head_path: str | None = None,
    small_draft_acc_head_path: str | None = None,
    draft_target_acc_head_path: str | None = None,
    main_rl_path: str | None = None,
    little_rl_path: str | None = None,
    main_rl_best_path: str | None = None,
    little_rl_best_path: str | None = None,
    dump_network_stats: bool = False,
    use_cuda_graph: bool = False,
) -> ExpConfig:
    eval_mode_value = eval_mode.value if isinstance(eval_mode, EvalMode) else eval_mode
    eval_dataset_value = (
        eval_dataset.value if isinstance(eval_dataset, EvalDataset) else eval_dataset
    )
    eval_dataset_name = (
        eval_dataset.name
        if isinstance(eval_dataset, EvalDataset)
        else Path(eval_dataset_value).stem.removeprefix("eval_")
    )

    # 使用微秒级时间戳确保唯一性
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_delay_ms = int(round(batch_delay * 1000))
    if eval_mode_value == EvalMode.adaptive_decoding.value:
        if acc_head_path is None:
            acc_head_path = resolve_acc_head_path(draft_model, target_model)

        # D3：默认路径解析单点化（原三处复制逻辑下沉 registry）
        main_rl_path, main_rl_best_path = resolve_rl_agent_paths(
            ROLE_MAIN,
            little_model=None,
            draft_model=draft_model,
            target_model=target_model,
            latest=main_rl_path,
            best=main_rl_best_path,
        )

        small_draft_acc_head_path = ""
        draft_target_acc_head_path = ""
        little_rl_path = ""
        little_rl_best_path = ""

    # ceesd, cee_cuhlm, cee_dsd, cee_dssd 需要用到 ARP 和 RL Adapter
    elif eval_mode_value in [
        EvalMode.ceesd.value,
        EvalMode.cee_cuhlm.value,
        EvalMode.cee_dsd.value,
        EvalMode.cee_dssd.value,
    ]:
        # Tri-decoding uses two prediction heads: little->draft and draft->target.
        if small_draft_acc_head_path is None:
            small_draft_acc_head_path = resolve_acc_head_path(little_model, draft_model)
        if draft_target_acc_head_path is None:
            draft_target_acc_head_path = resolve_acc_head_path(
                draft_model, target_model
            )

        main_rl_path, main_rl_best_path = resolve_rl_agent_paths(
            ROLE_MAIN,
            little_model=little_model,
            draft_model=draft_model,
            target_model=target_model,
            latest=main_rl_path,
            best=main_rl_best_path,
        )
        little_rl_path, little_rl_best_path = resolve_rl_agent_paths(
            ROLE_LITTLE,
            little_model=little_model,
            draft_model=draft_model,
            target_model=target_model,
            latest=little_rl_path,
            best=little_rl_best_path,
        )

        acc_head_path = ""

    else:
        # 其他模式不需要
        acc_head_path = ""
        small_draft_acc_head_path = ""
        draft_target_acc_head_path = ""
        main_rl_path = ""
        little_rl_path = ""
        main_rl_best_path = ""
        little_rl_best_path = ""

    return ExpConfig(
        eval_dataset=eval_dataset_value,
        run_full_dataset=run_full_dataset,
        random_sample=random_sample,
        sample_seed=sample_seed,
        CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
        eval_mode=eval_mode_value,
        edge_end_bandwidth=edge_end_bandwidth,
        edge_cloud_bandwidth=edge_cloud_bandwidth,
        cloud_end_bandwidth=cloud_end_bandwidth,
        min_bandwidth_mbps=min_bandwidth_mbps,
        comm_bw_model=comm_bw_model,
        transfer_top_k=transfer_top_k,
        gamma=gamma,
        gamma1=gamma1,
        gamma2=gamma2,
        num_shots=num_shots,
        num_samples_per_task=num_samples_per_task,
        eval_data_num=eval_data_num,
        max_tokens=max_tokens,
        small_draft_threshold=small_draft_threshold,
        draft_target_threshold=draft_target_threshold,
        uncertainty_threshold=uncertainty_threshold,
        exp_name=(
            f"{eval_mode_value}/{eval_dataset_name}/"
            f"{eval_mode_value}_{num_shots}shot_g1{gamma1}_g2{gamma2}_batchdelay{batch_delay_ms}ms"
            f"{'_cudagraph' if use_cuda_graph else ''}"
            f"_bw{edge_cloud_bandwidth:g}_{timestamp}"
        ),
        use_precise=use_precise,
        use_stochastic_comm=use_stochastic_comm,
        ntt_ms_edge_cloud=ntt_ms_edge_cloud,
        ntt_ms_edge_end=ntt_ms_edge_end,
        batch_delay=batch_delay,
        use_rl_adapter=use_rl_adapter,
        disable_rl_update=disable_rl_update,
        draft_model=draft_model,
        target_model=target_model,
        little_model=little_model,
        use_early_stopping=use_early_stopping,
        acc_head_path=acc_head_path,
        small_draft_acc_head_path=small_draft_acc_head_path,
        draft_target_acc_head_path=draft_target_acc_head_path,
        main_rl_path=main_rl_path,
        little_rl_path=little_rl_path,
        main_rl_best_path=main_rl_best_path,
        little_rl_best_path=little_rl_best_path,
        dump_network_stats=dump_network_stats,
        use_cuda_graph=use_cuda_graph,
    )


config_to_run = []

llama_series = ("llama-68m", "tiny-llama-1.1b", "llama-2-13b")
vicuna_series = ("vicuna-68m", "tiny-vicuna-1b", "vicuna-13b-v1.5")
qwen_series = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-14B")
qwen_series_large = ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B")
llama_chat_series = (
    "llama-68m",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
)
gemma_3_it_series = (
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
)
llama_3_series = (
    "llama-68m",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
)
qwen_1_5_series = (
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-7B-Chat",
)

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

# 主表 WAN 条件改为阶梯：46 Mbps（Table II 实测 46.9/46.0，协议统一取 46，
# 见 docs/protocol.md §2 #4）+ 两个带宽受限点 10/5 Mbps。
# 动机：46 下 NTT(50ms/轮) 主导通信，载荷是二阶量（tk_slt K=320 仅
# ~1.2ms/轮），压缩类方法的机制差异几乎不可见；10/5 Mbps 把载荷权重
# 提到一阶（实测外推：@10 Mbps 时 dsd(全词表) −45%、dssd(拒绝行) −16%、
# tk_slt −2%、cuhlm −1%）。5/10 也在论文 Fig.4 的 5–25 Mbps 鲁棒性
# 扫描范围内，属同一故事的延伸而非新设定。
# 注意：①带宽进 exp_name（_bw46/_bw10/_bw5）与 RUN_IDENTITY_FIELDS，
# 阶梯点互不冲突、resume/merge 安全；②低于 5 会被 min_bandwidth_mbps=5
# 钳位（cmd_temp 不透传该参数），如需更低先透传；③低带宽下 dsd/dssd
# 的最优 γ 会下移（载荷 ∝ 草稿数/拒绝行数），各带宽的 γ 需用
# scripts/gamma_sweep.py --bandwidth 重扫后再定。
edge_cloud_bandwidth = [46.0, 10.0, 5.0]

# 论文 §IV-A：每次云请求 50 ms 固定系统开销
batch_delay_values = [50e-3]

# 投机深度 γ：协议不冻结该项（docs/protocol.md §4，唯一待定项），因此一律显式传，
# 不允许靠 create_config 的签名默认值兜底。取值沿用仓库 Table V 对齐时的选择：
# 基线 γ=3（前向数与论文吻合，DSD 4113 vs 3980）、CEE-SD 取消融 static 的 γ1=γ2=5。
GAMMA_SINGLE = 3  # 单-γ 方法：dsd / dssd / cuhlm 读 --gamma
GAMMA1_CEESD = 5  # 三级方法 Draft→Target 推测窗口
GAMMA2_CEESD = 5  # 三级方法 Little→Draft 推测窗口

# top-k 压缩：**只有 CEE-SD 自己的设计里有**（DRA 选 top-k，上行传 top-k 压缩 logits）。
# 此前把 transfer_top_k=300 一律套在所有方法头上，等于把"压缩传输"这项本该被验证的
# 贡献免费送给了基线：
#   · DSSD 原文（§3 / 式 8）：拒绝时下行是**整词表分布** P_j(x)，即 |V|·bprob；
#   · DSD  原文：上行是 γ 个整词表分布（DSSD 论文 Alg.1 / 式 4）。
# 现在按各自原文计费。令 transfer_top_k=0 即可在不改基线代码的前提下切到整行载荷：
#   · CommunicationSimulator.transfer 的 is_compressed = (k is not None and k > 0)
#     ⇒ False，按 prob.numel()*element_size 计整行；
#   · reject_residual_payload_bytes(row, 0) 的 `0 < k < vocab` 不成立 ⇒ 返回
#     vocab*element，也是整行。
# 附带效果（是对齐而非副作用）：proposal_top_k(0) 返回 None，使
# rebuild_topk_uniform_probs 原样返回 —— "top-k + 均匀尾部" 的重建本身就是压缩机制的
# 一环，原文没有。temp=0 下草稿 token 取 argmax，top-300 与全量分布给出同一个 token，
# 所以生成的 token 序列不变。
TRANSFER_TOP_K_OURS = 300  # CEE-SD：自身设计，保留
TRANSFER_TOP_K_PAPER = 0  # dsd / dssd：原文无 top-k 压缩 ⇒ 整词表载荷
# TK-SLT：K 是该方法自己的超参（top-K 稀疏 logits 上行）。取 320 =
# 论文 §VI-B Fig 4 的最优工作点（T=0 与 T=1 两个温度下最大加速比都在
# K=320 取得，此时上行载荷 ≈ 1% 词表）；K=0 可退回其 vanilla DSD 基线。
TRANSFER_TOP_K_TKSLT = 320  # tk_slt：论文实测最优 K

# CUHLM 的不确定度阈值（越大越少上云、越快越差）。
#
# 0.8 是 create_config/argparse 的默认值，但它把 CUHLM 放在一个退化的工作点上：
# Llama/GSM8K 上 thr=0.8 的准确率是 0.0，而 CEE-SD 0.2375、DSD/DSSD 0.25 ——
# 也就是说 CUHLM 之所以"快 5.9 倍"，是靠几乎不调用云模型换来的。这也与论文
# Table V 自己标注的"baselines ≈target"矛盾。
#
# 取 0.08 的依据（仓库实测的阈值前沿，Llama-2-13b/GSM8K）：
#   thr 0.08 → acc 0.25（= target 水平，与 DSD/DSSD 相同、略高于 CEE-SD 0.2375）
#   thr 0.30 → acc 0.1375
#   thr 0.50 → acc 0.0375
#   thr 0.80 → acc 0.0
# 即 0.08 是"等准确率"工作点：让 CUHLM 在可比质量下参与吞吐比较，而不是拿
# 低质量换速度。Qwen1.5/Qwen3 两个系列还没有阈值前沿，0.08 是按同一判据外推的，
# 需在重跑后核对准确率是否与其他方法同量级。
UNCERTAINTY_THRESHOLD_CUHLM = 0.08
UNCERTAINTY_THRESHOLD_DEFAULT = 0.8  # 其余方法（主表里只有 CUHLM 消费该参数）

# 论文主实验（Table V）实验矩阵 = 三族模型 × 三个数据集 × 四个方法。
#
# 论文 §IV-A 明示的三族（little / draft / target）：
#   Llama   : llama-68m / tiny-llama-1.1b / llama-2-13b
#   Qwen1.5 : Qwen1.5-0.5B-Chat / Qwen1.5-1.8B-Chat / Qwen1.5-7B-Chat
#   Qwen3   : Qwen3-0.6B / Qwen3-1.7B / Qwen3-14B
paper_table5_series = (
    llama_series,
    qwen_1_5_series,
    qwen_series,
)

# Table V 的三个数据集；CNN/DM 只用于 Table VII 的精度评测，不属于吞吐主表。
paper_table5_datasets = (
    EvalDataset.mt_bench_noeval,
    EvalDataset.gsm8k,
    EvalDataset.humaneval,
)

# Table V 的方法列：四个基线 + 本方法。Table VI 的 CEE+ 变体与
# Table VIII 的 Static 消融都是独立实验，不进入主表矩阵。
paper_table5_modes = (
    EvalMode.dsd,  # DSD [7]
    EvalMode.dssd,  # DSSD [8]
    EvalMode.cuhlm,  # CUHLM [9]
    EvalMode.tk_slt,  # TK-SLT [10]：DSD + top-K 稀疏 logits 上行
    EvalMode.ceesd,  # CEE-SD（本文方法，= adaptive_tridecoding）
)

for little_model, draft_model, target_model in paper_table5_series:
    for dataset in paper_table5_datasets:
        for mode in paper_table5_modes:
            for edge_cloud_bw in edge_cloud_bandwidth:
                for batch_delay in batch_delay_values:
                    is_ceesd = mode == EvalMode.ceesd
                    config = create_config(
                        eval_mode=mode,
                        ntt_ms_edge_cloud=NTT_MS_EDGE_CLOUD,
                        ntt_ms_edge_end=NTT_MS_EDGE_END,
                        batch_delay=batch_delay,
                        use_precise=False,
                        use_stochastic_comm=True,
                        # 论文 §IV-A 的 edge-end 保守有效带宽 563 Mbps
                        edge_end_bandwidth=563,
                        edge_cloud_bandwidth=edge_cloud_bw,
                        cloud_end_bandwidth=edge_cloud_bw,
                        # 带宽下限随阶梯缩放：目标均值的 1/10（下限 0.5）。
                        # 固定 5 会在 5 Mbps 档把 trace 削成近似常数、
                        # 10 Mbps 档削掉下半段（重缩放后深衰采样密集），
                        # 阶梯三档的 trace 动态形状才可比。
                        min_bandwidth_mbps=max(0.5, edge_cloud_bw / 10),
                        # 载荷发射时长走流体模型（2026-10-09 决策，
                        # docs/protocol.md §3.4）
                        comm_bw_model="fluid",
                        small_draft_threshold=0.6,
                        draft_target_threshold=0.7,
                        # 不确定度阈值：主表里只有 CUHLM 消费（CEE-SD 只在
                        # cee_sd_opportunistic 变体里读，见 baselines.py:3781 的
                        # _opportunistic_first_stage 分支）。0.8 会让 CUHLM 退化到
                        # 0 准确率，故改用等准确率工作点 0.08。
                        uncertainty_threshold=(
                            UNCERTAINTY_THRESHOLD_CUHLM
                            if mode == EvalMode.cuhlm
                            else UNCERTAINTY_THRESHOLD_DEFAULT
                        ),
                        transfer_top_k=(
                            TRANSFER_TOP_K_OURS
                            if mode in (EvalMode.ceesd, EvalMode.cuhlm)
                            else (
                                TRANSFER_TOP_K_TKSLT
                                if mode == EvalMode.tk_slt
                                else TRANSFER_TOP_K_PAPER
                            )
                        ),
                        gamma=GAMMA_SINGLE,
                        gamma1=GAMMA1_CEESD if is_ceesd else GAMMA_SINGLE,
                        gamma2=GAMMA2_CEESD if is_ceesd else GAMMA_SINGLE,
                        max_tokens=128,
                        num_shots=3,
                        eval_dataset=dataset,
                        # DSSD 与 CEE 族用中间模型当草稿（与论文 Table V 的
                        # 接受率吻合）；DSD / CUHLM / TK-SLT 用端侧小模型
                        # （TK-SLT 原文 §VI-B：68M 草稿 + 7B 验证）。
                        draft_model=(
                            draft_model
                            if mode
                            in [
                                EvalMode.ceesd,
                                EvalMode.cee_cuhlm,
                                EvalMode.cee_dsd,
                                EvalMode.cee_dssd,
                                EvalMode.dssd,
                                EvalMode.adaptive_decoding,
                            ]
                            else little_model
                        ),
                        target_model=target_model,
                        little_model=little_model,
                        # 协议 §2 #12：RL adapter 只作用于 CEE-SD（tri 族），
                        # 不作用于基线，否则会改变基线的 top-k 行为。
                        use_rl_adapter=is_ceesd,
                        disable_rl_update=is_ceesd,
                        use_early_stopping=False,
                        use_cuda_graph=True,
                        eval_data_num=80,
                        run_full_dataset=False,
                        random_sample=True,
                        sample_seed=1234,
                    )
                    config_to_run.append(config)


# ---------------------------------------------------------------------------
# 断点续跑与结果合并
#
# exp_name 里带运行时间戳（..._20261005_125853_876008），同一个实验重跑一次就会得到
# 不同的 exp_name，所以**不能拿 exp_name 当合并键**。合并与续跑的键是「这次到底跑的
# 是什么」那组语义字段：数据集 + 方法 + 三档模型 + 通信口径 + 投机深度 + 采样设置。
# CUDA_VISIBLE_DEVICES 与 exp_name 这类每次运行都会变的字段必须排除在外。
RUN_IDENTITY_FIELDS = (
    "eval_dataset",
    "eval_mode",
    "little_model",
    "draft_model",
    "target_model",
    "edge_cloud_bandwidth",
    "min_bandwidth_mbps",
    "comm_bw_model",
    "edge_end_bandwidth",
    "cloud_end_bandwidth",
    "batch_delay",
    "gamma",
    "gamma1",
    "gamma2",
    "max_tokens",
    "num_shots",
    "eval_data_num",
    "num_samples_per_task",
    "sample_seed",
    "random_sample",
)


def _identity_value(value):
    """把枚举取成字符串、浮点归一，避免同一实验因类型不同算出两个键。"""
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 9)
    return value


def run_identity(config: dict) -> tuple:
    """把一个 config 压成与时间戳无关的语义键。"""
    return tuple(
        (field, _identity_value(config[field]))
        for field in RUN_IDENTITY_FIELDS
        if field in config
    )


def load_summary(path: str | Path) -> List[dict]:
    """读一份实验汇总（list[dict]）。"""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"找不到汇总文件: {resolved}")
    with resolved.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"汇总文件顶层不是 list: {resolved}")
    return data


def _summary_sort_key(entry: dict) -> tuple:
    config = entry.get("config") or {}
    return (
        str(config.get("eval_dataset", "")),
        str(config.get("eval_mode", "")),
        str(config.get("target_model", "")),
        str(entry.get("exp_name", "")),
    )


def merge_summaries(*summaries: List[dict]) -> List[dict]:
    """按语义键合并多份汇总，**后面的覆盖前面的**（新结果优先）。

    同一次实验在旧汇总里可能是 status=failed（result 里只有 error），
    新汇总里是 success，合并后保留 success 那条。
    """
    merged: dict = {}
    for summary in summaries:
        for entry in summary:
            config = entry.get("config") or {}
            if config:
                key = run_identity(config)
            else:
                # 没有 config 的条目（早期异常）无法定位语义，用 exp_name 兜底保留
                key = ("__exp_name__", str(entry.get("exp_name", "")))
            merged[key] = entry
    return sorted(merged.values(), key=_summary_sort_key)


def _mode_aliases(mode) -> set:
    """一个 eval_mode 的可写形式：枚举名（dsd）与枚举值（dist_spec）都接受。

    config 里存的是枚举的**值**（create_config 已转成字符串），所以名字要回查
    EvalMode 表，而不是从 config 元素本身取。
    """
    value = str(getattr(mode, "value", mode))
    names = {value}
    if isinstance(mode, EvalMode):
        names.add(mode.name)
    known = _MODE_NAME_BY_VALUE.get(value)
    if known:
        names.add(known)
    return names


def filter_configs_by_mode(
    configs: List[ExpConfig], only_modes: str | None = None
) -> List[ExpConfig]:
    """只保留指定方法（``--only-modes dsd,dssd``）。

    用于口径变更后只重跑受影响的方法，而不是整表 36 个。
    """
    if not only_modes:
        return list(configs)
    wanted = {token.strip() for token in only_modes.split(",") if token.strip()}
    available = {alias for c in configs for alias in _mode_aliases(c["eval_mode"])}
    unknown = wanted - available
    if unknown:
        raise ValueError(
            f"--only-modes 里有未知方法 {sorted(unknown)}；可用: {sorted(available)}"
        )
    return [
        config
        for config in configs
        if _mode_aliases(config["eval_mode"]) & wanted
    ]


def filter_configs_for_resume(
    configs: List[ExpConfig], resume_from: str | Path | None = None
) -> List[ExpConfig]:
    """挑出还没成功跑过的 config：已在汇总里 success 的不再重跑。"""
    if not resume_from:
        return list(configs)
    done = {
        run_identity(entry["config"])
        for entry in load_summary(resume_from)
        if entry.get("status") == "success" and entry.get("config")
    }
    return [config for config in configs if run_identity(config) not in done]


def describe_config(config: dict) -> str:
    dataset = Path(str(config.get("eval_dataset", "?"))).stem.replace("eval_", "")
    return (
        f"{str(config.get('eval_mode', '?')):22s} {dataset:16s} "
        f"{str(config.get('target_model', '?'))}"
    )


def parse_exp_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="论文 Table V 实验矩阵（三族模型 × 三数据集 × 四方法）"
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="SUMMARY.json",
        help="断点续跑：该汇总里 status=success 的实验不再重跑",
    )
    parser.add_argument(
        "--merge-from",
        action="append",
        default=None,
        metavar="SUMMARY.json",
        help="把本次结果合并进这份汇总（可重复传）；同语义键以本次结果为准",
    )
    parser.add_argument(
        "--only-modes",
        default=None,
        metavar="MODE[,MODE...]",
        help="只跑这些方法（枚举名或枚举值，如 dsd,dssd 或 dist_spec,dist_split_spec）",
    )
    parser.add_argument(
        "--merged-output",
        default=None,
        metavar="PATH",
        help="合并结果输出路径（默认在 experiment_results/ 下新建 merged 文件）",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        metavar="PATH",
        help="本次运行的汇总输出路径（默认按时间戳新建）",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印待跑清单，不真正运行",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_exp_args()

    # 创建日志目录
    log_dir = "exp_logs"
    Path(log_dir).mkdir(exist_ok=True)

    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)

    # 先按方法筛选（口径变更后只重跑受影响的方法），再按成功记录跳过
    selected = filter_configs_by_mode(config_to_run, args.only_modes)
    if args.only_modes:
        print(
            f"--only-modes {args.only_modes} ⇒ "
            f"从 {len(config_to_run)} 个中筛出 {len(selected)} 个"
        )
    pending = filter_configs_for_resume(selected, args.resume_from)
    skipped = len(selected) - len(pending)
    print(f"本次待跑 {len(pending)} 个")
    if skipped:
        print(f"（按 {args.resume_from} 的成功记录跳过 {skipped} 个）")
    for config in pending:
        print(f"  - {describe_config(config)}")

    if args.dry_run:
        print("\n[dry-run] 未执行任何实验。")
        sys.exit(0)
    if not pending:
        print("\n没有待跑的实验，无需运行。")
        sys.exit(0)

    # 本次运行的汇总文件；run_stamp 同时用于合并输出，让两份产物成对可辨
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = args.summary_file or str(
        results_dir / f"experiment_summary_{run_stamp}.json"
    )

    # 并行运行实验
    all_results = run_experiments_parallel(
        pending,
        max_workers=args.max_workers,
        log_dir=log_dir,
        summary_file=summary_file,
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
    print(f"\n本次汇总已保存到: {summary_file}")

    for result in all_results:
        print(f"\n实验: {result['exp_name']}")
        print(f"状态: {result['status']}")
        if result.get("log_file"):
            print(f"日志: {result['log_file']}")

    # 合并：旧汇总在前、本次结果在后，同语义键以本次为准
    merge_inputs = list(args.merge_from or [])
    if merge_inputs:
        print("\n" + "=" * 80)
        print("合并结果:")
        print("=" * 80)
        bases = [load_summary(path) for path in merge_inputs]

        # 合并前的状态，用来报「哪些实验从失败/缺失变成了成功」
        old_status: dict = {}
        for base in bases:
            for entry in base:
                if entry.get("config"):
                    old_status[run_identity(entry["config"])] = entry.get("status")

        merged = merge_summaries(*bases, all_results)

        resolved = sum(
            1
            for entry in merged
            if entry.get("config")
            and entry.get("status") == "success"
            and old_status.get(run_identity(entry["config"])) != "success"
        )

        merged_output = args.merged_output or str(
            results_dir / f"experiment_summary_merged_{run_stamp}.json"
        )
        tmp_path = merged_output + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, merged_output)

        merged_success = sum(1 for e in merged if e.get("status") == "success")
        merged_failed = len(merged) - merged_success
        print(f"合并自: {', '.join(merge_inputs)}")
        print(
            f"合并后条目数: {len(merged)}"
            f"（success {merged_success} / 非 success {merged_failed}）"
        )
        print(f"本次新转为 success: {resolved}")
        print(f"合并结果已保存到: {merged_output}")
