"""Command and metrics-file helpers for scripts that launch accelerate themselves."""

import os
import shutil
import sys
from pathlib import Path


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


# cmd_temp/add_args 保留给仍自行拼接 accelerate 命令的现用外部脚本
# （scripts/pilot_ab_network.py 渲染自己的 A/B 命令行）。exp.py 自身的
# 调度已改走 eval.worker 持久子进程（见 run_exp），不再使用该模板。
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
    value = value_of_extra_args if value_of_extra_args is not None else ""
    return f"{base_cmd.rstrip()} \\\n    --{extra_arg} {value}"


def get_file_path(exp_name: str) -> str:
    """Find metrics in a run directory, then try the old parent/prefix layout."""
    searches = [(f"exp/{exp_name}", "")]
    if "/" in exp_name:
        parent, prefix = os.path.split(exp_name)
        searches.append((os.path.join("exp", parent), prefix))
    for directory, prefix in searches:
        for root, _, files in os.walk(directory):
            for name in files:
                if prefix in name and name.endswith("_metrics.json"):
                    return os.path.join(root, name)

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
