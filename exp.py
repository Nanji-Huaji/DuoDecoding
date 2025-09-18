import subprocess
import os

from typing import TypedDict

from datetime import datetime

class ExpConfig(TypedDict):
    eval_mode: str
    edge_end_bandwidth: int
    edge_cloud_bandwidth: int
    cloud_end_bandwidth: int
    transfer_top_k: int
    exp_name: str
    use_precise: bool
    ntt_ms_edge_cloud: int | float
    ntt_ms_edge_end: int | float


cmd_temp = """
echo "Running experiment: {eval_mode}"
CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --num_processes 1 \
    --main_process_port 29051 \
    eval/eval_mt_bench.py \
    --eval_mode {eval_mode} \
    -e llama \
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
"""

def add_args(base_cmd: str, extra_arg: str) -> str:
    # 修复语法错误
    return base_cmd.rstrip() + f" \\\n    --{extra_arg}"

def get_file_path(exp_name: str) -> str:
    dir_path = f"exp/{exp_name}"
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith("_metrics.json"):  # 修复语法错误
                return os.path.join(root, file)
    print(f"File not found for {exp_name}")
    return ""

def run_exp(config: ExpConfig) -> dict:
    cmd = cmd_temp.format(**config)
    if config.get("use_precise", False):
        cmd = add_args(cmd, "use_precise")
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
    # 读取结果文件
    result_file = get_file_path(config["exp_name"])
    if result_file:
        with open(result_file, 'r') as f:
            result = f.read()
        return {"exp_name": config["exp_name"], "result": result}
    else:
        return {"exp_name": config["exp_name"], "result": "File not found"}


def create_config(
    eval_mode: str,
    ntt_ms_edge_cloud: int | float = 10,
    ntt_ms_edge_end: int | float = 1,
    use_precise: bool = True,
) -> ExpConfig:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return ExpConfig(
        eval_mode=eval_mode,
        edge_end_bandwidth=100,
        edge_cloud_bandwidth=100,
        cloud_end_bandwidth=100,
        transfer_top_k=300,
        exp_name=f"{eval_mode}/{eval_mode}_{timestamp}",
        use_precise=use_precise,
        ntt_ms_edge_cloud=ntt_ms_edge_cloud,
        ntt_ms_edge_end=ntt_ms_edge_end,
    )


config_to_run = [
    create_config("dist_spec"),
    create_config("dist_split_spec"),
    create_config("uncertainty_decoding"),
    create_config("tridecoding"),
]

if __name__ == "__main__":
    all_results = []
    for config in config_to_run:
        result = run_exp(config)
        all_results.append(result)
    
    # 保存所有结果到一个文件中
    with open("all_experiment_results.txt", "w") as f:
        for res in all_results:
            f.write(f"Experiment: {res['exp_name']}\n")
            f.write(f"Result: {res['result']}\n\n")
