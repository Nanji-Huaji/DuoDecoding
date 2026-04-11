from datetime import datetime
from pathlib import Path

import json

from exp import run_exp


CUDA_VISIBLE_DEVICES = "0,1"
LOG_DIR = "exp_logs"
RESULTS_DIR = Path("experiment_results")

CONTROLLED_SCRIPT = "eval/eval_cee_sd_controlled_topk.py"
CONTROLLED_TASK = "gsm8k"
CONTROLLED_TOPK_VALUES = "16,64,256,1024"
CONTROLLED_TOPK_STEP = 0
CONTROLLED_ENTROPY_QUANTILE = 0.8
CONTROLLED_MAX_HIGH_ENTROPY_STATES = 32

MODEL_SERIES = [
    ("llama-68m", "tiny-llama-1.1b", "llama-2-13b"),
]


def build_config(
    little_model: str,
    draft_model: str,
    target_model: str,
) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return {
        "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
        "eval_mode": "cee_sd",
        "edge_end_bandwidth": 563,
        "edge_cloud_bandwidth": 23.6,
        "cloud_end_bandwidth": 23.6,
        "transfer_top_k": 1024,
        "num_shots": 3,
        "num_samples_per_task": 1,
        "eval_data_num": 20,
        "exp_name": f"cee_sd_controlled_topk/{CONTROLLED_TASK}/{timestamp}",
        "use_precise": False,
        "use_stochastic_comm": True,
        "use_rl_adapter": True,
        "disable_rl_update": True,
        "small_draft_threshold": 0.8,
        "draft_target_threshold": 0.8,
        "ntt_ms_edge_cloud": 10,
        "ntt_ms_edge_end": 0,
        "eval_dataset": CONTROLLED_SCRIPT,
        "draft_model": draft_model,
        "target_model": target_model,
        "little_model": little_model,
        "acc_head_path": "",
        "small_draft_acc_head_path": "",
        "draft_target_acc_head_path": "",
        "main_rl_path": "",
        "little_rl_path": "",
        "main_rl_best_path": "",
        "little_rl_best_path": "",
        "max_tokens": 128,
        "gamma": 4,
        "gamma1": 5,
        "gamma2": 10,
        "use_early_stopping": False,
        "dump_network_stats": False,
        "batch_delay": 50e-3,
        "controlled_eval_task": CONTROLLED_TASK,
        "controlled_topk_values": CONTROLLED_TOPK_VALUES,
        "controlled_topk_step": CONTROLLED_TOPK_STEP,
        "controlled_entropy_quantile": CONTROLLED_ENTROPY_QUANTILE,
        "controlled_entropy_threshold": "",
        "controlled_max_high_entropy_states": CONTROLLED_MAX_HIGH_ENTROPY_STATES,
    }


def main() -> None:
    Path(LOG_DIR).mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_file = RESULTS_DIR / (
        f"cee_sd_controlled_topk_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    all_results = []
    for little_model, draft_model, target_model in MODEL_SERIES:
        config = build_config(little_model, draft_model, target_model)
        result = run_exp(config, log_dir=LOG_DIR)
        result["config"] = config.copy()
        all_results.append(result)
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Controlled CEE-SD scan summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
