from datetime import datetime
from pathlib import Path

import json

from exp import EvalDataset, EvalMode, create_config, run_exp


NTT_MS_EDGE_CLOUD = 10
NTT_MS_EDGE_END = 0
EDGE_END_BANDWIDTH = 563
EDGE_CLOUD_BANDWIDTH = 23.6
CLOUD_END_BANDWIDTH = EDGE_CLOUD_BANDWIDTH
TRANSFER_TOP_K = 1024
MAX_TOKENS = 128
NUM_SHOTS = 5
NUM_SAMPLES_PER_TASK = 1
EVAL_DATA_NUM = 30
CUDA_VISIBLE_DEVICES = "2,3"
FAIL_FAST = False

# Threshold sweep for the two uncertainty-gated stages in cee_cuhlm.
SMALL_DRAFT_THRESHOLDS = [round(i / 10, 1) for i in range(1, 10)]
DRAFT_TARGET_THRESHOLDS = [round(i / 10, 1) for i in range(1, 10)]

# Keep the sweep focused on a single model triplet by default.
MODEL_SERIES = [
    ("llama-68m", "tiny-llama-1.1b", "llama-2-13b"),
]

DATASETS = [
    EvalDataset.mt_bench_noeval,
]


def build_configs() -> list[dict]:
    configs: list[dict] = []
    for little_model, draft_model, target_model in MODEL_SERIES:
        for dataset in DATASETS:
            for small_draft_threshold in SMALL_DRAFT_THRESHOLDS:
                for draft_target_threshold in DRAFT_TARGET_THRESHOLDS:
                    config = create_config(
                        eval_mode=EvalMode.cee_cuhlm,
                        ntt_ms_edge_cloud=NTT_MS_EDGE_CLOUD,
                        ntt_ms_edge_end=NTT_MS_EDGE_END,
                        use_precise=False,
                        use_stochastic_comm=True,
                        edge_end_bandwidth=EDGE_END_BANDWIDTH,
                        edge_cloud_bandwidth=EDGE_CLOUD_BANDWIDTH,
                        cloud_end_bandwidth=CLOUD_END_BANDWIDTH,
                        small_draft_threshold=small_draft_threshold,
                        draft_target_threshold=draft_target_threshold,
                        transfer_top_k=TRANSFER_TOP_K,
                        max_tokens=MAX_TOKENS,
                        num_shots=NUM_SHOTS,
                        num_samples_per_task=NUM_SAMPLES_PER_TASK,
                        eval_data_num=EVAL_DATA_NUM,
                        CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
                        eval_dataset=dataset,
                        draft_model=draft_model,
                        target_model=target_model,
                        little_model=little_model,
                        use_rl_adapter=False,
                        disable_rl_update=True,
                        use_early_stopping=True,
                    )
                    configs.append(config)
    return configs


config_to_run = build_configs()


if __name__ == "__main__":
    log_dir = "exp_logs"
    Path(log_dir).mkdir(exist_ok=True)

    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)

    summary_file = str(
        results_dir
        / f"cee_cuhlm_threshold_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    print("=" * 80)
    print("CEE-CUHLM Threshold Sweep")
    print("=" * 80)
    print(f"Total configs: {len(config_to_run)}")
    print(f"CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}")
    print(f"num_samples_per_task: {NUM_SAMPLES_PER_TASK}")
    print(f"eval_data_num: {EVAL_DATA_NUM}")
    print(f"fail_fast: {FAIL_FAST}")

    all_results = []
    for config in config_to_run:
        result = run_exp(config, log_dir=log_dir)
        result["config"] = config.copy()
        all_results.append(result)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        if FAIL_FAST and result["status"] != "success":
            print("\n" + "=" * 80)
            print("Experiment failed, aborting immediately")
            print("=" * 80)
            print(f"exp_name: {config['exp_name']}")
            print(f"small_draft_threshold: {config['small_draft_threshold']}")
            print(f"draft_target_threshold: {config['draft_target_threshold']}")
            print(f"status: {result['status']}")
            print(f"log_file: {result['log_file']}")
            print(f"summary_file: {summary_file}")
            raise RuntimeError(
                f"Experiment failed for thresholds "
                f"({config['small_draft_threshold']}, {config['draft_target_threshold']}). "
                f"See log: {result['log_file']}"
            )

    print("\n" + "=" * 80)
    print("CEE-CUHLM Threshold Sweep Summary")
    print("=" * 80)
    print(f"Total experiments: {len(all_results)}")
    print(f"Success: {sum(1 for r in all_results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in all_results if r['status'] == 'failed')}")
    print(f"No result: {sum(1 for r in all_results if r['status'] == 'no_result')}")
    print(f"Exception: {sum(1 for r in all_results if r['status'] == 'exception')}")
    print(f"Summary saved to: {summary_file}")
