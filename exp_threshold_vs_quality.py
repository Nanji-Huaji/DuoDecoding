"""
Parallel cross-series threshold sweep with multi-GPU support.
"""
from datetime import datetime
from pathlib import Path
import json
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from exp import EvalDataset, EvalMode, ExpConfig, create_config, run_exp

THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
MODEL_SERIES = [
    ("llama-68m", "tiny-llama-1.1b", "Llama-2-13b-hf"),
    ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-14B"),
]
DATASETS = [EvalDataset.gsm8k]
GPUS = [0, 1, 2, 3]
EVAL_DATA_NUM = 30
RUN_FULL_DATASET = False
GAMMA1, GAMMA2, MAX_TOKENS = 3, 3, 128
NUM_SHOTS = 3
EDGE_END_BW, EDGE_CLOUD_BW = 563, 23.6
TRANSFER_TOP_K = 300

gpu_lock = threading.Lock()
gpu_available = sorted(GPUS)


def build_configs() -> list[tuple[int, ExpConfig]]:
    """Build configs with assigned GPU index."""
    configs: list[tuple[int, ExpConfig]] = []
    S1_VALS = [0.3, 0.5, 0.7]
    S2_VALS = [0.3, 0.5, 0.7, 0.9]
    for little, draft, target in MODEL_SERIES:
        for dataset in DATASETS:
            for s1 in S1_VALS:
                for s2 in S2_VALS:
                    configs.append(
                        create_config(
                            eval_mode=EvalMode.cee_cuhlm,
                            eval_dataset=dataset,
                            little_model=little, draft_model=draft, target_model=target,
                            CUDA_VISIBLE_DEVICES="0",  # placeholder, will be replaced
                            small_draft_threshold=s1, draft_target_threshold=s2,
                            eval_data_num=EVAL_DATA_NUM,
                            run_full_dataset=RUN_FULL_DATASET,
                            gamma1=GAMMA1, gamma2=GAMMA2, max_tokens=MAX_TOKENS,
                            num_shots=NUM_SHOTS, num_samples_per_task=1,
                            edge_end_bandwidth=EDGE_END_BW, edge_cloud_bandwidth=EDGE_CLOUD_BW,
                            cloud_end_bandwidth=EDGE_CLOUD_BW, transfer_top_k=TRANSFER_TOP_K,
                            use_precise=False, use_stochastic_comm=True,
                            use_rl_adapter=False, disable_rl_update=True,
                            small_draft_acc_head_path="", draft_target_acc_head_path="",
                            main_rl_path="", little_rl_path="",
                            main_rl_best_path="", little_rl_best_path="",
                        )
                    )
            for thr in THRESHOLDS:
                configs.append(
                    create_config(
                        eval_mode=EvalMode.cuhlm,
                        eval_dataset=dataset,
                        little_model=little, draft_model=little, target_model=target,
                        CUDA_VISIBLE_DEVICES="0",
                        uncertainty_threshold=thr,
                        eval_data_num=EVAL_DATA_NUM,
                        run_full_dataset=RUN_FULL_DATASET,
                        gamma1=GAMMA1, gamma2=GAMMA2, max_tokens=MAX_TOKENS,
                        num_shots=NUM_SHOTS, num_samples_per_task=1,
                        edge_end_bandwidth=EDGE_END_BW, edge_cloud_bandwidth=EDGE_CLOUD_BW,
                        cloud_end_bandwidth=EDGE_CLOUD_BW, transfer_top_k=TRANSFER_TOP_K,
                        use_precise=False, use_stochastic_comm=True,
                        use_rl_adapter=False, disable_rl_update=True,
                    )
                )
    return list(enumerate(configs))


def run_one(cfg: ExpConfig, gpu: int, idx: int) -> dict:
    """Run a single experiment on a specific GPU."""
    cfg["CUDA_VISIBLE_DEVICES"] = gpu
    result = run_exp(cfg, log_dir="exp_logs")
    result["config"] = {k: (v.value if hasattr(v, "value") else v) for k, v in cfg.items()}
    s1 = cfg.get("small_draft_threshold", "-")
    s2 = cfg.get("draft_target_threshold", cfg.get("uncertainty_threshold", "-"))
    marker = "✓" if result["status"] == "success" else "✗"
    series = cfg["little_model"].split("/")[-1]
    print(f"  [{gpu}] {marker} #{idx} {cfg['eval_mode']} S1={s1} S2={s2} {series} → {result['status']}")
    return result


def print_summary(results: list[dict]) -> None:
    from collections import defaultdict
    by_series = defaultdict(list)
    for r in results:
        if r["status"] != "success":
            continue
        cfg = r["config"]
        met = r["result"]
        series = f"{cfg['little_model'].split('/')[-1]}→{cfg['draft_model'].split('/')[-1]}→{cfg['target_model'].split('/')[-1]}"
        acc = met.get("accuracy", -1)
        tokens = met.get("generated_tokens", 0)
        tgt = met.get("target_forward_times", 0)
        wall = met.get("wall_time", 0)
        tptf = tokens / tgt if tgt else 0
        tps = tokens / wall if wall else 0
        method = cfg["eval_mode"]
        if method == EvalMode.cuhlm.value:
            s1, s2 = "-", cfg.get("uncertainty_threshold", "-")
            label = f"uncertainty thr={s2}"
        else:
            s1 = cfg.get("small_draft_threshold", "-")
            s2 = cfg.get("draft_target_threshold", "-")
            label = f"cee S1={s1} S2={s2}"
        by_series[series].append((acc, tptf, tps, tgt, wall, s1, s2, label))

    for series, entries in by_series.items():
        print(f"\n{'='*95}")
        print(f"Series: {series}")
        print(f"{'='*95}")
        entries.sort(key=lambda x: (-x[0], -x[1]))
        print(f"{'Rank':<5} {'Config':<28} {'Acc':>6} {'Tok/Tgt':>8} {'t/s':>7} {'TgtFwd':>8} {'Wall(s)':>8}")
        print("-" * 95)
        for i, (acc, tptf, tps, tgt, wall, s1, s2, label) in enumerate(entries[:15]):
            print(f"{i+1:<5} {label:<28} {acc:>6.3f} {tptf:>8.1f} {tps:>7.1f} {tgt:>8} {wall:>8.1f}")


if __name__ == "__main__":
    configs = build_configs()
    print(f"Configs: {len(configs)}, GPUs: {GPUS}")
    for _, c in configs[:3]:
        print(f"  {c['eval_mode']} {c['little_model'].split('/')[-1]}→{c['target_model'].split('/')[-1]}")

    results_dir = Path("experiment_results")
    results_dir.mkdir(exist_ok=True)
    summary_file = str(results_dir / f"cross_series_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    all_results = []
    gpu_iter = itertools.cycle(GPUS)

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = {}
        for idx, cfg in configs:
            gpu = next(gpu_iter)
            f = pool.submit(run_one, cfg, gpu, idx)
            futures[f] = idx

        for f in as_completed(futures):
            result = f.result()
            all_results.append(result)
            with open(summary_file, "w") as fh:
                json.dump(all_results, fh, indent=2, ensure_ascii=False)

    print_summary(all_results)
    print(f"\nFull results: {summary_file}")
