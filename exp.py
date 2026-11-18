"""Run the Table V experiment matrix. Existing sweeps can also import this module."""

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.experiments.config import EvalDataset, EvalMode, ExpConfig, create_config
from src.experiments.legacy import (
    add_args,
    cmd_temp,
    get_file_path,
    get_model_series,
    resolve_accelerate,
)
from src.experiments.matrix import (
    GAMMA_BY_MODE_BANDWIDTH,
    GAMMA_SINGLE,
    NTT_MS_EDGE_CLOUD,
    NTT_MS_EDGE_END,
    TRANSFER_TOP_K_OURS,
    TRANSFER_TOP_K_PAPER,
    TRANSFER_TOP_K_STATIC,
    TRANSFER_TOP_K_TKSLT,
    UNCERTAINTY_THRESHOLD_CUHLM,
    UNCERTAINTY_THRESHOLD_DEFAULT,
    batch_delay_values,
    build_matrix,
    edge_cloud_bandwidth,
    filter_configs_by_mode,
    gamma_for_matrix,
    llama_series,
    paper_table5_datasets,
    paper_table5_modes,
    paper_table5_series,
    qwen_1_5_series,
    qwen_series,
)
from src.experiments.scheduler import (
    GPUManager,
    get_available_gpus,
    run_experiments_parallel,
)
from src.experiments.summaries import (
    RUN_IDENTITY_FIELDS,
    filter_configs_for_resume,
    load_summary,
    merge_summaries,
    run_identity,
    save_summary,
    successful_runs,
)
from src.experiments.workers import run_exp

# Imports used by existing sweep scripts and notebooks. New callers should use
# build_matrix() rather than mutating the historical shared config_to_run list.
__all__ = [
    "EvalDataset",
    "EvalMode",
    "ExpConfig",
    "build_matrix",
    "create_config",
    "config_to_run",
    "filter_configs_by_mode",
    "filter_configs_for_resume",
    "run_exp",
    "run_experiments_parallel",
    "get_available_gpus",
    "GPUManager",
    "describe_config",
    "parse_exp_args",
    "main",
    "run_identity",
    "RUN_IDENTITY_FIELDS",
    "load_summary",
    "save_summary",
    "merge_summaries",
    "gamma_for_matrix",
    "llama_series",
    "qwen_1_5_series",
    "qwen_series",
    "paper_table5_series",
    "paper_table5_modes",
    "paper_table5_datasets",
    "edge_cloud_bandwidth",
    "batch_delay_values",
    "GAMMA_SINGLE",
    "GAMMA_BY_MODE_BANDWIDTH",
    "NTT_MS_EDGE_CLOUD",
    "NTT_MS_EDGE_END",
    "TRANSFER_TOP_K_OURS",
    "TRANSFER_TOP_K_PAPER",
    "TRANSFER_TOP_K_TKSLT",
    "TRANSFER_TOP_K_STATIC",
    "UNCERTAINTY_THRESHOLD_CUHLM",
    "UNCERTAINTY_THRESHOLD_DEFAULT",
    "add_args",
    "cmd_temp",
    "get_file_path",
    "get_model_series",
    "resolve_accelerate",
]
config_to_run: list[ExpConfig]


def __getattr__(name: str):
    if name == "config_to_run":
        configs = build_matrix()
        globals()[name] = configs
        return configs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def describe_config(config: dict) -> str:
    dataset = Path(str(config.get("eval_dataset", "?"))).stem.removeprefix("eval_")
    return (
        f"{str(config.get('eval_mode', '?')):22s} {dataset:16s} "
        f"{config.get('target_model', '?')}"
    )


def parse_exp_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="论文 Table V 实验矩阵：三族模型 × 三数据集 × 五方法 × 三档带宽"
    )
    parser.add_argument(
        "--only-modes",
        metavar="MODE[,MODE...]",
        help="只跑指定方法，接受枚举名或值（dsd / dist_spec）",
    )
    parser.add_argument(
        "--extra-modes",
        metavar="MODE[,MODE...]",
        help="向主表附加载入方法（如 tridecoding）；可与 --only-modes 合用",
    )
    parser.add_argument(
        "--resume-from", metavar="SUMMARY.json", help="跳过该汇总里已成功的配置"
    )
    parser.add_argument(
        "--summary-file", metavar="PATH", help="本次运行的增量汇总（默认按时间戳新建）"
    )
    parser.add_argument(
        "--merge-from",
        action="append",
        metavar="SUMMARY.json",
        help="合并旧汇总，可重复传；同一配置以本次结果为准",
    )
    parser.add_argument("--merged-output", metavar="PATH", help="合并后的输出文件")
    parser.add_argument(
        "--max-workers", type=int, default=4, help="最多并行运行的模型组数（默认 4）"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印待跑清单，不启动 worker 或写文件"
    )
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers 必须大于 0")
    return args


def _report(results: list[dict], summary_file: str) -> None:
    counts = Counter(r["status"] for r in results)
    print(f"\n实验汇总报告:\n总实验数: {len(results)}")
    for key, label in (
        ("success", "成功"),
        ("failed", "失败"),
        ("no_result", "无结果"),
        ("exception", "异常"),
    ):
        print(f"{label}: {counts[key]}")
    if counts["skipped_capacity_mismatch"]:
        print(f"所需 GPU 数量不足而跳过: {counts['skipped_capacity_mismatch']}")
    print(f"本次汇总已保存到: {summary_file}")
    for result in results:
        print(f"\n实验: {result['exp_name']}\n状态: {result['status']}")
        if result.get("log_file"):
            print(f"日志: {result['log_file']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_exp_args(argv)
    try:
        configs = build_matrix(args.extra_modes)
        selected = filter_configs_by_mode(configs, args.only_modes)
        pending = filter_configs_for_resume(selected, args.resume_from)
    except (ValueError, OSError) as exc:
        print(f"无法生成待跑清单: {exc}", file=sys.stderr)
        return 2

    if args.only_modes:
        print(
            f"--only-modes {args.only_modes} ⇒ "
            f"从 {len(configs)} 个中筛出 {len(selected)} 个"
        )
    print(f"本次待跑 {len(pending)} 个")
    if skipped := len(selected) - len(pending):
        print(f"（按 {args.resume_from} 的成功记录跳过 {skipped} 个）")
    for config in pending:
        print(f"  - {describe_config(config)}")
    if args.dry_run:
        print("\n[dry-run] 未执行任何实验。")
        return 0
    if not pending:
        print("\n没有待跑的实验，无需运行。")
        return 0

    # Read merge inputs before the first incremental write, even when a caller
    # reuses an old summary path for the current run.
    prior = merge_summaries(*(load_summary(p) for p in args.merge_from or ()))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = (
        args.summary_file or f"experiment_results/experiment_summary_{stamp}.json"
    )
    results = run_experiments_parallel(
        pending,
        max_workers=args.max_workers,
        log_dir="exp_logs",
        summary_file=summary_file,
    )
    if not results:
        print("没有得到本次运行结果，未生成新汇总。", file=sys.stderr)
        return 1
    _report(results, summary_file)

    if args.merge_from:
        merged = merge_summaries(prior, results)
        output = (
            args.merged_output
            or f"experiment_results/experiment_summary_merged_{stamp}.json"
        )
        save_summary(output, merged)
        resolved = len(successful_runs(merged) - successful_runs(prior))
        successes = sum(e.get("status") == "success" for e in merged)
        print(f"\n合并自: {', '.join(args.merge_from)}")
        print(
            f"合并后条目数: {len(merged)}"
            f"（success {successes} / 非 success {len(merged) - successes}）"
        )
        print(f"本次新转为 success: {resolved}\n合并结果已保存到: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
