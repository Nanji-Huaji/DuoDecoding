#!/usr/bin/env python3
"""γ（投机深度）扫描：为 paper_table5 主实验定各基线的 γ。

为什么需要这个
--------------
docs/protocol.md §4：γ 是协议里**唯一未冻结**的超参。主矩阵目前对所有
单-γ 基线统一取 ``GAMMA_SINGLE = 3``（对齐论文前向数），但 TK-SLT 的
理论（Theorem 2）表明 γ* 依赖 (α, b, c) = (接受率, 通信/验证比, 草稿/
验证比)——不同方法、不同模型对的 γ* 本来就不该一样。这个脚本把"γ 取
几"从拍脑袋变成可复现的实验：

  1. 在主矩阵的**同一套配置**下（NTT/带宽/batch_delay/数据集/K/阈值
     全部不动，协议冻结 paper_table5 + honest 计费），只扫 γ；
  2. 断点续跑（成功过的格子自动跳过，可随时 Ctrl-C 再续）；
  3. 聚合成 每方法 × γ 的吞吐/准确率表，并给出推荐：
       · 各方法 γ* = argmax 吞吐（准确率守恒约束下）；
       · 若 γ* 相对 γ=3 提升 < 5% 或呈平台 → 建议保持统一 γ=3
         （可控比较，少一档解释负担）；
       · TK-SLT 额外用实测 (α̂, b̂, ĉ) 回放论文 Theorem 2 的解析 γ*，
         与经验 argmax 互为交叉验证。

temp=0（cmd_temp 冻结）下投机解码与贪心解码 argmax 等价，准确率**不应**
随 γ 变化——表里的 acc 列只是漂移监控（若大幅波动，先查实现再谈调参）。

用法
----
    # 看计划（不占卡）：
    .venv/bin/python scripts/gamma_sweep.py --dry-run
    # 默认：4 个单-γ 基线 × γ∈1..8 × Llama 系 × GSM8K × 80 样本
    .venv/bin/python scripts/gamma_sweep.py
    # 粗扫（20 样本）/ 换系列 / 只扫两个方法 / 重复取均值：
    .venv/bin/python scripts/gamma_sweep.py --quick
    .venv/bin/python scripts/gamma_sweep.py --series qwen15 --dataset mt_bench_noeval
    .venv/bin/python scripts/gamma_sweep.py --modes dsd,tk_slt --repeats 3
    # 只重读汇总出报告（不跑）：
    .venv/bin/python scripts/gamma_sweep.py --analyze-only

汇总/报告写在 experiment_results/gamma_sweep_{tag}.{json,md}（tag 稳定，
跨次运行续跑合并）。K 不在扫描范围：TK-SLT 的 K=320 是论文 Fig 4 的
最优工作点（exp.py: TRANSFER_TOP_K_TKSLT），已有 Table II 交叉验证。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

# 直接执行本脚本时 sys.path[0] 是 scripts/，仓库根不在里面
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import exp  # noqa: E402 - 需先修好 sys.path
from exp import (  # noqa: E402
    EvalDataset,
    create_config,
    describe_config,
    filter_configs_for_resume,
    load_summary,
    merge_summaries,
    run_experiments_parallel,
)

# --------------------------------------------------------------------------- #
# 方法表：与 exp.py 主矩阵 dispatch 逐项对齐（uncertainty_threshold /
# transfer_top_k / draft_model 档位 / RL），扫描只动 γ。改主矩阵时同步这里。
# --------------------------------------------------------------------------- #
MODE_CHOICES: dict[str, dict] = {
    # 显示名: (eval_mode 值, uncertainty_threshold, transfer_top_k, 中模型当草稿)
    "dsd": ("dist_spec", 0.8, 0, False),
    "dssd": ("dist_split_spec", 0.8, 0, True),
    "cuhlm": ("uncertainty_decoding", exp.UNCERTAINTY_THRESHOLD_CUHLM, 300, False),
    "tk_slt": ("tk_slt", 0.8, exp.TRANSFER_TOP_K_TKSLT, False),
    # 可选：本方法（三级，γ1=γ2=γ 耦合粗扫；完整 2D 网格另做）
    "ceesd": ("adaptive_tridecoding", 0.8, 300, True),
}
DEFAULT_MODES = "dsd,dssd,cuhlm,tk_slt"

SERIES_CHOICES: dict[str, tuple[str, str, str]] = {
    "llama": exp.llama_series,
    "qwen15": exp.qwen_1_5_series,
    "qwen3": exp.qwen_series,
}

DATASET_CHOICES = {
    "gsm8k": EvalDataset.gsm8k,
    "mt_bench_noeval": EvalDataset.mt_bench_noeval,
    "humaneval": EvalDataset.humaneval,
}

# 推荐阈值：γ* 相对 γ=3 的提升小于该值 ⇒ 建议保持统一 γ
PLATEAU_GAIN_THRESHOLD = 0.05
# 准确率守恒容差：temp=0 下应严格不变；4-bit 目标 + 批量验证前向随 γ 改变形状，
# ±n/eval_data_num 量级的样本翻转（n=80 时 ±0.05 = 4 个样本）属数值漂移
ACC_TOLERANCE = 0.05


def _config_signature(cfg: dict) -> tuple:
    """配置的匹配签名：去掉 exp_name（时间戳）与 GPU 分配，其余逐字段参与。

    恢复与去重都靠它判断"这条日志/结果是不是本扫描的格子"。
    """
    return tuple(
        sorted(
            (k, v)
            for k, v in cfg.items()
            if k not in ("exp_name", "CUDA_VISIBLE_DEVICES")
        )
    )


def _parse_log_config(log_path: str) -> dict | None:
    """解析日志头部的"实验配置: {json}"（indent=2 多行，从全文偏移处解码）。

    配置不在首行（accelerate 横幅在前），且跨多行，必须整文定位后
    raw_decode 到配对的大括号为止。
    """
    try:
        with open(log_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    marker = "\n实验配置: "
    i = text.find(marker)
    offset = len(marker)
    if i < 0:
        if text.startswith("实验配置: "):
            i, offset = 0, len("实验配置: ")
        else:
            return None
    try:
        cfg, _ = json.JSONDecoder().raw_decode(text[i + offset :])
    except json.JSONDecodeError:
        return None
    return cfg if isinstance(cfg, dict) else None


def recover_from_logs(args, summary_path: Path) -> None:
    """从 exp_logs + exp/ 工件恢复本扫描的格子（summary 被覆盖时的安全网）。

    背景：2026-10-09 的覆盖事故里，扩展批（旧代码）把 summary 覆盖成了
    本批的几格；但每格的日志（头部带完整配置）与 metrics 工件都还在
    exp_logs/ 和 exp/ 下。本函数按"期望配置签名 + 日期守卫 + metrics
    存在"三条规则回收，并与现有 summary 合并写回。

    日期守卫（--since，默认 20261008 = 本脚本落地日）：排除 9 月的
    t5a 对齐跑——它们的参数与扫描格完全同 signature（配置里不含
    protocol/accounting 字段），但通信口径不同，数字不可比。
    """
    dataset_name = DATASET_CHOICES[args.dataset].name
    expected = {_config_signature(c) for c in build_configs(args)}
    current = load_summary(summary_path) if summary_path.exists() else []
    have = {
        _config_signature(e["config"])
        for e in current
        if e.get("status") == "success"
    }

    recovered = []
    for log in sorted(glob.glob(f"exp_logs/*{dataset_name}*.log")):
        cfg = _parse_log_config(log)
        if cfg is None:
            continue
        if _config_signature(cfg) not in expected:
            continue
        exp_name = cfg.get("exp_name", "")
        m = re.search(r"_(\d{8})_\d{6}_\d+$", exp_name)
        if not m or m.group(1) < args.since:
            continue
        metrics_path = exp.get_file_path(exp_name)
        if not metrics_path:
            print(f"  ⚠ metrics 工件缺失，跳过: {exp_name[:70]}")
            continue
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        recovered.append(
            {
                "exp_name": exp_name,
                "result": metrics,
                "log_file": log,
                "status": "success",
                "config": cfg,
            }
        )

    fresh = [e for e in recovered if _config_signature(e["config"]) not in have]
    merged = merge_summaries(recovered, current)
    tmp_summary = str(summary_path) + ".tmp"
    with open(tmp_summary, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    os.replace(tmp_summary, summary_path)
    print(
        f"[recover] 日志扫描命中 {len(recovered)} 格，其中 {len(fresh)} 格"
        f"不在 summary 里，合并后共 {len(merged)} 条 → {summary_path}"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="paper_table5 基线的 γ 扫描与超参推荐",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gammas",
        default="1,2,3,4,5,6,7,8",
        help="逗号分隔的 γ 取值（如 1,2,3,5,8,12）",
    )
    parser.add_argument(
        "--modes",
        default=DEFAULT_MODES,
        help=f"逗号分隔，可选: {','.join(MODE_CHOICES)}",
    )
    parser.add_argument(
        "--series",
        default="llama",
        choices=sorted(SERIES_CHOICES),
        help="模型系列（little/draft/target 三元组）",
    )
    parser.add_argument(
        "--dataset",
        default="gsm8k",
        choices=sorted(DATASET_CHOICES),
        help="评测集（gsm8k 有 accuracy，mt_bench_noeval 纯吞吐）",
    )
    parser.add_argument(
        "--eval-data-num", type=int, default=80, help="样本数（主矩阵为 80）"
    )
    parser.add_argument("--quick", action="store_true", help="粗扫：20 样本")
    parser.add_argument(
        "--repeats", type=int, default=1, help="每格重复次数（>1 时按 seed 交错取均值）"
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=46.0,
        help="edge-cloud/WAN 带宽 Mbps（主矩阵对齐值为 46；同时进文件 tag 与"
        "理论模型的 b̂）。注意：链路模拟器默认 min_bandwidth_mbps=5，"
        "低于 5 会被静默钳位",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="并行 GPU 数（默认 4）"
    )
    parser.add_argument(
        "--recover-from-logs",
        action="store_true",
        help="从 exp_logs 日志配置 + exp/ metrics 工件恢复本扫描的格子并"
        "合并写回 summary（覆盖事故的安全网），随后照常出报告",
    )
    parser.add_argument(
        "--since",
        default="20261008",
        help="恢复时只认 exp_name 日期 >= 该值的运行（YYYYMMDD）。默认"
        "20261008=本脚本落地日，用于排除 9 月 t5a 对齐跑（参数同 signature"
        "但通信口径不同）",
    )
    parser.add_argument(
        "--no-stochastic",
        action="store_true",
        help="关掉随机带宽 trace（默认与主矩阵一致开启；关掉更快但口径偏离主表）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印待跑清单")
    parser.add_argument(
        "--analyze-only", action="store_true", help="不跑实验，只重读汇总出报告"
    )
    parser.add_argument(
        "--summary-file", default=None, help="覆盖默认汇总路径（默认按 tag 命名）"
    )
    return parser.parse_args(argv)


def _resolve_modes(spec: str) -> list[tuple[str, str]]:
    wanted = [token.strip() for token in spec.split(",") if token.strip()]
    resolved = []
    for name in wanted:
        if name not in MODE_CHOICES:
            raise SystemExit(
                f"未知方法 {name!r}；可选: {','.join(MODE_CHOICES)}（也接受值形式，"
                f"如 dist_spec=dsd）"
            )
        resolved.append((name, MODE_CHOICES[name][0]))
    return resolved


def _parse_gammas(spec: str) -> list[int]:
    gammas = sorted({int(token) for token in spec.split(",") if token.strip()})
    if not gammas or any(g < 1 for g in gammas):
        raise SystemExit(f"γ 取值非法: {spec!r}")
    return gammas


#: 扫描使用的带宽模型（与主矩阵/paper_table5 冻结一致）。cmd_temp 会显式
#: 传 --comm_bw_model，而 apply_protocol 只填未显式设置的项——必须在
#: build_configs 里显式给 fluid，否则 create_config 的默认 instant 会
#: 压过协议冻结。summary 文件名的 _fluid 后缀也读这里。
SWEEP_BW_MODEL = "fluid"


def default_summary_path(args, tag_modes: str) -> Path:
    dataset_name = DATASET_CHOICES[args.dataset].name
    tag = f"{tag_modes}_{args.series}_{dataset_name}"
    if args.quick:
        tag += "_quick"
    # 带宽进 tag（46 = 主矩阵对齐值/历史默认，不后缀以保持旧文件名兼容）
    if args.bandwidth != 46.0:
        tag += f"_bw{args.bandwidth:g}"
    # 带宽模型进 tag（2026-10-09 统一口径 §3.4）：fluid 的通信数字与
    # 历史.instant 不可比，且 _config_signature 含 comm_bw_model/
    # min_bandwidth_mbps，混在一个文件里会出现同 (mode,γ) 双格。
    # 历史文件名（无后缀）= instant 口径，保持不动。
    if SWEEP_BW_MODEL == "fluid":
        tag += "_fluid"
    return Path("experiment_results") / f"gamma_sweep_{tag}.json"


def build_configs(args) -> list[exp.ExpConfig]:
    """按主矩阵口径构建扫描配置（只动 γ；repeats 用 seed 交错区分身份）。"""
    modes = _resolve_modes(args.modes)
    gammas = _parse_gammas(args.gammas)
    little_model, middle_model, target_model = SERIES_CHOICES[args.series]
    dataset = DATASET_CHOICES[args.dataset]
    eval_data_num = 20 if args.quick else args.eval_data_num

    configs = []
    for display, value in modes:
        _, uncertainty_threshold, transfer_top_k, use_middle = MODE_CHOICES[display]
        is_ceesd = value == "adaptive_tridecoding"
        for gamma in gammas:
            for repeat in range(args.repeats):
                configs.append(
                    create_config(
                        eval_mode=value,
                        ntt_ms_edge_cloud=exp.NTT_MS_EDGE_CLOUD,
                        ntt_ms_edge_end=exp.NTT_MS_EDGE_END,
                        batch_delay=50e-3,
                        use_precise=False,
                        use_stochastic_comm=not args.no_stochastic,
                        edge_end_bandwidth=563,
                        edge_cloud_bandwidth=args.bandwidth,
                        cloud_end_bandwidth=args.bandwidth,
                        # 与主矩阵同公式的带宽下限（见 exp.py 注释）：固定 5
                        # 会在 5 Mbps 档把 trace 削成近似常数
                        min_bandwidth_mbps=max(0.5, args.bandwidth / 10),
                        # cmd_temp 显式传 --comm_bw_model，apply_protocol 只填
                        # 未显式设置的项——必须显式给（见 SWEEP_BW_MODEL 注释）
                        comm_bw_model=SWEEP_BW_MODEL,
                        small_draft_threshold=0.6,
                        draft_target_threshold=0.7,
                        uncertainty_threshold=uncertainty_threshold,
                        transfer_top_k=transfer_top_k,
                        gamma=gamma,
                        # 基线 g1=g2=γ（矩阵约定）；ceesd 耦合粗扫
                        gamma1=gamma,
                        gamma2=gamma,
                        max_tokens=128,
                        num_shots=3,
                        eval_dataset=dataset,
                        draft_model=(
                            middle_model if use_middle else little_model
                        ),
                        target_model=target_model,
                        little_model=little_model,
                        use_rl_adapter=is_ceesd,
                        disable_rl_update=is_ceesd,
                        use_early_stopping=False,
                        use_cuda_graph=True,
                        eval_data_num=eval_data_num,
                        run_full_dataset=False,
                        random_sample=True,
                        sample_seed=1234 + repeat,
                    )
                )
    return configs


# --------------------------------------------------------------------------- #
# 聚合与分析
# --------------------------------------------------------------------------- #
def _num(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def load_cells(summary_path: Path) -> dict[tuple[str, int], dict]:
    """汇总 JSON → {(eval_mode, gamma): 聚合统计}。

    同一格子多次重复（--repeats）时取均值；status != success 的跳过。
    """
    if not summary_path.exists():
        raise SystemExit(
            f"找不到汇总文件: {summary_path}（先跑扫描，或检查 --summary-file）"
        )
    entries = json.loads(summary_path.read_text(encoding="utf-8"))
    buckets: dict[tuple[str, int], list[dict]] = {}
    for entry in entries:
        if entry.get("status") != "success":
            continue
        config = entry.get("config") or {}
        metrics = entry.get("result")
        if not isinstance(metrics, dict) or "throughput" not in metrics:
            continue
        mode = str(config.get("eval_mode", "?"))
        gamma = config.get("gamma")
        if not isinstance(gamma, int):
            continue
        buckets.setdefault((mode, gamma), []).append(
            {
                "accuracy": _num(metrics, "accuracy"),
                "throughput": _num(metrics, "throughput"),
                "wall_time": _num(metrics, "wall_time"),
                "communication_time": _num(metrics, "communication_time"),
                "queuing_time": _num(metrics, "queuing_time"),
                "draft_accepted": _num(metrics, "draft_accepted_tokens"),
                "draft_generated": _num(metrics, "draft_generated_tokens"),
                "draft_comp": _num(metrics, "draft_computation_time"),
                "draft_forward": _num(metrics, "draft_forward_times"),
                "target_comp": _num(metrics, "target_computation_time"),
                "target_forward": _num(metrics, "target_forward_times"),
                "generated_tokens": _num(metrics, "generated_tokens"),
                "avg_top_k": _num(metrics, "avg_top_k"),
            }
        )

    cells = {}
    for key, rows in buckets.items():
        agg = {}
        for field in rows[0]:
            values = [r[field] for r in rows if r[field] is not None]
            agg[field] = statistics.mean(values) if values else None
        agg["n_repeats"] = len(rows)
        # 接受率 α̂（每 token，方法对的性质，近似 γ 无关）
        if agg["draft_generated"]:
            agg["alpha"] = agg["draft_accepted"] / agg["draft_generated"]
        else:
            agg["alpha"] = None
        cells[key] = agg
    return cells


def _fmt(value, pattern="{:8.3f}", none="—"):
    return none if value is None else pattern.format(value)


def render_table(mode_display: str, mode_value: str, cells, gammas) -> str:
    lines = [
        f"\n== {mode_display} ({mode_value}) ==",
        f"{'γ':>3} | {'acc':>8} | {'tput(tok/s)':>11} | {'Δvs γ=3':>9} | "
        f"{'α̂':>6} | {'comm(s)':>8} | {'wall(s)':>8} | {'n':>2}",
        "-" * 78,
    ]
    base = cells.get((mode_value, 3), {}).get("throughput")
    for gamma in gammas:
        row = cells.get((mode_value, gamma))
        if row is None:
            lines.append(f"{gamma:>3} | {'(未跑/失败)':>8} |")
            continue
        delta = (
            row["throughput"] / base - 1 if base else None
        )
        lines.append(
            f"{gamma:>3} | {_fmt(row['accuracy'], '{:8.4f}')} | "
            f"{_fmt(row['throughput'], '{:11.3f}')} | "
            f"{_fmt(delta, '{:+8.1%}') if delta is not None else '—':>9} | "
            f"{_fmt(row['alpha'], '{:6.3f}')} | "
            f"{_fmt(row['communication_time'], '{:8.2f}')} | "
            f"{_fmt(row['wall_time'], '{:8.2f}')} | "
            f"{row['n_repeats']:>2}"
        )
    return "\n".join(lines)


def recommend_mode(
    mode_display: str, mode_value: str, cells, gammas
) -> tuple[list[str], int | None]:
    """决策阶梯（返回 (notes, 选定的 γ 或 None)）：

    1. 曲线全程平坦（极差 <1%）      ⇒ γ 不敏感，保持 3；
    2. 最优就是 γ=3                  ⇒ 保持 3；
    3. argmax 落在扫描边界            ⇒ 曲线仍在上升，先扩 --gammas，不定论；
    4. 相对 γ=3 提升 <5%             ⇒ 收益不抵解释成本，保持统一 3；
    5. 其余                          ⇒ 按方法取 γ*（最优簇内取最小 γ）。
    """
    rows = {
        gamma: cells[(mode_value, gamma)]
        for gamma in gammas
        if (mode_value, gamma) in cells
    }
    if not rows:
        return [f"[{mode_display}] 无成功数据，无法推荐。"], None
    notes = [f"\n[{mode_display}] 推荐："]
    pick: int | None = None
    acc_max = max((r["accuracy"] or 0.0) for r in rows.values())
    feasible = {
        gamma: r
        for gamma, r in rows.items()
        if r["accuracy"] is None or r["accuracy"] >= acc_max - ACC_TOLERANCE
    }
    gamma_star = max(feasible, key=lambda g: feasible[g]["throughput"])
    best_tput = feasible[gamma_star]["throughput"]

    tputs = [r["throughput"] for r in rows.values() if r["throughput"]]
    spread = (max(tputs) - min(tputs)) / max(tputs) if tputs else 0.0
    if spread < 0.01:
        notes.append(
            f"  γ 完全不敏感（全程极差 {spread:.2%}，通信/接受率逐 γ 相同）"
            f" ⇒ 保持 GAMMA_SINGLE = 3（该工作点下 γ 无行为影响）。"
        )
        return notes, None

    # 最优簇：与 best 差 <3% 的 γ 集合；同等收益取最小 γ（窗口小、边际成本低）
    cluster = sorted(
        g
        for g, r in feasible.items()
        if r["throughput"] and best_tput - r["throughput"] < 0.03 * best_tput
    )
    gamma_pick = cluster[0] if cluster else gamma_star

    base = rows.get(3, {}).get("throughput")
    # 增益一律按"采纳 pick 后相对 γ=3 的实际变化"报，边界提示用 argmax 的增益
    gain_pick = (
        feasible[gamma_pick]["throughput"] / base - 1 if base else None
    )
    gain_best = best_tput / base - 1 if base else None
    if gamma_star == 3:
        notes.append(
            f"  γ=3 即最优（tput {best_tput:.2f}）⇒ 保持 GAMMA_SINGLE = 3 不变。"
        )
    elif gamma_star == max(gammas):
        notes.append(
            f"  ⚠ 曲线在扫描边界 γ={max(gammas)} 仍在上升（边界 tput {best_tput:.2f}"
            + (f"，相对 γ=3 {gain_best:+.1%}" if gain_best is not None else "")
            + f"）⇒ 先扩 --gammas（如 --gammas {max(gammas)},"
            f"{max(gammas) + 2},{max(gammas) + 4},{max(gammas) + 8}）再定，"
            f"勿直接采用边界值。"
        )
    elif gain_pick is not None and gain_pick < PLATEAU_GAIN_THRESHOLD:
        notes.append(
            f"  γ* = {gamma_pick}（tput {feasible[gamma_pick]['throughput']:.2f}），"
            f"但相对 γ=3 仅 {gain_pick:+.1%} ⇒ 建议保持统一 GAMMA_SINGLE = 3"
            f"（可控比较，收益不抵解释成本）。"
        )
    else:
        notes.append(
            f"  γ* = {gamma_pick}（tput {feasible[gamma_pick]['throughput']:.2f}"
            + (f"，相对 γ=3 {gain_pick:+.1%}" if gain_pick is not None else "")
            + f"；最优簇 {cluster} 内差异 <3%）"
            f" ⇒ 建议主表按方法取 γ（最强基线原则：各基线调到自身最优再比）。"
        )
        pick = gamma_pick

    accs = [r["accuracy"] for r in rows.values() if r["accuracy"] is not None]
    if accs and max(accs) - min(accs) > ACC_TOLERANCE:
        notes.append(
            f"  ⚠ 准确率随 γ 波动 {max(accs) - min(accs):.4f}（temp=0 下应不变）——"
            f"超出 ±{ACC_TOLERANCE:.0%} 的数值漂移预期，查实现后再谈调参。"
        )
    return notes, pick


def _solve_acceptance_p(alpha_hat: float, gamma: int) -> float | None:
    """由 α̂ = E[accepted]/γ 反解每 token 接受率 p（几何接受模型）。

    E[accepted] = p(1−p^γ)/(1−p)，p∈(0,1) 上单调 ⇒ 二分。
    """
    target = alpha_hat * gamma

    def f(p: float) -> float:
        return p * (1 - p**gamma) / (1 - p) - target

    lo, hi = 1e-6, 0.9999
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _argmax_speedup(p: float, lam: float, L: float, cap: int = 64) -> tuple[int, float]:
    """S(γ) = (1−p^{γ+1})/((1−p)·(1+λ+γL)) 的整数 argmax（γ∈1..cap）。"""

    def s(g: int) -> float:
        return (1 - p ** (g + 1)) / ((1 - p) * (1 + lam + g * L))

    best_g, best_s = 1, s(1)
    for g in range(2, cap + 1):
        if s(g) > best_s:
            best_g, best_s = g, s(g)
    return best_g, best_s


def tk_slt_theory(
    mode_value: str, cells, gammas, bandwidth_mbps: float = 46.0
) -> list[str]:
    """理论交叉验证：论文模型 vs 仓库链路模型。

    论文 Theorem 2 的 S_inf 假设每轮成本 = T_LLM + γ(T_V + T_SLM)——**没有
    每轮固定开销**（Shannon 容量模型，无 per-message NTT、无云排队）。仓库
    链路模型对每个物理报文收 NTT（edge-cloud 上/下行各 50ms），主矩阵还叠
    加 batch_delay（每次目标前向 50ms 排队）——这些是 **γ 无关** 的每轮
    固定成本，记 λ = 固定开销/T_LLM。于是：

        S_paper(γ) = E[tokens]/(1+γL)，        L = b+c
        S_repo(γ)  = E[tokens]/(1+λ+γL)

    λ>0 会把最优 γ 显著推高（固定开销越大，越要用长草稿摊薄）。两条曲线
    都算出来与经验 argmax 对照：若经验贴合 S_repo 而不是 S_paper，说明
    差异来自链路模型而非实现问题。
    """
    rows = {
        gamma: cells[(mode_value, gamma)]
        for gamma in gammas
        if (mode_value, gamma) in cells
    }
    if not rows:
        return []
    if not rows[max(rows)]["alpha"]:
        return []

    # 参考点取 γ 最大的格子（前向数最多，T_LLM/T_SLM/轮数估计最稳）
    ref_gamma, ref = max(rows.items(), key=lambda kv: kv[0])
    if not ref["target_forward"] or not ref["draft_forward"]:
        return []
    rounds = ref["target_forward"]  # 每轮恰好一次验证前向（fallback 轮同样计）
    t_llm = ref["target_comp"] / ref["target_forward"]
    t_slm = ref["draft_comp"] / ref["draft_forward"]
    if t_llm <= 0 or not rounds:
        return []
    # b̂：每分布上行纯发射时长 / T_LLM。K·2 字节（FP16），带宽按本次扫描
    # 的 --bandwidth（主矩阵口径；推导见 docs/protocol.md §3.3）。
    k = exp.TRANSFER_TOP_K_TKSLT
    bandwidth_bps = bandwidth_mbps * 1e6 / 8
    b = (k * 2 / bandwidth_bps) / t_llm
    c = t_slm / t_llm
    # λ̂：每轮固定开销（通信含 NTT + 排队）/ T_LLM，扣除 γ 比例的载荷部分
    overhead = (ref["communication_time"] or 0.0) + (ref["queuing_time"] or 0.0)
    lam = max(0.0, overhead / rounds / t_llm - ref_gamma * b)
    L = b + c

    p = _solve_acceptance_p(ref["alpha"] or 0.0, ref_gamma)
    comm_ms = (ref["communication_time"] or 0) / rounds * 1e3
    queue_ms = (ref["queuing_time"] or 0) / rounds * 1e3
    if p is None:
        return [
            f"\n[TK-SLT 理论交叉验证]（参考格 γ={ref_gamma}）",
            f"  α̂ = {ref['alpha']:.3f}，b̂ = {b:.4f}，ĉ = {c:.4f} ⇒ L = {L:.4f}"
            f"（p̂ 反解失败，跳过模型对照）",
        ]
    gamma_paper, s_paper = _argmax_speedup(p, 0.0, L)
    gamma_repo, s_repo = _argmax_speedup(p, lam, L)
    empirical = max(rows, key=lambda g: rows[g]["throughput"] or 0)
    return [
        f"\n[TK-SLT 理论交叉验证]（参考格 γ={ref_gamma}，轮数 {rounds:.0f}）",
        f"  p̂ = {p:.3f}（由该格 α̂={ref['alpha']:.3f} 反解），b̂ = {b:.4f}，"
        f"ĉ = {c:.4f} ⇒ L = {L:.4f}",
        f"  λ̂ = {lam:.2f}（每轮固定开销 {comm_ms:.0f}ms 通信 + {queue_ms:.0f}ms 排队"
        f" ≈ {overhead / rounds * 1e3:.0f}ms，为 T_LLM 的 {lam:.1f} 倍）",
        f"  论文模型（无固定开销）  : γ* = {gamma_paper:2d}, S* = {s_paper:.3f}",
        f"  仓库链路模型（含 λ̂）   : γ* = {gamma_repo:2d}, S* = {s_repo:.3f}",
        f"  经验 argmax             : γ = {empirical}",
        "  ⇒ 若经验贴合仓库模型而非论文模型，差异来自链路模型（NTT+排队是"
        "论文 Shannon 模型没有的 γ 无关每轮成本），不是实现问题。",
    ]


def exp_patch_suggestion(per_mode_gamma: dict[str, int]) -> list[str]:
    lines = ["", "建议的 exp.py 改法（若按方法取 γ）：", "```python"]
    if per_mode_gamma:
        lines.append("GAMMA_BY_MODE = {")
        for value, gamma in sorted(per_mode_gamma.items()):
            lines.append(f'    "{value}": {gamma},')
        lines.append("}")
        lines.append(
            "# 主矩阵循环里: gamma=GAMMA_BY_MODE.get(mode.value, GAMMA_SINGLE)"
        )
    else:
        lines.append("# 保持 GAMMA_SINGLE = 3（扫描结论：统一 γ 更优）")
    lines.append("```")
    return lines


def analyze(args, summary_path: Path) -> str:
    cells = load_cells(summary_path)
    gammas = _parse_gammas(args.gammas)
    modes = _resolve_modes(args.modes)
    out = [f"γ 扫描报告  ←  {summary_path}"]

    per_mode_gamma = {}
    for display, value in modes:
        out.append(render_table(display, value, cells, gammas))
        notes, pick = recommend_mode(display, value, cells, gammas)
        out.extend(notes)
        if pick is not None:
            per_mode_gamma[value] = pick
        if value == "tk_slt":
            out.extend(tk_slt_theory(value, cells, gammas, args.bandwidth))

    out.extend(exp_patch_suggestion(per_mode_gamma))
    out.append(
        "\n注：temp=0 下准确率应与 γ 无关（贪心等价）；K=320 固定不扫"
        "（论文 Fig 4 最优，见 docs/protocol.md §3.3）。"
    )
    return "\n".join(out)


def main(argv=None):
    args = parse_args(argv)
    modes = _resolve_modes(args.modes)
    gammas = _parse_gammas(args.gammas)
    summary_path = (
        Path(args.summary_file)
        if args.summary_file
        else default_summary_path(args, "-".join(name for name, _ in modes))
    )

    if args.recover_from_logs:
        recover_from_logs(args, summary_path)
        print(analyze(args, summary_path))
        report_path = summary_path.with_suffix(".md")
        report_path.write_text(analyze(args, summary_path), encoding="utf-8")
        print(f"\n报告已写入: {report_path}")
        return

    if args.analyze_only:
        print(analyze(args, summary_path))
        return

    if args.bandwidth < 5.0:
        print(
            f"⚠ --bandwidth {args.bandwidth} 低于链路模拟器的 min_bandwidth_mbps=5"
            f"（cmd_temp 不传该参数），有效带宽会被钳到 5 Mbps；"
            f"如需更低请先在 cmd_temp 里透传 --min_bandwidth_mbps"
        )

    configs = build_configs(args)
    pending = filter_configs_for_resume(
        configs, summary_path if summary_path.exists() else None
    )
    skipped = len(configs) - len(pending)
    print(
        f"γ 扫描: {len(modes)} 方法 × γ∈{gammas} × {args.series} 系 × "
        f"{DATASET_CHOICES[args.dataset].name}"
        + (f" × {args.repeats} 重复" if args.repeats > 1 else "")
    )
    print(f"共 {len(configs)} 格，本次待跑 {len(pending)} 个")
    if skipped:
        print(f"（按 {summary_path} 的成功记录跳过 {skipped} 个）")
    for config in pending:
        print(f"  - {describe_config(config)}  γ={config['gamma']}")

    if args.dry_run:
        print("\n[dry-run] 未执行任何实验。")
        return
    if not pending:
        print("\n没有待跑的实验，直接出报告：")
        print(analyze(args, summary_path))
        return

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # 先留底旧汇总：run_experiments_parallel 只写"本次运行"的结果，
    # 不合并会把之前的格子覆盖掉（2026-10-09 实际踩过：扩展跑 1 格
    # 覆盖了之前的 32 格）。跑完后 merge 回写，同语义键以新结果为准。
    prior = load_summary(summary_path) if summary_path.exists() else []
    all_results = run_experiments_parallel(
        pending,
        max_workers=args.max_workers,
        log_dir="exp_logs",
        summary_file=str(summary_path),
    )
    merged = merge_summaries(prior, all_results)
    tmp_summary = str(summary_path) + ".tmp"
    with open(tmp_summary, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    os.replace(tmp_summary, summary_path)
    print(analyze(args, summary_path))

    report_path = summary_path.with_suffix(".md")
    report_path.write_text(analyze(args, summary_path), encoding="utf-8")
    print(f"\n报告已写入: {report_path}")


if __name__ == "__main__":
    main()
