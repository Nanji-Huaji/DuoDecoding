"""A/B 预实验：网络仿真口径切换（docs/protocol.md §3.4）前后各方法的变化。

两臂对比（同 20 个样本、同 seed、同 GPU，逐格顺序跑）：

- ``old`` 臂 = 历史口径：``--comm_bw_model instant``；tk_slt/cuhlm 额外回到
  逐报文往返（``--comm_round_trip_mode per_transfer``，每轮 2×NTT——它们
  2026-10-09 之前的真实计费方式；dsd/dssd 历史上本就是 per_round）。
- ``new`` 臂 = paper_table5 冻结口径：``--comm_bw_model fluid`` + 全方法
  per_round（每轮/每触发 1×NTT）。

预期信号（用于核对，不是结论）：
- 46 Mbps：fluid≈instant（载荷 < 0.2s 采样间隔），变化几乎全部来自
  tk_slt/cuhlm 的 NTT 合并（tk_slt 约 +31% 吞吐）；dsd/dssd 应基本不动。
- 5 Mbps：dsd（整词表上行）/dssd（整词表下行）的发射时长显著下修；
  tk_slt/cuhlm 载荷小，变化有限。
- 完整性检查：两臂的 accuracy 与纯计算 wall_time 应接近（口径不改解码
  路径，只改计费）——若出现大幅漂移说明实现有 bug 而不是口径差异。

用法::

    .venv/bin/python scripts/pilot_ab_network.py            # 46+5 Mbps, 20 样本
    .venv/bin/python scripts/pilot_ab_network.py --bandwidths 46 --samples 10

输出：``experiment_results/pilot_ab_network.json``（逐格原始结果，增量落盘，
中断后重跑跳过已完成格）+ ``experiment_results/pilot_ab_network.md``（对照表）。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

import exp as _exp  # noqa: F401,E402  import 副作用：矩阵构建 + create_config 工厂
from exp import (  # noqa: E402
    NTT_MS_EDGE_CLOUD,
    NTT_MS_EDGE_END,
    TRANSFER_TOP_K_OURS,
    TRANSFER_TOP_K_PAPER,
    TRANSFER_TOP_K_TKSLT,
    UNCERTAINTY_THRESHOLD_CUHLM,
    UNCERTAINTY_THRESHOLD_DEFAULT,
    EvalDataset,
    add_args,
    cmd_temp,
    create_config,
    get_file_path,
    resolve_accelerate,
)

#: 各方法的工作点 γ（预期 GAMMA_BY_MODE 的取值；cuhlm 对 γ 不敏感取 3）
MODE_GAMMA = {
    "dist_spec": 3,
    "dist_split_spec": 8,
    "uncertainty_decoding": 3,
    "tk_slt": 8,
}
MODE_LABEL = {
    "dist_spec": "dsd",
    "dist_split_spec": "dssd",
    "uncertainty_decoding": "cuhlm",
    "tk_slt": "tk_slt",
}
#: 旧口径 = 逐报文 2×NTT 的方法（2026-10-09 统一前的真实计费方式）
OLD_PER_TRANSFER = {"tk_slt", "uncertainty_decoding"}

LLAMA_SERIES = ("llama-68m", "tiny-llama-1.1b", "llama-2-13b")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="A/B 预实验：网络仿真口径（§3.4）切换前后各方法的变化"
    )
    p.add_argument(
        "--bandwidths",
        default="46,5",
        help="逗号分隔的 edge-cloud 带宽档（Mbps）。默认 46,5：46 看往返统一，"
        "5 看流体模型",
    )
    p.add_argument(
        "--modes",
        default="dsd,dssd,cuhlm,tk_slt",
        help=f"逗号分隔，可选: {','.join(MODE_LABEL)}",
    )
    p.add_argument("--samples", type=int, default=20, help="每格样本数（默认 20）")
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="固定 GPU 序号（默认自动检测空闲卡）",
    )
    p.add_argument(
        "--out",
        default="experiment_results/pilot_ab_network",
        help="输出文件前缀（.json/.md）",
    )
    return p.parse_args(argv)


def build_config(
    mode_value: str, gamma: int, bw: float, arm: str, samples: int
) -> dict:
    little, draft, target = LLAMA_SERIES
    is_cuhlm = mode_value == "uncertainty_decoding"
    is_tkslt = mode_value == "tk_slt"
    cfg = create_config(
        eval_mode=mode_value,
        ntt_ms_edge_cloud=NTT_MS_EDGE_CLOUD,
        ntt_ms_edge_end=NTT_MS_EDGE_END,
        batch_delay=50e-3,
        use_precise=False,
        use_stochastic_comm=True,
        edge_end_bandwidth=563,
        edge_cloud_bandwidth=bw,
        cloud_end_bandwidth=bw,
        min_bandwidth_mbps=max(0.5, bw / 10),
        comm_bw_model="fluid" if arm == "new" else "instant",
        small_draft_threshold=0.6,
        draft_target_threshold=0.7,
        uncertainty_threshold=(
            UNCERTAINTY_THRESHOLD_CUHLM if is_cuhlm else UNCERTAINTY_THRESHOLD_DEFAULT
        ),
        transfer_top_k=(
            TRANSFER_TOP_K_OURS
            if is_cuhlm
            else (TRANSFER_TOP_K_TKSLT if is_tkslt else TRANSFER_TOP_K_PAPER)
        ),
        gamma=gamma,
        gamma1=gamma,
        gamma2=gamma,
        max_tokens=128,
        num_shots=3,
        eval_dataset=EvalDataset.gsm8k,
        use_early_stopping=False,
        use_cuda_graph=True,
        eval_data_num=samples,
        run_full_dataset=False,
        random_sample=True,
        sample_seed=1234,
        draft_model=draft if mode_value == "dist_split_spec" else little,
        target_model=target,
        little_model=little,
    )
    # 臂别进 exp_name：两臂时间戳可能仅差毫秒，且工件必须能从文件名分辨口径
    cfg["exp_name"] = f"{cfg['exp_name']}_{arm}acc"
    return cfg


def render_cmd(cfg: dict, per_transfer: bool) -> str:
    cmd = cmd_temp.format(accelerate=resolve_accelerate(), **cfg)
    for flag in ("use_stochastic_comm",):
        if cfg.get(flag):
            cmd = add_args(cmd, flag)
    if per_transfer:
        # 旧口径（tk_slt/cuhlm）：去掉 --comm_accounting honest（其收敛块会把
        # 子开关改回 per_round），换成显式 per_transfer——每报文各付一次 NTT
        cmd = cmd.replace(
            "--comm_accounting honest", "--comm_round_trip_mode per_transfer"
        )
    return cmd


def run_one(cfg: dict, per_transfer: bool, gpu: int) -> dict:
    log_dir = Path("exp_logs")
    log_dir.mkdir(exist_ok=True)
    cfg = dict(cfg)
    cfg["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_file = log_dir / (
        cfg["exp_name"].replace("/", "_")
        + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    cmd = render_cmd(cfg, per_transfer)
    print(f"  ▶ {cfg['exp_name']}  (GPU {gpu})")
    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"实验配置: {json.dumps(cfg, indent=2, ensure_ascii=False)}\n")
        f.write(f"执行命令: {cmd}\n")
        f.write("=" * 80 + "\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        rc = proc.wait()
    status = "success" if rc == 0 else f"failed(rc={rc})"
    result = None
    if rc == 0:
        metrics_path = get_file_path(cfg["exp_name"])
        if metrics_path:
            with open(metrics_path, encoding="utf-8") as f:
                result = json.load(f)
    print(f"    {status}  ({time.time() - t0:.0f}s)")
    return {
        "exp_name": cfg["exp_name"],
        "status": status,
        "log_file": str(log_file),
        "result": result,
    }


def _num(d: dict, key: str):
    v = d.get(key) if isinstance(d, dict) else None
    return float(v) if isinstance(v, (int, float)) else None


def fmt(v, pattern="{:8.3f}"):
    return pattern.format(v) if isinstance(v, (int, float)) else "     n/a"


def main(argv=None):
    global args
    args = parse_args(argv)
    bandwidths = [float(x) for x in args.bandwidths.split(",")]
    labels = [m.strip() for m in args.modes.split(",")]
    valid_labels = set(MODE_LABEL.values())
    unknown = [m for m in labels if m not in valid_labels]
    if unknown:
        raise SystemExit(f"未知模式 {unknown}；可选: {sorted(valid_labels)}")
    mode_values = [
        (lbl, next(k for k, v in MODE_LABEL.items() if v == lbl)) for lbl in labels
    ]

    if args.gpu is not None:
        gpu = args.gpu
    else:
        from src.nvml import get_available_gpus

        free = get_available_gpus()
        if not free:
            raise SystemExit("没有检测到空闲 GPU（可用 --gpu 显式指定）")
        gpu = int(free[0])
    print(f"预实验 GPU: {gpu} | 样本数: {args.samples} | 带宽档: {bandwidths}")

    out_json = Path(f"{args.out}.json")
    out_md = Path(f"{args.out}.md")
    done = {}
    if out_json.exists():
        with open(out_json, encoding="utf-8") as f:
            for e in json.load(f):
                if e.get("status") == "success":
                    done[(e["bw"], e["mode"], e["arm"])] = e
        print(f"已有 {len(done)} 个完成格，跳过")

    entries = []
    for bw in bandwidths:
        for lbl, mv in mode_values:
            gamma = MODE_GAMMA[mv]
            for arm in ("old", "new"):
                key = (bw, lbl, arm)
                if key in done:
                    entries.append({**done[key], "bw": bw, "mode": lbl})
                    continue
                cfg = build_config(mv, gamma, bw, arm, args.samples)
                per_transfer = arm == "old" and mv in OLD_PER_TRANSFER
                entry = run_one(cfg, per_transfer, gpu)
                entry.update({"bw": bw, "mode": lbl, "gamma": gamma, "arm": arm})
                entries.append(entry)
                # 增量落盘：中断后重跑只补缺格
                if entry["status"] == "success":
                    done[key] = entry
                _atomic_write(out_json, entries)

    report = build_report(entries)
    out_md.write_text(report, encoding="utf-8")
    _atomic_write(out_json, entries)
    print(report)
    print(f"\n结果: {out_json}\n报告: {out_md}")


def _atomic_write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def build_report(entries) -> str:
    lines = [
        "# 网络口径 A/B 预实验（instant+逐报文 vs fluid+per_round）",
        "",
        f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}",
        "- 臂定义: old = instant（tk_slt/cuhlm 再加 per_transfer 2×NTT）；"
        "new = fluid + per_round（paper_table5 冻结口径）",
        "- 完整性检查: 同格两臂 accuracy / wall_time 应接近（口径不改解码）",
        "",
        "| 带宽 | 方法 | γ | 指标 | old | new | Δ |",
        "|---|---|---|---|---|---|---|",
    ]
    cells = {}
    for e in entries:
        if e.get("status") == "success" and e.get("result"):
            cells.setdefault((e["bw"], e["mode"], e.get("gamma")), {})[e["arm"]] = e[
                "result"
            ]
    for (bw, mode, gamma) in sorted(cells):
        arms = cells[(bw, mode, gamma)]
        if "old" not in arms or "new" not in arms:
            continue
        for key, pattern, unit in (
            ("throughput", "{:8.3f}", " tok/s"),
            ("communication_time", "{:8.2f}", " s"),
            ("wall_time", "{:8.2f}", " s"),
            ("accuracy", "{:8.4f}", ""),
        ):
            o, n = _num(arms["old"], key), _num(arms["new"], key)
            delta = (
                f"{(n - o) / o * 100:+.1f}%"
                if isinstance(o, (int, float))
                and isinstance(n, (int, float))
                and o != 0
                and key in ("throughput", "communication_time")
                else ""
            )
            lines.append(
                f"| {bw:g} | {mode} | {gamma} | {key}{unit} "
                f"| {fmt(o, pattern)} | {fmt(n, pattern)} | {delta} |"
            )
    # 口径自证：新臂 tk_slt/cuhlm 的 comm_accounting 应为 paper_rt
    lines += ["", "## 口径自证（metrics 的 comm_accounting 标签）", ""]
    for e in entries:
        r = e.get("result") or {}
        if r:
            lines.append(
                f"- {e['bw']:g} Mbps {e['mode']} [{e['arm']}]: "
                f"{r.get('comm_accounting')} / bw_model={r.get('comm_bw_model')}"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
