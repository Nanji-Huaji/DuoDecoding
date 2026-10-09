#!/usr/bin/env python
"""离线重放计费：把已落盘的通信记录换（带宽, NTT, 地板, 口径）重新结算。

前提（A/B 预实验已验证）：temp=0 且无 RL/ODLD 反馈时，解码与通信参数
无关——同一条逐消息记录可在任意网络配置下重新计费。带宽阶梯 / NTT
敏感性 / instant-fluid 口径对照因此变成纯 CPU 重放，不用重跑 GPU。

记录来源：metrics 的 ``comm_trace_edge_cloud``（[字节, 轮号] 列表，
``TransferUnit.round_idx`` 标注，dsd/dssd/tk_slt/cuhlm 均已接入）。

用法:
    # 把 46 Mbps 的工件重放到 5 Mbps（流体 + per_round + 阶梯地板）
    .venv/bin/python scripts/rebill.py exp/.../foo_metrics.json --bandwidth 5

    # NTT 敏感性（论文 Table II 的 76.36ms 档）
    .venv/bin/python scripts/rebill.py exp/a.json exp/b.json --ntt-ms 76.36

    # 口径对照：同一条记录按 instant + per_transfer 重算
    .venv/bin/python scripts/rebill.py exp/a.json --bw-model instant \
        --round-trip per_transfer

    # 引擎自检（无 GPU）：重放(记录, 配置X) == 直接以配置X记账
    .venv/bin/python scripts/rebill.py --self-test

局限: ① 只重放 edge_cloud 链路（矩阵基线只用它）；② 有通信反馈的方法
（RL 适配器开着的 CEE-SD、ODLD 开着的 tk_slt）解码依赖网络，重放只是
"同一动作序列在新网络下的成本"近似，不是真重跑；③ γ/数据集/模型变了
记录就失效（那是解码参数，不是网络参数）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.communication import CommunicationSimulator  # noqa: E402

# 引擎与 src/communication.py 的默认保持一致
DEF_NTT_MS = 50.0
DEF_BATCH_DELAY_MS = 50.0


def build_simulator(cfg: dict) -> CommunicationSimulator:
    """按重放配置构造与真实实验同构的链路模拟器。"""
    sim = CommunicationSimulator(
        cfg["bandwidth"],
        float("inf"),
        float("inf"),
        dimension="Mbps",
        ntt_ms_edge_cloud=cfg["ntt_ms"],
        ntt_ms_edge_end=0.317,
        use_stochastic=True,
        min_bandwidth_mbps=cfg["floor"],
        bw_model=cfg["bw_model"],
        mode=cfg.get("trace_mode", "static"),
    )
    sim.coalesce_rounds = cfg["round_trip"] == "per_round"
    return sim


def replay_comm(trace: list, cfg: dict) -> CommunicationSimulator:
    """把逐消息记录重放进给定配置的模拟器。

    样本边界 = 轮号**回退**（多样本拼接，每样本轮号从头计）或缺失段；
    边界处**重建模拟器**——真实实验每样本调用一次解码方法，模拟器
    （连续时钟、trace 游标）随样本归零，重放必须复刻这一点，否则后续
    样本的排水相位系统性偏移（实测 2 样本 dsd 偏 +1%）。
    样本内分组 = 轮号**相等**（per_transfer 记录里一轮多条消息）同组，
    变化切组；per_round 记录每轮一条（flush 单元），天然一组一轮。
    """
    samples: list[list] = []
    for item in trace:
        rid = item[1]
        prev = samples[-1][-1][1] if samples else None
        new_sample = (
            not samples or rid is None or prev is None or rid < prev
        )
        if new_sample:
            samples.append([item])
        else:
            samples[-1].append(item)

    holder = build_simulator(cfg)  # 只作返回载体（聚合口径）
    holder.stats = {"edge_cloud": [], "edge_end": [], "cloud_end": []}
    holder._pending_bytes = {}
    for sample in samples:
        sim = build_simulator(cfg)
        groups: list[list] = []
        for item in sample:
            rid = item[1]
            prev = groups[-1][-1][1] if groups else None
            new_group = not groups or rid is None or prev is None or rid != prev
            if new_group:
                groups.append([item])
            else:
                groups[-1].append(item)
        for group in groups:
            sim.set_round(int(group[0][1] or 0))
            for item in group:
                sim.simulate_transfer(item[0], "edge_cloud")
            sim.flush_round()
        sim.flush_round()
        # 样本级结果并入载体（clock/游标不跨样本延续；
        # edge_cloud_data/edge_cloud_comm_time 均为 stats 派生属性）
        holder.stats["edge_cloud"].extend(sim.stats["edge_cloud"])
        holder.connect_times["edge_cloud"] += sim.connect_times["edge_cloud"]
        holder.edge_cloud_bandwidth_history.extend(
            sim.edge_cloud_bandwidth_history
        )
    return holder


def rebill_artifact(path: Path, cfg: dict, batch_delay_ms: float) -> dict | None:
    """读工件 → 重放 → 新通信/墙钟/吞吐。返回对比行（缺记录返回 None）。"""
    m = json.loads(path.read_text())
    trace = m.get("comm_trace_edge_cloud")
    if not trace:
        return None
    sim = replay_comm(trace, cfg)

    old_comm = m["communication_time"]
    old_queue = m.get("queuing_time", 0.0)
    new_comm = sim.edge_cloud_comm_time
    new_queue = m.get("target_forward_times", 0) * batch_delay_ms / 1000.0
    new_wall = m["wall_time"] - old_comm - old_queue + new_comm + new_queue
    toks = m["generated_tokens"]
    return {
        "artifact": str(path),
        "bytes_logged": sum(u[0] for u in trace),
        "bytes_replayed": sim.edge_cloud_data,
        "bytes_metric": m.get("edge_cloud_data_bytes"),
        "old_comm": old_comm,
        "new_comm": new_comm,
        "old_throughput": m["throughput"],
        "new_throughput": toks / new_wall if new_wall > 0 else 0.0,
        "old_wall": m["wall_time"],
        "new_wall": new_wall,
        "rounds": m.get("connect_times", {}).get("edge_cloud"),
        "accuracy": m.get("accuracy"),
    }


def self_test() -> int:
    """引擎正确性：重放(记录, X) == 直接以 X 记账，对两个差异极大的 X。"""
    seq = []  # (轮号, 字节) —— 模拟 dsd 形状：8 轮 × (264KB 上行 + 0B 下行)
    for r in range(1, 9):
        seq.append((r, 264 * 1024))
        seq.append((r, 0))
    configs = [
        {"bandwidth": 46.0, "floor": 4.6, "bw_model": "fluid",
         "round_trip": "per_round", "ntt_ms": 50.0},
        {"bandwidth": 5.0, "floor": 0.5, "bw_model": "instant",
         "round_trip": "per_transfer", "ntt_ms": 76.36},
    ]
    fail = 0
    for cfg in configs:
        direct = build_simulator(cfg)
        trace = []
        for rid, group in itertools.groupby(seq, key=lambda u: u[0]):
            direct.set_round(rid)
            for _, nbytes in group:
                direct.simulate_transfer(nbytes, "edge_cloud")
                trace.append([nbytes, rid])
            direct.flush_round()
        direct.flush_round()
        replayed = replay_comm(trace, cfg)
        c1, c2 = direct.edge_cloud_comm_time, replayed.edge_cloud_comm_time
        ok = abs(c1 - c2) < 1e-9 and direct.edge_cloud_data == replayed.edge_cloud_data
        tag = f"{cfg['bandwidth']:g}Mbps {cfg['bw_model']}/{cfg['round_trip']}"
        print(f"  {tag:<34} direct={c1:.6f}s replay={c2:.6f}s "
              f"{'✓' if ok else '✗ MISMATCH'}")
        fail += 0 if ok else 1
    # 字节中立：per_round 合并不增减字节
    n1 = sum(u[0] for u in trace)
    n2 = replayed.edge_cloud_data
    print(f"  字节中立: 记录 {n1} == 重放 {n2} {'✓' if n1 == n2 else '✗'}")
    fail += 0 if n1 == n2 else 1
    print("自检通过" if fail == 0 else "自检失败")
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifacts", nargs="*", type=Path, help="metrics json 路径")
    ap.add_argument("--bandwidth", type=float, default=46.0, help="目标带宽 Mbps")
    ap.add_argument("--ntt-ms", type=float, default=DEF_NTT_MS)
    ap.add_argument("--bw-model", choices=("instant", "fluid"), default="fluid")
    ap.add_argument("--round-trip", choices=("per_round", "per_transfer"),
                    default="per_round")
    ap.add_argument("--floor", type=float, default=None,
                    help="带宽地板 Mbps；缺省 max(0.5, 带宽/10)（矩阵约定）")
    ap.add_argument("--batch-delay-ms", type=float, default=DEF_BATCH_DELAY_MS)
    ap.add_argument("--trace-mode", choices=("static", "driving", "walking"),
                    default="static")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.artifacts:
        ap.error("需要工件路径，或用 --self-test")

    cfg = {
        "bandwidth": args.bandwidth,
        "floor": args.floor if args.floor is not None
        else max(0.5, args.bandwidth / 10),
        "bw_model": args.bw_model,
        "round_trip": args.round_trip,
        "ntt_ms": args.ntt_ms,
        "trace_mode": args.trace_mode,
    }
    print(f"重放配置: {cfg['bandwidth']}Mbps {cfg['bw_model']}/{cfg['round_trip']} "
          f"NTT={cfg['ntt_ms']}ms 地板={cfg['floor']} "
          f"batch_delay={args.batch_delay_ms}ms\n")
    hdr = (f"{'工件':<58} {'通信s':>9} {'→':^0} {'新通信s':>9} "
           f"{'吞吐':>7} {'→':^0} {'新吞吐':>7} 字节核对")
    print(hdr)
    print("-" * len(hdr))
    rc = 0
    for p in args.artifacts:
        row = rebill_artifact(p, cfg, args.batch_delay_ms)
        if row is None:
            print(f"{str(p)[-58:]:<58} 无 comm_trace_edge_cloud（旧工件，"
                  f"需要带记录的新跑）")
            rc = 1
            continue
        ok_bytes = row["bytes_replayed"] == row["bytes_metric"]
        d_thr = (row["new_throughput"] / row["old_throughput"] - 1) * 100
        print(f"{str(p)[-58:]:<58} {row['old_comm']:>9.1f} → {row['new_comm']:>9.1f} "
              f"{row['old_throughput']:>7.3f} → {row['new_throughput']:>7.3f} "
              f"({d_thr:+5.1f}%) {'✓' if ok_bytes else '✗'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
