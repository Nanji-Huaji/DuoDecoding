"""口径协议与"生效口径"自述（治乱机制层，不改动任何测量数字）。

动机
----
参数一度有三个来源（CLI 默认 / `exp.py` 扫描配置 / `cmd_temp` 里的字面量），
且部分开关只被个别方法消费：`comm_round_trip_mode`、`charge_residual_payload`、
`transfer_top_k_cap`、`force_full_vocab_transfer` 曾长期只有 `adaptive_tridecoding`
（及其委托者 `cee_sd` / `cee_sd_opportunistic`）读取，而 `eval/utils.py` 又会把
**标称口径**无条件写进 metrics —— run 的标签因此可能与真实行为不一致。
2026-04 统一口径落地后（docs/protocol.md §3）基线族已接入同一套开关；
2026-10-09 起往返口径进一步统一（§3.4）：CUHLM 系与 TK-SLT 系的
**载荷字节**仍按各自论文计费（`cuhlm_uplink_payload_bytes` /
`tk_slt_uplink_payload_bytes`），但**往返次数**接入 `comm_round_trip_mode`
（主表 per_round = 每次云端交互 1×NTT）。分析见 `docs/param_ledger.md`，
最终口径见 `docs/protocol.md`。

本模块提供三件事：

1. `PROTOCOLS`：把每个协议的取值冻结成唯一真源，CLI 只负责"覆盖 + 告警"；
2. `mode_consumption()`：回答"当前 eval_mode 究竟消费了哪些开关"。优先用
   `Register` 注册表做运行时内省（含跨方法委托的传递闭包，例如
   `cee_sd_opportunistic` → `adaptive_tridecoding`）；注册表尚未填充时退化为
   `STATIC_MODE_CONSUMPTION` 快照。`test/test_protocol_spec.py` 断言两者一致，
   避免快照与实现漂移；
3. `emit_effective_report()`：run 开头打印一块自述，含"声明 vs 实际消费"。

默认 `--protocol none`：不写入任何参数，故本模块不影响任何既有数字。
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.mode_features import MODE_FEATURES

#: 通信计费开关（只有 tri 系 adaptive 协议消费，见 mode_consumption）
ACCOUNTING_SWITCHES: tuple[str, ...] = (
    "comm_round_trip_mode",
    "charge_residual_payload",
    "transfer_top_k_cap",
    "force_full_vocab_transfer",
)

#: 投机深度键：单-γ 方法与三级方法读的不是同一组
DEPTH_KEYS: tuple[str, ...] = ("gamma", "gamma1", "gamma2")

#: 基线族消费的计费开关（比 adaptive 族少 force_full_vocab_transfer：
#: 基线的上行压缩由 transfer_top_k 直接表达，没有"强制全词表"旁路）
_BASELINE_ACCOUNTING: tuple[str, ...] = (
    "comm_round_trip_mode",
    "charge_residual_payload",
    "transfer_top_k_cap",
)

#: 各模式消费的通信计费开关快照（运行时内省的离线替身，测试保证不漂移）。
#: 2026-04 统一口径落地（docs/protocol.md §3）：基线族接入与 ours 同一套
#: 计费开关；CUHLM 系（uncertainty_decoding/cuhlm/cee_cuhlm）不接统一
#: 开关，改按 CU-HLM 论文自身的计费口径计费（见 _CUHLM_MODES）。
_STATIC_ACCOUNTING: dict[str, tuple[str, ...]] = {
    # adaptive 族：全 4 开关（含 force_full_vocab_transfer 旁路）
    "adaptive_tridecoding": ACCOUNTING_SWITCHES,
    "cee_sd": ACCOUNTING_SWITCHES,
    "cee_sd_opportunistic": ACCOUNTING_SWITCHES,
    # 基线族（docs/protocol.md §3 已接线）
    "dsd": _BASELINE_ACCOUNTING,
    "dist_spec": _BASELINE_ACCOUNTING,
    "dssd": _BASELINE_ACCOUNTING,
    "dist_split_spec": _BASELINE_ACCOUNTING,
    "tridecoding": _BASELINE_ACCOUNTING,
    "ceesd_without_arp": _BASELINE_ACCOUNTING,
    "ceesd_w/o_arp": _BASELINE_ACCOUNTING,
    "adaptive_decoding": _BASELINE_ACCOUNTING,
    "cee_dssd": _BASELINE_ACCOUNTING,
    "cee_dsd": _BASELINE_ACCOUNTING,
    "speculative_decoding_with_bandwidth": _BASELINE_ACCOUNTING,
    # CUHLM / TK-SLT 系：2026-10-09 统一往返决策（docs/protocol.md §3.4）——
    # 载荷字节仍按各自论文的口径计费（cuhlm_uplink_payload_bytes /
    # tk_slt_uplink_payload_bytes），但**往返次数**接入统一开关
    # comm_round_trip_mode（per_round = 每次云端交互付 1×NTT）。
    # 此前它们逐报文付 NTT（每轮 2×50ms），同一主表内比 dsd/dssd/ours
    # 每轮多付 50ms 纯记账差异（tk_slt 合并后实测 +31% 吞吐）。
    # cee_cuhlm（三级变体，不在主矩阵）暂未接线，保持逐报文口径。
    "cuhlm": ("comm_round_trip_mode",),
    "uncertainty_decoding": ("comm_round_trip_mode",),
    "tk_slt": ("comm_round_trip_mode",),
    "tkslt": ("comm_round_trip_mode",),
}

#: 消费至少一个计费开关的注册名（由 _STATIC_ACCOUNTING 派生；报表文案用）
_ACCOUNTING_CONSUMERS: frozenset[str] = frozenset(_STATIC_ACCOUNTING)

#: 无通信阶段的模式（纯本地解码）：计费口径标签对它们无意义
_NO_COMM_MODES: frozenset[str] = frozenset({"small", "large", "target_only", "sd"})

#: CUHLM 系：载荷字节按 CU-HLM 论文自身的口径计费（式 (5)：上行
#: k·(b_prob+b_index) bits，b_prob=8、b_index=⌈log₂V⌉；token 索引/重同步
#: negligible；跳过 = 零通信。见 docs/protocol.md §3 的 CUHLM 小节与
#: src/communication.cuhlm_uplink_payload_bytes）。2026-10-09 起往返口径
#: 接入统一开关（§3.4）：per_round = 每次云端交互 1×NTT。
_CUHLM_MODES: frozenset[str] = frozenset(
    {"cuhlm", "uncertainty_decoding", "cee_cuhlm"}
)

#: TK-SLT 系：载荷字节同样按论文自身口径计费（WCSP'25 Zheng & Yang）——
#: 上行 γ·K·b_prob bits（b_prob=16，FP16；索引 negligible，论文 Table II
#: 的 L 值与该式逐位吻合），拒绝重采样在验证方基于稀疏 FP16 分布。见
#: src/communication.tk_slt_uplink_payload_bytes 与 docs/protocol.md §3.3。
#: 2026-10-09 起往返口径接入统一开关（§3.4）。
_TK_SLT_MODES: frozenset[str] = frozenset({"tk_slt", "tkslt"})

#: eval_mode → 投机深度读取键（运行时内省的快照，测试保证不漂移）
_STATIC_DEPTH_KEYS: dict[str, tuple[str, ...]] = {
    # 纯自回归基线：不读任何深度键
    "small": (),
    "large": (),
    "target_only": (),
    # 单-γ 族
    "adaptive_decoding": ("gamma",),
    "sd": ("gamma",),
    "dsd": ("gamma",),
    "dssd": ("gamma",),
    "dist_spec": ("gamma",),
    "dist_split_spec": ("gamma",),
    "uncertainty_decoding": ("gamma",),
    "cuhlm": ("gamma",),
    "tk_slt": ("gamma",),
    "tkslt": ("gamma",),
    "speculative_decoding_with_bandwidth": ("gamma",),
    # 三级族：只读 gamma1/gamma2
    "tridecoding": ("gamma1", "gamma2"),
    "adaptive_tridecoding": ("gamma1", "gamma2"),
    "cee_sd": ("gamma1", "gamma2"),
    "cee_sd_opportunistic": ("gamma1", "gamma2"),
    "cee_cuhlm": ("gamma1", "gamma2"),
    "cee_dsd": ("gamma1", "gamma2"),
    "cee_dssd": ("gamma1", "gamma2"),
    "ceesd_without_arp": ("gamma1", "gamma2"),
    "ceesd_w/o_arp": ("gamma1", "gamma2"),
}


@dataclass(frozen=True)
class ModeConsumption:
    """某个 eval_mode 实际消费了哪些口径开关。"""

    mode: str
    accounting: tuple[str, ...]
    depth_keys: tuple[str, ...]
    source: str  # "runtime" | "snapshot" | "unknown"

    @property
    def consumes_accounting(self) -> bool:
        return bool(self.accounting)


def _feature_names(code: Any) -> set[str]:
    """收集一个 code object 里出现的属性名/字符串常量（含嵌套函数）。"""
    names: set[str] = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, str):
            names.add(const)
        elif hasattr(const, "co_names"):
            names |= _feature_names(const)
    return names


def _collect(func: Any, registry: Mapping[str, Any], seen: set[int]) -> set[str]:
    """递归收集（含对其它注册方法的委托，带环保护）。"""
    # 解码方法被 @torch.no_grad() 等装饰器包了一层（functools.wraps），直接读
    # __code__ 只会拿到包装函数的代码对象，必须穿过 __wrapped__ 才能看到本体。
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    code = getattr(func, "__code__", None)
    if code is None or id(code) in seen:
        return set()
    seen.add(id(code))
    names = _feature_names(code)
    for name in list(names):
        target = registry.get(name)
        if target is not None:
            names |= _collect(target, registry, seen)
    return names


def live_mode_consumption(mode: str) -> ModeConsumption | None:
    """运行时内省：注册表可用时给出真实消费集，否则返回 None。"""
    try:
        from src.register import Register
    except Exception:  # pragma: no cover - 导入失败时退化为快照
        return None
    registry = Register._DECODING_REGISTRY
    func = registry.get(mode)
    if func is None:
        return None
    names = _collect(func, registry, set())
    return ModeConsumption(
        mode=mode,
        accounting=tuple(sorted(names & set(ACCOUNTING_SWITCHES))),
        depth_keys=tuple(sorted(names & set(DEPTH_KEYS))),
        source="runtime",
    )


def mode_consumption(mode: str) -> ModeConsumption:
    """查询某 eval_mode 的口径消费情况（优先运行时真值）。"""
    live = live_mode_consumption(mode)
    if live is not None:
        return live
    if mode in _STATIC_DEPTH_KEYS:
        return ModeConsumption(
            mode=mode,
            accounting=tuple(sorted(_STATIC_ACCOUNTING.get(mode, ()))),
            depth_keys=_STATIC_DEPTH_KEYS[mode],
            source="snapshot",
        )
    return ModeConsumption(mode=mode, accounting=(), depth_keys=(), source="unknown")


def accounting_consumers() -> tuple[str, ...]:
    """消费通信计费开关的模式名（按快照，供告警文案使用）。"""
    return tuple(sorted(_ACCOUNTING_CONSUMERS))


# --------------------------------------------------------------------------- #
# 协议冻结集
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProtocolSpec:
    """一个命名口径：`values` 会被写入 args，`declared` 是声明但未强制的项。"""

    summary: str
    values: Mapping[str, Any] = field(default_factory=dict)
    declared: tuple[str, ...] = ()
    for_tables: bool = True


#: honest / legacy 三个子开关的取值（与 src/utils.py 的 --comm_accounting 一致）。
#: 2026-04 全表解钳：两档口径的 transfer_top_k_cap 均为 0（不钳位）——
#: cap=16 原是 ours 的方法设计，钳位会改基线的提议分布，不属于计费口径。
_COMM_ACCOUNTING_PRESETS: dict[str, dict[str, Any]] = {
    "honest": {
        "charge_residual_payload": True,
        "comm_round_trip_mode": "per_round",
        "transfer_top_k_cap": 0,
    },
    "legacy": {
        "charge_residual_payload": False,
        "comm_round_trip_mode": "per_transfer",
        "transfer_top_k_cap": 0,
    },
}

PROTOCOLS: dict[str, ProtocolSpec] = {
    # 论文主表口径（docs/protocol.md §2）。通信数值取论文正文建模值：
    # edge-cloud NTT 50ms、edge-end 563Mbps（Table II 的 76.3/941 留作敏感性分析）。
    "paper_table5": ProtocolSpec(
        summary="论文 Table V 主表口径（全表同计费 + 论文正文通信数值）",
        values={
            # 通信数值
            "edge_end_bandwidth": 563,
            "edge_cloud_bandwidth": 46,
            "cloud_end_bandwidth": 46,
            "ntt_ms_edge_cloud": 50,
            "ntt_ms_edge_end": 0.317,
            "batch_delay": 0.05,
            # 载荷发射时长：流体排水（2026-10-09 决策，docs/protocol.md §3.4）。
            # 低带宽档（5/10 Mbps）长载荷横跨多个 trace 间隔，instant 的
            # 单采样冻结既失真又系统性多扣（E[S/B] > S/E[B]）。
            "comm_bw_model": "fluid",
            # 计费口径（全表同一套；基准是 adaptive_tridecoding 的实现）
            **_COMM_ACCOUNTING_PRESETS["honest"],
            # transfer_top_k 故意**不冻结**：它必须按方法区分，单个全局值表达不了。
            #   · CEE-SD = 300：top-k 压缩是它自身的设计
            #     （DRA 选 top-k，上行传压缩 logits）；
            #   · dsd / dssd = 0：这两篇原文都没有 top-k 压缩（DSSD 拒绝时下行是整词表
            #     分布 |V|·bprob，DSD 上行是 γ 个整词表分布），此前统一套 300 等于把
            #     "压缩传输"这项待验证的贡献免费送给基线。
            # 取 0 时 transfer() 的 is_compressed=False、reject_residual_payload_bytes
            # 走 vocab*element 分支，即整行载荷。由 exp.py 按方法显式传（见
            # TRANSFER_TOP_K_OURS / TRANSFER_TOP_K_PAPER）；argparse 默认仍是 300，
            # 供只跑 ours 的旧脚本沿用。决策记录见 docs/protocol.md §2 #8。
            # 推理设置
            "temp": 0.0,
            "max_tokens": 128,
            "num_shots": 3,
            "eval_data_num": 80,
            "random_sample": True,
            "sample_seed": 1234,
            "num_samples_per_task": 1,
            # 阈值
            "small_draft_threshold": 0.6,
            "draft_target_threshold": 0.7,
            # uncertainty_threshold 故意**不冻结**：它是 CUHLM 的工作点（越小越
            # 常上云、越准越慢），不是一个全表共享的 L0。0.8 会让 CUHLM 在
            # Llama/GSM8K 上退化到 0 准确率，与论文 Table V 自称的
            # "baselines ≈target" 矛盾；主表取值见 exp.py 的
            # UNCERTAINTY_THRESHOLD_CUHLM（0.08 = 实测的等准确率前沿点）。
            # 统计口径
            "use_early_stopping": False,
            "use_stochastic_comm": True,
        },
        declared=(
            "投机深度规则待定（docs/protocol.md §4）：单-γ 方法读 --gamma，"
            "三级方法读 --gamma1/--gamma2，必须显式传，协议不代填",
            "RL adapter 只作用于 tri 族、不作用于基线（L0 #12）：需 Step 2 实现，"
            "当前协议不强制",
            "cuda graph 必须成对报告（开/关各一列或脚注），故协议不代填该开关",
        ),
    ),
    "honest": ProtocolSpec(
        summary="只收敛通信计费口径（残差计费 + per_round；top-k 不钳位）",
        values=dict(_COMM_ACCOUNTING_PRESETS["honest"]),
    ),
    "legacy": ProtocolSpec(
        summary="复现 2026-09-24 前的历史数字（不收残差 + per_transfer）",
        values=dict(_COMM_ACCOUNTING_PRESETS["legacy"]),
    ),
    "smoke": ProtocolSpec(
        summary="冒烟/联调用的小规模口径（不用于出表）",
        values={
            "eval_data_num": 2,
            "max_tokens": 32,
            "num_shots": 0,
            "random_sample": False,
        },
        for_tables=False,
    ),
}

#: 不写入参数、也不需要偏离告警的口径项（供文档/报告引用）
NON_ARG_DECLARATIONS: tuple[str, ...] = (
    "stats: warmup 实跑声明次数 / 失败样本整体剔除 / 重跑速度只统计本次行",
)


# --------------------------------------------------------------------------- #
# 应用与偏离检测
# --------------------------------------------------------------------------- #


def _flag_present(cli_args: Sequence[str], flag: str) -> bool:
    """CLI 里是否显式出现该选项（支持 `--x`、`--x=v`）。"""
    return any(a == flag or a.startswith(flag + "=") for a in cli_args)


@dataclass
class ProtocolApplication:
    """一次协议应用的结果。"""

    name: str
    applied: list[str] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)
    unknown_keys: list[str] = field(default_factory=list)

    @property
    def is_named(self) -> bool:
        return self.name != "none"

    @property
    def clean(self) -> bool:
        """本 run 是否严格属于该协议（无 CLI 覆盖）。"""
        return self.is_named and not self.deviations


def apply_protocol(
    args: Any,
    name: str,
    cli_args: Sequence[str],
    dest_to_flags: Mapping[str, Sequence[str]] | None = None,
) -> ProtocolApplication:
    """把协议值写入 args：**只填未被命令行显式设置的项**。

    显式给出的 CLI 参数永远优先；若它偏离协议值，记入 `deviations`
    （调用方据此告警），保证"声明"与"实际"不会静默分叉。
    """
    result = ProtocolApplication(name=name)
    if name == "none":
        return result
    spec = PROTOCOLS.get(name)
    if spec is None:
        raise KeyError(f"未知协议 {name!r}；可用: {sorted(PROTOCOLS)}")

    dest_to_flags = dest_to_flags or {}
    for dest, want in spec.values.items():
        if not hasattr(args, dest):
            result.unknown_keys.append(dest)
            continue
        explicit = any(
            _flag_present(cli_args, flag) for flag in dest_to_flags.get(dest, ())
        )
        if explicit:
            have = getattr(args, dest)
            if have != want:
                result.deviations.append(f"{dest}={have!r}（协议值 {want!r}）")
            continue
        setattr(args, dest, want)
        result.applied.append(dest)

    setattr(args, "protocol", name)
    setattr(args, "protocol_applied", tuple(result.applied))
    setattr(args, "protocol_deviations", tuple(result.deviations))
    return result


# --------------------------------------------------------------------------- #
# 生效口径自述
# --------------------------------------------------------------------------- #


def _fmt(value: Any) -> str:
    return f"{value!r}" if isinstance(value, str) else str(value)


def render_effective_report(
    args: Any, application: ProtocolApplication
) -> str:
    """渲染"生效口径"自述；声明与实际消费不一致处显式标注。"""
    mode = str(getattr(args, "eval_mode", "?"))
    consumption = mode_consumption(mode)
    lines: list[str] = []

    name = application.name
    head = "未命名口径（未指定 --protocol）" if name == "none" else f"protocol={name}"
    lines.append(f"[protocol] 口径自述：{head}")
    if name != "none":
        lines.append(f"    {PROTOCOLS[name].summary}")

    # 通信数值
    lines.append(
        "    通信数值   "
        f"NTT edge_cloud={_fmt(getattr(args, 'ntt_ms_edge_cloud', '?'))}ms / "
        f"edge_end={_fmt(getattr(args, 'ntt_ms_edge_end', '?'))}ms · "
        f"带宽 {_fmt(getattr(args, 'edge_end_bandwidth', '?'))}/"
        f"{_fmt(getattr(args, 'edge_cloud_bandwidth', '?'))}/"
        f"{_fmt(getattr(args, 'cloud_end_bandwidth', '?'))} Mbps"
    )

    # 计费口径 + 真实消费
    lines.append(
        "    计费       "
        f"round_trip={_fmt(getattr(args, 'comm_round_trip_mode', '?'))} · "
        f"charge_residual={_fmt(getattr(args, 'charge_residual_payload', '?'))} · "
        f"topk_cap={_fmt(getattr(args, 'transfer_top_k_cap', '?'))}"
    )
    if consumption.source == "unknown":
        if mode in MODE_FEATURES:
            lines.append(
                f"    ⚠ eval_mode={mode!r} 在能力表中但没有已注册的解码实现，"
                f"消费情况未知（该模式当前不可用）"
            )
        else:
            lines.append(f"    ⚠ eval_mode={mode!r} 不在能力表内，消费情况未知")
    elif not consumption.consumes_accounting:
        if mode in _NO_COMM_MODES:
            lines.append(
                f"    · eval_mode={mode!r} 无通信阶段，计费开关与标签不适用"
            )
        elif mode in _CUHLM_MODES:
            lines.append(
                f"    · eval_mode={mode!r} [不消费] 上述开关：按 CU-HLM 论文"
                "自身口径计费（式 (5)：上行 k·(b_prob+b_index) bits，"
                "b_prob=8、b_index=⌈log₂V⌉；token 索引/重同步 negligible；"
                "跳过=零通信——docs/protocol.md §3.1）"
            )
        elif mode in _TK_SLT_MODES:
            lines.append(
                f"    · eval_mode={mode!r} [不消费] 上述开关：按 TK-SLT 论文"
                "自身口径计费（上行 γ·K·b_prob bits，b_prob=16/FP16，"
                "索引 negligible；下行=0 字节报文——docs/protocol.md §3.3）"
            )
        else:
            lines.append(
                f"    ⚠ 当前 eval_mode={mode!r} [不消费] 上述计费开关，"
                "本 run 的实际字节/往返由该方法内联实现决定"
            )

    # 投机深度
    depth = consumption.depth_keys
    if depth:
        vals = " ".join(f"{k}={_fmt(getattr(args, k, '?'))}" for k in depth)
        lines.append(f"    投机深度   读取键={','.join(depth)} · {vals}")
        if set(depth) != set(DEPTH_KEYS):
            lines.append(
                "    ⚠ γ 规则未定（docs/protocol.md §4）：单-γ 与三级方法读的不是"
                "同一组键，比较时须显式对齐"
            )
    else:
        lines.append(f"    投机深度   当前 eval_mode={mode!r} 不读任何 γ（无投机阶段）")

    # 推理与统计
    lines.append(
        "    推理       "
        f"temp={_fmt(getattr(args, 'temp', '?'))} · "
        f"max_tokens={_fmt(getattr(args, 'max_tokens', '?'))} · "
        f"num_shots={_fmt(getattr(args, 'num_shots', '?'))} · "
        f"N={_fmt(getattr(args, 'eval_data_num', '?'))} · "
        f"seed={_fmt(getattr(args, 'sample_seed', '?'))} · "
        f"random_sample={_fmt(getattr(args, 'random_sample', '?'))}"
    )
    lines.append(
        "    统计       "
        f"early_stopping={_fmt(getattr(args, 'use_early_stopping', '?'))} · "
        f"stochastic_comm={_fmt(getattr(args, 'use_stochastic_comm', '?'))} · "
        f"cuda_graph={_fmt(getattr(args, 'use_cuda_graph', '?'))}"
    )

    if application.unknown_keys:
        lines.append(
            f"    ⚠ 协议含未知参数（已跳过）: {', '.join(application.unknown_keys)}"
        )
    if application.deviations:
        lines.append(
            "    ⚠ CLI 覆盖了协议项 ⇒ 本 run 不属于该协议："
            + "; ".join(application.deviations)
        )
    if name != "none":
        for note in PROTOCOLS[name].declared:
            lines.append(f"    ⚠ 声明未强制: {note}")
    return "\n".join(lines)


def emit_effective_report(args: Any, application: ProtocolApplication) -> str:
    """打印并返回自述（调用方负责落盘/入 metrics）。"""
    report = render_effective_report(args, application)
    print(report, file=sys.stdout, flush=True)
    return report
