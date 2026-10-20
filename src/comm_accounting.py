"""通信计费口径的构造点（Phase 1 地基，已接线）。

迁移前
------
``comm_round_trip_mode`` / ``charge_residual_payload`` /
``transfer_top_k_cap`` / ``force_full_vocab_transfer`` /
``prob_payload_bits`` 在 13 个解码方法里各自 ``getattr(args, ...)``。
分散读取的成本不是性能，而是改口径时无法回答"哪些方法真的读了它"：
2026-04 之前这些开关只接线了 tri 系，metrics 标签却对全表生效——
run 的标签与真实行为不一致（src/protocols.py 模块头有完整记录）。

现状（2026-10-19）
----------------
32 个读取点全部收敛：解码方法经惰性 ``Decoding.comm_config``（基类，
Baselines 与 engine.py 的历史方法共用）读字段；模块助手
``_uplink_prob_payload`` 改收 ``bits: int``，由调用点传
``self.comm_config.prob_payload_bits``。``test/test_comm_config_reads.py``
守住"baselines.py / engine.py 方法体内不得再出现这些开关的 args 直读"。

本模块做三件事：

1. ``CommConfig``：把五个计费开关读成一个不可变对象；
2. ``CommEnv``：把通信数值层（带宽 / 地板 / 发射模型 / NTT trace / 排队）
   也收成一个对象——迁移前这些值在 ~20 处 ``CommunicationSimulator(...)``
   构造点各自 ``getattr(self.args, ..., 默认)``，默认值被复制了多份；
3. ``PROFILES``：从 ``src.protocols.PROTOCOLS`` 派生，不复制任何常量。

五个计费接口（上行/下行/残差/往返/RL 成本）留到下一个提交：它们需要
先按 call site 列清楚"谁在哪一步计了哪一笔"，否则接口形状是猜的。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.protocols import PROTOCOLS

#: 五个开关的 argparse 默认值（src/utils.py，2026-10 核对）。
#: 这里复制的是**默认值**而不是取值逻辑；test_comm_accounting 与
#: test_protocol_spec 共同保证它与 CLI 不漂移。
#: 注意 transfer_top_k_cap：2026-04 起全表解钳，CLI 默认是 0 而不是 16。
DEFAULTS: dict[str, Any] = {
    "comm_round_trip_mode": "per_round",
    "charge_residual_payload": True,
    "transfer_top_k_cap": 0,
    "force_full_vocab_transfer": False,
    "prob_payload_bits": 16,
}

#: 迁移前各方法 `getattr(..., <fallback>)` 的回退值。阴影里的按键必须
#: 逐字一致，否则手工构造的 Namespace（测试/工具）会静默换口径：
#: 例如 test_decoding_avg_topk_metrics 的 Namespace 没有
#: transfer_top_k_cap，旧代码回退 0（不钳位），CLI 默认 16 会把它钳到 16。
#: CLI 路径上属性总是存在，这些回退不可达；保留只为合成 args 零漂移。
FALLBACKS: dict[str, Any] = {
    "comm_round_trip_mode": "per_transfer",
    "charge_residual_payload": False,
    "transfer_top_k_cap": 0,
    "force_full_vocab_transfer": False,
    "prob_payload_bits": 16,
}


@dataclass(frozen=True)
class CommConfig:
    """一次运行生效的计费口径（不可变，可哈希）。"""

    comm_round_trip_mode: str
    charge_residual_payload: bool
    transfer_top_k_cap: int
    force_full_vocab_transfer: bool
    prob_payload_bits: int

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> "CommConfig":
        """从任意映射构造；缺失的键回落默认值（协议只冻结一部分）。"""
        return cls(
            comm_round_trip_mode=str(
                values.get("comm_round_trip_mode", DEFAULTS["comm_round_trip_mode"])
            ),
            charge_residual_payload=bool(
                values.get(
                    "charge_residual_payload",
                    DEFAULTS["charge_residual_payload"],
                )
            ),
            transfer_top_k_cap=int(
                values.get("transfer_top_k_cap", DEFAULTS["transfer_top_k_cap"])
            ),
            force_full_vocab_transfer=bool(
                values.get(
                    "force_full_vocab_transfer",
                    DEFAULTS["force_full_vocab_transfer"],
                )
            ),
            prob_payload_bits=int(
                values.get("prob_payload_bits", DEFAULTS["prob_payload_bits"])
            ),
        )

    @classmethod
    def from_args(cls, args: Any) -> "CommConfig":
        """从 argparse Namespace 读取——迁移后各方法的唯一入口。

        缺失属性回落 FALLBACKS（迁移前各方法自己的回退），不是 DEFAULTS：
        CLI 路径属性总存在，但测试里的合成 Namespace 未必，且旧回退与
        CLI 默认并不相同（docs/comm_call_sites.md §4.2）。
        """
        return cls.from_values(
            {key: getattr(args, key, value) for key, value in FALLBACKS.items()}
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: 命名协议 → 计费口径。真源是 src/protocols.PROTOCOLS；这里只做投影，
#: 不新增任何取值（协议没冻结的键回落 DEFAULTS）。
PROFILES: dict[str, CommConfig] = {
    name: CommConfig.from_values(spec.values) for name, spec in PROTOCOLS.items()
}


#: 通信数值层的回退值：逐字对迁移前各构造点的 ``getattr(self.args, ...,
#: <fallback>)``（CLI 路径属性总存在，这些只对合成 Namespace 生效）。
ENV_FALLBACKS: dict[str, Any] = {
    "edge_cloud_bandwidth": 20.0,
    "edge_end_bandwidth": 100.0,
    "cloud_end_bandwidth": 100.0,
    "min_bandwidth_mbps": 5.0,
    "comm_bw_model": "instant",
    "stochastic_ntt": False,
    "batch_delay": 0.0,
    "ntt_trace_file": "",
    "ntt_trace_scale": 1.0,
}


@dataclass(frozen=True)
class CommEnv:
    """一次运行生效的通信数值层（不可变，可哈希）。

    与 ``CommConfig`` 的分工：``CommConfig`` 决定"怎么计费"（往返/残差/
    位宽），``CommEnv`` 决定"链路长什么样"（带宽/NTT/发射模型/排队）。
    两者都是运行时唯一真源；``test_comm_config_reads.py`` 守卫这两个
    文件不得回退到 args 直读。
    """

    edge_cloud_bandwidth: float
    edge_end_bandwidth: float
    cloud_end_bandwidth: float
    min_bandwidth_mbps: float
    comm_bw_model: str
    stochastic_ntt: bool
    batch_delay: float
    ntt_trace_file: str
    ntt_trace_scale: float

    @classmethod
    def from_args(cls, args: Any) -> "CommEnv":
        """从 argparse Namespace 读取——构造通信模拟器的唯一入口。"""
        values = {
            key: getattr(args, key, fallback)
            for key, fallback in ENV_FALLBACKS.items()
        }
        return cls(
            edge_cloud_bandwidth=float(values["edge_cloud_bandwidth"]),
            edge_end_bandwidth=float(values["edge_end_bandwidth"]),
            cloud_end_bandwidth=float(values["cloud_end_bandwidth"]),
            min_bandwidth_mbps=float(values["min_bandwidth_mbps"]),
            comm_bw_model=str(values["comm_bw_model"]),
            stochastic_ntt=bool(values["stochastic_ntt"]),
            batch_delay=float(values["batch_delay"] or 0.0),
            ntt_trace_file=str(values["ntt_trace_file"] or ""),
            ntt_trace_scale=float(values["ntt_trace_scale"] or 1.0),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
