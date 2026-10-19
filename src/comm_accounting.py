"""通信计费口径的构造点（Phase 1 地基，尚未接线）。

现状
----
``comm_round_trip_mode`` / ``charge_residual_payload`` /
``transfer_top_k_cap`` / ``force_full_vocab_transfer`` /
``prob_payload_bits`` 在 13 个解码方法里各自 ``getattr(args, ...)``。
分散读取的成本不是性能，而是改口径时无法回答"哪些方法真的读了它"：
2026-04 之前这些开关只接线了 tri 系，metrics 标签却对全表生效——
run 的标签与真实行为不一致（src/protocols.py 模块头有完整记录）。

本模块只做两件事：

1. ``CommConfig``：把五个开关读成一个不可变对象；
2. ``PROFILES``：从 ``src.protocols.PROTOCOLS`` 派生，不复制任何常量。

尚未接线：解码方法仍直接读 args。按
``docs/refactor_baselines_plan.md`` Phase 1 逐方法迁移（一个方法一个 PR）。
在第一个方法迁移之前，本模块不被 src 的任何生产路径 import——这是
有意的：先让构造函数和 profile 有一处可测的真源，再接第一条线。

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
