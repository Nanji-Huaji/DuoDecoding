import json
import logging
import math
import os
import sys
import warnings
from typing import List, Literal, Optional, Tuple, TypedDict, cast, Protocol

import torch

from src.utils import read_trace_file, return_closest_mean_index


class TransferUnit(TypedDict):
    data_size_bytes: int | float
    transfer_time: float
    # 纯发射时间（不含传播延迟 NTT），用于能耗计算：
    # 传播时延期间无线电并不发射，能耗只应按发射时长计。
    tx_time: float
    # 记账轮号（set_round 标注；离线重放 rebill 按它重组 per_round 合并）
    round_idx: Optional[int]


class Statistics(TypedDict):
    edge_cloud: List[TransferUnit]
    edge_end: List[TransferUnit]
    cloud_end: List[TransferUnit]


LinkType = Literal["edge_cloud", "edge_end", "cloud_end"]

Dimension = Literal["Mbps", "MBps", "bps", "Bps"]


def _convert_to_bytes_per_second(bandwidth: float, dimension: Dimension) -> float:
    """
    将带宽转换为bytes/second
    """
    if dimension == "Mbps":
        return bandwidth * 1e6 / 8
    elif dimension == "MBps":
        return bandwidth * 1e6
    elif dimension == "bps":
        return bandwidth / 8
    elif dimension == "Bps":
        return bandwidth
    else:
        raise ValueError(f"Unknown dimension: {dimension}")


def _trace_comm(kind: str, link: str, nbytes: float, **extra) -> None:
    """环境变量 COMM_TRACE 门控的通信追踪（不设时零开销、不改变行为）。"""
    path = os.environ.get("COMM_TRACE")
    if not path:
        return
    try:
        frame = sys._getframe(2)
        caller = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
        rec = {"kind": kind, "link": link, "bytes": round(float(nbytes), 1),
               "caller": caller}
        rec.update(extra)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 真实 RTT trace 回放（sigcomm ping/ 实测数据，与 throughput/ 同 Campaign）
# 模块级只存**配置**（采样值/缩放/来源），不含游标：由 baselines 初始化处
# 一次性配置，10 个构造点无需逐一穿参。
# 游标必须按实例隔离（R7）：见 CommunicationSimulator._ntt_trace_index。
# 优先级: ping trace 回放 > 拥塞模型(L1) > 固定基值
# ---------------------------------------------------------------------------
_NTT_TRACE_STATE: dict = {"data": [], "scale": 1.0, "src": ""}


def configure_ntt_trace(values: list, scale: float = 1.0, src: str = "") -> None:
    _NTT_TRACE_STATE.update(data=list(values), scale=float(scale), src=src)


def _ntt_trace_active() -> bool:
    return len(_NTT_TRACE_STATE["data"]) > 0


# B16：概率载荷量化的"穿透阈值"——bits < 16 才真实量化（对数域），
# >= 16 视为不量化、按原始 element_size 计费。计费条件必须与此门
# 镜像（同一常量单源），否则会出现"收 bits 钱、传 element_size 数据"。
# 调用方（baselines）的 `< 16` 门与 _quantize_probs_logspace 的
# `>= 16 早退`均引用此语义。
PROB_QUANT_PASS_THROUGH_BITS = 16


class CommunicationSimulator:
    """
    用于模拟通信的类
    - 带宽单位：bytes/second
    - protocol_overhead_bytes: 每次传输的协议开销，单位bytes
    - transfer_top_k: Optional[int], 如果设置了top-k压缩，则在传输概率分布时只传输top-k概率，其他位置为0，假设传输时只传输非零部分
    - 统计信息保存在self.stats中
    - 统计信息包括每次传输的数据大小和传输时间
    - 统计信息分为三类链路：edge-cloud, edge-end, cloud-end
    """

    def __init__(
        self,
        bandwidth_edge_cloud,
        bandwidth_edge_end,
        bandwidth_cloud_end,
        protocol_overhead_bytes: int = 0,
        transfer_top_k: Optional[int] = None,
        dimension: Dimension = "Mbps",
        ntt_ms_edge_end: float = 20,
        ntt_ms_edge_cloud: float = 200,
        use_stochastic: bool = False,
        stochastic_ntt: bool = False,
        set_mean_bandwidth: bool = True,
        mode: Literal["driving", "static", "walking"] = "static",
        min_bandwidth_mbps: float = 5.0,
        trace_interval_s: float = 0.2,
        bw_model: Literal["instant", "fluid"] = "instant",
    ):
        # 带宽下限（Mbps）：低于该值的带宽会被钳制。设为 0 可禁用下限。
        # 注意：mmWave 等无线链路深衰时吞吐会跌到接近 0，5 Mbps 的默认
        # 下限会削平这些最有价值的低带宽时段，需要研究弱链路时请调低。
        self.min_bandwidth_mbps = min_bandwidth_mbps
        # 载荷发射时长的计算模型（2026-10-09 决策，docs/protocol.md §3.4）：
        #  · instant（历史默认，逐位可复现）：整条载荷用**起始时刻的瞬时
        #    trace 采样**计费，tx = S/B_t0。46 Mbps 下载荷普遍 <50ms（≪0.2s
        #    采样间隔）时近似成立；低带宽下长载荷（如 5 Mbps 时全词表窗口
        #    ~600ms，横跨 3+ 个间隔）会失真，且 E[S/B] > S/E[B]（Jensen）
        #    系统性多扣。
        #  · fluid（主表口径）：流体排水——载荷排空期间逐个 trace 间隔用
        #    各自的带宽积分，正确处理跨间隔长载荷，消除单采样冻结偏差。
        self.bw_model = bw_model
        # 流体模型的连续仿真时钟（秒，模 trace 全周期）：小载荷虽在单个
        # 间隔内排空，时钟仍按 tx+NTT 推进——反复计费最终会跨过间隔
        # 边界，trace 不会冻结在起始采样上（预实验实测过冻结会让 dsd
        # 的通信时间虚高 +140%）。
        self._trace_time_s: float = 0.0
        # 随机带宽 trace 的采样间隔（秒）。trace 是时间序列，必须按仿真
        # 时间推进索引，而不是按消息数推进，否则带宽的时间相关性失真。
        self.trace_interval_s = trace_interval_s
        self.bandwidth_edge_cloud = _convert_to_bytes_per_second(
            bandwidth_edge_cloud, dimension
        )
        self.bandwidth_edge_end = _convert_to_bytes_per_second(
            bandwidth_edge_end, dimension
        )
        self.bandwidth_cloud_end = _convert_to_bytes_per_second(
            bandwidth_cloud_end, dimension
        )
        self.protocol_overhead_bytes = protocol_overhead_bytes
        self.stats = Statistics(
            edge_cloud=[],
            edge_end=[],
            cloud_end=[],
        )
        self.transfer_top_k = transfer_top_k

        self.ntt_edge_end = ntt_ms_edge_end / 1000  # 转换为秒
        self.ntt_edge_cloud = ntt_ms_edge_cloud / 1000  # 转换为秒
        # 动态 NTT（L1）：与带宽 trace 同源的拥塞相关 RTT 模型。
        # 物理动机：5G/WAN 深衰时（带宽采样低于 trace 均值）链路排队，RTT 随
        # 拥塞程度放大；带宽充足时 NTT 保持基值。公式（确定性，无随机项）：
        #   ntt_t = ntt_base * (1 + (mean_bw / bw_t - 1)^+)   # bw 跌到均值一半 ⇒ RTT×2
        # 默认关闭（False）保证历史数字逐位可复现；开启要求 use_stochastic。
        self.stochastic_ntt = stochastic_ntt and use_stochastic
        self.ntt_edge_cloud_base = self.ntt_edge_cloud
        self.trace_mean_bw = 0.0  # trace 加载后计算（字节/秒）
        self.ntt_edge_cloud_history = []  # 每次计费观察到的 edge-cloud NTT（毫秒）

        # R7：RTT trace 回放的游标必须是**实例级**的。
        # 旧实现把游标放在模块级 `_NTT_TRACE_STATE["index"]`，同一进程里出现
        # 第二个模拟器时（测试即如此：8 个测试文件各建实例；同进程复用、
        # 标定脚本同理）两者会交替推进同一个游标，各自只拿到真实 trace 的
        # 隔一个采样。带宽 trace 的游标 `self.trace_index` 一向是实例级的。
        self._ntt_trace_index = 0

        # 按轮合并（block / 每次往返一次）：
        # 真实实现里一轮只需每条链路一次 WAN 往返；把同一轮内多条消息累积到
        # `_pending_bytes`，在 set_round()/flush_round() 时按链路各计一次。
        # 默认关闭，保持历史口径可复现。
        self.coalesce_rounds = False
        self._round_idx: Optional[int] = None
        self._pending_bytes: dict = {}

        self.connect_times = {"edge_end": 0, "cloud_end": 0, "edge_cloud": 0}

        # 用于记录 edge-cloud 的瞬时带宽、top-k 和起草长度历史
        self.edge_cloud_bandwidth_history = []
        self.edge_cloud_topk_history = []
        self.edge_cloud_draft_len_history = []

        self.use_stochastic = use_stochastic
        self.dimension = dimension
        if self.use_stochastic:
            if set_mean_bandwidth:
                assert (
                    bandwidth_edge_end is not None and bandwidth_edge_cloud is not None
                ), (
                    "When set_mean_bandwidth is True, bandwidth_edge_end and bandwidth_edge_cloud must not be None"
                )

            # Calculate conversion factor from Mbps to the current dimension unit
            # This is needed because trace data is always in Mbps
            if dimension == "Mbps":
                mbps_to_dim = 1.0
            elif dimension == "bps":
                mbps_to_dim = 1e6
            elif dimension == "MBps":
                mbps_to_dim = 1.0 / 8.0
            elif dimension == "Bps":
                mbps_to_dim = 1e6 / 8.0
            else:
                mbps_to_dim = 1.0

            floor_val = self.min_bandwidth_mbps * mbps_to_dim
            self.trace_file_dict = {
                "driving": "data/sigcomm-5gmemu-5g-mmWave-uplink-data/throughput/driving/5g/throughput.list",
                "static": "data/sigcomm-5gmemu-5g-mmWave-uplink-data/throughput/static/5g/away_p1.list",
                "walking": "data/sigcomm-5gmemu-5g-mmWave-uplink-data/throughput/walking/5g/away.list",
            }
            trace_file = self.trace_file_dict.get(mode, self.trace_file_dict["static"])
            self.trace_data = []
            self.trace_index = 0

            if set_mean_bandwidth and bandwidth_edge_cloud is not None:
                # target_mean is in the current dimension
                target_mean = max(0.1 * mbps_to_dim, bandwidth_edge_cloud)
                # return_closest_mean_index expects Mbps
                run_id = return_closest_mean_index(
                    trace_file, target_mean / mbps_to_dim
                )
                if run_id == -1:
                    run_id = 1

                raw_data = read_trace_file(trace_file, run_id)
                if raw_data:
                    current_mean = sum(raw_data) / len(raw_data)  # This is in Mbps
                    if current_mean > 0:
                        scale_factor = (target_mean / mbps_to_dim) / current_mean
                        # Apply scale factor and ensure a reasonable floor (5 Mbps in current dimension)
                        self.trace_data = [
                            max(floor_val, x * scale_factor * mbps_to_dim)
                            for x in raw_data
                        ]

                        # Re-calculate mean and adjust to match exactly if needed
                        actual_mean = sum(self.trace_data) / len(self.trace_data)
                        if actual_mean > 0:
                            re_scale = target_mean / actual_mean
                            self.trace_data = [
                                max(floor_val, x * re_scale) for x in self.trace_data
                            ]
                    else:
                        self.trace_data = [target_mean] * len(raw_data)
                else:
                    self.trace_data = [target_mean]
            else:
                self.trace_data = read_trace_file(trace_file, 1)  # This is in Mbps
                if not self.trace_data:
                    run_id = return_closest_mean_index(trace_file, None)
                    self.trace_data = read_trace_file(trace_file, run_id)
                # Convert from Mbps to target dimension and apply floor
                self.trace_data = [
                    max(floor_val, x * mbps_to_dim) for x in self.trace_data
                ]

            # 动态 NTT（L1）用：trace 均值（与 trace_data 同单位，dimension 口径）
            if self.trace_data:
                self.trace_mean_bw = sum(self.trace_data) / len(self.trace_data)

    def _next_ntt_trace_value(self) -> Optional[float]:
        """回放下一个 RTT trace 采样（毫秒）；无 trace 时返回 None。

        R7：游标是**实例级**的（`self._ntt_trace_index`），与带宽 trace 的
        `self.trace_index` 一致。不得改回模块级共享，否则同一进程里两个实例
        会交替推进同一个游标。
        """
        data = _NTT_TRACE_STATE["data"]
        if not data:
            return None
        value = float(data[self._ntt_trace_index]) * float(_NTT_TRACE_STATE["scale"])
        self._ntt_trace_index = (self._ntt_trace_index + 1) % len(data)
        return value

    @property
    def edge_cloud_comm_time(self):
        return sum(
            self.stats["edge_cloud"][i]["transfer_time"]
            for i in range(len(self.stats["edge_cloud"]))
        )

    @property
    def edge_cloud_trace(self) -> list[list]:
        """edge-cloud 的逐消息记账记录，供离线重放（scripts/rebill.py）。

        每项 [字节数, 轮号]。temp=0 且无 RL/ODLD 反馈时解码与通信参数
        无关（A/B 预实验验证：两臂 accuracy 逐位一致），同一条记录可在
        任意（带宽, NTT, 地板, bw_model, 往返口径）下重新计费——带宽
        阶梯/NTT 敏感性/口径对照都变成纯 CPU 重放，不用重跑 GPU。
        """
        return [
            [u["data_size_bytes"], u.get("round_idx")]
            for u in self.stats["edge_cloud"]
        ]

    @property
    def edge_end_comm_time(self):
        return sum(
            self.stats["edge_end"][i]["transfer_time"]
            for i in range(len(self.stats["edge_end"]))
        )

    @property
    def cloud_end_comm_time(self):
        return sum(
            self.stats["cloud_end"][i]["transfer_time"]
            for i in range(len(self.stats["cloud_end"]))
        )

    # ==========================================
    # Unit-explicit accessors for external consumers.
    # Internal storage is bytes/second (bandwidth) and seconds (NTT),
    # but the RL adapter and other consumers expect Mbps and milliseconds.
    # ==========================================
    @property
    def bandwidth_edge_cloud_mbps(self):
        return self.bandwidth_edge_cloud / (1e6 / 8)

    @property
    def bandwidth_edge_end_mbps(self):
        return self.bandwidth_edge_end / (1e6 / 8)

    @property
    def bandwidth_cloud_end_mbps(self):
        return self.bandwidth_cloud_end / (1e6 / 8)

    @property
    def ntt_edge_end_ms(self):
        return self.ntt_edge_end * 1000

    @property
    def ntt_edge_cloud_ms(self):
        return self.ntt_edge_cloud * 1000

    @property
    def edge_cloud_data(self):
        return sum(
            self.stats["edge_cloud"][i]["data_size_bytes"]
            for i in range(len(self.stats["edge_cloud"]))
        )

    @property
    def edge_end_data(self):
        return sum(
            self.stats["edge_end"][i]["data_size_bytes"]
            for i in range(len(self.stats["edge_end"]))
        )

    @property
    def cloud_end_data(self):
        return sum(
            self.stats["cloud_end"][i]["data_size_bytes"]
            for i in range(len(self.stats["cloud_end"]))
        )

    @property
    def get_connect_times(self) -> dict:
        return self.connect_times

    def set_round(self, round_idx: int) -> None:
        """标记进入新一轮；按轮合并模式下先把上一轮的累积量结算掉。

        轮号在两种模式下都记录（进 TransferUnit.round_idx）——离线重放
        （scripts/rebill.py）靠它把逐消息字节重组成任意往返口径。
        """
        if self.coalesce_rounds and (
            self._round_idx is None or round_idx != self._round_idx
        ):
            self.flush_round()
        self._round_idx = round_idx

    def flush_round(self) -> None:
        """结算本轮累积的字节：每条链路只计一次传输（一次往返、一次 NTT）。"""
        pending, self._pending_bytes = self._pending_bytes, {}
        for link_type, bucket in pending.items():
            if bucket.get("bytes", 0.0) > 0:
                self._charge_transfer(
                    bucket["bytes"],
                    cast(Literal["edge_cloud", "edge_end", "cloud_end"], link_type),
                    topk=int(bucket.get("topk", 0)),
                    draft_len=int(bucket.get("draft_len", 0)),
                    trace_kind="flushed",
                )

    def simulate_transfer(
        self,
        data_size_bytes: int | float,
        link_type: Literal["edge_cloud", "edge_end", "cloud_end"],
        add_to_stats=True,
        topk: int = 0,
        draft_len: int = 0,
    ) -> float:
        """
        执行一次传输模拟。
        - data_size_bytes: 传输的数据大小，单位bytes
        - link_type: 传输链路类型，"edge_cloud", "edge_end", "cloud_end"
        - topk: 可选，记录此次传输关联的 top-k 值
        - draft_len: 可选，记录此次传输关联的草稿长度

        `coalesce_rounds=True` 时只累积字节（往返与 NTT 推迟到 flush_round 结算），
        用来模拟"每轮每条链路一次往返"的成批实现；此时返回 0.0。
        """
        if self.coalesce_rounds:
            if os.environ.get("COMM_TRACE"):
                _trace_comm(
                    "pending", link_type, data_size_bytes, topk=topk, draft_len=draft_len
                )
            bucket = self._pending_bytes.setdefault(
                link_type, {"bytes": 0.0, "topk": 0, "draft_len": 0}
            )
            # add_to_stats=False 的字节不进桶（语义=不计入数据统计）。
            # 当前全仓无此调用点；这是为 Step 2 把 coalesce 接到所有方法后
            # 防潜伏错计的守卫——否则 flush 会把本该排除的字节计入。
            if add_to_stats:
                bucket["bytes"] += float(data_size_bytes)
            bucket["topk"] = max(int(bucket["topk"]), int(topk))
            bucket["draft_len"] += int(draft_len)
            return 0.0
        trace_kind = None
        if os.environ.get("COMM_TRACE"):
            # 来自 transfer() 的转发已在 transfer() 里打过 "transfer" 点
            trace_kind = (
                None
                if sys._getframe(1).f_code.co_name == "transfer"
                else "direct"
            )
        return self._charge_transfer(
            data_size_bytes, link_type, add_to_stats, topk, draft_len, trace_kind
        )

    def _floor_bandwidth_bps(self) -> float:
        """带宽下限（字节/秒），与 dimension 无关（显式用 Mbps 换算）。"""
        return _convert_to_bytes_per_second(self.min_bandwidth_mbps, "Mbps")

    def _trace_rate_bps(self, index: int) -> float:
        """trace 第 index 个采样的有效速率（字节/秒），已套带宽下限。"""
        rate = _convert_to_bytes_per_second(
            self.trace_data[index], cast(Dimension, self.dimension)
        )
        return max(self._floor_bandwidth_bps(), rate)

    def _drain_time_edge_cloud(
        self, data_size_bytes: float, start_time_s: float
    ) -> tuple[float, float]:
        """流体模型：S 字节从 ``start_time_s`` 时刻起排空所需的时间。

        排水从当前时刻在采样间隔内的**剩余部分**开始（不是整个间隔），
        逐间隔用各自带宽积分；跨间隔长载荷因此正确积分。返回
        (排空秒数, 排空结束时刻)——结束时刻包含起始偏移，调用方据此
        推进连续时钟 ``_trace_time_s``（再加 NTT），游标取
        ``int(时钟/间隔)``。小载荷虽在单间隔内排空，时钟仍前进，
        反复计费会自然跨过间隔边界，trace 不冻结。
        """
        if data_size_bytes <= 0:
            return 0.0, start_time_s
        remaining = float(data_size_bytes)
        tx = 0.0
        t = float(start_time_s)
        interval = self.trace_interval_s
        n = len(self.trace_data)
        # 防退化 trace（全 0 且 floor=0）死循环：间隔数封顶后按 1B/s 兜底
        max_intervals = max(1, n * 100)
        for _ in range(max_intervals):
            k = int(t / interval)
            # 剩余到下一边界的时间。浮点陷阱：t 累加贴近 (k+1)*interval 时，
            # t % interval 会返回 ≈interval 而非 0，得到 frac≈1e-16 的零容量
            # 且 int(t/interval) 不前进——排水循环原地空转直到兜底按 1B/s
            # 计费（预实验实测 dsd 通信时间虚高百万秒）。贴边（相对量
            # <1e-9）直接按"下一间隔完整容量"计，误差 ≤ 亚纳秒级空时。
            frac = (k + 1) * interval - t
            if frac < interval * 1e-9:
                k += 1
                t = k * interval
                frac = interval
            idx = k % n
            rate = self._trace_rate_bps(idx)
            capacity = rate * frac
            if remaining <= capacity:
                dt = remaining / rate
                return tx + dt, t + dt
            tx += frac
            t += frac
            remaining -= capacity
        return tx + remaining / 1.0, t + remaining / 1.0

    def _charge_transfer(
        self,
        data_size_bytes: int | float,
        link_type: Literal["edge_cloud", "edge_end", "cloud_end"],
        add_to_stats=True,
        topk: int = 0,
        draft_len: int = 0,
        trace_kind: str = "direct",
    ) -> float:
        """真正的计费入口（原 simulate_transfer 主体）。"""
        if trace_kind and os.environ.get("COMM_TRACE"):
            _trace_comm(
                trace_kind,
                link_type,
                data_size_bytes,
                topk=topk,
                draft_len=draft_len,
            )
        if self.use_stochastic and link_type == "edge_cloud" and self.trace_data:
            current_bw = self.trace_data[self.trace_index]
            self.bandwidth_edge_cloud = _convert_to_bytes_per_second(
                current_bw, cast(Dimension, self.dimension)
            )
            # NTT 三级优先: 真实 ping trace 回放 > 拥塞模型(L1) > 固定基值
            # trace 回放与带宽 trace 独立推进（同 Campaign 配对，非严格时间对齐）。
            # 注意: trace 值为毫秒，本字段为秒（历史记录×1000 回毫秒）。
            if _ntt_trace_active():
                self.ntt_edge_cloud = float(self._next_ntt_trace_value()) / 1000.0
            elif self.stochastic_ntt and self.trace_mean_bw > 0:
                congestion = max(
                    0.0, self.trace_mean_bw / max(current_bw, 1e-9) - 1.0
                )
                self.ntt_edge_cloud = self.ntt_edge_cloud_base * (1.0 + congestion)
            else:
                self.ntt_edge_cloud = self.ntt_edge_cloud_base
            self.ntt_edge_cloud_history.append(self.ntt_edge_cloud * 1000)
        elif link_type == "edge_cloud" and _ntt_trace_active():
            # 无带宽 trace 时仍可单独回放 RTT trace（毫秒→秒）
            self.ntt_edge_cloud = float(self._next_ntt_trace_value()) / 1000.0
            self.ntt_edge_cloud_history.append(self.ntt_edge_cloud * 1000)

        if link_type == "edge_cloud":
            bandwidth = self.bandwidth_edge_cloud
        elif link_type == "edge_end":
            bandwidth = self.bandwidth_edge_end
        elif link_type == "cloud_end":
            bandwidth = self.bandwidth_cloud_end
        else:
            raise ValueError(f"Unknown link type: {link_type}")

        # 带宽下限（默认 5 Mbps，可通过 min_bandwidth_mbps 配置，0 表示不设下限）。
        # 显式使用 "Mbps" 保证下限与 self.dimension 无关。
        bandwidth = max(self._floor_bandwidth_bps(), bandwidth)

        # ---- 载荷发射时长：两种带宽计算模型（见 __init__ 的 bw_model 注释）----
        # instant：整条载荷按起始采样计费（历史口径，逐位可复现）。
        fluid_end_time: Optional[float] = None
        if not (
            self.bw_model == "fluid"
            and self.use_stochastic
            and link_type == "edge_cloud"
            and self.trace_data
        ):
            tx_time = data_size_bytes / bandwidth
        else:
            # fluid：从连续时钟的当前时刻起跨 trace 间隔积分排水。
            tx_time, fluid_end_time = self._drain_time_edge_cloud(
                data_size_bytes, self._trace_time_s
            )
            if tx_time > 0:
                # stats 里的"瞬时带宽"记有效排水速率 S/tx，供 ODLD 等估计器读
                bandwidth = data_size_bytes / tx_time
        transfer_time = tx_time

        if link_type == "edge_end":
            ntt = self.ntt_edge_end
        elif link_type == "edge_cloud":
            ntt = self.ntt_edge_cloud
        elif link_type == "cloud_end":
            ntt = self.ntt_edge_cloud + self.ntt_edge_end

        self.connect_times[link_type] += 1

        transfer_time += ntt

        # 随机带宽 trace 按仿真时间推进：本次传输耗时 transfer_time 秒，
        # 对应 trace 上 round(transfer_time / trace_interval_s) 个采样点。
        # 之前按"每传输一次 +1"推进，带宽的时间相关性随传输时长漂移。
        if self.use_stochastic and link_type == "edge_cloud" and self.trace_data:
            if fluid_end_time is not None:
                # fluid：连续时钟推进到"排空 + NTT"时刻（NTT 是传播/握手，
                # 时间流逝但不传数据），游标取整数化后的采样；时钟模
                # trace 全周期避免长跑浮点漂移。
                self._trace_time_s = (fluid_end_time + ntt) % (
                    len(self.trace_data) * self.trace_interval_s
                )
                self.trace_index = int(self._trace_time_s / self.trace_interval_s) % (
                    len(self.trace_data)
                )
            else:
                steps = round(transfer_time / self.trace_interval_s)
                self.trace_index = (self.trace_index + max(1, steps)) % len(
                    self.trace_data
                )

        if add_to_stats:
            transfer_unit = TransferUnit(
                data_size_bytes=data_size_bytes,
                transfer_time=transfer_time,
                tx_time=tx_time,
                round_idx=self._round_idx,
            )
            self.stats[link_type].append(transfer_unit)

            # 记录 edge-cloud 的瞬时带宽、Top-K 和 Draft Length
            if link_type == "edge_cloud":
                bandwidth_mbps = bandwidth / (1e6 / 8)  # 转换为 Mbps (与 _convert_to_bytes_per_second 一致)
                self.edge_cloud_bandwidth_history.append(bandwidth_mbps)
                self.record_edge_cloud_draft_info(topk, draft_len)

        return transfer_time

    def record_edge_cloud_draft_info(self, topk: int, draft_len: int):
        """
        记录 edge-cloud 传输时的 top-k 和起草长度信息

        Args:
            topk: 传输使用的 top-k 值
            draft_len: 起草的序列长度
        """
        self.edge_cloud_topk_history.append(topk)
        self.edge_cloud_draft_len_history.append(draft_len)

    @staticmethod
    def _apply_top_k_compression(probs: torch.Tensor | None, k: int) -> torch.Tensor:
        """
        probs: torch.Tensor，当前词表概率分布，形状为(..., V)
        k: int，保留的top-k数量
        返回压缩后的概率分布，形状与输入相同，但仅保留top-k概率，其他位置为0，值得注意的是，这不是真正的压缩，但我们假设传输时只传输非零部分
        """
        if probs is None or probs.numel() == 0:
            return torch.empty(0)

        # len() 取的是第一维（batch 维），多维输入会误判；top-k 语义在最后一维。
        if k >= probs.shape[-1]:
            return probs

        # 获取top-k indices
        top_k_values, top_k_indices = torch.topk(probs, k, sorted=True)

        # 创建压缩的概率分布（scatter_ 沿最后一维，对 (..., V) 任意维数都正确）
        compressed_probs = torch.zeros_like(probs)
        compressed_probs.scatter_(-1, top_k_indices, top_k_values)

        return compressed_probs

    @staticmethod
    def rebuild_full_probs(compressed_probs: torch.Tensor) -> torch.Tensor:
        """
        接收一个稀疏的概率分布张量(compressed_probs)，并重建为完整的概率分布。
        compressed_probs: torch.Tensor，形状为(..., vocab_size)
        重建后形状同样为(..., vocab_size)
        """
        if compressed_probs is None or compressed_probs.numel() == 0:
            warnings.warn("警告：compressed_probs为空，无法重建完整概率分布")
            return torch.empty(0)

        rebuilt_probs = compressed_probs.clone()

        # top-k 位置 = 非零位置（压缩时只写入 top-k 值）
        top_k_mask = compressed_probs > 0
        top_k_sum = compressed_probs.sum(dim=-1, keepdim=True)

        # clamp：数值防御，top-k 之和不应超过 1（浮点误差/异常输入时防负"概率"
        # ——负 uniform 会让重建分布非法）
        residual_mass = (1.0 - top_k_sum).clamp_min(0.0)

        # 尾部 = 显式非 top-k 掩码。此前用 ==0 当尾部，会把本就为 0 的 top-k
        # 项也换成本轮 uniform，轻微污染重建分布
        tail_mask = ~top_k_mask
        tail_count = tail_mask.sum(dim=-1, keepdim=True)

        # 避免除零：没有尾部位置则无需重建
        uniform_prob = torch.where(
            tail_count > 0,
            residual_mass / tail_count,
            torch.zeros_like(residual_mass),
        )
        rebuilt_probs = torch.where(tail_mask, uniform_prob, rebuilt_probs)

        return rebuilt_probs

    @staticmethod
    def compress_rebuild_probs(probs: torch.Tensor, k: int) -> torch.Tensor:
        """
        Args:
            probs: 形状：(batch_size, seq_len, vocab_size)
            k: int, top-k数量
        Returns:
            rebuilt_probs: 形状：(batch_size, seq_len, vocab_size)
        """
        if probs is None or probs.numel() == 0:
            warnings.warn("警告：probs为空，无法进行压缩重建")
            return torch.empty(0)

        if probs.dim() != 3:
            raise ValueError(f"probs维度应为3，实际为{probs.dim()}，无法进行压缩重建")

        if k >= probs.shape[-1]:
            return probs  # 无需压缩

        batch_size, seq_len, vocab_size = probs.shape

        # 向量化处理：reshape为(batch_size * seq_len, vocab_size)
        flat_probs = probs.view(-1, vocab_size)

        # 批量获取top-k
        top_k_values, top_k_indices = torch.topk(flat_probs, k, dim=-1, sorted=True)

        # 创建压缩的概率分布
        compressed_probs = torch.zeros_like(flat_probs)
        batch_indices = torch.arange(flat_probs.shape[0]).unsqueeze(1).expand(-1, k)
        compressed_probs[batch_indices, top_k_indices] = top_k_values

        # 重建概率分布（tail = 显式非 top-k 掩码；residual clamp 防负"概率"）
        top_k_sum = compressed_probs.sum(dim=-1, keepdim=True)
        residual_mass = (1.0 - top_k_sum).clamp_min(0.0)
        top_k_mask = torch.zeros_like(compressed_probs, dtype=torch.bool)
        top_k_mask[batch_indices, top_k_indices] = True
        tail_mask = ~top_k_mask
        tail_count = tail_mask.sum(dim=-1, keepdim=True)

        uniform_prob = torch.where(
            tail_count > 0,
            residual_mass / tail_count,
            torch.zeros_like(residual_mass),
        )

        rebuilt_flat_probs = torch.where(tail_mask, uniform_prob, compressed_probs)

        # 恢复原始形状
        return rebuilt_flat_probs.view(batch_size, seq_len, vocab_size)

    def transfer(
        self,
        tokens: torch.Tensor | None,
        prob: torch.Tensor | None,
        link_type: Literal["edge_cloud", "edge_end", "cloud_end"],
        is_compressed: bool = False,
        compressed_k: Optional[int] = 300,
        prob_bits: Optional[int] = None,
    ) -> float:
        token_bytes = 0
        prob_bytes = 0

        # Token data size (int32 or int64)
        if tokens is not None and tokens.numel() > 0:
            token_bytes = tokens.element_size() * tokens.numel()

        # 概率载荷位宽（默认 None ⇒ 用 element_size()，与历史口径一致 ✓）
        # B16 缝隙关死：位宽计费只在真实量化的阈值内生效（<16 才量化，
        # 见 baselines._quantize_probs_logspace 的同款门）。原条件
        # `bits < 8*element_size` 对 fp32 存在 [16,32) 窗口——bits≥16 时
        # 数据不量化却按 bits/8 计费。当前所有调用点都以 `< 16` 严格门控，
        # 窗口不可达（潜伏缝隙而非活跃 bug）；此处镜像阈值后即使未来
        # 调用点漏门控也不会张开。argparse 默认值 16 恰在窗口左端点，
        # 此前全靠调用点的严格 < 挡住。
        prob_elem = None
        if prob is not None:
            prob_elem = prob.element_size()
            if (
                prob_bits is not None
                and 0 < int(prob_bits) < min(PROB_QUANT_PASS_THROUGH_BITS, 8 * prob_elem)
            ):
                prob_elem = int(prob_bits) / 8.0

        # Probability history data size (float32 or float16)
        if prob is not None and prob.numel() > 0 and prob_elem is not None:
            prob_bytes = prob_elem * prob.numel()

        total_bytes = token_bytes + prob_bytes

        total_bytes += self.protocol_overhead_bytes

        if (
            is_compressed
            and prob is not None
            and prob.numel() > 0
            and compressed_k is not None
        ):
            if prob.dim() == 3:
                seq_length = prob.shape[1]
            else:
                seq_length = 1
            compressed_payload_bytes = self._compressed_topk_payload_bytes(
                compressed_k=compressed_k,
                seq_length=seq_length,
                prob_element_size=(
                    prob_elem if prob_elem is not None else prob.element_size()
                ),
            )
            total_bytes = (
                token_bytes + compressed_payload_bytes + self.protocol_overhead_bytes
            )

        # 计算 topk 和 draft_len
        topk_val = 0
        draft_len_val = 0
        if link_type == "edge_cloud":
            topk_val = (
                compressed_k if (is_compressed and compressed_k is not None) else 0
            )
            draft_len_val = (
                tokens.numel() if (tokens is not None and tokens.numel() > 0) else 0
            )

        _trace_comm(
            "transfer",
            link_type,
            total_bytes,
            compressed=bool(is_compressed),
            k=compressed_k if is_compressed else 0,
            tokens=int(token_bytes),
            probs=int(prob_bytes),
        )
        transfer_time = self.simulate_transfer(
            total_bytes, link_type, topk=topk_val, draft_len=draft_len_val
        )

        return transfer_time

    @staticmethod
    def _compressed_topk_payload_bytes(
        *,
        compressed_k: int,
        seq_length: int,
        prob_element_size: int,
        index_element_size: int = 4,
    ) -> int:
        return seq_length * compressed_k * (prob_element_size + index_element_size)

    def send_reject_message(
        self, linktype: Literal["edge_cloud", "edge_end", "cloud_end"]
    ) -> None:
        self.simulate_transfer(6, linktype)

    def send_accept_message(
        self, linktype: Literal["edge_cloud", "edge_end", "cloud_end"]
    ) -> None:
        self.simulate_transfer(6, linktype)

    @property
    def total_comm_energy(self) -> float:
        """
        仅在子类中实现，用于占位，表示通信能耗，单位焦耳
        """
        return 0.0


def cuhlm_uplink_payload_bytes(
    compressed_k: int | None,
    vocab_size: int,
    prob_bits: int = 8,
) -> float:
    """CU-HLM 论文口径的上行载荷（式 (5) 的压缩形式），单位字节。

    论文 §II-B：上行只计**词表分布**的传输——

        B = k · (b_prob + b_index)  bits

    - ``b_prob = 8`` bits：单个概率的量化位宽（论文 §V 仿真参数
      "b_prob = 8"）；
    - ``b_index = ⌈log₂|V|⌉`` bits：词表索引的二进制编码宽度
      （V=32000 时为 15 bits）。

    交叉验证：全词表无压缩时 k=|V|=32000 ⇒ 32000×23/8 = 92000 B，
    正是论文摘要的 "up to 92kB of payload per token"；k*=30（论文离线
    最优）⇒ 86.25 B，对应论文的 "<0.1% of the full vocabulary payload"。

    论文 §II-B 与 §III-B Step 5 同时假设：draft token、响应 token 与
    重同步 token 的**索引传输开销 negligible，不计入成本分析**——所以
    本函数只含词表分布项，token 索引在各调用点按 0 字节报文计
    （报文仍发生，链路模型的 per-message NTT 照付；见
    ``baselines._send_downlink_index_only``）。

    ``compressed_k`` 为 ``None``/``<=0``（未启用压缩）时按全词表计，
    即论文 vanilla HLM 的上行口径。
    """
    if vocab_size is None or int(vocab_size) <= 1:
        return 0.0
    if compressed_k is None or int(compressed_k) <= 0:
        k = int(vocab_size)
    else:
        k = int(compressed_k)
        if k > int(vocab_size):
            k = int(vocab_size)
    b_index = max(1, (int(vocab_size) - 1).bit_length())
    return k * (int(prob_bits) + b_index) / 8.0


def tk_slt_uplink_payload_bytes(
    transfer_top_k: int | None,
    gamma: int,
    vocab_size: int,
    prob_bits: int = 16,
) -> float:
    """TK-SLT 论文口径的上行载荷（字节）。

    论文 "Communication-Efficient Collaborative LLM Inference via
    Distributed Speculative Decoding"（WCSP'25, Zheng & Yang）的 §II-B
    式 (2) 在 TK-SLT（§III Solution 1）下变为：每个草稿位置只传 top-K
    稀疏分布的 K 个概率值——

        D_V = K · b_prob bits，b_prob = 16（FP16，§VI-B "logits are
        quantized to half-precision (FP16) and transmitted"）

    γ 个草稿位置合计 D_up = γ·D_V（式 (2)）。``transfer_top_k`` 未启用
    （``None``/``<=0``）时退化为 vanilla DSD 的整词表载荷
    D_V = |V|·b_prob（§II-B），即该论文自己的不压缩基线。

    token 索引（草稿 token id 与 K 个词表索引）按 §II-B "the index size
    is insignificant relative to the vocabulary distribution, our analysis
    considers only the uplink transmission latency associated with the
    vocabulary distribution" **不计字节**——这与 CUHLM 系对 token 索引的
    处理一致（``_send_downlink_index_only``）。

    交叉验证 1（论文 Table II 逐位吻合）：论文实测 c≈0.07（SLM 计算）、
    b_full≈0.23（整词表传输），Table II 的 L 值满足
    L(K) = c + b_full·(K/32000)：
    K=3→0.0700、K=32→0.0702、K=320→0.0723、K=3200→0.093、
    K=32000→0.300。即上行载荷严格 ∝ K·b_prob、不含索引项（若含
    ⌈log₂V⌉ bits 索引，K=320 会得到 ≈0.0839 而非 0.0723）。

    交叉验证 2（论文 §I 的量级声明）：K=|V|=32000、FP16 ⇒
    32000×16/8 = 64000 B/token = 512 kbit，正是论文的
    "about 500 kbit per token"。
    """
    if vocab_size is None or int(vocab_size) <= 1 or int(gamma) <= 0:
        return 0.0
    if transfer_top_k is None or int(transfer_top_k) <= 0:
        k = int(vocab_size)
    else:
        k = min(int(transfer_top_k), int(vocab_size))
    return int(gamma) * k * int(prob_bits) / 8.0


class CUHLM(CommunicationSimulator):
    """
    基于不确定性进行机会传输的对比实验方法。

    超参数：
    - uncertainty_threshold: float, 不确定度阈值，默认0.8
    - M: int, 温度扰动采样数量，默认20
    - theta_max: float, 最大温度扰动，默认2.0
    """

    DEFAULT_COMPRESSED_VOCAB_SIZE = 300

    def __init__(
        self,
        bandwidth_edge_cloud,
        bandwidth_edge_end=float("inf"),
        bandwidth_cloud_end=float("inf"),
        uncertainty_threshold: float = 0.8,
        vocab_size: int = 32000,
        dimension: Dimension = "Mbps",
        ntt_ms_edge_end: float = 20,
        ntt_ms_edge_cloud: float = 200,
        use_stochastic: bool = False,
        stochastic_ntt: bool = False,
        set_mean_bandwidth: bool = True,
        mode: Literal["driving", "static", "walking"] = "static",
        min_bandwidth_mbps: float = 5.0,
        trace_interval_s: float = 0.2,
        bw_model: Literal["instant", "fluid"] = "instant",
    ):
        # 除了edge-cloud链路，其他链路假设无限带宽，因为不传输数据
        super().__init__(
            bandwidth_edge_cloud,
            bandwidth_edge_end,
            bandwidth_cloud_end,
            dimension=dimension,
            ntt_ms_edge_end=ntt_ms_edge_end,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            use_stochastic=use_stochastic,
            stochastic_ntt=stochastic_ntt,
            set_mean_bandwidth=set_mean_bandwidth,
            mode=mode,
            min_bandwidth_mbps=min_bandwidth_mbps,
            trace_interval_s=trace_interval_s,
            bw_model=bw_model,
        )
        self.uncertainty_threshold = uncertainty_threshold
        self.vocab_size = vocab_size

    @staticmethod
    def calculate_uncertainty(
        logits: torch.Tensor | None,
        M: int = 20,
        theta_max: float = 2.0,
        draft_token: Optional[int] = None,
    ) -> float:
        if logits is None or logits.numel() == 0:
            warnings.warn("警告：logits为空，无法计算不确定度，默认返回1.0")
            return 1.0
        if logits.dim() > 1:
            logits = logits[0]
        if draft_token is None:
            warnings.warn("警告：draft_token未提供，默认使用最高概率的token")
            draft_token = int(torch.argmax(logits).item())
        # 向量化温度采样
        temperatures = torch.rand(M, device=logits.device) * theta_max
        temperatures = torch.clamp(temperatures, min=1e-6)

        # 向量化扰动分布计算
        perturbed_logits = logits.unsqueeze(0) / temperatures.unsqueeze(
            1
        )  # [M, vocab_size]
        perturbed_logits = (
            perturbed_logits - perturbed_logits.max(dim=1, keepdim=True)[0]
        )
        perturbed_probs = torch.softmax(perturbed_logits, dim=-1)

        # 批量采样
        perturbed_tokens = torch.multinomial(perturbed_probs, 1).squeeze(1)  # [M]

        # 计算分歧
        disagreements = (perturbed_tokens != draft_token).sum().item()

        return disagreements / M

    @staticmethod
    def _get_current_probs(
        prob_history: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        返回当前时间步的概率分布，为一维张量，(...,vocab_size)
        """
        if prob_history is None or prob_history.numel() == 0:
            warnings.warn("警告：prob_history为空，无法获取当前概率分布")
            return torch.empty(0)
            # prob_history形状: (1, seq_len, 32000)
        if prob_history.dim() == 3:
            current_probs = prob_history[0, -1, :]
        elif prob_history.dim() == 2:
            current_probs = prob_history[-1, :]
        elif prob_history.dim() == 1:
            current_probs = prob_history
        else:
            raise ValueError("prob_history维度不支持")

        return current_probs

    @staticmethod
    def rebuild_full_probs(compressed_probs: torch.Tensor) -> torch.Tensor:
        """
        接收一个稀疏的概率分布张量(compressed_probs)，并重建为完整的概率分布。
        compressed_probs: torch.Tensor，形状为(..., vocab_size)
        重建后形状同样为(..., vocab_size)
        """
        if compressed_probs is None or compressed_probs.numel() == 0:
            warnings.warn("警告：compressed_probs为空，无法重建完整概率分布")
            return torch.empty(0)

        rebuilt_probs = compressed_probs.clone()

        # top-k 位置 = 非零位置（压缩时只写入 top-k 值）
        top_k_mask = compressed_probs > 0
        top_k_sum = compressed_probs.sum(dim=-1, keepdim=True)

        # clamp：数值防御，top-k 之和不应超过 1（浮点误差/异常输入时防负"概率"
        # ——负 uniform 会让重建分布非法）
        residual_mass = (1.0 - top_k_sum).clamp_min(0.0)

        # 尾部 = 显式非 top-k 掩码。此前用 ==0 当尾部，会把本就为 0 的 top-k
        # 项也换成本轮 uniform，轻微污染重建分布
        tail_mask = ~top_k_mask
        tail_count = tail_mask.sum(dim=-1, keepdim=True)

        # 避免除零：没有尾部位置则无需重建
        uniform_prob = torch.where(
            tail_count > 0,
            residual_mass / tail_count,
            torch.zeros_like(residual_mass),
        )
        rebuilt_probs = torch.where(tail_mask, uniform_prob, rebuilt_probs)

        return rebuilt_probs

    def determine_transfer_strategy(
        self, uncertainty: float, current_probs: torch.Tensor | None
    ) -> Tuple[bool, int]:
        "返回是否传输以及传输的词汇表大小"
        if current_probs is None or current_probs.numel() == 0:
            warnings.warn(
                "警告：current_probs为空，无法决定传输策略，默认不传输概率分布"
            )
            return False, 0
        if uncertainty >= self.uncertainty_threshold:  # 不确定度高，传输
            vocab_size = max(
                1,
                self._calculate_compressed_vocab_size(uncertainty, current_probs),
            )
            return True, vocab_size
        else:  # 大模型直接接受当前输出
            return False, 0

    @staticmethod
    def _apply_top_k_compression(probs: torch.Tensor | None, k: int) -> torch.Tensor:
        """
        probs: torch.Tensor，当前词表概率分布，形状为(..., V)
        k: int，保留的top-k数量
        返回压缩后的概率分布，形状与输入相同，但仅保留top-k概率，其他位置为0，值得注意的是，这不是真正的压缩，但我们假设传输时只传输非零部分
        """
        if probs is None or probs.numel() == 0:
            return torch.empty(0)

        # len() 取的是第一维（batch 维），多维输入会误判；top-k 语义在最后一维。
        if k >= probs.shape[-1]:
            return probs

        # 获取top-k indices
        top_k_values, top_k_indices = torch.topk(probs, k, sorted=True)

        # 创建压缩的概率分布（scatter_ 沿最后一维，对 (..., V) 任意维数都正确）
        compressed_probs = torch.zeros_like(probs)
        compressed_probs.scatter_(-1, top_k_indices, top_k_values)

        return compressed_probs

    @staticmethod
    def softplus(z, eta=1.0):
        return torch.log(1 + torch.exp(eta * z)) / eta

    def _calculate_compressed_vocab_size(
        self,
        uncertainty: float,
        current_probs: torch.Tensor,
        theta: float = 0.1,
        draft_token: Optional[int] = None,
    ) -> int:
        """按论文式 (26)（Proposition 2，在线逐轮规则）求最小压缩词表大小：

            k*(t) = arg min { k(t) | U_TV(β_d(t)) ≤ θ }

        （注：论文式 (24) 是离线时间平均版 k* = argmin{k | E_t[U_TV]≤θ}；
        本实现与其在线变体等价，即用式 (26) 的上界逐轮解约束。）

        所有量均定义在**词表概率分布** x(t) 上（论文第 II-A 节：x(t) 由
        logit 经 softmax 归一化而来）。喂原始 logits 会得到无意义的 k*。
        """
        if current_probs is None or current_probs.numel() == 0:
            return 0

        # 确保输入是完整的词汇表概率分布
        if current_probs.shape[-1] != self.vocab_size:
            warnings.warn(
                f"警告：概率分布长度({current_probs.shape[-1]})与词汇表大小({self.vocab_size})不匹配"
            )
            return max(1, min(300, self.vocab_size // 100))

        total_mass = float(current_probs.float().sum())
        if abs(total_mass - 1.0) > 0.05:
            warnings.warn(
                f"警告：输入的总质量 {total_mass:.4f} 偏离 1，"
                "请确认传入的是概率分布（softmax 后）而非原始 logits——"
                "论文式 (16)/(23)/(26) 均定义在概率分布上"
            )

        # Step 1: 计算rejection probability
        a, b = 0.815, -0.066
        beta_d = max(0, min(1, a * uncertainty + b))

        # Step 2: 对概率分布排序（float32 累加，避免半精度累加误差）
        sorted_probs, _ = torch.sort(
            current_probs.float().reshape(-1), descending=True
        )
        vocab_size = self.vocab_size

        # Step 3: 获取draft token概率
        if draft_token is None:
            x_d = float(sorted_probs[0])
        else:
            # draft_token是索引，获取对应概率
            if 0 <= draft_token < vocab_size:
                x_d = float(current_probs.reshape(-1)[draft_token])
            else:
                warnings.warn(f"警告：draft_token索引({draft_token})超出范围")
                x_d = float(sorted_probs[0])

        # Step 4: 计算softplus函数
        eta = 1.0
        l_neg_1 = self.softplus(torch.tensor(-1.0), eta).item()
        l_neg_beta = self.softplus(torch.tensor(-beta_d), eta).item()

        # Step 5: 计算分母（论文式 (26)）：(1-x_d)·ℓ(-1) + x_d·ℓ(-β_d)
        denominator = (1 - x_d) * l_neg_1 + x_d * l_neg_beta
        if denominator <= 0:
            return 30

        # Step 6: 向量化搜索最小的 k（与逐 k 暴力循环数学等价，O(V) 而非 O(V²)）
        #
        # 对每个 k：top_k_sum = Σ_{i≤k} x_i（降序前缀和），
        #   residual = max(1 - top_k_sum, 0)，uniform = residual / (V-k)
        #   numerator_k = Σ_{i>k} |x_i - uniform|（论文式 (23) 分子）
        # 利用降序排列的 crossover 性质（{x_i ≥ u} 恒为前缀）：
        #   m = 尾部中 ≥ uniform 的元素个数（searchsorted 一次求出全部）
        #   head = 尾部前 m 个元素之和（前缀和差分）
        #   numerator = 2·(head - m·uniform) + (V-k)·uniform - tail_sum
        prefix_sum = torch.cumsum(sorted_probs, dim=0)  # prefix_sum[i] = Σ_{j≤i} x_j
        ks = torch.arange(1, vocab_size, device=sorted_probs.device)
        top_k_sum = prefix_sum[ks - 1]
        tail_sum = prefix_sum[vocab_size - 1] - top_k_sum
        residual = (1.0 - top_k_sum).clamp_min(0.0)
        tail_count = vocab_size - ks
        uniform = residual / tail_count

        asc = -sorted_probs  # 升序，用于 searchsorted
        # pos[k] = 全数组中 x_i ≥ uniform_k 的个数；{x≥u} 是前缀，故 m = pos - k（≥0 截断）
        pos = torch.searchsorted(asc, -uniform, right=True)
        m = (pos - ks).clamp_min(0)
        idx = (ks + m - 1).clamp_max(vocab_size - 1)
        head = prefix_sum[idx] - top_k_sum  # 尾部前 m 大元素之和
        numerator = 2.0 * (head - m * uniform) + tail_count * uniform - tail_sum
        numerator = numerator.clamp_min(0.0)

        u_tv = numerator / denominator
        satisfied = u_tv <= theta
        if bool(satisfied.any()):
            return int(ks[satisfied][0])

        # 如果没找到满足条件的k，返回保守值
        return min(self.DEFAULT_COMPRESSED_VOCAB_SIZE, self.vocab_size // 100)

    def terminal_prob(
        self, current_probs: torch.Tensor, logits: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        返回先经过压缩再重建的终端概率分布
        形状为(vocab_size,)
        """
        if current_probs is None and logits is None:
            warnings.warn("警告：current_probs和logits均为空，无法获取终端概率分布")
            return torch.empty(0)

        if logits is None:
            # 按照贪心解码重建logits
            probs = torch.clamp(current_probs, min=1e-8)
            log_probs = torch.log(probs)
            # 减去最大值，使最大概率对应的logit为0
            if current_probs.dim() == 1:
                logits = log_probs - torch.max(log_probs)
            else:
                logits = log_probs - torch.max(log_probs, dim=-1, keepdim=True)[0]

        uncertainty = self.calculate_uncertainty(logits)
        should_transfer_prob, vocab_size = self.determine_transfer_strategy(
            uncertainty, current_probs
        )
        if not should_transfer_prob:
            return current_probs
        if vocab_size < self.vocab_size:
            compressed_probs = self._apply_top_k_compression(current_probs, vocab_size)
            rebuilt_probs = self.rebuild_full_probs(compressed_probs)
            return rebuilt_probs
        else:
            return current_probs


class PreciseCommunicationSimulator(CommunicationSimulator):
    """
    用于基于香农信道容量计算通信参数的通信模拟器
    Args:
        bandwidth_hz: 信道带宽，单位Hz
        channel_gain: 信道增益
        send_power_watt: 发送功率，单位瓦特
        noise_power_watt: 噪声功率，单位瓦特
    """

    def __init__(
        self,
        bandwidth_hz: int | float,
        channel_gain: float,
        send_power_watt: float,
        noise_power_watt: float,
        ntt_ms_edge_end: float = 20,
        ntt_ms_edge_cloud: float = 200,
        edge_cloud_args: dict | None = None,
        edge_end_args: dict | None = None,
        min_bandwidth_mbps: float = 5.0,
        bw_model: Literal["instant", "fluid"] = "instant",
    ):
        SNR = channel_gain * send_power_watt / noise_power_watt
        channel_capacity_bps = bandwidth_hz * math.log2(1 + SNR)
        if not getattr(PreciseCommunicationSimulator, "_has_logged", False):
            logging.info(
                f"信道容量: {channel_capacity_bps / 1e6:.2f} Mbps, 以 {channel_capacity_bps} bps, {channel_capacity_bps / 10} bps, {channel_capacity_bps / 10} bps 初始化 "
            )
            PreciseCommunicationSimulator._has_logged = True

        if edge_cloud_args is None:
            # edge-cloud 是承载概率分布上行数据的无线链路，默认取完整信道容量
            edge_cloud_bandwidth = channel_capacity_bps
        else:
            try:
                edge_cloud_SNR = (
                    edge_cloud_args["channel_gain"]
                    * edge_cloud_args["send_power_watt"]
                    / edge_cloud_args["noise_power_watt"]
                )
                edge_cloud_bandwidth = edge_cloud_args["bandwidth_hz"] * math.log2(
                    1 + edge_cloud_SNR
                )
            except KeyError:
                edge_cloud_bandwidth = channel_capacity_bps

        if edge_end_args is None:
            edge_end_bandwidth = channel_capacity_bps / 10
        else:
            try:
                edge_end_SNR = (
                    edge_end_args["channel_gain"]
                    * edge_end_args["send_power_watt"]
                    / edge_end_args["noise_power_watt"]
                )
                edge_end_bandwidth = edge_end_args["bandwidth_hz"] * math.log2(
                    1 + edge_end_SNR
                )
            except KeyError:
                edge_end_bandwidth = channel_capacity_bps / 10

        # 云端链路与边缘端链路带宽均为信道容量的十分之一
        cloud_end_bandwidth = channel_capacity_bps / 10

        super().__init__(
            edge_cloud_bandwidth,
            edge_end_bandwidth,
            cloud_end_bandwidth,
            dimension="bps",
            ntt_ms_edge_end=ntt_ms_edge_end,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            min_bandwidth_mbps=min_bandwidth_mbps,
            # 香农容量是恒定带宽：流体与 instant 数学等价（排空期间速率
            # 不变），接收参数只为让调用点统一（见 baselines.py 各构造点）
            bw_model=bw_model,
        )

        self.comm_energy = 0.0  # 通信能耗，单位焦耳
        self.send_power_watt = send_power_watt
        self.noise_power_watt = noise_power_watt
        self.bandwidth_hz = bandwidth_hz
        self.channel_gain = channel_gain

    @property
    def total_comm_energy(self):
        # 能耗只按纯发射时间 tx_time 计算（传播时延期间不发射，不计能耗）
        energy = 0.0
        for link_type in ["edge_cloud", "edge_end", "cloud_end"]:
            for unit in self.stats[link_type]:
                energy += unit["tx_time"] * self.send_power_watt
        return energy


class PreciseCUHLM(CUHLM):
    """
    CUHLM的复杂建模版本，基于香农信道容量计算实际通信参数

    参数：
    - bandwidth_hz: 信道带宽，单位Hz
    - channel_gain: 信道增益
    - send_power_watt: 发送功率，单位瓦特
    - noise_power_watt: 噪声功率，单位瓦特
    - uncertainty_threshold: 不确定度阈值，默认0.8
    - vocab_size: 词汇表大小，默认32000
    """

    def __init__(
        self,
        bandwidth_hz: int | float,
        channel_gain: float,
        send_power_watt: float,
        noise_power_watt: float,
        uncertainty_threshold: float = 0.8,
        vocab_size: int = 32000,
        ntt_ms_edge_cloud: float = 200,
        ntt_ms_edge_end: float = 20,
        min_bandwidth_mbps: float = 5.0,
        bw_model: Literal["instant", "fluid"] = "instant",
    ):
        # 计算信噪比
        SNR = channel_gain * send_power_watt / noise_power_watt
        # 根据香农公式计算信道容量（bits/second）
        channel_capacity_bps = bandwidth_hz * math.log2(1 + SNR)

        # 初始化CUHLM，使用计算得到的信道容量
        # edge-cloud使用完整信道容量，其他链路假设为容量的十分之一

        if not getattr(PreciseCUHLM, "_has_logged", False):
            print(
                f"信道容量: {channel_capacity_bps / 1e6:.2f} Mbps, 以 {channel_capacity_bps / 10} bps, {channel_capacity_bps} bps, {channel_capacity_bps / 10} bps 初始化 "
            )
            PreciseCUHLM._has_logged = True

        super().__init__(
            bandwidth_edge_cloud=channel_capacity_bps,
            bandwidth_edge_end=channel_capacity_bps / 10,
            bandwidth_cloud_end=channel_capacity_bps / 10,
            uncertainty_threshold=uncertainty_threshold,
            vocab_size=vocab_size,
            dimension="bps",
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            min_bandwidth_mbps=min_bandwidth_mbps,
            # 同 PreciseCommunicationSimulator：恒定带宽下两模型等价
            bw_model=bw_model,
        )

        # 存储通信物理参数
        self.bandwidth_hz = bandwidth_hz
        self.channel_gain = channel_gain
        self.send_power_watt = send_power_watt
        self.noise_power_watt = noise_power_watt
        self.SNR = SNR
        self.channel_capacity_bps = channel_capacity_bps

        # 通信能耗统计，单位焦耳
        self.comm_energy = 0.0

    @property
    def total_comm_energy(self) -> float:
        """计算总通信能耗（焦耳），只按纯发射时间 tx_time 计算"""
        energy = 0.0
        for link_type in ["edge_cloud", "edge_end", "cloud_end"]:
            for unit in self.stats[link_type]:
                energy += unit["tx_time"] * self.send_power_watt
        return energy
