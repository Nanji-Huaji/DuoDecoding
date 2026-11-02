"""Communication simulator factories extracted from core (S3a).

The factories keep their historical signatures and bodies. ``baselines.core``
privately re-exports them for compatibility; calls routed through the factories
resolve ``CommunicationSimulator``/``PreciseCommunicationSimulator`` here, while
direct constructor sites in core still resolve core's bindings.
"""

from ..comm_accounting import CommEnv
from ..communication import CommunicationSimulator, PreciseCommunicationSimulator


def _make_comm_simulator(
    env: CommEnv,
    *,
    precise: bool,
    use_stochastic_comm: bool,
    ntt_ms_edge_cloud: float,
    ntt_ms_edge_end: float,
    edge_end: float,
    cloud_end: float,
    transfer_top_k: int | None = None,
) -> CommunicationSimulator:
    """按统一口径构造通信模拟器（H2，tri/cee 族）。

    这 4 个方法（tridecoding / ceesd_without_arp / cee_dsd / cee_dssd）的
    ``if use_precise_comm_sim: ... else: ...`` 逐字重复。对称链路 4 个方法
    （DSSD/DSD/tk_slt/adaptive_decoding）走
    ``_make_symmetric_comm_simulator``：同样的 precise 分支，但
    edge_end/cloud_end 固定为 inf。

    仍保留显式块的只有 ``adaptive_tridecoding``（它在实参里读
    comm_trace_mode，抽进工厂会改变指纹事件序）与两个 CUHLM 方法
    （用 CUHLM 子类）。

    未给出的可选参数不传，让构造器用自身默认值（None / "static"），
    与抽取前逐字等价。
    """
    if precise:
        sim: CommunicationSimulator = PreciseCommunicationSimulator(
            bandwidth_hz=1e7,
            channel_gain=1e-8,
            send_power_watt=0.5,
            noise_power_watt=1e-10,
            min_bandwidth_mbps=env.min_bandwidth_mbps,
            bw_model=env.comm_bw_model,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
        )
    else:
        kwargs = {
            "bandwidth_edge_cloud": env.edge_cloud_bandwidth,
            "bandwidth_edge_end": edge_end,
            "bandwidth_cloud_end": cloud_end,
            "min_bandwidth_mbps": env.min_bandwidth_mbps,
            "bw_model": env.comm_bw_model,
            "dimension": "Mbps",
            "ntt_ms_edge_cloud": ntt_ms_edge_cloud,
            "ntt_ms_edge_end": ntt_ms_edge_end,
            "use_stochastic": use_stochastic_comm,
            "stochastic_ntt": env.stochastic_ntt,
        }
        if transfer_top_k is not None:
            kwargs["transfer_top_k"] = transfer_top_k
        sim = CommunicationSimulator(**kwargs)
    return sim


def _make_symmetric_comm_simulator(
    env: CommEnv,
    *,
    precise: bool,
    use_stochastic_comm: bool,
    ntt_ms_edge_cloud: float,
    ntt_ms_edge_end: float,
) -> CommunicationSimulator:
    """对称链路方法的构造点（DSSD/DSD/TK-SLT/adaptive_decoding）。

    这 4 个方法的建模只关心 edge-cloud 链路，edge_end 与 cloud_end 一律取
    无限带宽。``float("inf")`` 特意留在 else 分支里：它和迁移前一样出现在
    CommunicationSimulator 的实参上，逻辑指纹的事件序列因此不变。
    """
    if precise:
        sim: CommunicationSimulator = PreciseCommunicationSimulator(
            bandwidth_hz=1e7,
            channel_gain=1e-8,
            send_power_watt=0.5,
            noise_power_watt=1e-10,
            min_bandwidth_mbps=env.min_bandwidth_mbps,
            bw_model=env.comm_bw_model,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
        )
    else:
        sim = CommunicationSimulator(
            bandwidth_edge_cloud=env.edge_cloud_bandwidth,
            bandwidth_edge_end=float("inf"),
            bandwidth_cloud_end=float("inf"),
            min_bandwidth_mbps=env.min_bandwidth_mbps,
            bw_model=env.comm_bw_model,
            dimension="Mbps",
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            use_stochastic=use_stochastic_comm,
            stochastic_ntt=env.stochastic_ntt,
        )
    return sim
