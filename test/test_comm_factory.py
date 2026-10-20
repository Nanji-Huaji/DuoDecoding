"""H2 通信模拟器工厂的接线守卫。

工厂是 8 个方法的构造点（tri/cee 4 个 + 对称链路 4 个）；参数一旦漂移，golden 只能看到
"通信时间变了"，定位要翻回工厂。这里直接断言构造参数：
- precise 分支的 ``bandwidth_hz=1e7`` 是迁移前逐字值（1e6 会让信道容量
  缩小 10 倍，且 --use_precise 不在默认出表路径上，golden 覆盖不到）；
- regular 分支必须原样传 edge_end/cloud_end/transfer_top_k。
"""

from argparse import Namespace
from unittest.mock import patch

from src.baselines import (
    _make_comm_simulator,
    _make_symmetric_comm_simulator,
)
from src.comm_accounting import CommEnv


def _env() -> CommEnv:
    return CommEnv.from_args(
        Namespace(
            edge_cloud_bandwidth=46.0,
            edge_end_bandwidth=563.0,
            cloud_end_bandwidth=46.0,
            min_bandwidth_mbps=5.0,
            comm_bw_model="fluid",
            stochastic_ntt=True,
        )
    )


def _call(**overrides):
    kwargs = dict(
        precise=False,
        use_stochastic_comm=True,
        ntt_ms_edge_cloud=50.0,
        ntt_ms_edge_end=0.317,
        edge_end=563.0,
        cloud_end=46.0,
    )
    kwargs.update(overrides)
    return _make_comm_simulator(_env(), **kwargs)


def test_precise_path_keeps_historical_channel_bandwidth():
    with patch("src.baselines.PreciseCommunicationSimulator") as precise:
        _call(precise=True)
    kwargs = precise.call_args.kwargs
    assert kwargs["bandwidth_hz"] == 1e7  # 迁移前逐字值，勿改成 1e6
    assert kwargs["channel_gain"] == 1e-8
    assert kwargs["send_power_watt"] == 0.5
    assert kwargs["noise_power_watt"] == 1e-10
    assert kwargs["ntt_ms_edge_cloud"] == 50.0
    assert kwargs["ntt_ms_edge_end"] == 0.317
    assert kwargs["bw_model"] == "fluid"


def test_regular_path_passes_links_and_topk():
    with patch("src.baselines.CommunicationSimulator") as sim:
        _call(transfer_top_k=300)
    kwargs = sim.call_args.kwargs
    assert kwargs["bandwidth_edge_cloud"] == 46.0
    assert kwargs["bandwidth_edge_end"] == 563.0
    assert kwargs["bandwidth_cloud_end"] == 46.0
    assert kwargs["transfer_top_k"] == 300
    assert kwargs["use_stochastic"] is True
    assert kwargs["stochastic_ntt"] is True
    assert kwargs["dimension"] == "Mbps"


def test_transfer_top_k_omitted_when_none():
    """未给 top-k 时不传，交给构造器默认（与抽取前逐字等价）。"""
    with patch("src.baselines.CommunicationSimulator") as sim:
        _call(transfer_top_k=None)
    assert "transfer_top_k" not in sim.call_args.kwargs


def test_symmetric_path_forces_infinite_edge_links():
    """DSSD/DSD/TK-SLT/adaptive_decoding 只建模 edge-cloud；远端链路取 inf。"""
    with patch("src.baselines.CommunicationSimulator") as sim:
        _make_symmetric_comm_simulator(
            _env(),
            precise=False,
            use_stochastic_comm=True,
            ntt_ms_edge_cloud=50.0,
            ntt_ms_edge_end=0.317,
        )
    kwargs = sim.call_args.kwargs
    assert kwargs["bandwidth_edge_end"] == float("inf")
    assert kwargs["bandwidth_cloud_end"] == float("inf")
    assert kwargs["bandwidth_edge_cloud"] == 46.0
    assert kwargs["stochastic_ntt"] is True


def test_symmetric_precise_path_uses_historical_channel_bandwidth():
    with patch("src.baselines.PreciseCommunicationSimulator") as precise:
        _make_symmetric_comm_simulator(
            _env(),
            precise=True,
            use_stochastic_comm=False,
            ntt_ms_edge_cloud=50.0,
            ntt_ms_edge_end=0.317,
        )
    assert precise.call_args.kwargs["bandwidth_hz"] == 1e7
