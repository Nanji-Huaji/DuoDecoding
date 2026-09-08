"""Regression tests for the RL agent state-unit conversion bug.

Historical bug: baselines.py fed the RL adapter with raw simulator values —
bandwidth in bytes/second and NTT in seconds — while
``RLNetworkAdapter.select_config`` expects Mbps and milliseconds. As a
result the normalized bandwidth feature always saturated at 1.0 and the
normalized latency feature always collapsed to ~0.0, so the RL agent could
not perceive network conditions at all.

The fix exposes unit-explicit accessors on ``CommunicationSimulator``
(``bandwidth_*_mbps`` / ``ntt_*_ms``), switches all call sites to them, and
adds a runtime unit guard inside the RL adapter.
"""

import warnings

import pytest

from src.communication import CommunicationSimulator
from src.rl_adapter import RLNetworkAdapter, TASK_MAP


def _make_simulator() -> CommunicationSimulator:
    return CommunicationSimulator(
        bandwidth_edge_cloud=23.6,
        bandwidth_edge_end=563,
        bandwidth_cloud_end=23.6,
        dimension="Mbps",
        ntt_ms_edge_end=20,
        ntt_ms_edge_cloud=200,
    )


def test_simulator_accessors_return_mbps_and_ms():
    # Given: a simulator constructed from Mbps / ms arguments.
    sim = _make_simulator()

    # Then: the unit-explicit accessors round-trip the original values,
    # while the internal storage stays in bytes/second and seconds.
    assert sim.bandwidth_edge_cloud_mbps == pytest.approx(23.6)
    assert sim.bandwidth_edge_end_mbps == pytest.approx(563)
    assert sim.bandwidth_cloud_end_mbps == pytest.approx(23.6)
    assert sim.ntt_edge_end_ms == pytest.approx(20.0)
    assert sim.ntt_edge_cloud_ms == pytest.approx(200.0)

    assert sim.bandwidth_edge_cloud == pytest.approx(23.6e6 / 8)
    assert sim.ntt_edge_cloud == pytest.approx(0.2)


def test_simulator_accessors_handle_bps_dimension():
    # Given: a simulator constructed with the bps dimension (e.g. the precise
    # Shannon-capacity simulator path).
    sim = CommunicationSimulator(
        bandwidth_edge_cloud=10e6,
        bandwidth_edge_end=1e6,
        bandwidth_cloud_end=10e6,
        dimension="bps",
    )

    # Then: the Mbps accessors undo the bps->B/s conversion correctly.
    assert sim.bandwidth_edge_cloud_mbps == pytest.approx(10.0)
    assert sim.bandwidth_edge_end_mbps == pytest.approx(1.0)


def test_rl_feature_normalization_uses_mbps_and_ms():
    # Given: an adapter with the default normalization scales.
    adapter = RLNetworkAdapter(
        __import__("argparse").Namespace(),
        model_path="/tmp/unused_rl_unit_test.pth",
        best_model_path="/tmp/unused_rl_unit_best_test.pth",
        device="cpu",
        init_seed=73,
        init_strategy="fresh",
    )
    task_idx = TASK_MAP["gsm8k"]

    # When: the state is built from correctly-scaled units.
    features = adapter._get_current_feature_vector(
        bandwidth_mbps=23.6, latency_ms=200.0, entropy=2.0, last_acc_prob=0.7, task_name="gsm8k"
    )

    # Then: bandwidth and latency are well inside (0, 1) instead of saturating.
    assert features[0] == pytest.approx(23.6 / 1000.0)
    assert features[1] == pytest.approx(200.0 / 500.0)
    assert features[2] == pytest.approx(2.0 / 10.0)
    assert features[3] == pytest.approx(0.7)
    assert features[4:][task_idx] == pytest.approx(1.0)


def test_pre_fix_raw_simulator_units_saturate_features():
    # Given: the values as stored inside the simulator (B/s and seconds),
    # which is what the call sites erroneously passed before the fix.
    adapter = RLNetworkAdapter(
        __import__("argparse").Namespace(),
        model_path="/tmp/unused_rl_unit_test2.pth",
        best_model_path="/tmp/unused_rl_unit_best_test2.pth",
        device="cpu",
        init_seed=73,
        init_strategy="fresh",
    )
    sim = _make_simulator()

    # When: raw internal units are fed through the same normalization.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buggy = adapter._get_current_feature_vector(
            bandwidth_mbps=sim.bandwidth_edge_cloud,  # bytes/second
            latency_ms=sim.ntt_edge_cloud,  # seconds
            entropy=2.0,
            last_acc_prob=0.7,
            task_name="gsm8k",
        )

    # Then: the buggy features degenerate — bandwidth saturates at 1.0 and
    # latency collapses to ~0, i.e. the agent cannot perceive the network.
    assert buggy[0] == pytest.approx(1.0)
    assert buggy[1] < 0.01

    # While the fixed accessor path produces informative features.
    fixed = adapter._get_current_feature_vector(
        bandwidth_mbps=sim.bandwidth_edge_cloud_mbps,
        latency_ms=sim.ntt_edge_cloud_ms,
        entropy=2.0,
        last_acc_prob=0.7,
        task_name="gsm8k",
    )
    assert 0.0 < fixed[0] < 1.0
    assert 0.0 < fixed[1] < 1.0


def test_unit_guard_warns_on_misused_units():
    # Given: an adapter whose state would be built from raw simulator units.
    adapter = RLNetworkAdapter(
        __import__("argparse").Namespace(),
        model_path="/tmp/unused_rl_unit_test3.pth",
        best_model_path="/tmp/unused_rl_unit_best_test3.pth",
        device="cpu",
        init_seed=73,
        init_strategy="fresh",
    )

    # When: bytes/second bandwidth and seconds latency are passed.
    with pytest.warns(UserWarning, match="bytes/second"):
        adapter._check_state_units(bandwidth_mbps=23.6e6 / 8, latency_ms=0.2)
    with pytest.warns(UserWarning, match="seconds"):
        adapter._check_state_units(bandwidth_mbps=23.6, latency_ms=0.2)

    # Then: correctly-scaled values raise no warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adapter._check_state_units(bandwidth_mbps=23.6, latency_ms=200.0)
        adapter._check_state_units(bandwidth_mbps=1000.0, latency_ms=500.0)
