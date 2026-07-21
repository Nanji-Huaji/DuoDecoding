import math

import pytest
import torch

from src.communication import (
    CommunicationSimulator,
    PreciseCommunicationSimulator,
    PreciseCUHLM,
)
from src.utils import read_trace_file


def test_transfer_records_serialization_and_fixed_latency_components() -> None:
    # Given: a 1 Mbps edge-cloud link with 200 ms one-way fixed latency
    simulator = CommunicationSimulator(
        bandwidth_edge_cloud=1.0,
        bandwidth_edge_end=10.0,
        bandwidth_cloud_end=10.0,
        ntt_ms_edge_cloud=200.0,
        ntt_ms_edge_end=10.0,
    )

    # When: one megabit is transferred over edge-cloud
    simulator.simulate_transfer(125_000, "edge_cloud")

    # Then: the total is the measured model components with no hidden floor
    transfer = simulator.stats["edge_cloud"][0]
    assert transfer["serialization_time"] == pytest.approx(1.0)
    assert transfer["fixed_latency_time"] == pytest.approx(0.2)
    assert transfer["bandwidth_bytes_per_second"] == pytest.approx(125_000.0)
    assert transfer["transfer_time"] == pytest.approx(1.2)
    assert simulator.edge_cloud_serialization_time == pytest.approx(1.0)
    assert simulator.edge_cloud_fixed_latency_time == pytest.approx(0.2)
    assert simulator.total_serialization_time == pytest.approx(1.0)
    assert simulator.total_fixed_latency_time == pytest.approx(0.2)


def test_default_bandwidth_does_not_apply_a_floor() -> None:
    # Given: a finite link slower than the historical 5 Mbps implicit floor
    simulator = CommunicationSimulator(
        bandwidth_edge_cloud=1.0,
        bandwidth_edge_end=10.0,
        bandwidth_cloud_end=10.0,
        ntt_ms_edge_cloud=0.0,
        ntt_ms_edge_end=0.0,
    )

    # When: one megabit is transferred
    transfer_time = simulator.simulate_transfer(125_000, "edge_cloud")

    # Then: serialization follows the requested one Mbps bandwidth
    assert transfer_time == pytest.approx(1.0)


def test_explicit_minimum_bandwidth_is_applied() -> None:
    # Given: a one Mbps link with an explicit five Mbps floor
    simulator = CommunicationSimulator(
        bandwidth_edge_cloud=1.0,
        bandwidth_edge_end=10.0,
        bandwidth_cloud_end=10.0,
        ntt_ms_edge_cloud=0.0,
        ntt_ms_edge_end=0.0,
        min_bandwidth_mbps=5.0,
    )

    # When: one megabit is transferred
    transfer_time = simulator.simulate_transfer(125_000, "edge_cloud")

    # Then: serialization uses the requested explicit floor
    assert transfer_time == pytest.approx(0.2)


def test_finite_nonpositive_bandwidth_is_rejected() -> None:
    # Given: a finite link without capacity
    # When: the simulator is constructed
    # Then: its invalid physical input is rejected instead of being silently floored
    with pytest.raises(ValueError, match="greater than zero"):
        CommunicationSimulator(
            bandwidth_edge_cloud=0.0,
            bandwidth_edge_end=10.0,
            bandwidth_cloud_end=10.0,
        )


def test_compressed_topk_payload_counts_values_and_indices() -> None:
    # Given: two rows of float16 probabilities compressed to top three
    simulator = CommunicationSimulator(
        bandwidth_edge_cloud=10.0,
        bandwidth_edge_end=10.0,
        bandwidth_cloud_end=10.0,
        ntt_ms_edge_cloud=0.0,
        ntt_ms_edge_end=0.0,
    )
    probabilities = torch.zeros((1, 2, 10), dtype=torch.float16)

    # When: the compressed payload is transferred
    simulator.transfer(
        tokens=None,
        prob=probabilities,
        link_type="edge_cloud",
        is_compressed=True,
        compressed_k=3,
    )

    # Then: each top-k entry includes a value and an int32 index
    assert simulator.edge_cloud_data == 2 * 3 * (2 + 4)


def test_trace_reader_preserves_all_raw_samples(tmp_path) -> None:
    # Given: a trace containing low and trailing zero samples
    trace_file = tmp_path / "trace.list"
    trace_file.write_text("Run 1\n1.0,4.0,0.0\n", encoding="utf-8")

    # When: its first run is parsed
    samples = read_trace_file(str(trace_file), 1)

    # Then: the file is represented faithfully without trimming or clamping
    assert samples == [1.0, 4.0, 0.0]


def test_stochastic_trace_scales_raw_variation_to_target_mean(monkeypatch) -> None:
    # Given: a raw trace with non-zero variation and a two Mbps target mean
    monkeypatch.setattr("src.communication.return_closest_mean_index", lambda *_: 1)
    monkeypatch.setattr("src.communication.read_trace_file", lambda *_: [1.0, 3.0])

    # When: a stochastic simulator is initialized
    simulator = CommunicationSimulator(
        bandwidth_edge_cloud=2.0,
        bandwidth_edge_end=10.0,
        bandwidth_cloud_end=10.0,
        use_stochastic=True,
        set_mean_bandwidth=True,
    )

    # Then: raw variation remains and the scaled samples have the target mean
    assert simulator.trace_data == pytest.approx([1.0, 3.0])
    assert sum(simulator.trace_data) / len(simulator.trace_data) == pytest.approx(2.0)


def test_stochastic_all_zero_trace_has_clear_zero_bandwidth_contract(monkeypatch) -> None:
    # Given: a trace with no available capacity
    monkeypatch.setattr("src.communication.return_closest_mean_index", lambda *_: 1)
    monkeypatch.setattr("src.communication.read_trace_file", lambda *_: [0.0, 0.0])

    # When: a target mean is requested
    with pytest.raises(ValueError, match="zero-mean"):
        CommunicationSimulator(
            bandwidth_edge_cloud=2.0,
            bandwidth_edge_end=10.0,
            bandwidth_cloud_end=10.0,
            use_stochastic=True,
            set_mean_bandwidth=True,
        )


def test_stochastic_trace_exposes_explicitly_floored_samples(monkeypatch) -> None:
    # Given: a scaled trace below an explicitly requested floor
    monkeypatch.setattr("src.communication.return_closest_mean_index", lambda *_: 1)
    monkeypatch.setattr("src.communication.read_trace_file", lambda *_: [1.0, 3.0])

    # When: a stochastic simulator is configured with a five Mbps floor
    simulator = CommunicationSimulator(
        bandwidth_edge_cloud=2.0,
        bandwidth_edge_end=10.0,
        bandwidth_cloud_end=10.0,
        use_stochastic=True,
        set_mean_bandwidth=True,
        min_bandwidth_mbps=5.0,
        ntt_ms_edge_cloud=0.0,
    )

    # Then: its exposed samples show the explicitly altered trace
    assert simulator.trace_data == pytest.approx([5.0, 5.0])


def test_stochastic_zero_sample_models_an_outage_without_crashing(monkeypatch) -> None:
    # Given: a trace with one positive sample and one zero-capacity outage
    monkeypatch.setattr("src.communication.return_closest_mean_index", lambda *_: 1)
    monkeypatch.setattr("src.communication.read_trace_file", lambda *_: [2.0, 0.0])
    simulator = CommunicationSimulator(
        bandwidth_edge_cloud=2.0,
        bandwidth_edge_end=10.0,
        bandwidth_cloud_end=10.0,
        use_stochastic=True,
        set_mean_bandwidth=True,
        ntt_ms_edge_cloud=5.0,
    )

    # When: a positive-size transfer consumes the zero-capacity sample
    simulator.simulate_transfer(1, "edge_cloud")
    transfer_time = simulator.simulate_transfer(1, "edge_cloud")

    # Then: the outage is retained and modeled as unbounded transfer time
    assert simulator.trace_data == pytest.approx([4.0, 0.0])
    assert math.isinf(transfer_time)
    assert math.isinf(simulator.stats["edge_cloud"][1]["serialization_time"])


def test_stochastic_zero_sample_uses_explicit_floor(monkeypatch) -> None:
    # Given: a zero-capacity trace sample and an explicit five Mbps floor
    monkeypatch.setattr("src.communication.return_closest_mean_index", lambda *_: 1)
    monkeypatch.setattr("src.communication.read_trace_file", lambda *_: [2.0, 0.0])
    simulator = CommunicationSimulator(
        bandwidth_edge_cloud=2.0,
        bandwidth_edge_end=10.0,
        bandwidth_cloud_end=10.0,
        use_stochastic=True,
        set_mean_bandwidth=True,
        min_bandwidth_mbps=5.0,
        ntt_ms_edge_cloud=0.0,
    )

    # When: the floored outage sample is used for one megabit
    simulator.simulate_transfer(125_000, "edge_cloud")
    transfer_time = simulator.simulate_transfer(125_000, "edge_cloud")

    # Then: the explicit floor produces a finite serialization time
    assert transfer_time == pytest.approx(0.2)


@pytest.mark.parametrize("invalid_sample", [-1.0, math.inf, math.nan])
def test_stochastic_trace_rejects_invalid_samples(monkeypatch, invalid_sample) -> None:
    # Given: a trace with a nonphysical negative or non-finite sample
    monkeypatch.setattr("src.communication.return_closest_mean_index", lambda *_: 1)
    monkeypatch.setattr(
        "src.communication.read_trace_file", lambda *_: [2.0, invalid_sample]
    )

    # When: stochastic scaling is configured
    # Then: invalid trace data is rejected before simulation
    with pytest.raises(ValueError, match="non-negative and finite"):
        CommunicationSimulator(
            bandwidth_edge_cloud=2.0,
            bandwidth_edge_end=10.0,
            bandwidth_cloud_end=10.0,
            use_stochastic=True,
            set_mean_bandwidth=True,
        )


def test_infinite_bandwidth_has_zero_serialization_time() -> None:
    # Given: an idealized infinite-bandwidth edge-cloud link
    simulator = CommunicationSimulator(
        bandwidth_edge_cloud=math.inf,
        bandwidth_edge_end=10.0,
        bandwidth_cloud_end=10.0,
        ntt_ms_edge_cloud=7.0,
        ntt_ms_edge_end=0.0,
    )

    # When: data is transferred
    simulator.simulate_transfer(123, "edge_cloud")

    # Then: only the configured fixed latency remains
    transfer = simulator.stats["edge_cloud"][0]
    assert transfer["serialization_time"] == pytest.approx(0.0)
    assert transfer["transfer_time"] == pytest.approx(0.007)


@pytest.mark.parametrize(
    "simulator_type",
    [PreciseCommunicationSimulator, PreciseCUHLM],
)
def test_precise_simulators_apply_explicit_minimum_bandwidth(simulator_type) -> None:
    # Given: a precise simulator whose Shannon capacity is below five Mbps
    simulator = simulator_type(
        bandwidth_hz=1.0,
        channel_gain=1.0,
        send_power_watt=1.0,
        noise_power_watt=1.0,
        ntt_ms_edge_cloud=0.0,
        ntt_ms_edge_end=0.0,
        min_bandwidth_mbps=5.0,
    )

    # When: one megabit is transferred over edge-cloud
    transfer_time = simulator.simulate_transfer(125_000, "edge_cloud")

    # Then: the same explicit floor policy applies as in the simple simulator
    assert transfer_time == pytest.approx(0.2)
