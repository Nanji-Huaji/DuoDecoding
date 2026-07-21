import sys

from src import utils
from src.communication import CommunicationSimulator
from src.metrics import get_empty_metrics


def test_empty_metrics_include_communication_components() -> None:
    # Given: a fresh metrics record
    metrics = get_empty_metrics()

    # When: communication component fields are inspected
    # Then: both modeled components are explicitly zero initialized
    assert metrics["communication_serialization_time"] == 0.0
    assert metrics["communication_fixed_latency_time"] == 0.0


def test_batch_delay_defaults_to_zero_for_single_session(monkeypatch, tmp_path) -> None:
    # Given: the CLI receives no batch-delay override
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["duodecoding"])

    # When: arguments are parsed
    monkeypatch.setattr(utils, "model_zoo", lambda args: None)
    args = utils.parse_arguments()

    # Then: single-session runs have no synthetic serving queue delay
    assert args.batch_delay == 0.0
    assert args.min_bandwidth_mbps is None


def test_minimum_bandwidth_defaults_to_disabled(monkeypatch, tmp_path) -> None:
    # Given: the CLI receives no minimum-bandwidth override
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["duodecoding"])
    monkeypatch.setattr(utils, "model_zoo", lambda args: None)

    # When: arguments are parsed
    args = utils.parse_arguments()

    # Then: no floor is applied unless an experiment explicitly requests one
    assert args.min_bandwidth_mbps is None


def test_minimum_bandwidth_cli_value_reaches_simulator(monkeypatch, tmp_path) -> None:
    # Given: an experiment explicitly requests a five Mbps floor
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["duodecoding", "--min_bandwidth_mbps", "5"])
    monkeypatch.setattr(utils, "model_zoo", lambda args: None)

    # When: its parsed value configures a communication simulator
    args = utils.parse_arguments()
    simulator = CommunicationSimulator(
        bandwidth_edge_cloud=1.0,
        bandwidth_edge_end=10.0,
        bandwidth_cloud_end=10.0,
        ntt_ms_edge_cloud=0.0,
        ntt_ms_edge_end=0.0,
        min_bandwidth_mbps=args.min_bandwidth_mbps,
    )

    # Then: the experiment configuration applies the requested floor
    assert simulator.simulate_transfer(125_000, "edge_cloud") == 0.2
