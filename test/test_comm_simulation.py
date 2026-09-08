"""Regression tests for the network simulation fixes:

- P1: configurable bandwidth floor (min_bandwidth_mbps, 0 disables).
- P2: time-aligned advancement of the stochastic bandwidth trace.
- P5: PreciseCommunicationSimulator link-bandwidth mapping matches its
  documented intent (edge_cloud = full capacity, others = 1/10).
- P6: communication energy counts only transmit time (tx_time), not
  propagation delay (NTT).
"""

import math

import torch

from src.communication import (
    CommunicationSimulator,
    CUHLM,
    PreciseCommunicationSimulator,
)


class TestBandwidthFloor:
    def test_default_floor_clamps_to_5mbps(self):
        # 1 Mbps link: default floor raises it to 5 Mbps.
        sim = CommunicationSimulator(
            1.0, float("inf"), float("inf"), dimension="Mbps", ntt_ms_edge_cloud=0.0
        )
        # 5 Mbit of payload at the floored 5 Mbps -> 1.0 s
        t = sim.simulate_transfer(5e6 / 8, "edge_cloud")
        assert abs(t - 1.0) < 1e-6

    def test_zero_floor_disables_clamping(self):
        sim = CommunicationSimulator(
            1.0,
            float("inf"),
            float("inf"),
            dimension="Mbps",
            ntt_ms_edge_cloud=0.0,
            min_bandwidth_mbps=0.0,
        )
        # 1 Mbit of payload at 1 Mbps -> 1.0 s
        t = sim.simulate_transfer(1e6 / 8, "edge_cloud")
        assert abs(t - 1.0) < 1e-6

    def test_custom_floor(self):
        sim = CommunicationSimulator(
            0.1,
            float("inf"),
            float("inf"),
            dimension="Mbps",
            ntt_ms_edge_cloud=0.0,
            min_bandwidth_mbps=0.5,
        )
        # 0.5 Mbit of payload at floored 0.5 Mbps -> 1.0 s
        t = sim.simulate_transfer(0.5e6 / 8, "edge_cloud")
        assert abs(t - 1.0) < 1e-6

    def test_cuhlm_passthrough(self):
        sim = CUHLM(
            2.0,
            float("inf"),
            float("inf"),
            dimension="Mbps",
            ntt_ms_edge_cloud=0.0,
            min_bandwidth_mbps=0.0,
        )
        # 2 Mbit at 2 Mbps -> 1.0 s
        t = sim.simulate_transfer(2e6 / 8, "edge_cloud")
        assert abs(t - 1.0) < 1e-6

    def test_trace_scaling_respects_floor(self):
        sim = CommunicationSimulator(
            1.0,
            float("inf"),
            float("inf"),
            dimension="Mbps",
            use_stochastic=True,
            set_mean_bandwidth=True,
            min_bandwidth_mbps=0.1,
        )
        assert min(sim.trace_data) >= 0.1 - 1e-9

        sim_default = CommunicationSimulator(
            1.0,
            float("inf"),
            float("inf"),
            dimension="Mbps",
            use_stochastic=True,
            set_mean_bandwidth=True,
        )
        assert min(sim_default.trace_data) >= 5.0 - 1e-9


class TestTraceTimeAlignment:
    def _make_sim(self):
        sim = CommunicationSimulator(
            100.0,
            float("inf"),
            float("inf"),
            dimension="Mbps",
            ntt_ms_edge_cloud=0.0,
            min_bandwidth_mbps=0.0,
            trace_interval_s=0.001,
        )
        # Replace the trace with a flat known sequence (values in Mbps).
        sim.use_stochastic = True
        sim.trace_data = [10.0, 10.0, 10.0, 10.0, 10.0]
        sim.trace_index = 0
        return sim

    def test_long_transfer_advances_multiple_trace_steps(self):
        sim = self._make_sim()
        # 10 Mbps * 1 s = 10 Mbit -> transfer_time = 1.0 s -> 1000 trace steps
        sim.simulate_transfer(10e6 / 8, "edge_cloud")
        assert sim.trace_index == 1000 % 5

    def test_short_transfer_advances_at_least_one_step(self):
        sim = self._make_sim()
        # Tiny message: transfer_time ~ 0 -> still advances exactly 1 step
        sim.simulate_transfer(6, "edge_cloud")
        assert sim.trace_index == 1

    def test_advance_scales_with_duration(self):
        # Trace length 3 so that 1.0 s (1000 steps) and 2.0 s (2000 steps)
        # land on different indices: 1000 % 3 == 1, 2000 % 3 == 2.
        sim1 = self._make_sim()
        sim1.trace_data = [10.0, 10.0, 10.0]
        sim1.simulate_transfer(10e6 / 8, "edge_cloud")  # 1.0 s -> 1000 steps
        assert sim1.trace_index == 1

        sim2 = self._make_sim()
        sim2.trace_data = [10.0, 10.0, 10.0]
        sim2.simulate_transfer(20e6 / 8, "edge_cloud")  # 2.0 s -> 2000 steps
        assert sim2.trace_index == 2


class TestPreciseSimulator:
    @staticmethod
    def _capacity():
        return 1e7 * math.log2(1 + 1e-8 * 0.5 / 1e-10)  # bps

    def test_link_mapping_matches_intent(self):
        sim = PreciseCommunicationSimulator(
            bandwidth_hz=1e7,
            channel_gain=1e-8,
            send_power_watt=0.5,
            noise_power_watt=1e-10,
            min_bandwidth_mbps=0.0,
        )
        cap = self._capacity()
        assert abs(sim.bandwidth_edge_cloud - cap / 8) < 1e-6
        assert abs(sim.bandwidth_edge_end - cap / 10 / 8) < 1e-6
        assert abs(sim.bandwidth_cloud_end - cap / 10 / 8) < 1e-6

    def test_energy_counts_tx_time_not_propagation(self):
        sim = PreciseCommunicationSimulator(
            bandwidth_hz=1e7,
            channel_gain=1e-8,
            send_power_watt=0.5,
            noise_power_watt=1e-10,
            ntt_ms_edge_cloud=200.0,  # large propagation delay
            min_bandwidth_mbps=0.0,
        )
        cap = self._capacity()
        payload = 1e6  # bytes
        sim.simulate_transfer(payload, "edge_cloud")
        expected_tx = payload / (cap / 8)
        # 200 ms propagation delay must NOT contribute energy
        assert abs(sim.total_comm_energy - expected_tx * 0.5) < 1e-9
        assert sim.stats["edge_cloud"][0]["transfer_time"] >= 0.2

    def test_stats_carry_tx_time(self):
        sim = CommunicationSimulator(
            100.0, float("inf"), float("inf"), dimension="Mbps", ntt_ms_edge_cloud=0.0
        )
        sim.simulate_transfer(1e6, "edge_cloud")
        unit = sim.stats["edge_cloud"][0]
        assert unit["tx_time"] == unit["transfer_time"]  # ntt == 0 here
        assert unit["tx_time"] == 1e6 / (100e6 / 8)
