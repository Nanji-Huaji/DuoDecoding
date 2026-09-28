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


class TestCuhlmCompressedVocab:
    """CUHLM 压缩词表公式（论文 arXiv:2505.11788 式 (26) 在线规则）的回归测试。

    覆盖 2026-03 修复：
    - O(V) 向量化实现与逐 k 暴力循环数学等价；
    - 输入必须是概率分布（喂原始 logits 会触发告警）；
    - `_apply_top_k_compression` 对 (..., V) 多维输入按最后一维压缩；
    - `uncertainty_threshold` 边界语义（u > u_th 才传输）。
    """

    @staticmethod
    def _bruteforce_k_star(uncertainty, probs, theta=0.1):
        """修复前的逐 k 暴力循环（作为等价性参照）。"""
        a, b = 0.815, -0.066
        beta_d = max(0.0, min(1.0, a * uncertainty + b))
        sorted_probs, _ = torch.sort(probs.float().reshape(-1), descending=True)
        vocab = sorted_probs.numel()
        x_d = float(sorted_probs[0])
        l_neg_1 = float(torch.log1p(torch.exp(torch.tensor(-1.0))))
        l_neg_beta = float(torch.log1p(torch.exp(torch.tensor(-beta_d))))
        denom = (1 - x_d) * l_neg_1 + x_d * l_neg_beta
        if denom <= 0:
            return 30
        for k in range(1, vocab):
            top = float(sorted_probs[:k].sum())
            residual = 1.0 - top
            u = residual / (vocab - k) if residual > 0 else 0.0
            num = float((sorted_probs[k:] - u).abs().sum())
            if num / denom <= theta:
                return k
        return min(300, vocab // 100)

    def test_vectorized_matches_bruteforce(self):
        torch.manual_seed(0)
        for vocab in (50, 300):
            for alpha in (0.05, 1.0, 10.0):
                dist = torch.distributions.Dirichlet(
                    torch.full((vocab,), float(alpha))
                )
                probs = dist.sample()
                sim = CUHLM(
                    20.0,
                    vocab_size=vocab,
                    ntt_ms_edge_cloud=0.0,
                )
                for uncertainty in (0.0, 0.3, 0.6, 0.9):
                    ref = self._bruteforce_k_star(uncertainty, probs)
                    new = sim._calculate_compressed_vocab_size(uncertainty, probs)
                    assert abs(new - ref) <= 1, (vocab, alpha, uncertainty, ref, new)

    def test_uniform_distribution_gives_k1(self):
        # 均匀分布：top-1 + 均匀尾重构恰好无损 ⇒ 分子为 0 ⇒ k* = 1。
        vocab = 64
        probs = torch.full((vocab,), 1.0 / vocab)
        sim = CUHLM(20.0, vocab_size=vocab, ntt_ms_edge_cloud=0.0)
        assert sim._calculate_compressed_vocab_size(0.5, probs) == 1

    def test_warns_when_fed_raw_logits(self):
        vocab = 64
        torch.manual_seed(1)
        raw_logits = torch.randn(vocab) * 5  # 总质量远离 1
        sim = CUHLM(20.0, vocab_size=vocab, ntt_ms_edge_cloud=0.0)
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            sim._calculate_compressed_vocab_size(0.5, raw_logits)
        assert any("概率分布" in str(w.message) for w in caught)

    def test_threshold_boundary_semantics(self):
        sim = CUHLM(
            20.0,
            vocab_size=64,
            uncertainty_threshold=0.8,
            ntt_ms_edge_cloud=0.0,
        )
        probs = torch.full((64,), 1.0 / 64)
        should, _ = sim.determine_transfer_strategy(0.79, probs)
        assert should is False  # u <= u_th：机会跳过，不上传
        should, k = sim.determine_transfer_strategy(0.81, probs)
        assert should is True and 1 <= k < 64  # u > u_th：压缩上传

    def test_apply_top_k_compression_on_2d(self):
        torch.manual_seed(2)
        probs = torch.softmax(torch.randn(3, 32), dim=-1)
        from src.communication import CUHLM as _CUHLM

        compressed = _CUHLM._apply_top_k_compression(probs, 5)
        # 每行恰好保留 5 个非零项，且非零值与该行 top-5 概率一致
        # （恢复总质量为 1 是 rebuild_full_probs 的职责，压缩本身只保留 top-k）
        assert compressed.shape == probs.shape
        top5_values = torch.topk(probs, 5, dim=-1).values
        for row in compressed:
            assert int((row > 0).sum()) == 5
        assert torch.allclose(
            torch.sort(compressed, dim=-1, descending=True).values[:, :5],
            top5_values,
            atol=1e-7,
        )
        # k 曾与 batch 维（3）比较：k=5 >= 3 会原样返回；现在应真正压缩
        assert not torch.equal(compressed, probs)
