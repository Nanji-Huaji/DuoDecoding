"""Regression tests for the network simulation fixes:

- P1: configurable bandwidth floor (min_bandwidth_mbps, 0 disables).
- P2: time-aligned advancement of the stochastic bandwidth trace.
- P5: PreciseCommunicationSimulator link-bandwidth mapping matches its
  documented intent (edge_cloud = full capacity, others = 1/10).
- P6: communication energy counts only transmit time (tx_time), not
  propagation delay (NTT).
"""

import math

import pytest

import torch

from src.metrics import INT_SIZE

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


class TestProbBitsBillingWindow:
    """B16：位宽计费窗口语义（缝隙已关死）。

    计费条件与量化穿透阈值镜像后：bits >= 16 永远按 element_size
    全宽计费（数据不量化）。argparse 默认值 16 恰在旧窗口左端点，
    此处锁死其贴边行为。
    """

    def _sim(self):
        # 8 MBps = 1 B/ms，传输时间毫秒数即字节数
        return CommunicationSimulator(8, float("inf"), float("inf"), dimension="MBps")

    def test_fp32_bits16_bills_full_width(self):
        # 旧条件 `0 < 16 < 8*4=32` 会按 2B/项计费，数据却未量化
        prob = torch.zeros(1, 1, 100, dtype=torch.float32)
        sim = self._sim()
        t16 = sim.transfer(None, prob, "edge_cloud", False, None, 16)
        t32 = sim.transfer(None, prob, "edge_cloud", False, None, 32)
        t8 = sim.transfer(None, prob, "edge_cloud", False, None, 8)
        assert t16 == pytest.approx(t32)          # 400B：全宽
        # 差值断言（扣除公共 NTT 底噪）：400B − 100B = 300B @ 8MB/s
        assert t16 - t8 == pytest.approx(300 / 8e6, rel=1e-6)

    def test_fp16_bits16_bills_full_width(self):
        prob = torch.zeros(1, 1, 100, dtype=torch.float16)
        sim = self._sim()
        t16 = sim.transfer(None, prob, "edge_cloud", False, None, 16)
        t12 = sim.transfer(None, prob, "edge_cloud", False, None, 12)
        # fp16 全宽 2B/项=200B；bits=12 → 1.5B/项=150B（差值扣底噪）
        assert t16 - t12 == pytest.approx(50 / 8e6, rel=1e-6)

    def test_zero_or_negative_bits_ignored(self):
        prob = torch.zeros(1, 1, 100, dtype=torch.float32)
        sim = self._sim()
        t0 = sim.transfer(None, prob, "edge_cloud", False, None, 0)
        tnone = sim.transfer(None, prob, "edge_cloud", False, None, None)
        assert t0 == pytest.approx(tnone)


class TestNttTraceCursorIsolation:
    """R7: RTT trace 回放游标必须按模拟器实例隔离。

    旧实现把它放在模块级 `_NTT_TRACE_STATE["index"]`，于是同一进程里出现第二个
    实例（本测试文件就是——每个测试各建实例；将来同进程复用/标定脚本同理）时，
    两者交替推进同一个游标，各自只拿到真实 trace 的隔一个采样。
    `Baselines.__init__` 每次都会 `configure_ntt_trace(...)` 重置游标，所以这条
    在单实例的 eval 路径上与旧行为逐位一致；差别只在多实例。
    """

    @staticmethod
    def _reset():
        from src.communication import configure_ntt_trace

        configure_ntt_trace([])

    @pytest.fixture(autouse=True)
    def _clean_trace_state(self):
        self._reset()
        yield
        self._reset()

    @staticmethod
    def _sim():
        return CommunicationSimulator(1.0, 1.0, 1.0, dimension="Mbps")

    def test_two_instances_do_not_share_the_cursor(self):
        from src.communication import configure_ntt_trace

        configure_ntt_trace([10.0, 20.0, 30.0, 40.0])
        a, b = self._sim(), self._sim()
        # 旧实现：a=10, b=20, a=30（共享游标）；现在两者各自从 10 开始
        assert a._next_ntt_trace_value() == 10.0
        assert b._next_ntt_trace_value() == 10.0
        assert a._next_ntt_trace_value() == 20.0
        assert b._next_ntt_trace_value() == 20.0

    def test_cursor_wraps_and_applies_scale(self):
        from src.communication import configure_ntt_trace

        configure_ntt_trace([10.0, 20.0], scale=2.0)
        sim = self._sim()
        assert [sim._next_ntt_trace_value() for _ in range(3)] == [20.0, 40.0, 20.0]

    def test_no_trace_returns_none(self):
        self._reset()
        assert self._sim()._next_ntt_trace_value() is None
        assert self._sim()._next_ntt_trace_value() is None

    def test_reconfigure_then_build_starts_from_zero(self):
        """真实生命周期：configure 在前、建模拟器在后（Baselines.__init__）。

        所以"每次重新配置都从 trace 头开始"这一点仍然成立——靠的是新实例的游标
        初始为 0，而不是靠 configure 去重置别人的游标（旧实现是后者）。
        """
        from src.communication import configure_ntt_trace

        configure_ntt_trace([1.0, 2.0])
        assert self._sim()._next_ntt_trace_value() == 1.0
        configure_ntt_trace([7.0, 8.0])
        fresh = self._sim()
        assert fresh._next_ntt_trace_value() == 7.0
        assert fresh._next_ntt_trace_value() == 8.0


class TestDownlinkMergeIsByteNeutral:
    """B17：把"token 一次 + 位置索引一次"合并成一次传输，只改往返次数。

    旧写法（本仓库 10 处）把下行拆成两次 simulate/transfer 调用，而
    `_charge_transfer` 每调用一次就加一遍 NTT + connect_times，于是同一次
    WAN 往返被计费两次。合并后字节数应当逐位不变，只少一次 NTT。
    """

    @staticmethod
    def _sim():
        return CommunicationSimulator(
            1651.0,
            float("inf"),
            float("inf"),
            dimension="Mbps",
            ntt_ms_edge_cloud=200.0,
            use_stochastic=False,
        )

    def test_merge_saves_one_ntt_and_changes_no_bytes(self):
        t = torch.randint(0, 1000, (1, 1))
        token_bytes = t.element_size() * t.numel()

        legacy = self._sim()
        legacy.transfer(t, None, "edge_cloud")  # 旧：第一步
        legacy.simulate_transfer(INT_SIZE, "edge_cloud")  # 旧：第二步

        merged = self._sim()
        merged.simulate_transfer(INT_SIZE + token_bytes, "edge_cloud")  # 新：一次

        legacy_bytes = sum(u["data_size_bytes"] for u in legacy.stats["edge_cloud"])
        merged_bytes = sum(u["data_size_bytes"] for u in merged.stats["edge_cloud"])
        assert merged_bytes == legacy_bytes, "合并必须字节中立"

        # 恰好省下 ntt_ms_edge_cloud（200 ms）
        saved = legacy.edge_cloud_comm_time - merged.edge_cloud_comm_time
        assert saved == pytest.approx(0.2)
