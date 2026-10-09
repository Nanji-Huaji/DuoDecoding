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


class TestFluidBwModel:
    """载荷发射时长的流体模型（2026-10-09 统一口径 §3.4，bw_model="fluid"）。

    instant（历史口径）：整条载荷按起始时刻的瞬时 trace 采样计费；
    fluid：排空期间逐 trace 间隔积分——低带宽下长载荷横跨多个间隔时，
    instant 既失真（冻结在起始采样）又系统性多扣（E[S/B] > S/E[B]）。
    """

    BW_MBPS = [10.0, 30.0]  # 注入的确定性 trace（Mbps）
    INTERVAL = 0.2

    def _sim(self, bw_model: str) -> CommunicationSimulator:
        sim = CommunicationSimulator(
            46.0,
            float("inf"),
            float("inf"),
            dimension="Mbps",
            ntt_ms_edge_cloud=50.0,
            ntt_ms_edge_end=0.0,
            use_stochastic=True,
            min_bandwidth_mbps=5.0,
            bw_model=bw_model,
        )
        # 注入确定性 trace（跳过 data/ 依赖），并显式对齐采样间隔
        sim.trace_data = list(self.BW_MBPS)
        sim.trace_interval_s = self.INTERVAL
        sim.trace_index = 0
        return sim

    def test_small_payload_stays_in_first_interval(self):
        """0.125 MB @10 Mbps = 0.1s < 0.2s 间隔 ⇒ 单采样内完成，与 instant 同值。"""
        sim = self._sim("fluid")
        t = sim.simulate_transfer(0.125 * 1e6, "edge_cloud")
        # tx = 0.125MB / 1.25MB/s = 0.1s；NTT = 0.05s
        assert t == pytest.approx(0.1 + 0.05)
        assert sim.trace_index == 0  # 未越过当前采样

    def test_long_payload_integrates_across_intervals(self):
        """0.5 MB：间隔 0（10 Mbps）送 0.25MB 用满 0.2s，余 0.25MB 落到
        间隔 1（30 Mbps）⇒ tx = 0.2 + 0.25/3.75 ≈ 0.2667s，而非 instant 的
        0.5/1.25 = 0.4s（起始采样冻结的失真）。"""
        sim = self._sim("fluid")
        t = sim.simulate_transfer(0.5 * 1e6, "edge_cloud")
        assert t == pytest.approx(0.2 + 0.25 / 3.75 + 0.05)
        assert sim.trace_index == 1  # 排空落在采样 1；NTT 0.05s 不足一个间隔

    def test_instant_model_bit_compatible_with_history(self):
        """instant：整条载荷按起始采样计费，游标按 round(总时长/间隔) 跳。"""
        sim = self._sim("instant")
        t = sim.simulate_transfer(0.5 * 1e6, "edge_cloud")
        assert t == pytest.approx(0.5 / 1.25 + 0.05)
        # total = 0.45s ⇒ round(0.45/0.2) = 2 ⇒ max(1,2) = 2 ⇒ (0+2)%2 = 0
        assert sim.trace_index == 0

    def test_declining_trace_makes_fluid_slower_than_instant_start(self):
        """后采样更低时 fluid 反而更慢——积分是双向修正，不是单向优惠。"""
        sim = self._sim("fluid")
        sim.trace_data = [30.0, 10.0]
        t_fluid = sim.simulate_transfer(0.5 * 1e6, "edge_cloud")
        sim2 = self._sim("instant")
        sim2.trace_data = [30.0, 10.0]
        t_instant = sim2.simulate_transfer(0.5 * 1e6, "edge_cloud")
        # instant：0.5MB/3.75MB/s ≈ 0.133s（全冻结在 30 Mbps 采样）
        # fluid：0.2s 内送 0.75MB 上限 ⇒ 0.5MB < 0.75MB ⇒ 单间隔完成，同值
        assert t_fluid == pytest.approx(t_instant)
        # 再大一点跨过间隔边界：0.8MB ⇒ fluid = 0.2 + 0.05/1.25 = 0.24s
        # （连续时钟须一并归零——游标只是它的整数化投影）
        sim.trace_index = 0
        sim._trace_time_s = 0.0
        t_big = sim.simulate_transfer(0.8 * 1e6, "edge_cloud")
        assert t_big == pytest.approx(0.2 + 0.05 / 1.25 + 0.05)

    def test_zero_byte_payload_has_zero_tx(self):
        sim = self._sim("fluid")
        t = sim.simulate_transfer(0, "edge_cloud")
        assert t == pytest.approx(0.05)  # 只有 NTT

    def test_floor_applied_per_sample(self):
        """trace 采样低于 min_bandwidth_mbps 时逐采样钳位。"""
        sim = self._sim("fluid")
        sim.trace_data = [0.1, 30.0]  # 采样 0 被钳到 5 Mbps
        # 0.3 MB：间隔 0 只能送 5Mbps×0.2s = 0.125MB，余 0.175MB 落到采样 1
        t = sim.simulate_transfer(0.3 * 1e6, "edge_cloud")
        assert t == pytest.approx(0.2 + 0.175 / 3.75 + 0.05)

    def test_non_stochastic_falls_back_to_instant(self):
        """无 trace 时 fluid 与 instant 同值（恒定带宽下两者本就等价）。"""
        sim = CommunicationSimulator(
            46.0,
            float("inf"),
            float("inf"),
            dimension="Mbps",
            ntt_ms_edge_cloud=0.0,
            ntt_ms_edge_end=0.0,
            use_stochastic=False,
            bw_model="fluid",
        )
        t = sim.simulate_transfer(5.75 * 1e6, "edge_cloud")
        assert t == pytest.approx(1.0)

    def test_effective_rate_recorded_in_history(self):
        """stats 的带宽历史记有效排水速率 S/tx（ODLD 等估计器的口径）。"""
        sim = self._sim("fluid")
        sim.simulate_transfer(0.5 * 1e6, "edge_cloud")
        tx = sim.stats["edge_cloud"][-1]["tx_time"]
        assert tx == pytest.approx(0.2 + 0.25 / 3.75)
        eff_mbps = 0.5e6 / tx * 8 / 1e6
        assert sim.edge_cloud_bandwidth_history[-1] == pytest.approx(eff_mbps)

    def test_repeated_small_payloads_advance_clock(self):
        """冻结 bug 回归（2026-10-09 预实验实测）：小载荷虽在单个间隔内
        排空，连续时钟仍按 tx+NTT 推进，反复计费会跨过间隔边界——
        trace 不冻结在起始采样，有效速率随时间切换到后续采样。"""
        sim = self._sim("fluid")
        sim.trace_data = [30.0, 10.0]
        rates = []
        for _ in range(10):
            sim.simulate_transfer(0.1 * 1e6, "edge_cloud")  # 0.1MB
            rates.append(round(sim.edge_cloud_bandwidth_history[-1], 1))
        # 第一条按采样 0（30 Mbps）排空；时钟累计越过 0.2s 后后续消息
        # 会看到采样 1（10 Mbps）——出现至少两档有效速率即"未冻结"
        assert rates[0] == pytest.approx(30.0)
        assert len(set(rates)) >= 2, f"trace 冻结了: {rates}"

    def test_float_boundary_clock_does_not_stall(self):
        """浮点边界回归（2026-10-09 预实验实测）：时钟累加贴近间隔边界时
        t % interval 返回 ≈interval 而非 0，旧实现得到 frac≈1e-16 的零容量
        且采样索引不前进——排水原地空转直到 max_intervals 兜底按 1B/s
        计费（dsd 通信时间虚高百万秒）。贴边必须按下一间隔完整容量排水。"""
        sim = self._sim("fluid")
        sim.trace_data = [30.0, 30.0]
        # 4.999999999999999 % 0.2 = 0.19999...（≈interval）而非 0
        sim._trace_time_s = 4.999999999999999
        t = sim.simulate_transfer(192 * 1024, "edge_cloud")
        # 192KB @30Mbps(3.75MB/s) ≈ 0.051s + NTT 0.05s；绝不能是 1B/s 兜底
        assert t == pytest.approx(192 * 1024 / (3.75 * 1e6) + 0.05, rel=0.01)

    def test_long_run_small_payloads_no_explosion(self):
        """60 轮 192KB 上行（dsd 的真实载荷形状）：任何一轮的 tx 都必须
        在物理量级（<1s），总通信时间 <30s——冻结/兜底 bug 的端到端签名。"""
        sim = CommunicationSimulator(
            46.0,
            float("inf"),
            float("inf"),
            dimension="Mbps",
            ntt_ms_edge_cloud=50.0,
            ntt_ms_edge_end=0.317,
            use_stochastic=True,
            min_bandwidth_mbps=4.6,
            bw_model="fluid",
        )
        for _ in range(60):
            sim.simulate_transfer(192 * 1024, "edge_cloud")
            sim.simulate_transfer(0, "edge_cloud")
        assert max(u["tx_time"] for u in sim.stats["edge_cloud"]) < 1.0
        assert sim.edge_cloud_comm_time < 30.0


class TestCommTraceReplay:
    """逐消息记账记录（comm_trace）与离线重放的语义锁定。

    rebill 的前提：同一条 [字节, 轮号] 记录在任意网络配置下重放，
    结果 == 以该配置直接记账（A/B 预实验证明解码与网络无关）。
    """

    def _mk(self, **kw):
        return CommunicationSimulator(
            46.0, float("inf"), float("inf"), dimension="Mbps",
            ntt_ms_edge_cloud=50.0, ntt_ms_edge_end=0.317,
            use_stochastic=True, min_bandwidth_mbps=4.6,
            **kw,
        )

    def test_round_idx_stamped_both_modes(self):
        """per_transfer 也标注轮号（set_round 不再提前 return）。"""
        for coalesce in (True, False):
            sim = self._mk(bw_model="fluid")
            sim.coalesce_rounds = coalesce
            sim.set_round(7)
            sim.simulate_transfer(1024, "edge_cloud")
            sim.flush_round()
            u = sim.stats["edge_cloud"][-1]
            assert u["round_idx"] == 7, f"coalesce={coalesce}: {u['round_idx']}"

    def test_trace_replay_equals_direct(self):
        """记录在配置 X 采集、配置 Y 重放 == 直接以 Y 记账。"""
        import itertools

        seq = [(r, b) for r in range(1, 6) for b in (264 * 1024, 0)]
        for y in (
            dict(bw=5.0, floor=0.5, model="fluid", rt="per_round"),
            dict(bw=10.0, floor=1.0, model="instant", rt="per_transfer"),
        ):
            def mk():
                sim = CommunicationSimulator(
                    y["bw"], float("inf"), float("inf"), dimension="Mbps",
                    ntt_ms_edge_cloud=50.0, ntt_ms_edge_end=0.317,
                    use_stochastic=True, min_bandwidth_mbps=y["floor"],
                    bw_model=y["model"],
                )
                sim.coalesce_rounds = y["rt"] == "per_round"
                return sim

            direct, replay = mk(), mk()
            trace = []
            for rid, group in itertools.groupby(seq, key=lambda u: u[0]):
                for sim in (direct, replay):
                    sim.set_round(rid)
                for _, nbytes in group:
                    direct.simulate_transfer(nbytes, "edge_cloud")
                    trace.append([nbytes, rid])
                direct.flush_round()
            for rid, group in itertools.groupby(trace, key=lambda u: u[1]):
                replay.set_round(rid)
                for item in group:
                    replay.simulate_transfer(item[0], "edge_cloud")
                replay.flush_round()
            replay.flush_round()
            assert replay.edge_cloud_comm_time == pytest.approx(
                direct.edge_cloud_comm_time, abs=1e-9
            ), f"y={y}"
            assert replay.edge_cloud_data == direct.edge_cloud_data


class TestCommTraceReplaySamples:
    """多样本重放：样本边界 = 轮号回退，且模拟器随样本归零（连续时钟
    不跨样本延续）——不归零会让后续样本的排水相位系统性偏移。"""

    def test_two_samples_reset_clock(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from rebill import replay_comm

        # 两个样本：轮号各从 1 起（样本 1: 1..3, 样本 2: 1..2）
        one = [[200 * 1024, r] for r in (1, 2, 3)]
        two = [[200 * 1024, r] for r in (1, 2)]
        cfg = {"bandwidth": 5.0, "floor": 0.5, "bw_model": "fluid",
               "round_trip": "per_round", "ntt_ms": 50.0}
        # ① 拼接重放
        joined = replay_comm(one + two, cfg)
        # ② 两个样本独立重放（= 真实实验的逐样本调用形状）
        sep = replay_comm(one, cfg)
        sep2 = replay_comm(two, cfg)
        want = sep.edge_cloud_comm_time + sep2.edge_cloud_comm_time
        assert joined.edge_cloud_comm_time == pytest.approx(want, abs=1e-9), (
            "拼接重放必须按样本归零时钟："
            f"{joined.edge_cloud_comm_time:.6f} != {want:.6f}"
        )
        assert joined.edge_cloud_data == 5 * 200 * 1024
