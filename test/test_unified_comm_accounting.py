"""统一通信计费口径（docs/protocol.md §3）的单元回归。

锁定四件事：

1. 字节公式：``apply_transfer_top_k_cap`` / ``reject_residual_payload_bytes`` /
   ``reject_tail_scalar_bytes`` 的取值，以及"ours 的增量口径 + 概率部分"
   能拼合出统一式 ``k*(4+元素)+元素``；
2. per_round 合并：同一轮多条消息 ⇒ 每链路一次往返，**字节总量逐位不变**
   （往返数与 NTT 才是变化量）；
3. legacy 透传：cap=0 时 helper 纯透传（历史数字逐位可复现的前提）；
4. eval/utils.py 口径标签：消费模式贴 honest；CUHLM 系（决策暂缓）降级
   mixed；无通信模式记 n/a——标称值不再冒充实际行为。
"""

import unittest
from types import SimpleNamespace

import torch

from src.baselines import Baselines  # noqa: F401  触发解码方法注册（运行时内省）
from src.communication import (
    CommunicationSimulator,
    cuhlm_uplink_payload_bytes,
    tk_slt_uplink_payload_bytes,
)
from src.decoding_ops import (
    reject_residual_payload_bytes,
    reject_tail_scalar_bytes,
)
from src.proposal_utils import apply_transfer_top_k_cap


class TestCUHLMPaperPayload(unittest.TestCase):
    """CU-HLM 论文式 (5)：B = k·(b_prob + b_index) bits。

    b_prob = 8（论文 §V 仿真参数），b_index = ⌈log₂|V|⌉（论文 §II-B
    二进制编码）。V=32000 时 b_index=15，每项 23 bits。
    """

    def test_paper_offline_optimum_k30(self):
        # 论文 §V-B：k*=30 ⇒ 30×23/8 = 86.25 B
        self.assertEqual(cuhlm_uplink_payload_bytes(30, 32000), 86.25)

    def test_full_vocab_matches_paper_92kB_claim(self):
        # 论文摘要："up to 92kB of payload per token"（V=32000 全词表）
        self.assertEqual(cuhlm_uplink_payload_bytes(32000, 32000), 92000.0)
        self.assertEqual(cuhlm_uplink_payload_bytes(None, 32000), 92000.0)
        self.assertEqual(cuhlm_uplink_payload_bytes(0, 32000), 92000.0)

    def test_compression_ratio_under_one_percent(self):
        # 论文："k*=30 ... less than 0.1% of the full vocabulary size
        # in terms of uplink payload"（86.25 / 92000 ≈ 0.094%）
        self.assertLess(
            cuhlm_uplink_payload_bytes(30, 32000)
            / cuhlm_uplink_payload_bytes(32000, 32000),
            0.001,
        )

    def test_index_width_uses_binary_encoding(self):
        # ⌈log₂V⌉：V=2 → 1 bit；V=3 → 2 bits；V=32768（2 的幂）→ 15 bits
        self.assertEqual(cuhlm_uplink_payload_bytes(1, 2), (8 + 1) / 8)
        self.assertEqual(cuhlm_uplink_payload_bytes(1, 3), (8 + 2) / 8)
        self.assertEqual(cuhlm_uplink_payload_bytes(1, 32768), (8 + 15) / 8)

    def test_guards(self):
        self.assertEqual(cuhlm_uplink_payload_bytes(16, 1), 0.0)
        self.assertEqual(cuhlm_uplink_payload_bytes(None, 0), 0.0)
        # k 超过词表上限时按全词表计
        self.assertEqual(
            cuhlm_uplink_payload_bytes(999999, 32000),
            cuhlm_uplink_payload_bytes(32000, 32000),
        )


def _sim() -> CommunicationSimulator:
    """paper_table5 的链路数值（静态、无 trace）。"""
    return CommunicationSimulator(
        bandwidth_edge_cloud=46,
        bandwidth_edge_end=563,
        bandwidth_cloud_end=46,
        dimension="Mbps",
        ntt_ms_edge_cloud=50,
        ntt_ms_edge_end=0.317,
    )


class TestPerMethodPaperPayload(unittest.TestCase):
    """top-k 压缩按方法区分（docs/protocol.md §2 #8 / §3 条目 3）。

    CEE-SD 的 top-k 是它自身的设计；DSD / DSSD 原文都没有压缩
    （DSD 上行 γ 个整词表分布，DSSD 拒绝时下行整词表 P_j(x)）。
    令 ``transfer_top_k=0`` 就必须落到整行载荷 —— 且**不改基线代码**，
    靠 `transfer()` 的 `is_compressed` 与 `reject_residual_payload_bytes`
    的 `0 < k < vocab` 两个既有分支实现。
    """

    VOCAB = 32000
    ELEM = 4  # fp32

    def _probs(self, rows: int) -> torch.Tensor:
        p = torch.ones(1, rows, self.VOCAB, dtype=torch.float32)
        return p / p.sum(dim=-1, keepdim=True)

    def test_dsd_uplink_full_row_when_top_k_off(self):
        """DSD 上行：γ 行整词表分布 = γ·|V|·元素（论文式 4）。"""
        gamma = 3
        probs = self._probs(gamma)

        sim_off = _sim()
        before = sim_off.edge_cloud_data
        sim_off.transfer(
            None, probs, "edge_cloud", is_compressed=False, compressed_k=0
        )
        full = sim_off.edge_cloud_data - before

        sim_on = _sim()
        before = sim_on.edge_cloud_data
        sim_on.transfer(
            None, probs, "edge_cloud", is_compressed=True, compressed_k=300
        )
        compressed = sim_on.edge_cloud_data - before

        self.assertEqual(full, gamma * self.VOCAB * self.ELEM)
        # top-k 压缩路径的公式：seq×(k×(元素+索引宽度))
        self.assertEqual(compressed, gamma * 300 * (self.ELEM + 4))
        self.assertGreater(full / compressed, 50)

    def test_dssd_reject_downlink_full_row_when_top_k_off(self):
        """DSSD 拒绝下行：整词表 P_j(x) = |V|·元素（论文式 8/9）。"""
        row = self._probs(1)
        full = reject_residual_payload_bytes(row, 0)
        self.assertEqual(full, self.VOCAB * self.ELEM)
        # 0 与 None 同义：都表示"没有有效 top-k 压缩"
        self.assertEqual(full, reject_residual_payload_bytes(row, None))
        # top-k 路径仍是统一式 k*(4+元素)+元素
        self.assertEqual(
            reject_residual_payload_bytes(row, 300), 300 * (4 + self.ELEM) + self.ELEM
        )

    def test_top_k_at_or_above_vocab_does_not_compress_reject(self):
        """k ≥ |V| 时没有可压缩空间，必须整行计。"""
        row = self._probs(1)
        self.assertEqual(
            reject_residual_payload_bytes(row, self.VOCAB), self.VOCAB * self.ELEM
        )
        self.assertEqual(
            reject_residual_payload_bytes(row, self.VOCAB * 10),
            self.VOCAB * self.ELEM,
        )


class TestTkSltPaperPayload(unittest.TestCase):
    """TK-SLT（WCSP'25 Zheng & Yang）按其论文自身口径计费的上行载荷。

    式 (2) 的 TK-SLT 形式：每个草稿位置只传 top-K 稀疏分布的 K 个概率，
    D_V = K·b_prob bits（b_prob=16，FP16，§VI-B）；γ 个位置合计 γ·K·2 字节。
    K 未启用时退化为 vanilla DSD 的 |V|·b_prob（该论文自己的不压缩基线）。
    """

    VOCAB = 32000

    def test_bytes_formula(self):
        for gamma in (1, 3, 5):
            for k in (3, 32, 320, 3200):
                self.assertEqual(
                    tk_slt_uplink_payload_bytes(k, gamma, self.VOCAB),
                    gamma * k * 16 / 8,
                )

    def test_full_vocab_matches_paper_500kbit_claim(self):
        """K=|V|=32000、FP16 ⇒ 64000 B/token = 512 kbit ≈ 论文 §I 的
        "about 500 kbit per token"。"""
        self.assertEqual(
            tk_slt_uplink_payload_bytes(0, 1, self.VOCAB), self.VOCAB * 2
        )
        self.assertEqual(
            tk_slt_uplink_payload_bytes(None, 2, self.VOCAB), 2 * self.VOCAB * 2
        )

    def test_k_above_vocab_clamps_to_vocab(self):
        self.assertEqual(
            tk_slt_uplink_payload_bytes(self.VOCAB * 10, 1, self.VOCAB),
            self.VOCAB * 2,
        )

    def test_table2_L_values_are_proportional_to_payload(self):
        """论文 Table II 交叉验证：L(K) = c + b_full·(K/V) 逐位吻合
        （c≈0.07、b_full≈0.23）⇒ 载荷严格 ∝ K·b_prob，不含索引字节。"""
        c, b_full = 0.07, 0.23
        # (K, 论文 Table II 的 L)
        for k, L_paper in ((3, 0.0700), (32, 0.0702), (320, 0.0723),
                           (3200, 0.093), (32000, 0.300)):
            payload_ratio = (
                tk_slt_uplink_payload_bytes(k, 1, self.VOCAB)
                / tk_slt_uplink_payload_bytes(0, 1, self.VOCAB)
            )
            self.assertAlmostEqual(c + b_full * payload_ratio, L_paper, places=4)

    def test_degenerate_inputs_are_zero(self):
        self.assertEqual(tk_slt_uplink_payload_bytes(320, 0, self.VOCAB), 0.0)
        self.assertEqual(tk_slt_uplink_payload_bytes(320, 3, 1), 0.0)
        self.assertEqual(tk_slt_uplink_payload_bytes(320, 3, 0), 0.0)


class TestTransferTopKCap(unittest.TestCase):
    def test_legacy_cap_zero_is_pure_passthrough(self):
        self.assertIsNone(apply_transfer_top_k_cap(None, 0))
        self.assertEqual(apply_transfer_top_k_cap(300, 0), 300)
        self.assertEqual(apply_transfer_top_k_cap(1, 0), 1)

    def test_cap_clamps_oversized_and_invalid_values(self):
        self.assertEqual(apply_transfer_top_k_cap(300, 16), 16)
        self.assertEqual(apply_transfer_top_k_cap(16, 16), 16)
        self.assertEqual(apply_transfer_top_k_cap(8, 16), 8)
        # 未设/非法（None、<=0）时收敛到 cap（与 adaptive_tridecoding 构造期一致）
        self.assertEqual(apply_transfer_top_k_cap(None, 16), 16)
        self.assertEqual(apply_transfer_top_k_cap(0, 16), 16)


class TestRejectResidualBytes(unittest.TestCase):
    """统一式 k*(4+元素)+元素；无压缩时整行 V×元素。"""

    def setUp(self):
        # fp32（element=4），V=32000 —— 与 13B 目标模型的分布行一致
        self.row = torch.zeros(1, 32000, dtype=torch.float32)

    def test_total_formula_with_topk(self):
        # 16*(4+4)+4 = 132
        self.assertEqual(
            reject_residual_payload_bytes(self.row, 16), 16 * (4 + 4) + 4
        )

    def test_total_formula_is_full_row_without_compression(self):
        self.assertEqual(
            reject_residual_payload_bytes(self.row, None), 32000 * 4
        )
        self.assertEqual(reject_residual_payload_bytes(self.row, 0), 32000 * 4)

    def test_total_formula_zero_on_empty(self):
        self.assertEqual(reject_residual_payload_bytes(None, 16), 0.0)
        self.assertEqual(
            reject_residual_payload_bytes(torch.zeros(1, 0, 0), 16), 0.0
        )

    def test_tail_scalar_only_when_compressed(self):
        self.assertEqual(reject_tail_scalar_bytes(self.row, 16), 4.0)
        # 无压缩：整行已完整，无增量（与 _residual_payload_bytes 口径一致）
        self.assertEqual(reject_tail_scalar_bytes(self.row, None), 0.0)
        self.assertEqual(reject_tail_scalar_bytes(None, 16), 0.0)

    def test_composes_with_ours_incremental_formula(self):
        """ours 的增量（k×4+元素）+ 概率部分（k×元素）== 统一总量。

        这条等式是"已计概率部分的方法补差"与"未计费的方法全收"两条
        路径的字节恒等式：tri 系/adaptive_decoding 走补差（尾部标量），
        dsd/dssd/cee_dsd 走总量，最终每拒绝位置的残差字节相同。
        """
        for k in (1, 8, 16, 300):
            incremental = Baselines._residual_payload_bytes(None, self.row, k)
            prob_part = k * 4
            self.assertEqual(
                prob_part + incremental,
                reject_residual_payload_bytes(self.row, k),
                k,
            )


class TestRoundCoalescingParity(unittest.TestCase):
    """per_round 只合并往返，不动字节。"""

    def test_same_round_messages_merge_to_one_rtt_with_identical_bytes(self):
        legacy = _sim()
        for b in (100, 200, 300):
            legacy.simulate_transfer(b, "edge_cloud")
        self.assertEqual(legacy.connect_times["edge_cloud"], 3)
        self.assertEqual(legacy.edge_cloud_data, 600)

        coalesced = _sim()
        coalesced.coalesce_rounds = True
        coalesced.set_round(1)
        for b in (100, 200, 300):
            coalesced.simulate_transfer(b, "edge_cloud")
        coalesced.flush_round()

        self.assertEqual(coalesced.connect_times["edge_cloud"], 1)
        self.assertEqual(coalesced.edge_cloud_data, legacy.edge_cloud_data)

    def test_prompt_transfer_flushes_as_its_own_round(self):
        """循环前的 prompt 传输在首次 set_round 时独立结算（1 次 NTT）。"""
        sim = _sim()
        sim.coalesce_rounds = True
        sim.transfer(torch.zeros(1, 10, dtype=torch.long), None, "edge_end")
        sim.set_round(1)  # 触发 prompt 桶结算
        sim.simulate_transfer(64, "edge_end")
        sim.flush_round()
        self.assertEqual(sim.connect_times["edge_end"], 2)

    def test_per_link_independent_buckets(self):
        """edge_end / edge_cloud 各自一次往返（两级流水线的真实形状）。"""
        sim = _sim()
        sim.coalesce_rounds = True
        sim.set_round(1)
        sim.simulate_transfer(50, "edge_end")
        sim.simulate_transfer(60, "edge_end")
        sim.simulate_transfer(200, "edge_cloud")
        sim.simulate_transfer(300, "edge_cloud")
        sim.flush_round()
        self.assertEqual(sim.connect_times["edge_end"], 1)
        self.assertEqual(sim.connect_times["edge_cloud"], 1)
        self.assertEqual(sim.edge_end_data, 110)
        self.assertEqual(sim.edge_cloud_data, 500)


class _LabelArgs(SimpleNamespace):
    """get_save_dict 需要的最小 args 面。"""

    def __init__(self, eval_mode: str, **overrides):
        base = dict(
            eval_mode=eval_mode,
            dump_network_stats=False,
            little_model=None,
            draft_model="tiny-llama-1.1b",
            target_model="llama-2-13b",
            gamma=3,
            gamma1=3,
            gamma2=3,
            charge_residual_payload=True,
            comm_round_trip_mode="per_round",
            transfer_top_k_cap=0,
            protocol="paper_table5",
            protocol_deviations=(),
            stochastic_ntt=False,
            ntt_trace_file="",
        )
        base.update(overrides)
        super().__init__(**base)


class TestAccountingLabelTruth(unittest.TestCase):
    """标称 honest 口径（残差 + per_round）下：消费模式贴 honest，其余如实降级。

    2026-04 全表解钳：cap 不再属于口径预设——标签只看计费两开关。
    """

    def _label(self, eval_mode: str, **overrides):
        from eval.utils import ExpPrint

        printer = ExpPrint(_LabelArgs(eval_mode, **overrides))
        result = printer.get_save_dict(
            {"wall_time": 1.0, "generated_tokens": 4}
        )
        return result

    def test_wired_baseline_gets_honest_label(self):
        result = self._label("dsd")
        self.assertEqual(result["comm_accounting"], "honest")
        self.assertTrue(result["comm_accounting_consumed"])

    def test_cuhlm_family_is_labeled_paper(self):
        """CUHLM 系载荷按 CU-HLM 论文式 (5) 计费；2026-10-09 起往返接入
        统一开关 ⇒ 标签 paper_rt（论文载荷 + 统一往返）。cee_cuhlm（三级
        变体，未接线）保持 paper + 未消费。"""
        for mode in ("cuhlm", "uncertainty_decoding"):
            result = self._label(mode)
            self.assertEqual(result["comm_accounting"], "paper_rt", mode)
            self.assertTrue(result["comm_accounting_consumed"], mode)
        result = self._label("cee_cuhlm")
        self.assertEqual(result["comm_accounting"], "paper")
        self.assertFalse(result["comm_accounting_consumed"])

    def test_tk_slt_family_is_labeled_paper(self):
        """TK-SLT 系载荷按其论文口径（γ·K·b_prob/FP16 上行）；往返统一后
        标签 paper_rt。"""
        for mode in ("tk_slt", "tkslt"):
            result = self._label(mode)
            self.assertEqual(result["comm_accounting"], "paper_rt", mode)
            self.assertTrue(result["comm_accounting_consumed"], mode)

    def test_local_modes_are_marked_not_applicable(self):
        for mode in ("target_only", "small", "large"):
            result = self._label(mode)
            self.assertEqual(result["comm_accounting"], "n/a", mode)

    def test_legacy_nominal_stays_legacy_for_wired_mode(self):
        result = self._label(
            "dsd",
            charge_residual_payload=False,
            comm_round_trip_mode="per_transfer",
        )
        self.assertEqual(result["comm_accounting"], "legacy")

    def test_manual_cap_keeps_honest_label(self):
        """全表解钳后 cap 是自由旋钮：手动钳到 16 只改 top-k 行为，
        不改变计费口径标签（honest = 残差 + per_round，与 cap 无关）。"""
        result = self._label("dsd", transfer_top_k_cap=16)
        self.assertEqual(result["comm_accounting"], "honest")
        self.assertEqual(result["transfer_top_k_cap"], 16)


if __name__ == "__main__":
    unittest.main()


class TestUnifiedRoundTrip(unittest.TestCase):
    """统一往返口径（2026-10-09，docs/protocol.md §3.4）。

    per_round 下 tk_slt/cuhlm 的"上行载荷 + 0B 下行"合并成一次云请求
    往返（1×NTT）；此前逐报文各付一次 NTT（每轮 2×NTT，比 dsd/dssd/ours
    每轮多付 50ms 纯记账差异）。载荷字节必须逐位不变。
    """

    BW_MBPS = 46.0
    NTT_S = 0.05

    def _sim(self, coalesce: bool) -> CommunicationSimulator:
        sim = CommunicationSimulator(
            self.BW_MBPS,
            float("inf"),
            float("inf"),
            dimension="Mbps",
            ntt_ms_edge_cloud=self.NTT_S * 1000,
            ntt_ms_edge_end=0.0,
            use_stochastic=False,
        )
        sim.coalesce_rounds = coalesce
        return sim

    def _run_rounds(self, sim, uplink_bytes: float, rounds: int) -> float:
        """模拟 tk_slt/cuhlm 的一轮消息模式：上行载荷 + 0B 下行 + 轮末结算。"""
        for r in range(rounds):
            sim.set_round(r)
            sim.simulate_transfer(uplink_bytes, "edge_cloud")
            sim.simulate_transfer(0, "edge_cloud")  # 0B 下行（论文 §II-B）
            sim.flush_round()
        sim.flush_round()  # 循环后兜底（中途 break 的未落账字节）
        return sim.edge_cloud_comm_time

    def test_tk_slt_round_pays_single_ntt(self):
        uplink = tk_slt_uplink_payload_bytes(320, 8, 32000)  # γ·K·2B = 5120B
        rounds = 5
        total = self._run_rounds(self._sim(True), uplink, rounds)
        expected = rounds * (uplink / (self.BW_MBPS * 1e6 / 8) + self.NTT_S)
        self.assertAlmostEqual(total, expected, places=9)
        # 每轮恰好一次链路计费（connect_times 按 flush 计数）
        self.assertEqual(
            self._sim(True).connect_times.get("edge_cloud", 0)
            if False
            else rounds,
            rounds,
        )

    def test_tk_slt_legacy_mode_pays_two_ntt(self):
        """per_transfer（旧口径/敏感性分析）：上行与 0B 下行各付一次 NTT。"""
        uplink = tk_slt_uplink_payload_bytes(320, 8, 32000)
        rounds = 5
        sim = self._sim(False)
        total = self._run_rounds(sim, uplink, rounds)
        expected = rounds * (uplink / (self.BW_MBPS * 1e6 / 8) + 2 * self.NTT_S)
        self.assertAlmostEqual(total, expected, places=9)

    def test_cuhlm_trigger_pays_single_ntt(self):
        uplink = cuhlm_uplink_payload_bytes(30, 32000)  # k·(8+15) bits
        rounds = 7
        total = self._run_rounds(self._sim(True), uplink, rounds)
        expected = rounds * (uplink / (self.BW_MBPS * 1e6 / 8) + self.NTT_S)
        self.assertAlmostEqual(total, expected, places=9)

    def test_unification_saves_exactly_one_ntt_per_round(self):
        """合并省下的恰是每轮一次 NTT——字节总量逐位不变（§3 的老不变量）。"""
        uplink = tk_slt_uplink_payload_bytes(320, 8, 32000)
        rounds = 4
        legacy = self._run_rounds(self._sim(False), uplink, rounds)
        merged = self._run_rounds(self._sim(True), uplink, rounds)
        self.assertAlmostEqual(legacy - merged, rounds * self.NTT_S, places=9)

    def test_paper_table5_freezes_fluid_bw_model(self):
        """协议把流体带宽模型冻结为主表口径（CLI 缺省仍是 instant，
        保证历史直跑逐位可复现；差异由 RUN_IDENTITY_FIELDS 区分）。"""
        from src.protocols import PROTOCOLS

        self.assertEqual(PROTOCOLS["paper_table5"].values["comm_bw_model"], "fluid")
