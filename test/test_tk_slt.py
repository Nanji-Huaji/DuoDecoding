"""TK-SLT（Top-K Sparse Logits Transmission）基线的回归测试。

论文 "Communication-Efficient Collaborative LLM Inference via Distributed
Speculative Decoding"（WCSP'25, Zheng & Yang）。锁定三件事：

1. **纯数学**：ODLD/AS²（Theorem 2 的 Lambert-W 闭式 γ*）必须逐格复现
   论文 Table I；S_inf 复现式 (11)；W_{−1} 二分实现解 w·e^w = z。
2. **通信口径**（论文自身，非仓库统一开关）：上行 = γ·K·b_prob bits
   （b_prob=16，FP16，且概率值真实量化——验证判据与计费看到同一个 q̂）；
   草稿 token 索引与下行响应 negligible ⇒ 0 字节报文。字节公式的数值
   断言在 test_unified_comm_accounting.py，这里锁解码循环的实际调用序列。
3. **解码语义**（概率-wire 变体）：草稿分布先显式选 min(K,V) 个坐标、
   按保留质量重归一化、写成真实 FP16 payload；草稿**从该 payload 重建的
   dense-V FP16 行采样**，接收端只从同一 payload 重建并走 strict
   ProposalWindow / prepare_verification_window / decide_verification。
   采样所用的律与验证所用的律逐位相同（旧实现是 raw Q 采样 + FP16 q̂
   验证，TVD 9.76502895e-5）；被采样候选的 wire 支撑必为正。
"""

import math
import unittest
from argparse import Namespace
from fractions import Fraction
from unittest.mock import patch

import torch

import src.verification as verification_module
from src.baselines import Baselines
from src.baselines_helpers import (
    _last_edge_cloud_tx_seconds,
    _quantize_probs_fp16,
)
from src.communication import CommunicationSimulator, tk_slt_uplink_payload_bytes
from src.sampling import sampling_weights
from src.tk_slt_policy import (
    _lambert_w_minus_one,
    tk_slt_optimal_draft_length,
    tk_slt_select_speculative,
    tk_slt_speedup_ratio,
)

VOCAB = 4
K = 2  # top-K 稀疏 logits 的 K（< VOCAB 才有压缩意义）


class _TestBaselines(Baselines):
    def load_data(self):
        return None

    def preprocess(self, input_text):
        return input_text

    def postprocess(self, input_text, output_text):
        return output_text

    def eval(self):
        return None


class _FakeCudaEvent:
    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing

    def record(self, stream=None):
        return None

    def elapsed_time(self, other):
        return 0.0


class _FakeCommSimulator:
    """记录 transfer/simulate_transfer 调用序列的最小通信模拟器。

    可选 stats 列表模拟真实链路模型的 TransferUnit.tx_time（ODLD 的 b̂
    估计用）；不提供时 _last_edge_cloud_tx_seconds 应回退 None。
    镜像统一往返口径（2026-10-09）用到的接口：coalesce_rounds 默认 False
    （per_transfer 直充），flush_round/set_round 为 no-op。
    """

    instances = []

    # 统一往返口径：默认 per_transfer（直充）；测试 per_round 时改 True
    coalesce_rounds = False

    def flush_round(self):
        """per_round 合并模式的轮末结算；fake 不做字节合并。"""

    def set_round(self, round_idx):
        """per_round 合并模式的轮界标记；fake 不结算。"""

    def __init__(self, *args, **kwargs):
        self.edge_cloud_comm_time = 0.0
        self.edge_end_comm_time = 0.0
        self.edge_cloud_data = 0
        self.edge_end_data = 0
        self.cloud_end_data = 0
        self.total_comm_energy = 0.0
        self.connect_times = {}
        self.edge_cloud_bandwidth_history = []
        self.edge_cloud_topk_history = []
        self.edge_cloud_draft_len_history = []
        self.transfer_calls = []
        self.stats = {"edge_cloud": [], "edge_end": [], "cloud_end": []}
        self.transfer_top_k = None
        self.tx_time_per_uplink = 0.0
        _FakeCommSimulator.instances.append(self)

    def transfer(self, tokens, probs, link_type="edge_cloud", **kwargs):
        self.transfer_calls.append(
            {
                "tokens": None if tokens is None else tokens.clone(),
                "probs": None if probs is None else probs.clone(),
                "link_type": link_type,
                "kwargs": dict(kwargs),
            }
        )
        return 0.0

    def simulate_transfer(self, data_size_bytes, link_type="edge_cloud", **kwargs):
        self.transfer_calls.append(
            {
                "tokens": None,
                "probs": None,
                "link_type": link_type,
                "kwargs": dict(kwargs),
                "data_size_bytes": int(data_size_bytes),
            }
        )
        if link_type == "edge_cloud" and kwargs.get("draft_len", 0) > 0:
            # 只给"分布上行"这条消息配 tx_time（论文 T_V 的口径）——
            # ODLD 集成测试可用可控值喂 b̂。
            self.stats["edge_cloud"].append(
                {
                    "data_size_bytes": int(data_size_bytes),
                    "tx_time": self.tx_time_per_uplink,
                }
            )
        return 0.0


class _FakeDraftCache:
    """端侧 SLM 缓存：每个草稿步返回脚本化的模型概率行。

    解码循环不再调用 ``generate``：草稿 token 由循环从 FP16 wire 行采样。
    默认行是 token 循环上的 one-hot（采样确定），需要非 one-hot 律的测试
    通过 router 的 ``draft_rows`` 覆盖。``forward_context_calls`` 记录每步
    输入长度，``forward_context_contents`` 记录被推进的真实前缀内容。
    """

    instances = []

    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        self.init_args = dict(temperature=temperature, top_k=top_k, top_p=top_p)
        self.model = model
        self.device = model.device
        self.vocab_size = VOCAB
        self.rollback_calls = []
        self.forward_context_calls = []
        self.forward_context_contents = []
        self.rows = [self._one_hot(token) for token in (0, 1, 2, 3)]
        _FakeDraftCache.instances.append(self)

    @staticmethod
    def _one_hot(token: int) -> torch.Tensor:
        row = torch.zeros(1, VOCAB)
        row[0, token % VOCAB] = 1.0
        return row

    def forward_context(self, prefix):
        self.forward_context_calls.append(int(prefix.shape[1]))
        self.forward_context_contents.append(prefix.clone())
        index = min(len(self.forward_context_calls) - 1, len(self.rows) - 1)
        return self.rows[index].clone()

    def generate(self, *args, **kwargs):
        raise AssertionError(
            "TK-SLT draft must sample the reconstructed wire row via forward_context"
        )

    def rollback(self, end_pos):
        self.rollback_calls.append(end_pos)

    def reset_for_new_sample(self):
        pass


class _FakeTargetCache:
    """BS 侧 LLM 缓存：pos+1 在 x 内时把 1.0 放在该 token 上（接受友好），

    最后一行（x 之后的位置）用固定分布供 bonus token 采样。
    extra_rows 可覆盖指定位置的行（构造必拒/非法行场景）。
    """

    instances = []

    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        self.init_args = dict(temperature=temperature, top_k=top_k, top_p=top_p)
        self.model = model
        self.device = model.device
        self.vocab_size = VOCAB
        self.rollback_calls = []
        self.generate_calls = []
        self.forward_context_calls = []
        self.prob_history = None
        self.logits_history = None
        self.extra_rows = {}
        _FakeTargetCache.instances.append(self)

    def forward_context(self, x):
        """第 pos 行 = P(·|x[:pos+1])：预测位置 pos+1 的 token。

        pos+1 落在 x 内时把 1.0 放在该 token 上（接受判据 p/q 恰好抵消已接受
        的 wire 质量），否则用固定 bonus 分布（x 之后的位置）。本 fake 与真实
        ``KVCacheModel.forward_context`` 一样不采样。
        """
        self.forward_context_calls.append(x.shape[1])
        rows = []
        for pos in range(x.shape[1]):
            if pos in self.extra_rows:
                rows.append(list(self.extra_rows[pos]))
                continue
            row = [0.0] * VOCAB
            if pos + 1 < x.shape[1]:
                row[int(x[0, pos + 1]) % VOCAB] = 1.0
            else:
                row = [0.4, 0.3, 0.2, 0.1]
            rows.append(row)
        self.prob_history = torch.tensor([rows])
        return self.prob_history[:, -1, :]

    def generate(self, x, gamma):
        """Legacy seam kept observable: production must not call it."""
        self.generate_calls.append((x.shape[1], gamma))
        self.forward_context(x)
        return x

    def rollback(self, end_pos):
        self.rollback_calls.append(end_pos)

    def reset_for_new_sample(self):
        pass


class _FakeCacheRouter:
    """按 model.kind 分发到 draft/target 假缓存（替代 KVCacheModel 构造）。

    draft_rows：草稿缓存每步返回的模型概率行（默认 one-hot token 循环）。
    target_extra_rows：注入到 target 缓存的指定位置概率行。
    """

    def __init__(self, draft_rows=None, target_extra_rows=None):
        self.draft_rows = None if draft_rows is None else list(draft_rows)
        self.target_extra_rows = dict(target_extra_rows or {})

    def __call__(self, model, temperature, top_k, top_p, **kwargs):
        if getattr(model, "kind", None) == "draft":
            cache = _FakeDraftCache(model, temperature, top_k, top_p, **kwargs)
            if self.draft_rows is not None:
                cache.rows = list(self.draft_rows)
            return cache
        cache = _FakeTargetCache(model, temperature, top_k, top_p, **kwargs)
        cache.extra_rows = dict(self.target_extra_rows)
        return cache


class TkSltLoopTestBase(unittest.TestCase):
    """解码循环测试的公共脚手架（fakes + patches，仿 test_uncertainty_decoding）。"""

    def setUp(self):
        _FakeDraftCache.instances = []
        _FakeTargetCache.instances = []
        _FakeCommSimulator.instances = []
        torch.manual_seed(1234)
        args = Namespace(
            eval_mode="tk_slt",
            edge_cloud_bandwidth=46,
            max_tokens=2,
            temp=1.0,
            top_k=0,
            top_p=0.0,
            batch_delay=0.0,
            seed=0,
            exp_name="test",
            eval_dataset="unit",
            little_model="little",
            draft_model="draft",
            target_model="target",
            gamma=2,
            gamma1=1,
            gamma2=1,
            use_early_stopping=False,
            dump_network_stats=False,
            protocol_deviations=(),
        )
        self.args = args
        self.instance = object.__new__(_TestBaselines)
        self.instance.args = args
        self.instance.accelerator = type("Accel", (), {"is_main_process": True})()
        self.instance.vocab_size = VOCAB
        self.instance.num_acc_tokens = []
        self.instance.draft_forward_times = 0
        self.instance.target_forward_times = 0
        self.instance.draft_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "draft"}
        )()
        self.instance.target_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "target"}
        )()

    def _patched_loop(self, router):
        return (
            patch("src.baselines.tk_slt.KVCacheModel", router),
            patch(
                "src.baselines.simulators.CommunicationSimulator",
                _FakeCommSimulator,
            ),
            patch(
                "src.baselines.simulators.PreciseCommunicationSimulator",
                _FakeCommSimulator,
            ),
            patch("src.baselines.tk_slt.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.tk_slt.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.tk_slt.torch.cuda.synchronize", return_value=None),
        )

    def _run(self, prefix, transfer_top_k=K, **overrides):
        return self._run_router(_FakeCacheRouter(), prefix, transfer_top_k, **overrides)

    def _run_router(self, router, prefix, transfer_top_k=K, **overrides):
        self.args.__dict__.update(overrides)
        patches = self._patched_loop(router)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            return self.instance.tk_slt(prefix, transfer_top_k=transfer_top_k)

    # -- recorders: the loop's declared seams, observed without altering them --

    def _record_codec(self):
        payloads = []
        module = __import__("src.tk_slt_proposal", fromlist=["encode_proposal_row"])
        real = module.encode_proposal_row

        def wrapper(*args, **kwargs):
            payload = real(*args, **kwargs)
            payloads.append(payload)
            return payload

        return payloads, patch("src.tk_slt_proposal.encode_proposal_row", wrapper)

    def _record_windows(self):
        windows = []
        real = verification_module.prepare_verification_window

        def wrapper(proposal, target_history):
            window = real(proposal, target_history)
            windows.append(window)
            return window

        patcher = patch(
            "src.baselines.tk_slt.prepare_verification_window", wrapper
        )
        return windows, patcher

    def _record_multinomial(self):
        samples = []
        original = torch.multinomial

        def wrapper(probabilities, *args, **kwargs):
            samples.append(probabilities.detach().clone())
            return original(probabilities, *args, **kwargs)

        return samples, patch("torch.multinomial", wrapper)

    def _uplinks(self):
        """分布上行：本方法的上行报文总是带 draft_len=γ（topk 可为 0）。"""
        sim = _FakeCommSimulator.instances[0]
        return [
            c
            for c in sim.transfer_calls
            if "data_size_bytes" in c and c["kwargs"].get("draft_len", 0) > 0
        ]

    def _downlinks(self):
        sim = _FakeCommSimulator.instances[0]
        return [
            c
            for c in sim.transfer_calls
            if "data_size_bytes" in c
            and c["data_size_bytes"] == 0
            and c["kwargs"].get("draft_len", 0) == 0
        ]


class TestTkSltDecodeLoop(TkSltLoopTestBase):
    def test_cloned_role_defaults_cannot_correlate_proposal_and_acceptance(self):
        """A valid categorical dependency with cloned default role tapes.

        This CPU toy uses inverse CDF, not CUDA's exponential race. Default
        draft/target streams restart the same tape; an explicit CPU generator
        instead shares one advancing tape across all draws in this invocation.
        Q=[1/2,1/2], P=[1/4,3/4] must yield token0 in 1/4 of the grid, not 1/2.
        """
        emitted_zero = 0
        grid = 8
        for i in range(grid):
            for j in range(grid):
                tape = ((i + 0.5) / grid, (j + 0.5) / grid, 0.125)
                positions = {}
                sample_count = 0

                def next_uniform(role, generator, *, positions=positions, tape=tape):
                    key = role if generator is None else generator
                    offset = positions.get(key, 0)
                    positions[key] = offset + 1
                    return tape[min(offset, len(tape) - 1)]

                def categorical(weights, num_samples=1, *, generator=None):
                    nonlocal sample_count
                    role = "draft" if sample_count == 0 else "target"
                    sample_count += 1
                    value = next_uniform(role, generator)
                    threshold = torch.tensor([[value]], dtype=weights.dtype)
                    return torch.searchsorted(weights.cumsum(-1), threshold, right=True)

                def uniform(shape, *, device=None, generator=None, **kwargs):
                    value = next_uniform("target", generator)
                    return torch.full(shape, value, device=device, **kwargs)

                raw = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
                router = _FakeCacheRouter(
                    draft_rows=[raw],
                    target_extra_rows={
                        0: [0.25, 0.75, 0.0, 0.0],
                        1: [0.25, 0.75, 0.0, 0.0],
                    },
                )
                with (
                    patch("torch.multinomial", categorical),
                    patch("torch.rand", uniform),
                ):
                    output, _ = self._run_router(
                        router, torch.tensor([[3]]), max_tokens=2, gamma=1
                    )
                emitted_zero += int(output[0, 1] == 0)

        self.assertEqual(emitted_zero, 16)  # independent 8x8 grid: 1/4, not 32/64

    def test_draft_sampler_law_is_the_transmitted_wire_law(self):
        """The checked counterexample: Q=[.6,.4], P=wire(Q).

        The old loop sampled raw Q and verified P=wire(Q): every ratio was
        exactly 1, so the first token kept the raw Q law while P differs by TVD
        9.76502895e-5.  The active loop must draw from the reconstructed FP16
        wire row, which is also the receiver's reference law.
        """
        prefix = torch.tensor([[3]], dtype=torch.long)
        raw_row = torch.tensor([[0.6, 0.4, 0.0, 0.0]], dtype=torch.float32)
        wire_row = torch.tensor([[0.60009765625, 0.39990234375, 0.0, 0.0]])
        router = _FakeCacheRouter(
            draft_rows=[raw_row],
            target_extra_rows={0: wire_row[0].tolist(), 1: wire_row[0].tolist()},
        )
        payloads, codec_patch = self._record_codec()
        windows, window_patch = self._record_windows()
        samples, sample_patch = self._record_multinomial()

        with codec_patch, window_patch, sample_patch:
            output, metrics = self._run_router(
                router, prefix, max_tokens=2, gamma=1
            )

        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertEqual(tuple(payload.values.shape), (1, K))
        self.assertEqual(payload.indices.tolist(), [[0, 1]])
        self.assertEqual(payload.values.tolist(), wire_row[:, :K].tolist())
        self.assertEqual(payload.dense().tolist(), wire_row.tolist())
        self.assertEqual(payload.nbytes, K * 2)

        # the draft draw used exactly the payload reconstruction
        self.assertTrue(torch.equal(samples[0], sampling_weights(wire_row)))
        # the receiver's reference law is that same reconstruction, bit for bit
        proposal = windows[0].proposal
        self.assertTrue(torch.equal(proposal.probabilities[0, 0], payload.dense()[0]))
        self.assertTrue(torch.equal(proposal.sampling_weights[0, 0], samples[0][0]))
        # target P is the wire row, so acceptance is exactly 1 and the bonus law
        # is the wire law: the first-token law is P, not raw Q.
        self.assertTrue(torch.equal(windows[0].selected_q, windows[0].selected_p))
        self.assertTrue(torch.equal(samples[1], sampling_weights(wire_row)))

        # exact rational law distance, as in the read-only wire-law probe
        def _law(tensor):
            weights = [
                Fraction(float(v)) for v in sampling_weights(tensor).reshape(-1)
            ]
            total = sum(weights)
            return [w / total for w in weights]

        tvd = float(
            sum(
                abs(a - b)
                for a, b in zip(_law(wire_row), _law(raw_row), strict=True)
            )
            / 2
        )
        self.assertAlmostEqual(tvd, 9.765028953570009e-05, delta=1e-16)
        self.assertFalse(torch.equal(samples[0], sampling_weights(raw_row)))

        self.assertIn(int(output[0, 1]), (0, 1))
        self.assertEqual(metrics["draft_generated_tokens"], 1)
        self.assertEqual(metrics["draft_accepted_tokens"], 1)
        self.assertEqual(metrics["generated_tokens"], 2)
        self.assertEqual(len(samples), 2)  # one draft draw + one bonus draw

    def test_sampled_candidate_always_has_positive_wire_support(self):
        """A positive raw tail that FP16 cannot represent is never a candidate.

        The old bridge accepted a sampled token whose wire support was 0.  The
        wire law can only draw tokens it actually carries.
        """
        prefix = torch.tensor([[3]], dtype=torch.long)
        raw_row = torch.tensor([[1e-8, 1.0 - 1e-8, 0.0, 0.0]], dtype=torch.float64)
        router = _FakeCacheRouter(
            draft_rows=[raw_row],
            target_extra_rows={0: [0.0, 1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0, 0.0]},
        )
        payloads, codec_patch = self._record_codec()
        windows, window_patch = self._record_windows()

        with codec_patch, window_patch:
            output, metrics = self._run_router(
                router, prefix, max_tokens=3, gamma=1, transfer_top_k=1
            )

        payload = payloads[0]
        self.assertEqual(payload.indices.tolist(), [[1]])
        self.assertEqual(payload.values.tolist(), [[1.0]])
        # token 0 has positive raw support but is not on the wire at all
        legacy = _quantize_probs_fp16(raw_row.unsqueeze(1))
        self.assertEqual(legacy[0, 0, 0].item(), 0.0)
        receiver = windows[0].proposal.probabilities[0, 0]
        self.assertEqual(receiver[0].item(), 0.0)
        self.assertEqual(receiver[1].item(), 1.0)
        self.assertGreater(
            windows[0].proposal.selected_probabilities()[0, 0].item(), 0.0
        )
        self.assertEqual(int(output[0, 1]), 1)
        self.assertEqual(metrics["draft_accepted_tokens"], 1)
        self.assertEqual(_FakeDraftCache.instances[0].forward_context_calls, [1])

    def test_underflowed_wire_tail_is_not_a_drawable_support(self):
        """A raw 1e-8 tail that FP16 drops is zero *in the sampling law*.

        The legacy bridge accepted a sampled token with zero wire support; the
        wire law can only draw mass it carries, and the receiver's row shows the
        underflow explicitly instead of an invented one-hot.
        """
        prefix = torch.tensor([[3]], dtype=torch.long)
        raw_row = torch.tensor(
            [[0.5, 0.5 - 1e-8, 1e-8, 0.0]], dtype=torch.float64
        )
        router = _FakeCacheRouter(
            draft_rows=[raw_row],
            target_extra_rows={0: [1.0, 0.0, 0.0, 0.0], 1: [1.0, 0.0, 0.0, 0.0]},
        )
        payloads, codec_patch = self._record_codec()
        windows, window_patch = self._record_windows()
        samples, sample_patch = self._record_multinomial()

        with codec_patch, window_patch, sample_patch:
            output, metrics = self._run_router(
                router, prefix, max_tokens=2, gamma=1, transfer_top_k=0
            )

        payload = payloads[0]
        self.assertIsNone(payload.indices)  # full vocabulary: implicit indices
        self.assertGreater(float(raw_row[0, 2].item()), 0.0)
        self.assertEqual(payload.values[0, 2].item(), 0.0)  # real FP16 underflow
        receiver = windows[0].proposal.probabilities[0, 0]
        self.assertEqual(receiver[2].item(), 0.0)
        self.assertEqual(float(sampling_weights(payload.dense())[0, 2].item()), 0.0)
        self.assertTrue(
            torch.equal(samples[0], sampling_weights(payload.dense()))
        )
        self.assertGreater(
            float(windows[0].proposal.selected_probabilities()[0, 0].item()), 0.0
        )
        self.assertIn(int(output[0, 1]), (0, 1))
        # one draft draw + one continuation; a rejection spends one more budget
        # token, so the target-only fallback may add exactly one retained draw.
        accepted = metrics["draft_accepted_tokens"]
        self.assertIn(accepted, (0, 1))
        self.assertEqual(len(samples), 2 + (1 if accepted == 0 else 0))

    def test_invalid_model_row_fails_before_any_draw(self):
        """The draft model row is validated by the codec before sampling."""
        prefix = torch.tensor([[3]], dtype=torch.long)
        router = _FakeCacheRouter(
            draft_rows=[torch.tensor([[2.0, 1.0, 0.0, 0.0]], dtype=torch.float64)]
        )
        with patch(
            "torch.multinomial",
            side_effect=AssertionError("draw before validation"),
        ):
            with self.assertRaises(ValueError):
                self._run_router(router, prefix, max_tokens=3, gamma=1)

    def test_topk_tie_row_bills_exactly_the_sparse_payload(self):
        """Tied top-K logits keep more positives than K; a K-slot wire re-selects.

        ``sampling.py`` masks ``logits < threshold``, so a uniform row keeps all
        V positives with top_k=2.  The payload must still be exactly min(K, V)
        FP16 values, and the billed bytes must equal those values.
        """
        prefix = torch.tensor([[3]], dtype=torch.long)
        uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float64)
        wire_uniform = [0.5, 0.5, 0.0, 0.0]
        router = _FakeCacheRouter(
            draft_rows=[uniform], target_extra_rows={0: wire_uniform, 1: wire_uniform}
        )
        payloads, codec_patch = self._record_codec()
        windows, window_patch = self._record_windows()

        with codec_patch, window_patch:
            output, metrics = self._run_router(
                router, prefix, max_tokens=3, gamma=1
            )

        payload = payloads[0]
        self.assertEqual(int((uniform > 0).sum().item()), VOCAB)  # the code fact
        self.assertEqual(tuple(payload.values.shape), (1, K))
        self.assertEqual(payload.indices.tolist(), [[0, 1]])  # ties by lowest id
        self.assertEqual(payload.values.tolist(), [[0.5, 0.5]])
        self.assertEqual(self._uplinks()[0]["data_size_bytes"], payload.nbytes)
        receiver = windows[0].proposal.probabilities[0, 0]
        self.assertEqual(int((receiver > 0).sum().item()), K)
        self.assertEqual(float(receiver[2:].sum().item()), 0.0)
        self.assertIn(int(output[0, 1]), (0, 1))

    def test_accept_path_bills_paper_payload_and_downlink(self):
        """单轮全接受：上行 = γ·K·2 B（FP16），下行 = 0 B 报文。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        output, metrics = self._run(prefix, max_tokens=3)

        self.assertEqual(tuple(output.shape), (1, 4))
        self.assertTrue(torch.equal(output[:, :1], prefix))
        self.assertEqual(metrics["generated_tokens"], 3)
        self.assertEqual(metrics["draft_generated_tokens"], 2)
        self.assertEqual(metrics["draft_accepted_tokens"], 2)
        self.assertEqual(metrics["avg_top_k"], K)
        self.assertEqual(metrics["avg_draft_len"], 2)
        self.assertEqual(self.instance.num_acc_tokens, [2])

        # 缓存构造：草稿采样 top-k = K（softmax 只作用于 top-K logits），
        # 目标不压缩——TK-SLT 的定义性行为
        draft_cache, target_cache = (
            _FakeDraftCache.instances[0],
            _FakeTargetCache.instances[0],
        )
        self.assertEqual(draft_cache.init_args["top_k"], K)
        self.assertEqual(target_cache.init_args["top_k"], 0)
        self.assertEqual(target_cache.init_args["top_p"], 0)
        self.assertEqual(target_cache.generate_calls, [])
        self.assertEqual(target_cache.forward_context_calls, [3])
        # 草稿用 forward_context 逐步推进（不再 generate）：输入长度 1、2
        self.assertEqual(draft_cache.forward_context_calls, [1, 2])
        self.assertEqual(
            [c.tolist() for c in draft_cache.forward_context_contents],
            [[[3]], [[3, 0]]],
        )

        sim = _FakeCommSimulator.instances[0]
        # 1) 首轮 prompt 上传（全仓一次性约定）
        self.assertIsNotNone(sim.transfer_calls[0]["tokens"])
        self.assertIsNone(sim.transfer_calls[0]["probs"])
        # 2) 分布上行：γ·K·(16/8) 字节 = 2·2·2 = 8，附 topk/draft_len 历史
        uplink = self._uplinks()[0]
        self.assertEqual(uplink["data_size_bytes"], 2 * K * 2)
        self.assertEqual(uplink["kwargs"].get("topk"), K)
        self.assertEqual(uplink["kwargs"].get("draft_len"), 2)
        # 3) 下行（结果 token + 位置 j）：§II-B negligible ⇒ 0 字节报文
        self.assertEqual(len(self._downlinks()), 1)
        self.assertEqual(self._downlinks()[0]["data_size_bytes"], 0)
        # 回滚：全接受 ⇒ draft n+1 / target n+2
        self.assertEqual(draft_cache.rollback_calls, [3])
        self.assertEqual(target_cache.rollback_calls, [4])

    def test_verification_window_is_reconstructed_from_payloads_only(self):
        """接收端只用已计费 payload 重建：没有未计费的 dense raw Q 旁路。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        raw_row = torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float32)
        target_row = [0.4, 0.35, 0.2, 0.05]
        router = _FakeCacheRouter(
            draft_rows=[raw_row], target_extra_rows={0: target_row, 1: target_row}
        )
        payloads, codec_patch = self._record_codec()
        windows, window_patch = self._record_windows()
        samples, sample_patch = self._record_multinomial()

        with codec_patch, window_patch, sample_patch:
            self._run_router(router, prefix, max_tokens=3, gamma=1)

        payload = payloads[0]
        self.assertEqual(payload.indices.tolist(), [[0, 1]])
        self.assertEqual(payload.values.tolist(), [[0.625, 0.375]])
        receiver = windows[0].proposal.probabilities[0, 0]
        self.assertTrue(torch.equal(receiver, payload.dense()[0]))
        self.assertEqual(float(receiver[2:].sum().item()), 0.0)
        # not the raw model row, and not the retired full-row FP16 roundtrip
        self.assertFalse(torch.equal(receiver.float(), raw_row[0]))
        self.assertFalse(
            torch.equal(
                receiver.float(),
                _quantize_probs_fp16(raw_row.unsqueeze(1))[0, 0].float(),
            )
        )
        # the draft draw used the same reconstruction
        self.assertTrue(torch.equal(samples[0], sampling_weights(payload.dense())))
        self.assertTrue(
            torch.equal(windows[0].proposal.sampling_weights[0, 0], samples[0][0])
        )

    def test_reject_path_resamples_from_strict_wire_residual(self):
        """拒绝路径 = strict norm((P−Q̂)+)，无 legacy max_fn 兜底。

        构造必拒场景：目标在 drafted token 上概率为 0 ⇒ p/q = 0。
        Q̂ 是 wire 行（one-hot token 0），残差恰为 P 在其余位置的归一化。
        """
        prefix = torch.tensor([[3]], dtype=torch.long)
        reject_router = _FakeCacheRouter(
            target_extra_rows={0: [0.0, 0.9, 0.1, 0.0]}
        )
        samples, sample_patch = self._record_multinomial()
        windows, window_patch = self._record_windows()

        with sample_patch, window_patch:
            output, metrics = self._run_router(reject_router, prefix)

        self.assertEqual(tuple(output.shape), (1, 3))
        self.assertTrue(torch.equal(output[:, :1], prefix))
        self.assertIn(int(output[0, 1]), (1, 2))
        self.assertEqual(metrics["draft_accepted_tokens"], 0)
        self.assertEqual(self.instance.num_acc_tokens, [0, 1])
        # 3 draws: draft + strict residual + fallback (no extra sampling)
        self.assertEqual(len(samples), 3)
        one_hot_weights = sampling_weights(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        )
        self.assertTrue(torch.equal(samples[0], one_hot_weights))
        residual = samples[1].reshape(-1)
        self.assertEqual(float(residual[0].item()), 0.0)
        self.assertAlmostEqual(float(residual[1].item()), 0.9, places=6)
        self.assertAlmostEqual(float(residual[2].item()), 0.1, places=6)
        self.assertEqual(float(residual.sum().item()), 1.0)
        self.assertTrue(
            torch.equal(
                windows[0].proposal.probabilities[0, 0],
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float16),
            )
        )
        # 上行按 γ_eff=1 计费：1·K·2 = 4 字节
        self.assertEqual(self._uplinks()[0]["data_size_bytes"], K * 2)
        self.assertEqual(self._uplinks()[0]["kwargs"].get("draft_len"), 1)
        self.assertEqual(len(self._downlinks()), 1)
        target_cache = _FakeTargetCache.instances[0]
        self.assertEqual(target_cache.generate_calls, [])
        self.assertEqual(target_cache.forward_context_calls, [2, 2])

    def test_invalid_target_row_fails_before_acceptance_rng(self):
        """Strict validation stays an error: no max_fn / one-hot fallback."""
        prefix = torch.tensor([[3]], dtype=torch.long)
        router = _FakeCacheRouter(target_extra_rows={0: [0.2, 0.2, 0.2, 0.2]})
        samples, sample_patch = self._record_multinomial()

        with sample_patch, self.assertRaises(ValueError):
            self._run_router(router, prefix, max_tokens=3, gamma=1)

        # only the draft draw happened; acceptance/continuation never ran
        self.assertEqual(len(samples), 1)

    def test_multi_round_window_slicing_and_single_prompt_upload(self):
        """两轮 DSD：prompt 只传一次；每轮 payload 独立、前缀内容真实。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        payloads, codec_patch = self._record_codec()
        windows, window_patch = self._record_windows()

        with codec_patch, window_patch:
            output, metrics = self._run(prefix, max_tokens=4, gamma=1)

        draft_cache = _FakeDraftCache.instances[0]
        self.assertEqual(len(payloads), 2)
        self.assertEqual(len(windows), 2)
        self.assertEqual([w.proposal.prefix_len for w in windows], [1, 3])
        for window, payload in zip(windows, payloads, strict=True):
            self.assertTrue(
                torch.equal(
                    window.proposal.probabilities[0, 0], payload.dense()[0]
                )
            )
        # 第二轮草稿缓存从第 1 轮真实结尾继续推进（input length 3）
        self.assertEqual(draft_cache.forward_context_calls, [1, 3])
        self.assertEqual(
            draft_cache.forward_context_contents[0].tolist(), [[3]]
        )
        self.assertEqual(
            draft_cache.forward_context_contents[1].shape[1], 3
        )
        self.assertEqual(
            draft_cache.forward_context_contents[1][0, :1].tolist(), [3]
        )
        # prompt 只在首轮上传一次
        prompt_uploads = [
            c for c in _FakeCommSimulator.instances[0].transfer_calls
            if c["tokens"] is not None
        ]
        self.assertEqual(len(prompt_uploads), 1)
        self.assertEqual(len(self._uplinks()), 2)
        self.assertEqual(metrics["generated_tokens"], 4)
        self.assertEqual(output.shape[1], 5)

    def test_k_disabled_degenerates_to_vanilla_dsd_billing(self):
        """K<=0/None ⇒ 全量 softmax 提议 + 整词表载荷（论文自己的 DSD 基线）。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        payloads, codec_patch = self._record_codec()
        with codec_patch:
            output, metrics = self._run(prefix, transfer_top_k=0, max_tokens=3)

        draft_cache = _FakeDraftCache.instances[0]
        self.assertEqual(draft_cache.init_args["top_k"], 0)
        # γ=2、|V|=4、FP16 ⇒ 2·4·2 = 16 字节（整词表）
        self.assertEqual(self._uplinks()[0]["data_size_bytes"], 2 * VOCAB * 2)
        self.assertEqual(self._uplinks()[0]["kwargs"].get("topk"), 0)
        self.assertEqual(metrics["avg_top_k"], 0)
        self.assertEqual(len(payloads), 2)
        for payload in payloads:
            self.assertIsNone(payload.indices)
            self.assertEqual(tuple(payload.values.shape), (1, VOCAB))
            self.assertEqual(payload.nbytes, VOCAB * 2)


class TestTkSltOdldWiring(TkSltLoopTestBase):
    def test_gamma_star_replaces_cli_gamma_from_round_two(self):
        """ODLD 开启：首轮用 --gamma 兜底，此后逐轮用 γ*。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        with patch(
            "src.baselines.tk_slt.tk_slt_select_speculative",
            return_value=(True, 3, 1.5),
        ) as sel:
            output, metrics = self._run(
                prefix, max_tokens=8, gamma=2, tk_slt_odld=True
            )

        self.assertEqual(sel.call_count, 2)  # 首轮无估计不咨询；第 2/3 轮咨询
        # （第 3 轮 remaining=1，咨询后走 fallback：target 直出）
        draft_cache = _FakeDraftCache.instances[0]
        self.assertEqual(draft_cache.forward_context_calls, [1, 2, 4, 5, 6])
        self.assertEqual(self.instance.num_acc_tokens, [2, 3, 1])
        self.assertTrue(metrics["tk_slt_odld"])
        self.assertEqual(metrics["tk_slt_gamma_star_final"], 3)

    def test_as2_standalone_round_has_no_uplink(self):
        """AS² 判 S*<=1 的轮次：target 直出、无上行分布，只回 0 字节下行。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        with patch(
            "src.baselines.tk_slt.tk_slt_select_speculative",
            return_value=(False, 1, 0.5),
        ):
            output, metrics = self._run(
                prefix, max_tokens=3, gamma=1, tk_slt_odld=True
            )

        # 首轮 DSD（draft 1 次）+ 第 2 轮 standalone（target 直出）
        draft_cache = _FakeDraftCache.instances[0]
        target_cache = _FakeTargetCache.instances[0]
        self.assertEqual(len(draft_cache.forward_context_calls), 1)
        self.assertEqual(target_cache.generate_calls, [])
        self.assertEqual(len(target_cache.forward_context_calls), 2)
        self.assertEqual(metrics["tk_slt_standalone_rounds"], 1)
        # 上行分布只有首轮那一次；两轮各有一次 0 字节下行
        self.assertEqual(len(self._uplinks()), 1)
        self.assertEqual(len(self._downlinks()), 2)
        self.assertEqual(metrics["generated_tokens"], 3)

    def test_odld_off_by_default_keeps_fixed_gamma(self):
        """默认关闭：所有轮都用 --gamma（与其它基线同口径）。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        output, metrics = self._run(prefix, max_tokens=8, gamma=2)
        draft_cache = _FakeDraftCache.instances[0]
        # 第 3 轮 remaining=2 ⇒ γ_eff=1（越界钳位是既有约定）
        self.assertEqual(draft_cache.forward_context_calls, [1, 2, 4, 5, 7])
        self.assertFalse(metrics["tk_slt_odld"])
        self.assertEqual(metrics["tk_slt_standalone_rounds"], 0)


class TestRetiredFp16Codec(unittest.TestCase):
    """The retired full-row FP16 roundtrip kept for the historical bridge only.

    The active TK-SLT loop no longer uses this codec (see
    :mod:`src.tk_slt_proposal`); these assertions pin the historical helper's
    documented lossiness so the retained bridge cannot be mistaken for a
    lossless wire.
    """

    def test_values_change_and_rows_stay_normalized(self):
        probs = torch.tensor([[[0.1, 0.3, 0.6, 0.0]]])
        out = _quantize_probs_fp16(probs)
        self.assertFalse(torch.equal(out, probs))
        self.assertTrue(
            torch.allclose(out.sum(dim=-1), torch.ones(1, 1), atol=1e-6)
        )

    def test_zeros_and_one_hot_survive(self):
        one_hot = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
        self.assertTrue(torch.equal(_quantize_probs_fp16(one_hot), one_hot))
        # 稀疏支撑（0 值位置）保持为 0
        sparse = torch.tensor([[[0.6, 0.4, 0.0, 0.0]]])
        out = _quantize_probs_fp16(sparse)
        self.assertTrue(torch.equal(out[:, :, 2:], torch.zeros(1, 1, 2)))

    def test_none_and_empty_pass_through(self):
        self.assertIsNone(_quantize_probs_fp16(None))
        empty = torch.zeros(1, 0, VOCAB)
        self.assertTrue(torch.equal(_quantize_probs_fp16(empty), empty))

    def test_positive_fp16_tail_can_underflow_and_change_the_wire_law(self):
        """The retired wire representation is lossy; docs must not call it lossless."""
        raw = torch.tensor([[[0.6, 0.4, 1e-8, 0.0]]], dtype=torch.float64)
        wire = _quantize_probs_fp16(raw)
        # 1e-8 is positive in the proposal law but not representable in fp16.
        self.assertEqual(wire[0, 0, 2].item(), 0.0)
        self.assertFalse(torch.equal(wire, raw))
        # Round-to-nearest + row renormalization moves the retained masses.
        self.assertNotEqual(wire[0, 0, 0].item(), raw[0, 0, 0].item())
        self.assertAlmostEqual(float(wire.sum().item()), 1.0, places=6)


class TestCompatibilityReExports(unittest.TestCase):
    def test_policy_names_resolve_from_helpers_and_package(self):
        import src.baselines as baselines
        import src.baselines_helpers as helpers
        import src.tk_slt_policy as policy

        for name in (
            "_lambert_w_minus_one",
            "tk_slt_optimal_draft_length",
            "tk_slt_select_speculative",
            "tk_slt_speedup_ratio",
        ):
            self.assertIs(getattr(helpers, name), getattr(policy, name))
            self.assertIs(getattr(baselines, name), getattr(policy, name))


class TestLambertWMinusOne(unittest.TestCase):
    def test_solves_we_w_equals_z(self):
        for z in (-1e-30, -1e-10, -1e-4, -0.1, -0.3, -0.367):
            w = _lambert_w_minus_one(z)
            self.assertLessEqual(w, -1.0)
            self.assertAlmostEqual(w * math.exp(w), z, delta=abs(z) * 1e-9 + 1e-40)

    def test_boundary_and_domain_errors(self):
        with self.assertRaises(ValueError):
            _lambert_w_minus_one(0.0)
        with self.assertRaises(ValueError):
            _lambert_w_minus_one(-1.0)  # < -1/e
        with self.assertRaises(ValueError):
            _lambert_w_minus_one(0.5)

    def test_minus_one_branch_not_principal(self):
        # W_{-1} 必须取 ≤ -1 的分支（主分支 W_0 ∈ [-1, 0) 解同一方程）
        w = _lambert_w_minus_one(-0.1)
        self.assertLess(w, -3.0)


class TestOdldTableI(unittest.TestCase):
    """论文 Table I 的 γ* 逐格复现（α 行 × L 列）。

    1* = S_inf < 1（AS² 会退回 standalone LLM）。注意两个精确平局格
    （α=0.4/L=0.4 与 α=0.6/L=0.6）：数学上 S_inf(1) = (1+α)/(1+L) = 1
    恰好成立，论文自己的表格因浮点落点不同把前者标成无星、后者标成
    1*——本实现按 Algorithm 2 的字面判据 S* > 1 取 standalone，平局格
    只断言 γ* 与 S*≈1。
    """

    TABLE_I = {
        0.4: [4, 2, 1, "1", "1*"],
        0.6: [7, 3, 2, 1, "1*"],
        0.8: [14, 6, 4, 2, 1],
    }
    L_VALUES = [0.01, 0.1, 0.2, 0.4, 0.6]

    def test_gamma_star_matches_every_cell(self):
        for alpha, row in self.TABLE_I.items():
            for L, expected in zip(self.L_VALUES, row, strict=True):
                with self.subTest(alpha=alpha, L=L):
                    starred = isinstance(expected, str) and expected.endswith("*")
                    gamma_exp = int(str(expected).rstrip("*"))
                    use_dsd, gamma, s = tk_slt_select_speculative(alpha, L, 0.0)
                    self.assertEqual(gamma, gamma_exp)
                    if abs(s - 1.0) < 1e-9:
                        continue  # 精确平局格：论文自身的浮点落点决定分支
                    self.assertEqual(use_dsd, not starred)

    def test_speedup_ratio_formula(self):
        # 式 (11)：L=0 时 S = E[#tokens] = (1-α^{γ+1})/(1-α)（式 12）
        self.assertEqual(tk_slt_speedup_ratio(5, 0.5, 0.0), 1.96875)
        self.assertAlmostEqual(tk_slt_speedup_ratio(3, 0.0, 0.2), 1 / 1.6)
        self.assertAlmostEqual(tk_slt_speedup_ratio(3, 1.0, 0.2), 4 / 1.6)
        # 论文 Fig 2 的量级：α=0.8、L=0.2、γ=4 ⇒ S≈1.868
        self.assertAlmostEqual(
            tk_slt_speedup_ratio(4, 0.8, 0.2), 0.67232 / 0.36, places=6
        )

    def test_degenerate_guards(self):
        # α ≥ 1 / L ≤ 0：γ 取上限
        self.assertEqual(tk_slt_optimal_draft_length(1.0, 0.1, 0.1)[0], 64)
        self.assertEqual(tk_slt_optimal_draft_length(0.5, 0.0, 0.0)[0], 64)
        # α ≤ 0：γ*=1（AS² → standalone）
        g, s = tk_slt_optimal_draft_length(0.0, 0.1, 0.1)
        self.assertEqual((g, s), (1, 1 / 1.2))
        # L ≥ 1：论文约束外，γ*=1 且 S<1 ⇒ standalone
        g, s = tk_slt_optimal_draft_length(0.5, 0.7, 0.5)
        self.assertEqual(g, 1)
        self.assertLess(s, 1.0)


class TestTxTimeHelper(unittest.TestCase):
    def test_reads_last_edge_cloud_tx_time(self):
        sim = CommunicationSimulator(
            bandwidth_edge_cloud=46,
            bandwidth_edge_end=float("inf"),
            bandwidth_cloud_end=float("inf"),
            dimension="Mbps",
            ntt_ms_edge_cloud=200,
            ntt_ms_edge_end=20,
        )
        sim.simulate_transfer(8, "edge_cloud", topk=K, draft_len=1)
        expected = 8 / (46 * 1e6 / 8)
        self.assertAlmostEqual(
            _last_edge_cloud_tx_seconds(sim), expected, places=15
        )

    def test_returns_none_without_stats(self):
        self.assertIsNone(_last_edge_cloud_tx_seconds(_FakeCommSimulator()))


class TestPaperPayloadReachesSimulator(TkSltLoopTestBase):
    def test_uplink_bytes_follow_gamma_k_and_vocab(self):
        """上行字节 = tk_slt_uplink_payload_bytes（γ·K·b_prob/8），单源。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        self._run(prefix, max_tokens=3)
        uplink = self._uplinks()[0]
        self.assertEqual(
            uplink["data_size_bytes"],
            int(tk_slt_uplink_payload_bytes(K, 2, VOCAB)),
        )


if __name__ == "__main__":
    unittest.main()
