"""CU-HLM (`uncertainty_decoding` / `cuhlm`) public-loop CPU contracts.

The single-token lifecycle under test (paper II-A / IV-B + Algorithm 1): one
round finalizes exactly one token.  A trigger forwards the *current prefix*
through the target (no draft token, no discarded sample), then either keeps the
draft token (accept) or appends one residual token (reject).  A skip appends
the draft token without any target forward.  There is **no full-accept bonus**.

These tests drive the real registered public method.  The KV caches are fakes
whose history lengths mirror the real `KVCacheModel` endpoint semantics (the
forwarded prefix has `prefix_len` rows; the finalized token is *not* forwarded).
`test_cuhlm_cpu_contracts.py` repeats the endpoint assertions with a real tiny
Llama + real `KVCacheModel`.
"""

import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.baselines import Baselines
from src.communication import cuhlm_uplink_payload_bytes


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
    instances = []

    # 统一往返口径（2026-10-09）：默认 per_transfer；flush/set_round 为 no-op
    coalesce_rounds = False

    #: 每个触发轮是否真的需要上行；按 determine_transfer_strategy 调用序号取。
    transfer_flags = [True]

    #: 可选的脚本化重建分布（每个触发轮消费一项）；空表示原样返回 q。
    reconstructed_rows = []

    def flush_round(self):
        """per_round 合并模式的轮末结算；fake 不做字节合并。"""
        self.round_events.append(("flush", None))

    def set_round(self, round_idx):
        """per_round 合并模式的轮界标记；fake 不结算。"""
        self.round_events.append(("set_round", int(round_idx)))

    def __init__(self, *args, **kwargs):
        self.init_kwargs = dict(kwargs)
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
        self.round_events = []
        self.strategy_calls = []
        self.uncertainty_draft_tokens = []
        self.accept_messages = 0
        self.reject_messages = 0
        _FakeCommSimulator.instances.append(self)

    def transfer(self, tokens, probs, link_type="edge_cloud", **kwargs):
        self.transfer_calls.append(
            {
                "kind": "context_transfer",
                "tokens": None if tokens is None else tokens.clone(),
                "probs": None if probs is None else probs.clone(),
                "link_type": link_type,
                "kwargs": kwargs,
            }
        )
        return 0.0

    def calculate_uncertainty(self, logits, M=20, theta_max=2.0, draft_token=None):
        self.uncertainty_draft_tokens.append(draft_token)
        return 1.0

    def determine_transfer_strategy(self, uncertainty, current_probs, draft_token=None):
        idx = len(self.strategy_calls)
        self.strategy_calls.append(
            (
                uncertainty,
                None if current_probs is None else tuple(current_probs.shape),
                draft_token,
            )
        )
        flags = _FakeCommSimulator.transfer_flags
        should_transfer = flags[idx] if idx < len(flags) else flags[-1]
        return should_transfer, 2

    def _get_current_probs(self, prob_history):
        return prob_history[0, -1, :]

    def _apply_top_k_compression(self, probs, k):
        return probs

    def compress_rebuild_probs(self, probs, k):
        # F-CUHLM：reject 重采样改用压缩重构分布 x̂。镜像真实实现的
        # 三维约束（communication.py:584），防止调用方再退回二维输入。
        if probs.dim() != 3:
            raise ValueError(f"probs维度应为3，实际为{probs.dim()}")
        if _FakeCommSimulator.reconstructed_rows:
            row = _FakeCommSimulator.reconstructed_rows.pop(0)
            return torch.tensor(row, dtype=probs.dtype).reshape(1, 1, -1)
        return probs

    def simulate_transfer(self, data_size_bytes, link_type="edge_cloud", **kwargs):
        # 与 transfer 同样记录一次调用：F-CUHLM 论文口径下，触发上行
        # （式 (5) 字节）与 0 字节下行都经 simulate_transfer 计费。
        self.transfer_calls.append(
            {
                "kind": (
                    "downlink" if int(data_size_bytes) == 0 else "uplink"
                ),
                "tokens": None,
                "probs": None,
                "link_type": link_type,
                "kwargs": dict(kwargs),
                "data_size_bytes": int(data_size_bytes),
            }
        )
        return 0.0

    def send_accept_message(self, linktype):
        self.accept_messages += 1

    def send_reject_message(self, linktype):
        self.reject_messages += 1


def _row(length, index_value, vocab_size):
    row = torch.zeros(vocab_size, dtype=torch.float)
    row[index_value] = 1.0
    return row


class _FakeCache:
    """Endpoint-faithful KVCacheModel stand-in.

    Real semantics after `generate(prefix, 1)` / `forward_context(prefix)`:
    the cache endpoint and every history row number is ``prefix_len`` (the
    forwarded prefix).  The returned/new token is *not* forwarded yet.  The old
    fake returned ``prefix_len + 1`` history rows for a 1-token prefix, which
    hid off-by-one endpoint bugs.
    """

    instances = []

    vocab = 2
    draft_tokens = [0]
    draft_rows = [[1.0, 0.0]]
    target_rows = [[0.0, 1.0]]

    def __init__(self, model, temperature, top_k, top_p):
        self.model = model
        self.device = model.device
        self.vocab_size = _FakeCache.vocab
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.rollback_calls = []
        self.generate_calls = []
        self.forward_context_calls = []
        self._current_seq_len = 0
        self.prob_history = None
        self.logits_history = None
        _FakeCache.instances.append(self)

    def _history(self, length, row):
        probs = torch.zeros(1, length, self.vocab_size, dtype=torch.float)
        probs[:, -1, :] = torch.tensor(row, dtype=torch.float)
        logits = torch.zeros(1, length, self.vocab_size, dtype=torch.float)
        return probs, logits

    def _at(self, rows, idx):
        return rows[idx] if idx < len(rows) else rows[-1]

    def _set_endpoint(self, length):
        self._current_seq_len = length
        self.current_length = length

    def generate(self, prefix, gamma):
        idx = len(self.generate_calls)
        self.generate_calls.append((prefix.clone(), gamma))
        if self.model.kind != "draft":
            raise AssertionError(
                "target.generate must not be used by the single-token CU-HLM "
                "loop; the target only needs forward_context(prefix)"
            )
        if gamma != 1:
            raise AssertionError(f"CU-HLM draft gamma must be 1, got {gamma}")
        token = self._at(_FakeCache.draft_tokens, idx)
        self.prob_history, self.logits_history = self._history(
            prefix.shape[1], self._at(_FakeCache.draft_rows, idx)
        )
        self._set_endpoint(prefix.shape[1])
        return torch.cat(
            (
                prefix,
                torch.tensor([[token]], dtype=prefix.dtype, device=prefix.device),
            ),
            dim=1,
        )

    def forward_context(self, prefix):
        idx = len(self.forward_context_calls)
        self.forward_context_calls.append(prefix.clone())
        if self.model.kind == "draft":
            raise AssertionError("draft cache must use generate(), not forward_context")
        self.prob_history, self.logits_history = self._history(
            prefix.shape[1], self._at(_FakeCache.target_rows, idx)
        )
        self._set_endpoint(prefix.shape[1])
        return self.prob_history[:, -1, :]

    def rollback(self, end_pos):
        self.rollback_calls.append(end_pos)


class UncertaintyDecodingTests(unittest.TestCase):
    def setUp(self):
        _FakeCache.instances = []
        _FakeCommSimulator.instances = []
        _FakeCommSimulator.transfer_flags = [True]
        _FakeCommSimulator.reconstructed_rows = []
        _FakeCache.vocab = 2
        _FakeCache.draft_tokens = [0]
        _FakeCache.draft_rows = [[1.0, 0.0]]
        _FakeCache.target_rows = [[0.0, 1.0]]
        args = Namespace(
            eval_mode="uncertainty_decoding",
            edge_cloud_bandwidth=10,
            max_tokens=1,
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
            gamma=1,
            gamma1=1,
            gamma2=1,
            use_early_stopping=False,
            dump_network_stats=False,
        )

        self.instance = object.__new__(_TestBaselines)
        self.instance.args = args
        self.instance.accelerator = type("Accel", (), {"is_main_process": True})()
        self.instance.vocab_size = 2
        self.instance.num_acc_tokens = []
        self.instance.draft_forward_times = 0
        self.instance.target_forward_times = 0
        self.instance.color_print = lambda *args, **kwargs: None
        self.instance.draft_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "draft"}
        )()
        self.instance.target_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "target"}
        )()

    def _run(self, prefix):
        with (
            patch("src.baselines.cuhlm.KVCacheModel", _FakeCache),
            patch("src.baselines.cuhlm.CUHLM", _FakeCommSimulator),
            patch("src.baselines.cuhlm.torch.cuda.Event", _FakeCudaEvent),
            patch(
                "src.baselines.cuhlm.torch.cuda.current_stream", return_value=None
            ),
            patch("src.baselines.cuhlm.torch.cuda.synchronize", return_value=None),
        ):
            output, metrics = self.instance.uncertainty_decoding(prefix)
        return output, metrics, _FakeCache.instances, _FakeCommSimulator.instances[0]

    # ------------------------------------------------------------------
    # Core lifecycle: one round finalizes exactly one token
    # ------------------------------------------------------------------
    def test_rejection_path_keeps_single_token_and_uses_prefix_forward(self):
        # draft p(0)=1, target p(0)=0 -> deterministic reject; residual token 1.
        prefix = torch.tensor([[1]], dtype=torch.long)
        output, metrics, caches, comm_simulator = self._run(prefix)

        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertTrue(torch.equal(output[:, :1], prefix))
        self.assertEqual(int(output[0, 1]), 1)  # residual token, not a bonus sample
        self.assertEqual(metrics["generated_tokens"], 1)
        self.assertEqual(metrics["draft_generated_tokens"], 1)
        self.assertEqual(metrics["draft_accepted_tokens"], 0)
        self.assertEqual(metrics["target_forward_times"], 1)

        approx_cache, target_cache = caches
        # The target only forwarded the old prefix (len 1); no draft token, no
        # discarded sample.  Neither cache is rolled back: the finalized token
        # has not been forwarded yet.
        self.assertEqual(len(target_cache.forward_context_calls), 1)
        self.assertEqual(
            tuple(target_cache.forward_context_calls[0].shape), (1, 1)
        )
        self.assertEqual(target_cache.generate_calls, [])
        self.assertEqual(approx_cache.rollback_calls, [])
        self.assertEqual(target_cache.rollback_calls, [])
        self.assertEqual(approx_cache.current_length, 1)
        self.assertEqual(target_cache.current_length, 1)
        self.assertEqual(tuple(target_cache.prob_history.shape), (1, 1, 2))

        self.assertEqual(comm_simulator.reject_messages, 0)
        self.assertEqual(comm_simulator.accept_messages, 0)
        kinds = [call["kind"] for call in comm_simulator.transfer_calls]
        self.assertEqual(kinds, ["context_transfer", "uplink", "downlink"])

    def test_accept_keeps_draft_without_bonus_sample(self):
        # Equal draft/target laws make y_d == x_d, i.e. deterministic accept for
        # every sampled draft token.  The old code appended a target bonus and
        # produced two tokens for a budget of one.
        _FakeCache.draft_rows = [[0.5, 0.5]]
        _FakeCache.target_rows = [[0.5, 0.5]]
        prefix = torch.tensor([[1]], dtype=torch.long)
        output, metrics, caches, comm_simulator = self._run(prefix)

        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertEqual(int(output[0, 1]), 0)  # the draft token itself
        self.assertEqual(metrics["generated_tokens"], 1)
        self.assertEqual(metrics["draft_accepted_tokens"], 1)
        self.assertEqual(metrics["target_forward_times"], 1)

        approx_cache, target_cache = caches
        self.assertEqual(len(approx_cache.generate_calls), 1)
        self.assertEqual(len(target_cache.forward_context_calls), 1)
        self.assertEqual(target_cache.forward_context_calls[0].tolist(), [[1]])
        self.assertEqual(target_cache.generate_calls, [])
        self.assertEqual(approx_cache.rollback_calls, [])
        self.assertEqual(target_cache.rollback_calls, [])
        self.assertEqual(approx_cache.current_length, 1)
        self.assertEqual(target_cache.current_length, 1)

        kinds = [call["kind"] for call in comm_simulator.transfer_calls]
        self.assertEqual(kinds, ["context_transfer", "uplink", "downlink"])

    def test_skip_appends_draft_without_target_forward_or_fake_comm(self):
        _FakeCommSimulator.transfer_flags = [False]
        prefix = torch.tensor([[1]], dtype=torch.long)
        output, metrics, caches, comm_simulator = self._run(prefix)

        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertEqual(int(output[0, 1]), 0)
        self.assertEqual(metrics["generated_tokens"], 1)
        self.assertEqual(metrics["draft_forward_times"], 1)
        self.assertEqual(metrics["target_forward_times"], 0)
        # A skipped draft is directly finalized, so it counts as accepted
        # (draft-finalization accounting), NOT as target accuracy.
        self.assertEqual(metrics["draft_accepted_tokens"], 1)
        approx_cache, target_cache = caches
        self.assertEqual(target_cache.forward_context_calls, [])
        self.assertEqual(target_cache.generate_calls, [])
        # Only loop_idx == 1's initial context transfer: no uplink, no downlink.
        self.assertEqual(
            [call["kind"] for call in comm_simulator.transfer_calls],
            ["context_transfer"],
        )

    def test_zero_budget_performs_no_forward_or_sampling(self):
        self.instance.args.max_tokens = 0
        prefix = torch.tensor([[1, 0]], dtype=torch.long)
        output, metrics, caches, comm_simulator = self._run(prefix)

        self.assertTrue(torch.equal(output, prefix))
        self.assertEqual(metrics["generated_tokens"], 0)
        self.assertEqual(metrics["draft_forward_times"], 0)
        self.assertEqual(metrics["target_forward_times"], 0)
        for cache in caches:
            self.assertEqual(cache.generate_calls, [])
            self.assertEqual(cache.forward_context_calls, [])
        self.assertEqual(comm_simulator.transfer_calls, [])

    def test_lossy_reconstructed_law_drives_the_rejection_token(self):
        # q(d)=.5, P(d)=.3 -> ratio .6; draw .9 rejects.  (P - qhat)+ puts all
        # mass on token 0, while the exact (P - q)+ would put it on token 2.
        _FakeCache.vocab = 3
        self.instance.vocab_size = 3
        _FakeCache.draft_tokens = [0]
        _FakeCache.draft_rows = [[0.5, 0.2, 0.3]]
        _FakeCache.target_rows = [[0.3, 0.0, 0.7]]
        _FakeCommSimulator.transfer_flags = [True]
        _FakeCommSimulator.reconstructed_rows = [[0.1, 0.2, 0.7]]
        prefix = torch.tensor([[1]], dtype=torch.long)

        with patch(
            "src.baselines.cuhlm.torch.rand",
            return_value=torch.tensor([[0.9]]),
        ):
            output, metrics, _caches, comm_simulator = self._run(prefix)

        self.assertEqual(output[0, 1:].tolist(), [0])
        self.assertEqual(metrics["draft_accepted_tokens"], 0)
        self.assertEqual(metrics["generated_tokens"], 1)
        self.assertEqual(metrics["target_forward_times"], 1)
        self.assertEqual(_FakeCommSimulator.reconstructed_rows, [])
        self.assertEqual(comm_simulator.reject_messages, 0)
        self.assertEqual(
            [call["kind"] for call in comm_simulator.transfer_calls],
            ["context_transfer", "uplink", "downlink"],
        )

    def test_invalid_reconstructed_law_fails_before_acceptance_rng(self):
        _FakeCache.vocab = 3
        self.instance.vocab_size = 3
        _FakeCache.draft_tokens = [0]
        _FakeCache.draft_rows = [[0.5, 0.2, 0.3]]
        _FakeCache.target_rows = [[0.3, 0.0, 0.7]]
        _FakeCommSimulator.transfer_flags = [True]
        _FakeCommSimulator.reconstructed_rows = [[float("nan"), 0.5, 0.4]]
        prefix = torch.tensor([[1]], dtype=torch.long)

        with patch(
            "src.baselines.cuhlm.torch.rand",
            side_effect=AssertionError("acceptance RNG consumed before validation"),
        ):
            with self.assertRaisesRegex(ValueError, "reconstructed"):
                self._run(prefix)

    def test_skip_does_not_hide_an_invalid_proposal(self):
        # The sampler could not have produced draft token 0 from this row; the
        # skip branch must not silently finalize it.
        _FakeCache.draft_tokens = [0]
        _FakeCache.draft_rows = [[0.0, 1.0]]
        _FakeCommSimulator.transfer_flags = [False]
        prefix = torch.tensor([[1]], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "selected support"):
            self._run(prefix)

    def test_consecutive_accepts_never_insert_bonus_tokens(self):
        _FakeCache.draft_rows = [[0.5, 0.5]]
        _FakeCache.target_rows = [[0.5, 0.5]]
        _FakeCache.draft_tokens = [0, 1, 0]
        _FakeCommSimulator.transfer_flags = [True, True, True]
        self.instance.args.max_tokens = 3
        prefix = torch.tensor([[1]], dtype=torch.long)
        output, metrics, caches, _ = self._run(prefix)

        self.assertEqual(tuple(output.shape), (1, 4))
        self.assertEqual(output[0, 1:].tolist(), [0, 1, 0])
        self.assertEqual(metrics["generated_tokens"], 3)
        self.assertEqual(metrics["draft_generated_tokens"], 3)
        self.assertEqual(metrics["draft_accepted_tokens"], 3)
        self.assertEqual(metrics["target_forward_times"], 3)
        approx_cache, target_cache = caches
        self.assertEqual(len(approx_cache.generate_calls), 3)
        self.assertEqual(len(target_cache.forward_context_calls), 3)
        # Every target prefix forward sees the finalized prefix before the
        # current round: 1, 2, 3 tokens.
        self.assertEqual(
            [call.shape[1] for call in target_cache.forward_context_calls],
            [1, 2, 3],
        )
        self.assertEqual(target_cache.generate_calls, [])

    def test_accept_reject_mixed_rounds_conserve_tokens(self):
        # round 1 reject (draft 0, target prefers 1) -> token 1
        # round 2 accept (equal laws, draft 0) -> token 0
        # round 3 skip (draft 1) -> token 1
        _FakeCache.draft_tokens = [0, 0, 1]
        _FakeCache.draft_rows = [[1.0, 0.0], [0.5, 0.5], [0.5, 0.5]]
        _FakeCache.target_rows = [[0.0, 1.0], [0.5, 0.5], [0.5, 0.5]]
        _FakeCommSimulator.transfer_flags = [True, True, False]
        self.instance.args.max_tokens = 3
        prefix = torch.tensor([[1]], dtype=torch.long)
        output, metrics, caches, comm_simulator = self._run(prefix)

        self.assertEqual(output[0, 1:].tolist(), [1, 0, 1])
        self.assertEqual(metrics["generated_tokens"], 3)
        self.assertEqual(metrics["draft_generated_tokens"], 3)
        # Two drafts were directly finalized (one accept + one skip); the
        # rejected round contributed no draft token.
        self.assertEqual(metrics["draft_accepted_tokens"], 2)
        self.assertEqual(metrics["target_forward_times"], 2)
        # Conservation: generated = accepted drafts + rejected rounds.
        self.assertEqual(
            metrics["generated_tokens"] - metrics["draft_accepted_tokens"], 1
        )

        # The skip round must not fabricate comm: round 1 (trigger) and round 2
        # (accept) bill one uplink + one index-only downlink each; round 3
        # (skip) bills nothing beyond the loop-1 context transfer.
        self.assertEqual(
            [call["kind"] for call in comm_simulator.transfer_calls],
            ["context_transfer", "uplink", "downlink", "uplink", "downlink"],
        )
        # set_round once per loop iteration, then one settlement flush.
        self.assertEqual(
            comm_simulator.round_events,
            [
                ("set_round", 1),
                ("set_round", 2),
                ("set_round", 3),
                ("flush", None),
            ],
        )

        approx_cache, target_cache = caches
        # target cache advanced on trigger rounds only; the skip round did not
        # forward it.  Each trigger forward covers the full finalized prefix.
        self.assertEqual(
            [call.shape[1] for call in target_cache.forward_context_calls],
            [1, 2],
        )
        # Final prefix is prompt(1) + 3 finalized tokens = 4.  The last
        # finalized token was not forwarded by either cache, and the target is
        # additionally two rounds behind after the skip (it will catch up over
        # both missing tokens on its next trigger).
        self.assertEqual(target_cache.current_length, 2)
        self.assertEqual(tuple(target_cache.prob_history.shape), (1, 2, 2))
        # The draft cache is always exactly at the prefix it forwarded this
        # round (3), one token behind the finalized output (4).
        self.assertEqual(approx_cache.current_length, 3)
        self.assertEqual(tuple(approx_cache.prob_history.shape), (1, 3, 2))

    # ------------------------------------------------------------------
    # EOS: only a finalized EOS stops; a rejected draft EOS does not.
    # ------------------------------------------------------------------
    def _set_eos(self, eos_id, max_tokens, prompt):
        _FakeCache.vocab = 5
        self.instance.vocab_size = 5
        self.instance.args.max_tokens = max_tokens
        self.instance.tokenizer = SimpleNamespace(eos_token_id=eos_id)
        return prompt

    def test_rejected_draft_eos_does_not_stop_generation(self):
        _FakeCache.draft_tokens = [3, 0]
        _FakeCache.draft_rows = [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
        _FakeCache.target_rows = [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
        _FakeCommSimulator.transfer_flags = [True, False]
        prefix = torch.tensor([[1, 2]], dtype=torch.long)
        self._set_eos(3, 2, prefix)
        output, metrics, caches, _ = self._run(prefix)

        # Round 1 rejects draft EOS 3 -> residual 4; round 2 skips token 0.
        self.assertEqual(output[0, 2:].tolist(), [4, 0])
        self.assertEqual(metrics["generated_tokens"], 2)
        self.assertEqual(metrics["draft_accepted_tokens"], 1)  # skip only
        approx_cache, _target_cache = caches
        self.assertEqual(len(approx_cache.generate_calls), 2)

    def test_finalized_eos_stops_before_another_round(self):
        _FakeCache.draft_tokens = [3]
        _FakeCache.draft_rows = [[0.0, 0.0, 0.0, 1.0, 0.0]]
        _FakeCommSimulator.transfer_flags = [False]
        prefix = torch.tensor([[1, 2]], dtype=torch.long)
        self._set_eos(3, 5, prefix)
        output, metrics, caches, _ = self._run(prefix)

        self.assertEqual(output[0, 2:].tolist(), [3])
        self.assertEqual(metrics["generated_tokens"], 1)
        self.assertEqual(metrics["draft_forward_times"], 1)
        approx_cache, _target_cache = caches
        self.assertEqual(len(approx_cache.generate_calls), 1)

    def test_prompt_eos_does_not_stop_generation(self):
        _FakeCache.draft_tokens = [0]
        _FakeCache.draft_rows = [[1.0, 0.0, 0.0, 0.0, 0.0]]
        _FakeCommSimulator.transfer_flags = [False]
        prefix = torch.tensor([[3, 3]], dtype=torch.long)
        self._set_eos(3, 1, prefix)
        output, _metrics, caches, _ = self._run(prefix)

        self.assertEqual(output[0, 2:].tolist(), [0])
        approx_cache, _target_cache = caches
        self.assertEqual(len(approx_cache.generate_calls), 1)

    def test_zero_budget_with_prompt_eos_does_not_sample(self):
        prefix = torch.tensor([[3, 3]], dtype=torch.long)
        self._set_eos(3, 0, prefix)
        output, metrics, caches, comm_simulator = self._run(prefix)

        self.assertTrue(torch.equal(output, prefix))
        self.assertEqual(metrics["generated_tokens"], 0)
        for cache in caches:
            self.assertEqual(cache.generate_calls, [])
            self.assertEqual(cache.forward_context_calls, [])
        self.assertEqual(comm_simulator.transfer_calls, [])

    # ------------------------------------------------------------------
    # Communication ordering / paper accounting (unchanged by this batch)
    # ------------------------------------------------------------------
    def test_trigger_comm_order_and_pending_resync_payload(self):
        # skip round 1 buffers one resync token; round 2 triggers.
        _FakeCommSimulator.transfer_flags = [False, True]
        _FakeCache.draft_tokens = [0, 0]
        _FakeCache.draft_rows = [[1.0, 0.0], [1.0, 0.0]]
        _FakeCache.target_rows = [[0.0, 1.0]]
        self.instance.args.max_tokens = 2
        prefix = torch.tensor([[1]], dtype=torch.long)
        _output, metrics, _caches, comm_simulator = self._run(prefix)

        kinds = [call["kind"] for call in comm_simulator.transfer_calls]
        self.assertEqual(
            kinds, ["context_transfer", "uplink", "downlink"]
        )
        uplink = comm_simulator.transfer_calls[1]
        self.assertEqual(
            uplink["data_size_bytes"],
            int(cuhlm_uplink_payload_bytes(2, self.instance.vocab_size)),
        )
        self.assertEqual(uplink["kwargs"].get("topk"), 2)
        # one buffered skipped token + this round's draft token
        self.assertEqual(uplink["kwargs"].get("draft_len"), 2)
        downlink = comm_simulator.transfer_calls[2]
        self.assertEqual(downlink["data_size_bytes"], 0)
        self.assertEqual(metrics["target_forward_times"], 1)

    # ------------------------------------------------------------------
    # Existing constructor/strategy regressions (kept intact)
    # ------------------------------------------------------------------
    def test_plain_cuhlm_constructor_receives_runner_vocab_size(self):
        """Qwen 等 151936 词表不能被 CUHLM 默认的 32000 覆盖。"""
        self.instance.vocab_size = 151936
        self.instance.args.max_tokens = 0
        prefix = torch.tensor([[1]], dtype=torch.long)

        self._run(prefix)

        self.assertEqual(
            _FakeCommSimulator.instances[0].init_kwargs["vocab_size"], 151936
        )

    def test_precise_cuhlm_constructor_receives_runner_vocab_size(self):
        self.instance.vocab_size = 151936
        self.instance.args.max_tokens = 0
        prefix = torch.tensor([[1]], dtype=torch.long)

        with (
            patch("src.baselines.cuhlm.KVCacheModel", _FakeCache),
            patch("src.baselines.cuhlm.PreciseCUHLM", _FakeCommSimulator),
            patch("src.baselines.cuhlm.torch.cuda.Event", _FakeCudaEvent),
            patch(
                "src.baselines.cuhlm.torch.cuda.current_stream", return_value=None
            ),
            patch("src.baselines.cuhlm.torch.cuda.synchronize", return_value=None),
        ):
            self.instance.uncertainty_decoding(prefix, use_precise_comm_sim=True)

        self.assertEqual(
            _FakeCommSimulator.instances[0].init_kwargs["vocab_size"], 151936
        )

    def test_uncertainty_and_strategy_consume_the_same_real_draft_token(self):
        _FakeCache.draft_tokens = [1]
        _FakeCache.draft_rows = [[0.1, 0.9]]
        _FakeCache.target_rows = [[0.9, 0.1]]
        _FakeCommSimulator.transfer_flags = [True]
        prefix = torch.tensor([[1, 0]], dtype=torch.long)
        _output, _metrics, _caches, comm_simulator = self._run(prefix)

        self.assertEqual(comm_simulator.uncertainty_draft_tokens, [1])
        self.assertEqual(comm_simulator.strategy_calls, [(1.0, (2,), 1)])


if __name__ == "__main__":
    unittest.main()
