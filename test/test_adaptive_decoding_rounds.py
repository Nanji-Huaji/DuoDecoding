"""Public-entry CPU differential gate for the two-model adaptive decoder.

The oracle compiles ONLY the frozen ``git show a9a61cc:src/baselines/core.py``
adaptive_decoding AST (no registration or old-module execution). The checked-in
SHA-256-pinned method works in shallow clones, and intentionally cannot call
helpers introduced by the refactor. Both executions use fresh fixtures,
deterministic clocks and real torch RNG. The oracle is auxiliary orchestration
coverage on deterministic toy laws, NOT a same-seed trajectory contract: A3
intentionally removes discarded target samples. Its extra sample/generate events
and acceptance draw VALUES are excluded, but draw count/order, real forwards,
all metrics, billing, rollback, policy and stop/trace actions remain compared.
Target rollback requests (old ``n + 2`` vs new ``n + 1``) compare the actual
``min(end, before)`` truncations. Nondegenerate laws use independent assertions
rather than an obsolete RNG oracle; acceptance and residual sampling stay real.
"""

import ast
import builtins
import copy
import hashlib
import json
import math
import os
import tempfile
import textwrap
import unittest
from argparse import Namespace
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src import communication
from src.baselines import Baselines, core
from src.communication import CommunicationSimulator
from src.utils import sample

ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "test/golden/adaptive_decoding/baseline_a9a61cc.py.txt"
ORACLE_SHA256 = "66085f5510bf9f622691583fe1e617daf17e6823d49d72b7ccc850161faf8a52"
VOCAB = 4
ONE = (0.0, 1.0, 0.0, 0.0)
TWO = (0.0, 0.0, 1.0, 0.0)


@lru_cache(maxsize=1)
def _oracle_code():
    source = ORACLE_PATH.read_bytes()
    assert hashlib.sha256(source).hexdigest() == ORACLE_SHA256, "frozen oracle changed"
    module = ast.parse(textwrap.dedent(source.decode("utf-8")))
    assert len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef)
    method = copy.deepcopy(module.body[0])
    assert method.name == "adaptive_decoding"
    method.decorator_list = [
        d
        for d in method.decorator_list
        if not (
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and isinstance(d.func.value, ast.Name)
            and d.func.value.id == "Register"
        )
    ]
    assert len(method.decorator_list) == 1  # retain torch.no_grad()
    return compile(
        ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
        "<a9a61cc:adaptive_decoding>",
        "exec",
    )


def _oracle():
    # Separate locals avoid replacing the current core adaptive_decoding binding.
    namespace = {}
    exec(_oracle_code(), core.__dict__, namespace)
    return namespace["adaptive_decoding"]


def _plain(value):
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), value.detach().cpu().tolist())
    return value


@dataclass
class _Case:
    gamma: int = 3
    budget: int = 4
    topk: object = 2
    cap: int = 0
    mode: str = "per_transfer"
    residual: bool = False
    target_rows: dict = field(default_factory=dict)
    draft_rows: dict = field(default_factory=dict)
    rl_choices: tuple = ()
    frozen_rl: bool = False
    main_process: bool = True
    adapter_limit: object = None
    history_limit: dict = field(default_factory=dict)
    cache_start_lengths: dict = field(default_factory=dict)
    eos: int = -1
    early_stop: bool = False
    check_stopping: bool = False
    trace: str = "off"  # off, file, invalid
    reward_cost: str = "nominal"
    prompt: tuple = (0,)
    graph: bool = False


def _case(**overrides):
    """Pure case fixture shared with the independent contract tests."""
    return _Case(**overrides)


class _Decoder(Baselines):
    def load_data(self):
        return None

    def preprocess(self, value):
        return value

    def postprocess(self, input_text, output_text):
        return output_text

    def eval(self):
        return None

    def color_print(self, *args, **kwargs):
        pass

    def _stop_at_eos(self, prefix, prompt_len):
        result, hit = super()._stop_at_eos(prefix, prompt_len)
        self.actions.append(("eos", _plain(prefix), _plain(result), hit))
        return result, hit

    def _check_stopping_criteria(self, prefix, stop_sequences=None, prompt_len=None):
        result = super()._check_stopping_criteria(prefix, stop_sequences, prompt_len)
        self.actions.append(
            ("early_stop", _plain(prefix), tuple(stop_sequences or ()), result)
        )
        return result


class _Adapter:
    def __setattr__(self, name, value):
        if name == "threshold" and hasattr(self, "actions"):
            self.actions.append(("adapter.threshold", value))
        super().__setattr__(name, value)

    def __init__(self, actions, limit):
        self.actions = actions
        self.limit = limit
        self.threshold = 0.5
        self.step_acc_probs = []

    def reset_step(self):
        self.actions.append(("adapter.reset",))
        self.step_acc_probs = []

    def predict(self, hidden_states):
        self.step_acc_probs.append(0.875)
        stop = self.limit is not None and len(self.step_acc_probs) >= self.limit
        self.actions.append(("adapter.predict", len(self.step_acc_probs), stop))
        return stop


class _Policy:
    def __init__(self, actions, choices):
        self.actions = actions
        self.choices = choices
        self.calls = 0

    def select_config(self, bandwidth, latency, probs, entropy, task, training):
        choice = self.choices[min(self.calls, len(self.choices) - 1)]
        self.calls += 1
        self.actions.append(
            (
                "rl.select",
                bandwidth,
                latency,
                tuple(probs),
                entropy,
                task,
                training,
                choice,
            )
        )
        return choice

    def step(self, reward):
        self.actions.append(("rl.step", reward))

    def save(self, throughput):
        self.actions.append(("rl.save", throughput))


def _run(method, case, *, expect_error=None):
    """Run a decoder entry on scripted caches and a real CPU simulator.

    ``expect_error`` is used by the contract tests that assert a malformed
    proposal is rejected before RNG draws or communication effects; the actions
    are still returned so callers can prove nothing happened.
    """
    actions = []
    caches = []
    simulators = []
    real_multinomial = torch.multinomial
    real_rand = torch.rand
    real_open = builtins.open
    ticks = iter(range(100000))

    class Cache:
        def __init__(self, model, temperature, top_k, top_p, **kwargs):
            self.model = model
            self.kind = model.kind
            self.device = model.device
            self.vocab_size = VOCAB
            self.prob_history = torch.empty((1, 0, VOCAB))
            self.hidden_states = torch.zeros((1, 1, 1))
            self.current_length = case.cache_start_lengths.get(self.kind, 0)
            caches.append(self)
            actions.append(("cache.init", self.kind, temperature, top_k, top_p, kwargs))

        def _row(self, index):
            rows = case.draft_rows if self.kind == "draft" else case.target_rows
            return torch.tensor([rows.get(index, ONE)], dtype=torch.float32)

        def _forward_with_kvcache(self, prefix):
            assert not torch.is_grad_enabled()
            length = prefix.shape[1]
            before = self.prob_history.shape[1]
            if length > before:
                tail = torch.stack(
                    [self._row(i)[0] for i in range(before, length)], dim=0
                )
                self.prob_history = torch.cat(
                    (self.prob_history, tail.unsqueeze(0)), dim=1
                )
            self.current_length = length
            q = self._row(length - 1)
            limit = case.history_limit.get(self.kind)
            if limit is not None:
                self.prob_history = self.prob_history[:, :limit, :]
            actions.append(
                (
                    "cache.forward",
                    self.kind,
                    _plain(prefix),
                    before,
                    self.prob_history.shape[1],
                )
            )
            return q

        def forward_context(self, prefix):
            assert prefix.shape[1] >= self.current_length
            return self._forward_with_kvcache(prefix)

        def generate(self, prefix, gamma):
            actions.append(("cache.generate", self.kind, _plain(prefix), gamma))
            x = prefix.clone()
            for _ in range(gamma):
                q = self._forward_with_kvcache(x)
                x = torch.cat((x, sample(q)), dim=1)
            return x

        def rollback(self, end_pos):
            before = self.prob_history.shape[1]
            self.prob_history = self.prob_history[:, :end_pos, :]
            self.current_length = min(self.current_length, end_pos)
            actions.append(
                ("rollback", self.kind, end_pos, before, self.prob_history.shape[1])
            )

    class Simulator(CommunicationSimulator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.record_topk = True
            simulators.append(self)
            actions.append(("sim.init",))

        def __setattr__(self, name, value):
            if name == "transfer_top_k" and getattr(self, "record_topk", False):
                actions.append(("sim.topk", value))
            super().__setattr__(name, value)

        def set_round(self, index):
            actions.append(("round", index))
            return super().set_round(index)

        def flush_round(self):
            actions.append(("flush", self._round_idx))
            return super().flush_round()

        def transfer(
            self,
            tokens,
            prob,
            link_type,
            is_compressed=False,
            compressed_k=300,
            prob_bits=None,
        ):
            actions.append(
                (
                    "transfer",
                    self._round_idx,
                    _plain(tokens),
                    _plain(prob),
                    link_type,
                    is_compressed,
                    compressed_k,
                    prob_bits,
                )
            )
            return super().transfer(
                tokens, prob, link_type, is_compressed, compressed_k, prob_bits
            )

        def simulate_transfer(
            self, size, link_type, add_to_stats=True, topk=0, draft_len=0
        ):
            actions.append(
                ("payload", self._round_idx, link_type, size, topk, draft_len)
            )
            return super().simulate_transfer(
                size, link_type, add_to_stats, topk, draft_len
            )

        def _charge_transfer(
            self,
            size,
            link_type,
            add_to_stats=True,
            topk=0,
            draft_len=0,
            trace_kind="direct",
        ):
            actions.append(
                (
                    "charge",
                    self._round_idx,
                    link_type,
                    size,
                    topk,
                    draft_len,
                    trace_kind,
                )
            )
            return super()._charge_transfer(
                size, link_type, add_to_stats, topk, draft_len, trace_kind
            )

        def send_reject_message(self, linktype):
            actions.append(("reject", self._round_idx, linktype))
            return super().send_reject_message(linktype)

        def nominal_comm_s(self, link_type):
            value = super().nominal_comm_s(link_type)
            actions.append(("reward.cost", self._round_idx, link_type, value))
            return value

    class Event:
        def __init__(self, enable_timing=True):
            actions.append(("event.init", enable_timing))

        def record(self, stream=None):
            actions.append(("event.record",))

        def elapsed_time(self, other):
            actions.append(("event.elapsed",))
            return 17.0

    def clock():
        # Binary-exact step (1/128 s) so equal wall-time deltas have equal
        # float rewards whether the decoder takes one or two clock reads.
        tick = next(ticks)
        actions.append(("clock", tick))
        return tick / 128.0

    def multinomial(*args, **kwargs):
        result = real_multinomial(*args, **kwargs)
        actions.append(("sample", _plain(args[0]), _plain(result)))
        return result

    def rand(*args, **kwargs):
        result = real_rand(*args, **kwargs)
        actions.append(("rand", _plain(result)))
        return result

    decoder = object.__new__(_Decoder)
    decoder.actions = actions
    decoder.args = Namespace(
        max_tokens=case.budget,
        gamma=case.gamma,
        top_k=0,
        temp=1.0,
        top_p=0.0,
        edge_cloud_bandwidth=10.0,
        edge_end_bandwidth=10.0,
        cloud_end_bandwidth=10.0,
        batch_delay=0.125,
        min_bandwidth_mbps=0.0,
        comm_bw_model="instant",
        comm_round_trip_mode=case.mode,
        charge_residual_payload=case.residual,
        transfer_top_k_cap=case.cap,
        disable_rl_update=case.frozen_rl,
        rl_comm_cost_mode=case.reward_cost,
        use_cuda_graph=case.graph,
    )
    decoder.accelerator = SimpleNamespace(is_main_process=case.main_process)
    decoder.vocab_size = VOCAB
    decoder.num_acc_tokens = []
    decoder.draft_forward_times = decoder.target_forward_times = 0
    decoder.draft_model = SimpleNamespace(device=torch.device("cpu"), kind="draft")
    decoder.target_model = SimpleNamespace(device=torch.device("cpu"), kind="target")
    decoder.adapter = _Adapter(actions, case.adapter_limit)
    decoder.rl_adapter = _Policy(actions, case.rl_choices) if case.rl_choices else None
    decoder.little_rl_adapter = None
    decoder.task = "differential-cpu"
    decoder.tokenizer = SimpleNamespace(
        eos_token_id=case.eos,
        decode=lambda ids, **kwargs: "STOP" if case.early_stop else "continue",
    )

    with tempfile.TemporaryDirectory(prefix="adaptive-rounds-") as temp:
        trace_path = Path(temp) / (
            "absent/trace.jsonl" if case.trace == "invalid" else "trace.jsonl"
        )

        class TraceFile:
            def __init__(self, fh):
                self.fh = fh

            def __enter__(self):
                self.fh.__enter__()
                return self

            def __exit__(self, *exc):
                return self.fh.__exit__(*exc)

            def write(self, text):
                actions.append(("trace.write", json.loads(text)))
                return self.fh.write(text)

        def trace_open(file, *args, **kwargs):
            if os.fspath(file) == str(trace_path):
                actions.append(("trace.open", case.trace))
                return TraceFile(real_open(file, *args, **kwargs))
            return real_open(file, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {
                        "APPEND_TRACE": str(trace_path) if case.trace != "off" else "",
                        "COMM_TRACE": "",
                        "CACHE_CHECK_TRACE": "",
                        "ARP_CALIB_TRACE": "",
                        "TOPK_TRACE": "",
                    },
                )
            )
            stack.enter_context(patch("src.baselines.runtime.KVCacheModel", Cache))
            stack.enter_context(
                patch("src.baselines.simulators.CommunicationSimulator", Simulator)
            )
            stack.enter_context(patch.object(torch.cuda, "Event", Event))
            stack.enter_context(
                patch.object(torch.cuda, "current_stream", return_value=None)
            )
            stack.enter_context(
                patch.object(torch.cuda, "synchronize", return_value=None)
            )
            stack.enter_context(patch.object(core.time, "time", clock))
            stack.enter_context(patch.object(torch, "multinomial", multinomial))
            stack.enter_context(patch.object(torch, "rand", rand))
            stack.enter_context(patch("builtins.open", trace_open))
            stack.enter_context(
                patch.dict(
                    communication._NTT_TRACE_STATE,
                    {"data": [], "scale": 1.0, "src": ""},
                )
            )
            # Preserve ambient test-suite RNG while comparing the real end states.
            stack.enter_context(torch.random.fork_rng(devices=[]))
            torch.random.default_generator.manual_seed(8128)
            error = None
            try:
                tokens, metrics = method(
                    decoder,
                    torch.tensor([case.prompt], dtype=torch.long),
                    transfer_top_k=case.topk,
                    ntt_ms_edge_cloud=11.0,
                    use_early_stopping=case.early_stop or case.check_stopping,
                    stop_sequences=["STOP"]
                    if case.early_stop or case.check_stopping
                    else None,
                )
            except BaseException as exc:
                if expect_error is None:
                    raise
                tokens, metrics = None, None
                error = exc
            rng = torch.get_rng_state().clone()
        trace = trace_path.read_text() if trace_path.exists() else ""

    simulator = simulators[0]
    if expect_error is not None and not isinstance(error, expect_error):
        raise AssertionError(
            f"expected {expect_error.__name__}, got "
            f"{type(error).__name__ if error is not None else 'no exception'}: {error}"
        )
    return {
        "tokens": tokens,
        "metrics": metrics,
        "rng": rng,
        "actions": actions,
        "stats": copy.deepcopy(simulator.stats),
        "trace": trace,
        "accepted": decoder.num_acc_tokens,
        "error": error,
        "global_forwards": (decoder.draft_forward_times, decoder.target_forward_times),
        "caches": [(c.kind, c.current_length, _plain(c.prob_history)) for c in caches],
        "pending": copy.deepcopy(simulator._pending_bytes),
    }


def _normalize_actions(actions):
    """Compare protocol effects, not the intentionally corrected RNG trajectory.

    Only the frozen oracle's discarded target sample is removed. Genuine forward
    lengths, sampling laws/results, acceptance draw count/order, communication,
    rewards and stop/trace events remain observable. These comparisons use delta
    laws, so different uniform values cannot alter their acceptance decisions.
    """
    normalized = []
    discarded_target_sample = False
    for action in actions:
        if action[0] == "clock":
            continue
        if action[:2] == ("cache.generate", "target"):
            assert action[-1] == 1 and not discarded_target_sample
            discarded_target_sample = True
            continue
        if action[0] == "sample" and discarded_target_sample:
            discarded_target_sample = False
            continue
        if action[0] == "rand":
            normalized.append(("rand", action[1][:2]))  # dtype/shape, not RNG value
        elif action[0] == "rollback":
            kind, requested, before, after = action[1:]
            normalized.append(("rollback", kind, min(requested, before), before, after))
        else:
            normalized.append(action)
    assert not discarded_target_sample
    return normalized


class AdaptiveDecodingRoundsTests(unittest.TestCase):
    def assertDifferential(self, case):
        old = _run(_oracle(), case)
        new = _run(Baselines.adaptive_decoding, case)
        self.assertTrue(torch.equal(old["tokens"], new["tokens"]))
        old_rest = {k: v for k, v in old.items() if k not in ("tokens", "rng")}
        new_rest = {k: v for k, v in new.items() if k not in ("tokens", "rng")}
        old_rest["actions"] = _normalize_actions(old_rest["actions"])
        new_rest["actions"] = _normalize_actions(new_rest["actions"])
        self.assertEqual(old_rest, new_rest)
        # Includes ALL timing metrics: no whitelist that could hide metric drift.
        return new

    def test_public_context_forward_never_draws_a_discarded_target_sample(self):
        for budget, expected_samples in ((1, 1), (4, 4), (9, 9)):
            with self.subTest(budget=budget):
                result = _run(Baselines.adaptive_decoding, _Case(budget=budget))
                samples = [a for a in result["actions"] if a[0] == "sample"]
                self.assertEqual(len(samples), expected_samples)
                self.assertFalse(any(
                    a[0] == "cache.generate" for a in result["actions"]
                ))
                self.assertEqual(result["tokens"].tolist(), [[0] + [1] * budget])

    def test_full_accept_public_entry(self):
        result = self.assertDifferential(_Case())
        self.assertEqual(result["metrics"]["draft_accepted_tokens"], 3)
        self.assertEqual(result["accepted"], [3])
        self.assertEqual(len([a for a in result["actions"] if a[0] == "rand"]), 3)

    def test_full_accept_multiple_rounds_and_target_only_tail(self):
        for mode in ("per_transfer", "per_round"):
            with self.subTest(mode=mode):
                result = self.assertDifferential(_Case(budget=9, mode=mode, graph=True))
                self.assertEqual(result["accepted"], [3, 3, 1])
                self.assertEqual(result["metrics"]["generated_tokens"], 9)
                self.assertEqual(result["metrics"]["draft_generated_tokens"], 6)
                self.assertEqual(result["metrics"]["avg_top_k"], 2)
                self.assertEqual(result["pending"], {})
                actions = result["actions"]
                tail = actions[actions.index(("round", 3)) + 1 :]
                self.assertFalse(
                    any(
                        a[0]
                        in (
                            "rollback",
                            "transfer",
                            "payload",
                            "rand",
                            "adapter.reset",
                            "rl.select",
                        )
                        for a in tail
                    )
                )
                target_forwards = [
                    a for a in tail if a[:2] == ("cache.forward", "target")
                ]
                self.assertEqual(len(target_forwards), 1)
                self.assertEqual(
                    target_forwards[0][2], _plain(result["tokens"][:, :-1])
                )
                self.assertEqual(len([a for a in tail if a[0] == "sample"]), 1)
                self.assertFalse(any(a[0] == "cache.generate" for a in tail))
                self.assertEqual(result["global_forwards"], (6, 3))

    def test_cache_length_equal_to_prefix_is_a_legal_prefilled_hit(self):
        # A full-prefix cache is a legitimate hit: the entry guard must allow
        # current_length == prefix_len (only strictly ahead is stale state).
        result = self.assertDifferential(
            _Case(
                cache_start_lengths={"draft": 1, "target": 1},
            )
        )
        self.assertEqual(result["accepted"], [3])
        self.assertEqual(result["metrics"]["generated_tokens"], 4)
        self.assertEqual(result["tokens"].shape[1], 5)

    def test_all_accept_target_rolls_back_to_forwarded_length_and_reforwards_bonus(
        self,
    ):
        result = _run(Baselines.adaptive_decoding, _case(budget=9))
        actions = result["actions"]
        target_rollbacks = [
            a for a in actions if a[0] == "rollback" and a[1] == "target"
        ]
        self.assertEqual(len(target_rollbacks), 2)
        for action in target_rollbacks:
            # The request is exactly n + 1, which for a full accept equals the
            # genuinely forwarded history length; it never asks past the cache.
            self.assertEqual(action[2], action[3])
            self.assertEqual(action[4], action[3])
        # The round-2 target forward starts after the accepted prefix and thus
        # must feed the bonus token sampled at the end of round 1 again: that
        # token was never forwarded by the target (rollback kept n + 1 rows).
        round2 = actions.index(("round", 2))
        bonus = next(a for a in reversed(actions[:round2]) if a[0] == "sample")[2][2][
            0
        ][0]
        target_forward = next(
            a for a in actions[round2:] if a[0] == "cache.forward" and a[1] == "target"
        )
        self.assertEqual(target_forward[3], target_rollbacks[0][2])
        self.assertEqual(target_forward[2][2][0][target_rollbacks[0][2]], bonus)

    def test_first_and_later_rejection_stop_consuming_acceptance_rng(self):
        for row, accepted in ((0, 0), (1, 1), (2, 2)):
            for mode in ("per_transfer", "per_round"):
                with self.subTest(row=row, mode=mode):
                    result = self.assertDifferential(
                        _Case(
                            target_rows={row: TWO},
                            eos=2,
                            mode=mode,
                            residual=True,
                        )
                    )
                    actions = result["actions"]
                    self.assertEqual(result["accepted"], [accepted])
                    self.assertEqual(
                        len([a for a in actions if a[0] == "rand"]), row + 1
                    )
                    self.assertEqual(result["tokens"].tolist(), [[0] + [1] * row + [2]])
                    self.assertEqual(
                        [a[2] for a in actions if a[0] == "rollback"],
                        [row + 1, row + 1],
                    )
                    reject = next(i for i, a in enumerate(actions) if a[0] == "reject")
                    self.assertFalse(any(a[0] == "rand" for a in actions[reject + 1 :]))
                    # Residual row really samples token 2, not a mocked constant.
                    residual_samples = [a for a in actions[reject:] if a[0] == "sample"]
                    self.assertEqual(residual_samples[0][1][2], [[0.0, 0.0, 1.0, 0.0]])
                    self.assertEqual(result["pending"], {})

    def test_rejection_then_next_round_uses_truncated_cache(self):
        for row in (0, 1):
            with self.subTest(row=row):
                result = self.assertDifferential(
                    _Case(target_rows={row: TWO}, budget=8)
                )
                self.assertEqual(result["metrics"]["generated_tokens"], 8)
                rollbacks = [a for a in result["actions"] if a[0] == "rollback"]
                self.assertEqual(rollbacks[0][1:3], ("draft", row + 1))
                self.assertEqual(rollbacks[0][4], row + 1)
                self.assertGreater(rollbacks[0][3], rollbacks[0][4])
                forwards = [
                    a
                    for a in result["actions"]
                    if a[0] == "cache.forward" and a[1] == "draft"
                ]
                # Round 2 observes the genuinely truncated history.
                self.assertEqual(forwards[3][3], row + 1)
                self.assertGreater(len(result["accepted"]), 1)

    def test_rl_cap_changes_reject_payload_and_next_proposal_not_current_or_simulator(
        self,
    ):
        for mode in ("per_transfer", "per_round"):
            for residual in (False, True):
                for cost in ("nominal", "off"):
                    with self.subTest(mode=mode, residual=residual, cost=cost):
                        result = self.assertDifferential(
                            _Case(
                                topk=1,
                                cap=3,
                                budget=5,
                                target_rows={0: TWO},
                                rl_choices=((99, 0.8), (2, 0.4)),
                                mode=mode,
                                residual=residual,
                                reward_cost=cost,
                            )
                        )
                        actions = result["actions"]
                        self.assertEqual(
                            [a for a in actions if a[0] == "sim.topk"],
                            [("sim.topk", 1)],
                        )
                        self.assertEqual(result["accepted"], [0, 3])
                        self.assertEqual(
                            result["metrics"]["avg_top_k"], 2.0
                        )  # (1 + 3)/2, before RL
                        rejection_payload = next(
                            a for a in actions if a[0] == "transfer" and a[2] is None
                        )
                        self.assertEqual(rejection_payload[5:7], (True, 3))
                        reward = next(
                            i for i, a in enumerate(actions) if a[0] == "rl.step"
                        )
                        rollback = next(
                            i for i, a in enumerate(actions) if a[0] == "rollback"
                        )
                        payload = actions.index(rejection_payload)
                        resample = next(
                            i
                            for i in range(payload + 1, len(actions))
                            if actions[i][0] == "sample"
                        )
                        self.assertLess(reward, rollback)
                        self.assertLess(rollback, payload)
                        self.assertLess(payload, resample)
                        self.assertEqual(
                            actions[reward][1], 0.0
                        )  # gamma>1 and zero accepted
                        nominal = [a for a in actions if a[0] == "reward.cost"]
                        self.assertEqual(len(nominal), 2 if cost == "nominal" else 0)
                        round1_payloads = [
                            a[3] for a in actions if a[0] == "payload" and a[1] == 1
                        ]
                        self.assertEqual(
                            round1_payloads,
                            [12, 6, 24] + ([4] if residual else []) + [12],
                        )

    def test_rl_uses_previous_topk_for_this_sampling_and_selected_topk_next_round(self):
        row = (0.125, 0.5, 0.25, 0.125)
        # Independent distribution/policy assertions: changed RNG consumption
        # intentionally makes the frozen oracle's same-seed trajectory obsolete.
        result = _run(
            Baselines.adaptive_decoding,
            _Case(
                draft_rows={i: row for i in range(20)},
                target_rows={i: row for i in range(20)},
                topk=1,
                cap=3,
                budget=8,
                adapter_limit=1,
                rl_choices=((99, 0.8),),
            )
        )
        actions = result["actions"]
        draft_samples = {}
        current_round = None
        for i, action in enumerate(actions):
            if action[0] == "round":
                current_round = action[1]
            if action[0] == "cache.forward" and action[1] == "draft":
                self.assertEqual(actions[i + 1][0], "sample")
                draft_samples.setdefault(current_round, []).append(
                    actions[i + 1][1][2][0]
                )
        self.assertGreater(len(draft_samples), 1)
        for actual, expected in zip(
            draft_samples[1][0], (1 / 6, 0.5, 1 / 6, 1 / 6), strict=True
        ):
            self.assertAlmostEqual(actual, expected, places=6)
        for probs in draft_samples[2]:
            for actual, expected in zip(probs, row, strict=True):
                self.assertAlmostEqual(actual, expected, places=6)
        selects = [a for a in actions if a[0] == "rl.select"]
        self.assertAlmostEqual(
            result["metrics"]["avg_top_k"], (1 + 3 * (len(selects) - 1)) / len(selects)
        )
        self.assertTrue(any(a[0] == "rand" and 0 < a[1][2][0] < 1 for a in actions))

    def test_initial_cap_and_rl_uncompressed_selection_keep_distinct_semantics(self):
        for initial in (None, 0, -1, 99):
            with self.subTest(initial=initial):
                result = self.assertDifferential(_Case(topk=initial, cap=2))
                self.assertEqual(result["metrics"]["avg_top_k"], 2)
                self.assertEqual(
                    [a for a in result["actions"] if a[0] == "sim.topk"],
                    [("sim.topk", 2)],
                )
        for selected in (None, 0, -1):
            with self.subTest(selected=selected):
                result = self.assertDifferential(
                    _Case(
                        topk=2,
                        cap=2,
                        budget=5,
                        target_rows={0: TWO},
                        rl_choices=((selected, 0.7),),
                        residual=True,
                    )
                )
                self.assertEqual(result["metrics"]["avg_top_k"], 1)
                payload = next(
                    a for a in result["actions"] if a[0] == "transfer" and a[2] is None
                )
                self.assertEqual(payload[5:7], (False, selected))
                self.assertEqual(
                    [a for a in result["actions"] if a[0] == "sim.topk"],
                    [("sim.topk", 2)],
                )

    def test_positive_reward_prices_only_pre_resample_communication(self):
        for mode in ("per_transfer", "per_round"):
            for cost in ("nominal", "off"):
                with self.subTest(mode=mode, cost=cost):
                    result = self.assertDifferential(
                        _Case(
                            budget=2,
                            adapter_limit=1,
                            mode=mode,
                            reward_cost=cost,
                            rl_choices=((2, 0.5),),
                        )
                    )
                    # Clock delta=1/128s (binary exact). At reward time exactly
                    # one 12B token+prob has crossed the 10Mbps link, plus the
                    # fixed 11ms NTT.
                    comm = 0.011 + 12 / 1250000 if cost == "nominal" else 0
                    expected = math.exp(min(1 / (1 / 128 + comm + 1e-9), 100) / 20)
                    reward = next(a[1] for a in result["actions"] if a[0] == "rl.step")
                    self.assertAlmostEqual(reward, expected, places=12)

    def test_disable_rl_update_and_nonmain_process_preserve_observation_not_updates(
        self,
    ):
        for frozen in (False, True):
            for main in (False, True):
                with self.subTest(frozen=frozen, main=main):
                    result = self.assertDifferential(
                        _Case(
                            rl_choices=((3, 0.7),),
                            frozen_rl=frozen,
                            main_process=main,
                        )
                    )
                    selects = [a for a in result["actions"] if a[0] == "rl.select"]
                    self.assertEqual(len(selects), 1)
                    self.assertEqual(selects[0][-2], not frozen)
                    updates = [a for a in result["actions"] if a[0] == "rl.step"]
                    saves = [a for a in result["actions"] if a[0] == "rl.save"]
                    self.assertEqual(len(updates), 0 if frozen else 1)
                    self.assertEqual(len(saves), 0 if frozen else 1)
                    self.assertEqual(
                        result["global_forwards"], (3, 1) if main else (0, 0)
                    )
                    self.assertEqual(result["metrics"]["draft_forward_times"], 3)
                    self.assertEqual(result["metrics"]["target_forward_times"], 1)

    def test_budget_zero_one_and_nonpositive_gamma_have_no_rollback_or_downlink(self):
        for budget, gamma in ((0, 3), (1, 3), (5, 0), (5, -1)):
            for mode in ("per_transfer", "per_round"):
                with self.subTest(budget=budget, gamma=gamma, mode=mode):
                    result = self.assertDifferential(
                        _Case(
                            budget=budget,
                            gamma=gamma,
                            mode=mode,
                            trace="file",
                            rl_choices=((2, 0.5),),
                        )
                    )
                    actions = result["actions"]
                    self.assertEqual(
                        result["metrics"]["generated_tokens"], 0 if budget == 0 else 1
                    )
                    self.assertEqual(result["accepted"], [] if budget == 0 else [1])
                    self.assertEqual(result["metrics"]["avg_top_k"], 0)
                    self.assertFalse(
                        any(
                            a[0]
                            in (
                                "rollback",
                                "rand",
                                "payload",
                                "transfer",
                                "rl.select",
                                "rl.step",
                                "trace.open",
                                "early_stop",
                            )
                            for a in actions
                        )
                    )
                    self.assertEqual(result["trace"], "")

    def test_adapter_shortened_proposals_account_actual_not_requested_gamma(self):
        result = self.assertDifferential(_Case(adapter_limit=1, budget=5, trace="file"))
        self.assertEqual(result["accepted"], [1, 1, 1])
        self.assertEqual(result["metrics"]["avg_draft_len"], 1)
        self.assertEqual(result["metrics"]["draft_generated_tokens"], 2)
        self.assertEqual(
            [json.loads(line)["gamma"] for line in result["trace"].splitlines()], [1, 1]
        )

    def test_append_eos_trace_earlystop_downlink_order_and_invalid_trace_is_nonfatal(
        self,
    ):
        for trace in ("file", "invalid"):
            for early in (False, True):
                with self.subTest(trace=trace, early=early):
                    result = self.assertDifferential(
                        _Case(
                            trace=trace,
                            early_stop=early,
                            check_stopping=True,
                        )
                    )
                    actions = result["actions"]
                    eos = next(i for i, a in enumerate(actions) if a[0] == "eos")
                    opened = next(
                        i for i, a in enumerate(actions) if a[0] == "trace.open"
                    )
                    self.assertLess(eos, opened)
                    self.assertEqual(
                        actions[eos][1][1], (1, 5)
                    )  # bonus was appended first
                    stopped = next(
                        i for i, a in enumerate(actions) if a[0] == "early_stop"
                    )
                    self.assertLess(opened, stopped)
                    self.assertEqual(actions[stopped][-1], early)
                    if early:
                        self.assertFalse(
                            any(a[0] == "payload" for a in actions[stopped + 1 :])
                        )
                    else:
                        self.assertTrue(
                            any(a[0] == "payload" for a in actions[stopped + 1 :])
                        )
                    if trace == "file":
                        self.assertEqual(
                            json.loads(result["trace"]),
                            {
                                "iter": 1,
                                "prefix_len_before": 1,
                                "gamma": 3,
                                "n": 3,
                                "accepted": 3,
                                "appended_token": 1,
                                "prefix_len_after": 5,
                            },
                        )
                    else:
                        self.assertEqual(result["trace"], "")
                        self.assertFalse(any(a[0] == "trace.write" for a in actions))

    def test_eos_in_draft_or_bonus_skips_trace_earlystop_and_downlink(self):
        for target_rows, eos, prompt, expected in (
            ({}, 1, (1, 0), [[1, 0, 1]]),
            ({3: TWO}, 2, (0,), [[0, 1, 1, 1, 2]]),
        ):
            with self.subTest(target_rows=target_rows):
                result = self.assertDifferential(
                    _Case(
                        target_rows=target_rows,
                        eos=eos,
                        prompt=prompt,
                        trace="file",
                        early_stop=True,
                        mode="per_round",
                    )
                )
                self.assertEqual(result["tokens"].tolist(), expected)
                actions = result["actions"]
                first_eos = next(i for i, a in enumerate(actions) if a[0] == "eos")
                self.assertTrue(actions[first_eos][-1])
                self.assertFalse(
                    any(
                        a[0] in ("trace.open", "early_stop", "payload")
                        for a in actions[first_eos + 1 :]
                    )
                )
                self.assertEqual(result["trace"], "")

    def test_truncated_target_history_raises_before_rand_or_transfer(self):
        # New contract: the target must cover proposal.tokens in full, including
        # the all-accept bonus row. A short target history is a programming
        # error, not a silent end-of-draft accept.
        for limit in (0, 1, 2):
            with self.subTest(limit=limit):
                result = _run(
                    Baselines.adaptive_decoding,
                    _case(history_limit={"target": limit}),
                    expect_error=ValueError,
                )
                actions = result["actions"]
                self.assertIsInstance(result["error"], ValueError)
                self.assertFalse(any(a[0] == "rand" for a in actions))
                self.assertFalse(any(a[0] == "transfer" for a in actions))

    def test_short_raw_draft_cache_does_not_reject_complete_proposal(self):
        # New contract: rejection ratios/payloads use proposal.probabilities
        # (the q actually sampled from), not the raw draft cache length. A
        # rolled-back/short raw cache must not manufacture a reject.
        result = _run(
            Baselines.adaptive_decoding,
            _case(history_limit={"draft": 0}),
        )
        actions = result["actions"]
        self.assertEqual(result["accepted"], [3])
        self.assertEqual(result["tokens"].tolist(), [[0, 1, 1, 1, 1]])
        self.assertFalse(any(a[0] == "reject" for a in actions))
        self.assertEqual(len([a for a in actions if a[0] == "rand"]), 3)
        self.assertEqual(len([a for a in actions if a[0] == "transfer"]), 3)


if __name__ == "__main__":
    unittest.main()
