"""Adaptive decoding must not sample from cloned per-role default RNG streams.

The bug: the adaptive proposal sampled its draft token from the proposal cache's
default stream (through the shared adapter loop in ``operations``), acceptance
sampled ``torch.rand`` on the draft device, and the all-accepted bonus and the
target-only tail sampled the target cache's default stream.  ``torch.manual_seed``
seeds every CUDA default generator identically, so two draws that happen on
*different* logical devices but at the same stream offset are the same draw.  For
a uniform iid law with ``gamma=1`` the first proposal and the first target bonus
can therefore be one cloned categorical variate: their joint law collapses onto
the diagonal instead of being the product of two iid draws.

These tests are CPU-only.  They exercise the real public entry
``Baselines.adaptive_decoding`` with a forwarded-cache adapter, real
``sampling_weights`` preprocessing and communication/event test adapters.
One test additionally uses real tiny CPU Llama models, caches and communication;
the other model forwards are fakes. CUDA timing is replaced in all cases, so
this is not a GPU acceptance test.  The native dependency seam is ``torch.multinomial``
and ``torch.rand``.  No private RNG field, private adaptive helper signature or
``sample`` hook is inspected: the tests only require that every token/acceptance
draw asks Torch for an explicit non-``None`` generator and that one invocation
reuses one CPU generator.

The exact-law regression substitutes a deterministic tape for the RNG output.
On the pre-fix code an explicit generator is absent, so the tape models the
cloned CUDA default streams: the draft/default tape and the target/default tape
are equal clones that each start at offset zero.  On the fixed code the single
explicit per-invocation CPU generator owns one advancing tape.  This models only
the *dependency structure* of the draws (equal-ordinal cloned variates); it does
not assume anything about CPU-vs-CUDA Philox uniform/exponential alignment.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 4
CPU_DEVICE = torch.device("cpu")
UNIFORM_ROW = (0.25, 0.25, 0.25, 0.25)
UNIFORM_SCRIPT = {"draft": [UNIFORM_ROW], "target": [UNIFORM_ROW]}


def _load_cee_fixture():
    """Load the existing forwarded-cache/entry fixtures by file location.

    The ``test`` directory is not a package and ``test/test_cee_refactor.py`` is
    shadowed by ``test.py`` at the repository root, so a private importlib name is
    the stable form.  Only fixture classes for the public entry wiring and the
    realistic forwarded cache are reused; no test case is collected from it.
    """
    path = ROOT / "test" / "test_cee_refactor.py"
    spec = importlib.util.spec_from_file_location("adaptive_rng_cee_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cee = _load_cee_fixture()


class _ScriptedCache:
    """Constant/scripted-law forwarded cache used by the adaptive entry.

    It exposes the cache surface adaptive consumes: ``forward_context`` returns
    the next-position model law and keeps a position-indexed ``prob_history``;
    ``rollback`` truncates it.  ``script[kind][index]`` is the law for position
    ``index`` (the last row repeats), so a test can make proposal and target laws
    agree (always accept) or disagree (forced rejection and residual draw).
    """

    instances = []
    script = {}
    # Which cache most recently forwarded; the cloned-default tape substitute
    # needs the logical device of a draw that carries no explicit generator.
    phase = None

    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        self.model = model
        self.kind = model.kind
        self.device = model.device
        self.vocab_size = VOCAB
        self.prob_history = torch.empty((1, 0, VOCAB))
        # Adapter.predict consumes a hidden-state placeholder only.
        self.hidden_states = torch.zeros((1, 1, 1))
        self.current_length = 0
        self.rollback_calls = []
        type(self).instances.append(self)

    def _law(self, index):
        rows = type(self).script[self.kind]
        return torch.tensor([rows[min(index, len(rows) - 1)]], dtype=torch.float32)

    def forward_context(self, prefix):
        type(self).phase = self.kind
        self.current_length = prefix.shape[1]
        self.prob_history = torch.cat(
            [self._law(index) for index in range(prefix.shape[1])], dim=0
        ).unsqueeze(0)
        return self.prob_history[:, -1, :]

    def rollback(self, end_pos):
        self.prob_history = self.prob_history[:, :end_pos, :]
        self.current_length = min(self.current_length, end_pos)
        self.rollback_calls.append(end_pos)


class _ClonedDefaultsTape:
    """Deterministic native dependency tape for the two token draws.

    ``proposal`` is the first uniform of the proposal's logical default stream and
    ``bonus`` is the first uniform of the target's logical default stream.  Pre-fix
    both streams are clones seeded identically, so the bonus restarts at the
    proposal's first value.  A draw carrying an explicit generator instead keys
    one shared advancing tape for the invocation: proposal, acceptance, bonus.
    """

    def __init__(self, proposal, bonus, acceptance=0.5):
        self._shared = [proposal, acceptance, bonus, acceptance]
        self._defaults = {
            ("default", "draft"): list(self._shared),
            ("default", "target"): list(self._shared),
        }
        self._offsets = {}

    def next_value(self, key):
        offset = self._offsets.get(key, 0)
        self._offsets[key] = offset + 1
        tape = self._defaults.get(key, self._shared)
        return tape[min(offset, len(tape) - 1)]


class AdaptiveRngTestBase(unittest.TestCase):
    def setUp(self):
        cee._FakeCommSimulator.instances = []
        _ScriptedCache.instances = []
        _ScriptedCache.script = {}
        _ScriptedCache.phase = None
        self.draws = []

    def _instance(self, *, max_tokens, gamma, tokenizer=None):
        instance = cee.CeeRefactorTests()._make_instance("adaptive_decoding")
        instance.args.max_tokens = max_tokens
        instance.args.gamma = gamma
        instance.args.gamma1 = gamma
        instance.args.gamma2 = gamma
        instance.adapter = cee._FakeAdapter()
        instance.color_print = lambda *args, **kwargs: None
        if tokenizer is not None:
            instance.tokenizer = tokenizer
        return instance

    def _run(self, *, script, max_tokens, gamma=1, transfer_top_k=None,
             tokenizer=None, torch_patches, fork_rng=True):
        """Run the public adaptive entry on scripted caches and record draws.

        ``fork_rng`` keeps a decode's real draws from perturbing the ambient
        suite RNG.  The zero-budget test disables it so its before/after default
        CPU state comparison is meaningful.
        """
        self.draws = []
        _ScriptedCache.instances = []
        _ScriptedCache.script = script
        _ScriptedCache.phase = None
        instance = self._instance(
            max_tokens=max_tokens, gamma=gamma, tokenizer=tokenizer
        )
        with ExitStack() as stack:
            for target, value in _ENTRY_PATCHES:
                stack.enter_context(patch(target, value))
            for target, value in torch_patches:
                stack.enter_context(patch(target, value))
            if fork_rng:
                stack.enter_context(torch.random.fork_rng(devices=[]))
            output, metrics = instance.adaptive_decoding(
                torch.tensor([[0]], dtype=torch.long),
                transfer_top_k=transfer_top_k,
            )
        return output, metrics

    # -- arm the RNG seam either with the real functions or with the tape --

    def _real_rng_patches(self):
        original_multinomial = torch.multinomial
        original_rand = torch.rand

        def multinomial(weights, *args, **kwargs):
            self.draws.append(
                ("multinomial", weights.detach().clone(), kwargs.get("generator"))
            )
            return original_multinomial(weights, *args, **kwargs)

        def rand(*args, **kwargs):
            self.draws.append(("rand", None, kwargs.get("generator")))
            return original_rand(*args, **kwargs)

        return [("torch.multinomial", multinomial), ("torch.rand", rand)]

    def _tape_patches(self, tape):
        def multinomial(weights, *args, num_samples=1, generator=None, **kwargs):
            role = _ScriptedCache.phase or "target"
            key = generator if generator is not None else ("default", role)
            self.draws.append(("multinomial", weights.detach().clone(), generator))
            thresholds = torch.full(
                (weights.shape[0], num_samples),
                tape.next_value(key),
                dtype=weights.dtype,
            )
            return torch.searchsorted(weights.cumsum(-1), thresholds, right=True)

        def rand(*args, device=None, generator=None, **kwargs):
            key = generator if generator is not None else ("default", "draft")
            self.draws.append(("rand", None, generator))
            shape = args[0] if args else kwargs.get("size", ())
            size = (shape,) if isinstance(shape, int) else tuple(shape)
            return torch.full(
                size,
                tape.next_value(key),
                dtype=torch.float32,
                device=device,
            )

        return [("torch.multinomial", multinomial), ("torch.rand", rand)]

    # -- assertions shared by the provenance tests --

    def _assert_explicit_private_generator(self, expected_ops):
        self.assertEqual([op for op, _, _ in self.draws], expected_ops)
        generators = [generator for _, _, generator in self.draws]
        self.assertTrue(
            all(generator is not None for generator in generators),
            "every token/acceptance draw must request an explicit generator; "
            f"got {generators!r}",
        )
        self.assertEqual(
            len({id(generator) for generator in generators}),
            1,
            "one decode invocation on one actual device must reuse one generator",
        )
        self.assertEqual(generators[0].device, CPU_DEVICE)

    def _token_weights(self):
        return [weights for op, weights, _ in self.draws if op == "multinomial"]

    def _assert_law(self, actual, row):
        torch.testing.assert_close(
            actual,
            torch.tensor([list(row)], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-6,
        )


_ENTRY_PATCHES = (
    ("src.baselines.core.KVCacheModel", _ScriptedCache),
    ("src.baselines.runtime.KVCacheModel", _ScriptedCache),
    ("src.baselines.simulators.CommunicationSimulator", cee._FakeCommSimulator),
    (
        "src.baselines.simulators.PreciseCommunicationSimulator",
        cee._FakeCommSimulator,
    ),
    ("torch.cuda.Event", cee._FakeCudaEvent),
    ("torch.cuda.current_stream", lambda *args, **kwargs: None),
    ("torch.cuda.synchronize", lambda *args, **kwargs: None),
)


class AdaptiveRngLawTests(AdaptiveRngTestBase):
    def test_proposal_and_bonus_pair_law_is_not_cloned_defaults(self):
        """Full 8x8 grid of two categorical variates, Q=P=uniform V4.

        With matching proposal/target laws every acceptance is certain, so the
        only randomness in the emitted pair is the proposal draw and the
        all-accepted bonus draw.  Each grid cell must appear exactly four times
        (2 proposal coordinates x 2 bonus coordinates).  Pre-fix, the bonus
        restarts the proposal's default tape, so only the four diagonal cells
        appear - a duplicated variate, not two iid draws.
        """
        grid = 8
        observed = Counter()
        for proposal_index in range(grid):
            for bonus_index in range(grid):
                tape = _ClonedDefaultsTape(
                    proposal=(proposal_index + 0.5) / grid,
                    bonus=(bonus_index + 0.5) / grid,
                )
                output, _ = self._run(
                    script=UNIFORM_SCRIPT,
                    max_tokens=2,
                    torch_patches=self._tape_patches(tape),
                )
                observed[(int(output[0, 1]), int(output[0, 2]))] += 1
        expected = Counter(
            {
                (first, second): 4
                for first in range(VOCAB)
                for second in range(VOCAB)
            }
        )
        self.assertEqual(
            dict(observed),
            dict(expected),
            "proposal/bonus joint law collapsed away from the uniform 8x8 grid",
        )


class AdaptiveRngProvenanceTests(AdaptiveRngTestBase):
    def test_proposal_acceptance_and_bonus_use_one_private_cpu_generator(self):
        """All-accepted gamma=1 step: proposal q, acceptance, target bonus."""
        output, metrics = self._run(
            script=UNIFORM_SCRIPT,
            max_tokens=2,
            torch_patches=self._real_rng_patches(),
        )

        self.assertEqual(output.shape[1], 3)  # prompt + proposal + bonus
        self.assertEqual(metrics["generated_tokens"], 2)
        self._assert_explicit_private_generator(
            ["multinomial", "rand", "multinomial"]
        )
        # Proposal and bonus keep the native normalized sampling_weights row.
        self.assertEqual(len(self._token_weights()), 2)
        for weights in self._token_weights():
            self._assert_law(weights, UNIFORM_ROW)

    def test_rejection_residual_draw_uses_private_generator(self):
        """Forced rejection resamples norm((P-Q)+) with an explicit generator."""
        script = {
            "draft": [(1.0, 0.0, 0.0, 0.0)],
            "target": [(0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)],
        }
        output, metrics = self._run(
            script=script,
            max_tokens=2,
            tokenizer=SimpleNamespace(eos_token_id=1),
            torch_patches=self._real_rng_patches(),
        )

        self.assertEqual(output.tolist()[0][1:], [1])
        self.assertEqual(metrics["generated_tokens"], 1)
        # Proposal draw, acceptance draw, residual draw.  The residual token is
        # EOS, so the loop stops before the target-only tail.
        self._assert_explicit_private_generator(
            ["multinomial", "rand", "multinomial"]
        )
        proposal_weights, residual_weights = self._token_weights()
        self._assert_law(proposal_weights, (1.0, 0.0, 0.0, 0.0))
        self._assert_law(residual_weights, (0.0, 1.0, 0.0, 0.0))

    def test_target_only_tail_draw_uses_private_generator(self):
        """Zero draft budget: only the target tail samples, still explicitly."""
        script = {
            "draft": [UNIFORM_ROW],
            "target": [(0.1, 0.2, 0.3, 0.4)],
        }
        output, metrics = self._run(
            script=script,
            max_tokens=1,
            torch_patches=self._real_rng_patches(),
        )

        self.assertEqual(output.shape[1], 2)
        self.assertEqual(metrics["generated_tokens"], 1)
        self._assert_explicit_private_generator(["multinomial"])
        self._assert_law(self._token_weights()[0], (0.1, 0.2, 0.3, 0.4))

    def test_multiple_rounds_and_tail_reuse_the_same_private_generator(self):
        """Two accepted rounds and a target-only tail share one invocation stream."""
        output, metrics = self._run(
            script=UNIFORM_SCRIPT,
            max_tokens=5,
            torch_patches=self._real_rng_patches(),
        )
        self.assertEqual(output.shape[1], 6)
        self.assertEqual(metrics["generated_tokens"], 5)
        self._assert_explicit_private_generator(
            ["multinomial", "rand", "multinomial"] * 2 + ["multinomial"]
        )
        for weights in self._token_weights():
            self._assert_law(weights, UNIFORM_ROW)

    def test_real_tiny_cpu_models_and_caches_use_the_private_stream(self):
        """No scripted probability history: real model/cache, fake CUDA timing only."""
        from transformers import LlamaConfig, LlamaForCausalLM

        with torch.random.fork_rng(devices=[]):
            runtime = self._instance(max_tokens=5, gamma=1)
            config = LlamaConfig(
                vocab_size=VOCAB, hidden_size=8, intermediate_size=16,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=32, output_hidden_states=True,
            )
            runtime.draft_model = LlamaForCausalLM(config).eval()
            runtime.target_model = LlamaForCausalLM(config).eval()
            with torch.no_grad():
                runtime.draft_model.lm_head.weight.zero_()
                runtime.target_model.lm_head.weight.zero_()
            self.draws = []
            with ExitStack() as stack:
                for target, value in _ENTRY_PATCHES:
                    if target.startswith("torch.cuda."):
                        stack.enter_context(patch(target, value))
                for target, value in self._real_rng_patches():
                    stack.enter_context(patch(target, value))
                output, metrics = runtime.adaptive_decoding(
                    torch.tensor([[0]], dtype=torch.long), transfer_top_k=None,
                )
        self.assertEqual(output.shape[1], 6)
        self.assertEqual(metrics["generated_tokens"], 5)
        self._assert_explicit_private_generator(
            ["multinomial", "rand", "multinomial"] * 2 + ["multinomial"]
        )
        for weights in self._token_weights():
            self._assert_law(weights, UNIFORM_ROW)

    def test_zero_budget_draws_nothing_and_leaves_default_cpu_rng_untouched(self):
        """No token budget: no draw, no entropy from the default CPU stream."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261111)
            before = torch.get_rng_state().clone()
            output, metrics = self._run(
                script=UNIFORM_SCRIPT,
                max_tokens=0,
                torch_patches=self._real_rng_patches(),
                fork_rng=False,
            )
            after = torch.get_rng_state().clone()

        self.assertEqual(self.draws, [])
        self.assertTrue(
            torch.equal(before, after),
            "zero-budget decode consumed entropy from the default CPU generator",
        )
        self.assertEqual(output.tolist(), [[0]])
        self.assertEqual(metrics["generated_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
