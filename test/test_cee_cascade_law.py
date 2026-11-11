"""Public CEE first-token cap-law regression on real tiny CPU models.

This promotes the Lead's supported real-cache counterexample to a durable
regression. It drives each public baseline entry (``cee_dsd`` / ``cee_dssd``)
with real tiny Llama models, the real ``KVCacheModel``, and the real
communication simulator. Only CUDA timing is replaced by the shared fixture's
fake event/stream/synchronize. Categorical and acceptance coordinates are
replaced by a deterministic inverse-CDF/acceptance grid; this is finite-grid
evidence, not a CUDA-kernel/Philox simulation and not a native-population claim.

Fixture: transfer_top_k=4 is passed to cache construction, but
``args.transfer_top_k_cap=1`` is applied afterwards.  gamma1=gamma2=0 and budget
2 isolate the direct draft bonus/carried first token.  The draft cache therefore
retains raw D ~ [1/2, 1/4, 1/8, 1/8], while the outer codec's selected draft law
is the k=1 wire law Q_D = Decode(Encode(D)) = [1/2, 1/6, 1/6, 1/6].

The earlier 8x4 diagnostic is not durable here: an 8-center source grid cannot
represent Q_D's 1/6 boundary.  We cross U24 source centers (the exact common
refinement of denominators 8 and 6), R4 acceptance centers, and S8
target/residual centers.  Under the confirmed semantic fix, carried/fresh draft
proposals for the target verifier target Q_D (not raw D), and the inner
verifier/residual/bonus target Q_D; final targetP is unchanged.

Independent first-token oracle for exactly one U/R/S cell, cap=1:
  Q_D source support over U24: token0=12, token1=4, token2=4, token3=4.
  Acceptance R4: token0/token1 accept 4/4; token2/token3 accept 3/4.
  Rejection sends the k=1 residual to token1.
  Expected = [12*4, 4*4 + (4+4)*1, 4*3, 4*3] * 8 S-centers
           = [384, 192, 96, 96].
Current production instead samples the carried proposal from raw D while
verifying against Q_D, hence [384, 240, 72, 72].  With cap=0, Q_D=D and the same
literal oracle is [384, 192, 96, 96] in both modes (positive control).

Only the first committed output token is asserted.  Later outputs and unused
verifier draws share the chosen U/S coordinate within a role; they cannot rewrite
the first committed token, so this fixture is deliberately NOT a joint-law
oracle.  Global CPU RNG is preserved via ``torch.random.fork_rng``.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_cee_cascade_law_fixture", ROOT / "test/test_cee_refactor.py"
)
_fixture = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _fixture
_spec.loader.exec_module(_fixture)

# Centers strictly inside each inverse-CDF bin, so no draw sits on a boundary.
U24_CENTERS = tuple((index + 0.5) / 24 for index in range(24))
R4_CENTERS = (0.125, 0.375, 0.625, 0.875)
S8_CENTERS = tuple((index + 0.5) / 8 for index in range(8))

# Literal, independently derived oracle: one grid cell mass is 1/768.
EXPECTED_FIRST_TOKEN_COUNTS = [384, 192, 96, 96]

_ROLE = {"phase": None}


def _mark_role(role):
    def _hook(module, inputs):
        _ROLE["phase"] = role

    return _hook


def _tiny_model(role):
    config = LlamaConfig(
        vocab_size=4,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        output_hidden_states=True,
    )
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(1.0)
        for layer in model.model.layers:
            for parameter in layer.self_attn.parameters():
                parameter.zero_()
            for parameter in layer.mlp.parameters():
                parameter.zero_()
        hidden_sum = model.model(torch.tensor([[0]])).last_hidden_state[0, -1].sum()
        # Context-independent row D ~ [1/2, 1/4, 1/8, 1/8].
        log_probs = torch.tensor([0.5, 0.25, 0.125, 0.125]).log()
        model.lm_head.weight.copy_(log_probs[:, None].expand(4, 8) / hidden_sum)
    model.register_forward_pre_hook(_mark_role(role))
    return model


@pytest.fixture(scope="module")
def tiny_models():
    # Reusable actual model objects; every grid case gets a fresh baseline and
    # fresh KVCacheModel instances from the eager cache-acquisition path.
    with torch.random.fork_rng(devices=[]):
        return {role: _tiny_model(role) for role in ("little", "draft", "target")}


def _inverse_cdf(weights, coordinate, num_samples):
    cumulative = weights.cumsum(dim=-1)
    threshold = cumulative.new_full((*cumulative.shape[:-1], num_samples), coordinate)
    return torch.searchsorted(cumulative, threshold, right=True)


def _first_generated_token(
    models, mode, cap, u_coordinate, r_coordinate, s_coordinate, transfer_top_k=4
):
    runtime = _fixture.CeeRefactorTests()._make_instance(mode)
    runtime.args.max_tokens = 2
    runtime.args.gamma1 = 0
    runtime.args.gamma2 = 0
    runtime.args.transfer_top_k_cap = cap
    runtime.color_print = lambda *args, **kwargs: None
    for role, model in models.items():
        setattr(runtime, f"{role}_model", model)

    _ROLE["phase"] = None

    def categorical(weights, num_samples=1, **kwargs):
        phase = _ROLE["phase"]
        if phase == "draft":
            coordinate = u_coordinate
        elif phase == "target":
            coordinate = s_coordinate
        else:
            raise AssertionError(
                f"categorical draw without a draft/target model-forward role: {phase!r}"
            )
        return _inverse_cdf(weights, coordinate, num_samples)

    def uniform(*args, device=None, **kwargs):
        shape = args[0] if len(args) == 1 else args
        if isinstance(shape, int):
            shape = (shape,)
        return torch.full(shape, r_coordinate, device=device)

    with ExitStack() as stack:
        # Do not leak the deterministic tape into the global CPU RNG state.
        stack.enter_context(torch.random.fork_rng(devices=[]))
        for target, value in (
            ("torch.cuda.Event", _fixture._FakeCudaEvent),
            ("torch.cuda.current_stream", lambda *args, **kwargs: None),
            ("torch.cuda.synchronize", lambda *args, **kwargs: None),
            ("torch.multinomial", categorical),
            ("torch.rand", uniform),
        ):
            stack.enter_context(patch(target, value))
        output, _ = getattr(runtime, mode)(
            torch.tensor([[0]], dtype=torch.long),
            transfer_top_k=transfer_top_k,
        )

    assert output.shape[1] >= 2, output
    return int(output[0, 1])


@pytest.mark.parametrize("mode", ("cee_dsd", "cee_dssd"))
def test_cee_cap_first_token_law_is_target_p(tiny_models, mode):
    # Positive control first: with no cap, Encode/Decode is the identity for
    # k=4 >= vocab, so Q_D=D. This validates the real-model/coordinate fixture.
    for cap in (0, 1):
        first_tokens = Counter(
            _first_generated_token(
                tiny_models, mode, cap, u_coordinate, r_coordinate, s_coordinate
            )
            for u_coordinate in U24_CENTERS
            for r_coordinate in R4_CENTERS
            for s_coordinate in S8_CENTERS
        )
        observed = [first_tokens[token] for token in range(4)]
        assert observed == EXPECTED_FIRST_TOKEN_COUNTS, (
            f"{mode} cap={cap} first-token counts violate the target-P law: "
            f"expected {EXPECTED_FIRST_TOKEN_COUNTS}, observed {observed}. "
            "With cap=1 the carried draft proposal must come from "
            "Q_D=[1/2,1/6,1/6,1/6], not raw D=[1/2,1/4,1/8,1/8]; raw-D carries "
            "produce [384,240,72,72]."
        )


class _NeedCoordinate(Exception):
    def __init__(self, options):
        self.options = options


class _StoppedChunkCache(_fixture._CeeForwardedCache):
    """Constant-law cache adapter; constructor top-k4 is identity, target top-k0.

    Q_L after the cap is [3/4,1/12,1/12,1/12], Q_D is
    [1/2,1/6,1/6,1/6], and P is uniform. Both rejection stages have
    positive probability. An inner rejection yields a shorter carried chunk,
    so the protocol must append a fresh draft row rather than an inner bonus.
    """

    phase = None
    last_tokens = {}

    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        assert top_k in (0, 4) and top_p == 0
        super().__init__(model, temperature, top_k, top_p, **kwargs)

    def _set_uniform_history(self, prefix):
        role = self.model.kind
        type(self).phase = role
        type(self).last_tokens[role] = prefix.clone()
        probabilities = {
            "little": [0.75, 0.125, 0.0625, 0.0625],
            "draft": [0.5, 0.25, 0.125, 0.125],
            "target": [0.25, 0.25, 0.25, 0.25],
        }
        row = torch.tensor(probabilities[role])
        self.prob_history = row.expand(1, prefix.shape[1], 4).clone()
        self.logits_history = self.prob_history.log()


class _StoppedChunkCoordinates:
    """Explore the finite dependency tree, with equivalent cells merged.

    Categorical weights are exercised on a24-center grid. Inner acceptance has
    only threshold2/3 (token0), outer acceptance only1/2 (token0); other tokens
    always accept. Merge those R6 cells into representatives with exact rational
    weights. No internal baseline helper or private RNG state is observed.
    """

    def __init__(self, path):
        self.path = path
        self.position = 0
        self.generator = None

    def _record_generator(self, kwargs):
        generator = kwargs.get("generator")
        assert generator is not None
        assert generator is not torch.random.default_generator
        if self.generator is None:
            self.generator = generator
        else:
            assert generator is self.generator

    def _pick(self, options):
        if self.position == len(self.path):
            raise _NeedCoordinate(options)
        choice = self.path[self.position]
        self.position += 1
        assert any(choice == value for value, _ in options)
        return choice

    def multinomial(self, weights, num_samples=1, **kwargs):
        self._record_generator(kwargs)
        assert num_samples == 1 and weights.shape == (1, 4)
        bins = Counter(
            int(_inverse_cdf(weights, coordinate, 1)[0, 0])
            for coordinate in U24_CENTERS
        )
        choice = self._pick(
            [
                (("categorical", token), Fraction(count, 24))
                for token, count in sorted(bins.items())
            ]
        )
        return torch.tensor([[choice[1]]], dtype=torch.long, device=weights.device)

    def rand(self, *args, device=None, **kwargs):
        self._record_generator(kwargs)
        shape = args[0] if len(args) == 1 else args
        if isinstance(shape, int):
            shape = (shape,)
        count = int(torch.tensor(shape).prod())
        role = _StoppedChunkCache.phase
        tokens = _StoppedChunkCache.last_tokens[role][0, -count:].tolist()
        per_token = []
        for token in tokens:
            if token != 0:
                per_token.append([(0.5, Fraction(1))])
            elif role == "draft":
                per_token.append([(1 / 3, Fraction(2, 3)), (5 / 6, Fraction(1, 3))])
            else:
                assert role == "target"
                per_token.append([(0.25, Fraction(1, 2)), (0.75, Fraction(1, 2))])
        options = []
        for parts in product(*per_token):
            probability = Fraction(1)
            for _, mass in parts:
                probability *= mass
            options.append(
                (("acceptance", tuple(value for value, _ in parts)), probability)
            )
        choice = self._pick(options)
        return torch.tensor(choice[1], device=device).reshape(shape)


def _run_stopped_chunk_path(mode, path, gamma1=1):
    runtime = _fixture.CeeRefactorTests()._make_instance(mode)
    runtime.args.max_tokens = 3
    runtime.args.gamma1 = gamma1
    runtime.args.gamma2 = 1
    runtime.args.transfer_top_k_cap = 1
    runtime.color_print = lambda *args, **kwargs: None
    _StoppedChunkCache.instances = []
    _StoppedChunkCache.phase = None
    _StoppedChunkCache.last_tokens = {}
    coordinates = _StoppedChunkCoordinates(path)
    with ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))
        for destination, replacement in (
            ("src.baselines.runtime.KVCacheModel", _StoppedChunkCache),
            (
                "src.baselines.simulators.CommunicationSimulator",
                _fixture._FakeCommSimulator,
            ),
            ("torch.cuda.Event", _fixture._FakeCudaEvent),
            ("torch.cuda.current_stream", lambda *args, **kwargs: None),
            ("torch.cuda.synchronize", lambda *args, **kwargs: None),
            ("torch.multinomial", coordinates.multinomial),
            ("torch.rand", coordinates.rand),
        ):
            stack.enter_context(patch(destination, replacement))
        output, _ = getattr(runtime, mode)(torch.tensor([[0]]), transfer_top_k=4)
    assert output.shape == (1, 4)
    return tuple(output[0, 1:].tolist())


def _stopped_chunk_joint_law(mode, gamma1=1):
    frontier = [((), Fraction(1))]
    probabilities = Counter()
    while frontier:
        path, mass = frontier.pop()
        try:
            tokens = _run_stopped_chunk_path(mode, path, gamma1)
        except _NeedCoordinate as branch:
            assert sum(weight for _, weight in branch.options) == 1
            frontier.extend(
                (path + (choice,), mass * weight) for choice, weight in branch.options
            )
        else:
            probabilities[tokens] += mass
    return probabilities


@pytest.mark.parametrize("mode", ("cee_dsd", "cee_dssd"))
@pytest.mark.parametrize("gamma1", (0, 1))
def test_stopped_inner_chunk_and_fresh_draft_preserve_target_joint_law(mode, gamma1):
    # gamma1=0 makes the outer window length itself depend on the inner stop;
    # gamma1=1 exercises fresh Q_D proposals after shorter carried chunks.
    observed = _stopped_chunk_joint_law(mode, gamma1)
    # Target is context-independent uniform V4, not a distribution derived from
    # the implementation's carried-row or stopping decisions.
    expected = {tokens: Fraction(1, 64) for tokens in product(range(4), repeat=3)}
    assert sum(observed.values()) == 1
    assert observed == expected


@pytest.fixture(scope="module")
def uniform_tiny_models():
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        models = {role: _tiny_model(role) for role in ("little", "draft", "target")}
        for model in models.values():
            model.lm_head.weight.zero_()
        return models


@pytest.mark.parametrize("mode", ("cee_dsd", "cee_dssd"))
@pytest.mark.parametrize(
    "budget, expected_forwards, expected_lengths, expected_candidates",
    [
        (1, (0, 0, 1), (0, 0, 1), (0, 0)),
        (2, (0, 1, 1), (0, 1, 2), (0, 1)),
        (3, (1, 1, 1), (1, 2, 3), (1, 2)),
        (8, (3, 3, 2), (6, 7, 8), (3, 6)),
    ],
)
def test_native_cpu_cache_budget_and_rng_lifetime(
    uniform_tiny_models,
    mode,
    budget,
    expected_forwards,
    expected_lengths,
    expected_candidates,
):
    """Real models/cache/native samplers: no discarded draws or forwarded tail.

    Uniform identical model laws make every candidate genuinely accepted, so
    the literal call counts and cache endpoints have no stochastic tolerance.
    Budget8 spans two rounds and must retain one advancing private CPU stream.
    """
    from src.model_gpu import KVCacheModel

    runtime = _fixture.CeeRefactorTests()._make_instance(mode)
    runtime.args.max_tokens = budget
    runtime.args.gamma1, runtime.args.gamma2 = 1, 2
    runtime.args.transfer_top_k_cap = 0
    runtime.args.disable_eos_stop = True
    runtime.color_print = lambda *args, **kwargs: None
    for role, model in uniform_tiny_models.items():
        setattr(runtime, f"{role}_model", model)
    roles = ("little", "draft", "target")
    role_by_model = {id(model): role for role, model in uniform_tiny_models.items()}
    forward_counts = Counter()
    caches = {}
    categorical_generators, uniform_generators = [], []
    native_multinomial, native_rand = torch.multinomial, torch.rand

    def cache_factory(model, *args, **kwargs):
        cache = KVCacheModel(model, *args, **kwargs)
        caches[role_by_model[id(model)]] = cache
        return cache

    def categorical(*args, **kwargs):
        categorical_generators.append(kwargs.get("generator"))
        return native_multinomial(*args, **kwargs)

    def uniform(*args, **kwargs):
        uniform_generators.append(kwargs.get("generator"))
        return native_rand(*args, **kwargs)

    def record_forward(model, args):
        forward_counts[role_by_model[id(model)]] += 1

    with ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))
        torch.manual_seed(20261111)
        for model in uniform_tiny_models.values():
            handle = model.register_forward_pre_hook(record_forward)
            stack.callback(handle.remove)
        for destination, replacement in (
            ("src.baselines.runtime.KVCacheModel", cache_factory),
            ("torch.cuda.Event", _fixture._FakeCudaEvent),
            ("torch.cuda.current_stream", lambda *args, **kwargs: None),
            ("torch.cuda.synchronize", lambda *args, **kwargs: None),
            ("torch.multinomial", categorical),
            ("torch.rand", uniform),
        ):
            stack.enter_context(patch(destination, replacement))
        output, metrics = getattr(runtime, mode)(torch.tensor([[0]]), transfer_top_k=4)
    assert output.shape == (1, budget + 1)
    assert len(categorical_generators) == budget
    generators = categorical_generators + uniform_generators
    assert generators[0] is not None
    assert generators[0] is not torch.random.default_generator
    assert all(generator is generators[0] for generator in generators)
    assert tuple(forward_counts[role] for role in roles) == expected_forwards
    assert (
        tuple(metrics[f"{role}_forward_times"] for role in roles) == expected_forwards
    )
    assert tuple(caches[role].current_length for role in roles) == expected_lengths
    for cache in caches.values():
        if cache.current_length:
            assert cache.prob_history.shape[1] == cache.current_length
        assert cache.current_length < output.shape[1]
    assert (
        metrics["little_generated_tokens"],
        metrics["draft_generated_tokens"],
    ) == expected_candidates
    assert metrics["generated_tokens"] == budget


@pytest.mark.parametrize("mode", ("cee_dsd", "cee_dssd"))
@pytest.mark.parametrize(
    "dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
def test_wire_precision_tail_support_and_actual_message_bytes(mode, dtype):
    # Independent literal K=1 payload: retained .5, omitted mass .5 shared by
    # three tail entries, each rounded to the actual payload type. Target P is
    # this same valid low-precision row, so every proposal really accepts.
    decoded = torch.tensor([[0.5, 1 / 6, 1 / 6, 1 / 6]], dtype=dtype)
    expected_weights = decoded.float() / decoded.float().sum(-1, keepdim=True)
    raw_draft = torch.tensor([[0.5, 0.25, 0.125, 0.125]], dtype=dtype)
    samplers = []

    class TypedCache(_fixture._CeeForwardedCache):
        instances = []

        def _set_uniform_history(self, prefix):
            row = decoded if self.model.kind == "target" else raw_draft
            self.prob_history = row[:, None, :].expand(1, prefix.shape[1], 4).clone()
            self.logits_history = self.prob_history.log()

    class WireComm(_fixture._CeeCommunicationSimulator):
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.probability_payloads = []

        def transfer(self, tokens, prob, link_type, *args, **kwargs):
            if prob is not None:
                self.probability_payloads.append((link_type, prob.clone()))
            return super().transfer(tokens, prob, link_type, *args, **kwargs)

    def categorical(weights, num_samples=1, **kwargs):
        assert kwargs.get("generator") is not None and num_samples == 1
        samplers.append(weights.clone())
        # A real positive-probability tail token, outside the single kept index.
        return torch.tensor([[2 if len(samplers) == 1 else 1]], device=weights.device)

    def uniform(shape, device=None, **kwargs):
        assert kwargs.get("generator") is not None
        return torch.full(shape, 0.5, device=device)

    runtime = _fixture.CeeRefactorTests()._make_instance(mode)
    runtime.args.gamma1 = runtime.args.gamma2 = 0
    runtime.args.max_tokens = 2
    runtime.args.transfer_top_k_cap = 1
    runtime.args.disable_eos_stop = True
    runtime.color_print = lambda *args, **kwargs: None
    with ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))
        for destination, replacement in (
            ("src.baselines.runtime.KVCacheModel", TypedCache),
            ("src.baselines.simulators.CommunicationSimulator", WireComm),
            ("torch.cuda.Event", _fixture._FakeCudaEvent),
            ("torch.cuda.current_stream", lambda *args, **kwargs: None),
            ("torch.cuda.synchronize", lambda *args, **kwargs: None),
            ("torch.multinomial", categorical),
            ("torch.rand", uniform),
        ):
            stack.enter_context(patch(destination, replacement))
        output, metrics = getattr(runtime, mode)(torch.tensor([[0]]), transfer_top_k=4)
    assert output.tolist() == [[0, 2, 1]]
    assert len(samplers) == 2
    assert all(torch.equal(weights, expected_weights) for weights in samplers)
    draft_cache = next(
        cache for cache in TypedCache.instances if cache.model.kind == "draft"
    )
    assert torch.equal(draft_cache.prob_history[:, -1], raw_draft)
    comm = WireComm.instances[-1]
    if mode == "cee_dsd":
        # int64 prompt8 + candidate8 + int32 index4 + two probability values
        # + reply(position4 + int64 token8). No implicit int32 token packing.
        expected_cloud_bytes = 32 + 2 * decoded.element_size()
    else:
        expected_cloud_bytes = 36  # prompt8 + candidate8 + selected-Q8 + reply12
        assert len(comm.probability_payloads) == 1
        link, scalar = comm.probability_payloads[0]
        expected_q = expected_weights.double() / expected_weights.double().sum(
            -1, keepdim=True
        )
        assert link == "edge_cloud" and scalar.dtype == torch.float64
        assert torch.equal(scalar, expected_q[:, 2:3])
    assert metrics["edge_cloud_data_bytes"] == expected_cloud_bytes
    assert metrics["edge_end_data_bytes"] == 32
    assert len(comm.edge_cloud_topk_history) == len(comm.edge_cloud_bandwidth_history)
    assert len(comm.edge_cloud_draft_len_history) == len(
        comm.edge_cloud_bandwidth_history
    )


@pytest.mark.parametrize("mode", ("cee_dsd", "cee_dssd"))
@pytest.mark.parametrize("disable_eos_stop", (False, True))
def test_confirmed_eos_crops_real_caches_and_stops_further_rounds(
    uniform_tiny_models,
    mode,
    disable_eos_stop,
):
    from src.model_gpu import KVCacheModel

    runtime = _fixture.CeeRefactorTests()._make_instance(mode)
    runtime.args.max_tokens = 6
    runtime.args.gamma1, runtime.args.gamma2 = 1, 2
    runtime.args.transfer_top_k_cap = 0
    runtime.args.disable_eos_stop = disable_eos_stop
    runtime.tokenizer = SimpleNamespace(eos_token_id=3)
    runtime.color_print = lambda *args, **kwargs: None
    roles = ("little", "draft", "target")
    role_by_model = {id(model): role for role, model in uniform_tiny_models.items()}
    for role, model in uniform_tiny_models.items():
        setattr(runtime, f"{role}_model", model)
    caches, draws = {}, []

    def cache_factory(model, *args, **kwargs):
        cache = KVCacheModel(model, *args, **kwargs)
        caches[role_by_model[id(model)]] = cache
        return cache

    def categorical(weights, num_samples=1, **kwargs):
        assert kwargs.get("generator") is not None
        draws.append(weights)
        return torch.tensor([[3 if len(draws) == 1 else 1]], device=weights.device)

    with ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))
        for destination, replacement in (
            ("src.baselines.runtime.KVCacheModel", cache_factory),
            ("torch.cuda.Event", _fixture._FakeCudaEvent),
            ("torch.cuda.current_stream", lambda *args, **kwargs: None),
            ("torch.cuda.synchronize", lambda *args, **kwargs: None),
            ("torch.multinomial", categorical),
        ):
            stack.enter_context(patch(destination, replacement))
        output, metrics = getattr(runtime, mode)(torch.tensor([[0]]), transfer_top_k=4)
    expected = [[0, 3, 1, 1, 1, 1, 1]] if disable_eos_stop else [[0, 3]]
    assert output.tolist() == expected
    assert len(draws) == (6 if disable_eos_stop else 5)
    assert metrics["target_forward_times"] == (2 if disable_eos_stop else 1)
    assert metrics["generated_tokens"] == (6 if disable_eos_stop else 1)
    assert tuple(caches[role].current_length for role in roles) == (
        (2, 4, 6) if disable_eos_stop else (1, 1, 1)
    )
    for cache in caches.values():
        assert cache.prob_history.shape[1] == cache.current_length
        assert cache.current_length < output.shape[1]


@pytest.mark.parametrize("mode", ("cee_dsd", "cee_dssd"))
def test_zero_budget_uses_no_rng_model_cache_or_cuda_timer(mode):
    runtime = _fixture.CeeRefactorTests()._make_instance(mode)
    runtime.args.max_tokens = 0
    prefix = torch.tensor([[0]])
    with (
        torch.random.fork_rng(devices=[]),
        patch(
            "src.baselines.runtime.KVCacheModel",
            side_effect=AssertionError("cache allocation"),
        ),
        patch("torch.cuda.Event", side_effect=AssertionError("CUDA timer")),
        patch("torch.multinomial", side_effect=AssertionError("categorical RNG")),
        patch("torch.rand", side_effect=AssertionError("acceptance RNG")),
    ):
        before = torch.random.get_rng_state().clone()
        output, metrics = getattr(runtime, mode)(prefix)
        assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(output, prefix)
    assert metrics["generated_tokens"] == 0


@pytest.mark.parametrize("mode", ("cee_dsd", "cee_dssd"))
@pytest.mark.parametrize("role", ("little", "draft", "target"))
@pytest.mark.parametrize(
    "bad_row",
    (
        [float("nan"), 0.25, 0.25, 0.25],
        [float("inf"), 0, 0, 0],
        [-0.125, 0.5, 0.5, 0.125],
        [0.25, 0.25, 0, 0],
    ),
    ids=("nan", "inf", "negative", "nonunit-mass"),
)
def test_invalid_first_probability_row_fails_before_rng(mode, role, bad_row):
    class InvalidCache(_fixture._CeeForwardedCache):
        instances = []

        def _set_uniform_history(self, prefix):
            row = torch.tensor(bad_row if self.model.kind == role else [0.25] * 4)
            self.prob_history = row.expand(1, prefix.shape[1], 4).clone()
            self.logits_history = self.prob_history.log()

    runtime = _fixture.CeeRefactorTests()._make_instance(mode)
    runtime.args.max_tokens = {"little": 3, "draft": 2, "target": 1}[role]
    runtime.args.gamma1 = 0
    runtime.args.gamma2 = int(role == "little")
    runtime.args.transfer_top_k_cap = 1
    runtime.color_print = lambda *args, **kwargs: None
    with ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))
        for destination, replacement in (
            ("src.baselines.runtime.KVCacheModel", InvalidCache),
            ("torch.cuda.Event", _fixture._FakeCudaEvent),
            ("torch.cuda.current_stream", lambda *args, **kwargs: None),
            ("torch.cuda.synchronize", lambda *args, **kwargs: None),
        ):
            stack.enter_context(patch(destination, replacement))
        stack.enter_context(
            patch("torch.multinomial", side_effect=AssertionError("categorical RNG"))
        )
        stack.enter_context(
            patch("torch.rand", side_effect=AssertionError("acceptance RNG"))
        )
        before = torch.random.get_rng_state().clone()
        with pytest.raises(ValueError):
            getattr(runtime, mode)(torch.tensor([[0]]), transfer_top_k=4)
        assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("mode", ("cee_dsd", "cee_dssd"))
def test_real_topk_cutoff_ties_use_uniform_wire_tail_without_a_cap(mode):
    from src.model_gpu import KVCacheModel

    with torch.random.fork_rng(devices=[]), torch.no_grad():
        models = {role: _tiny_model(role) for role in ("little", "draft", "target")}
        for model in models.values():
            model.lm_head.weight.zero_()
        models["draft"].lm_head.weight[2:].fill_(-1)
        # Real threshold top-k=1 retains TWO tied maxima. This is not the cap
        # fixture and does not rely on a cache adapter ignoring filtering.
        draft = KVCacheModel(models["draft"], 1.0, 1, 0.0)
        draft.vocab_size = 4
        raw = draft.forward_context(torch.tensor([[0]]))
        assert torch.equal(raw, torch.tensor([[0.5, 0.5, 0.0, 0.0]]))
        # Stable kept index0 and uniform tail give Q_D=[.5,1/6,1/6,1/6].
        # Source coordinate.8 selects token2, impossible under raw filtered D.
        # Uniform final target accepts token2 at independent R=.25.
        first = _first_generated_token(
            models,
            mode,
            0,
            0.8,
            0.25,
            0.125,
            transfer_top_k=1,
        )
    assert first == 2


@pytest.mark.parametrize("mode", ("cee_dsd", "cee_dssd"))
@pytest.mark.parametrize("charge_residual", (False, True))
def test_both_rejections_bill_effective_inner_and_full_target_payloads(
    mode, charge_residual
):
    runtime = _fixture.CeeRefactorTests()._make_instance(mode)
    runtime.args.max_tokens = 3
    runtime.args.gamma1 = runtime.args.gamma2 = 1
    runtime.args.transfer_top_k_cap = 1
    runtime.args.charge_residual_payload = charge_residual
    runtime.args.disable_eos_stop = True
    runtime.color_print = lambda *args, **kwargs: None
    _StoppedChunkCache.instances = []
    _fixture._CeeCommunicationSimulator.instances = []
    tokens = iter([0, 1, 0, 1, 1])
    generators = []

    def categorical(weights, num_samples=1, **kwargs):
        generators.append(kwargs.get("generator"))
        token = next(tokens)
        assert num_samples == 1 and weights[0, token] > 0
        return torch.tensor([[token]], device=weights.device)

    def uniform(shape, device=None, **kwargs):
        generators.append(kwargs.get("generator"))
        return torch.full(shape, 0.9, device=device)

    with ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))
        for destination, replacement in (
            ("src.baselines.runtime.KVCacheModel", _StoppedChunkCache),
            (
                "src.baselines.simulators.CommunicationSimulator",
                _fixture._CeeCommunicationSimulator,
            ),
            ("torch.cuda.Event", _fixture._FakeCudaEvent),
            ("torch.cuda.current_stream", lambda *args, **kwargs: None),
            ("torch.cuda.synchronize", lambda *args, **kwargs: None),
            ("torch.multinomial", categorical),
            ("torch.rand", uniform),
        ):
            stack.enter_context(patch(destination, replacement))
        output, metrics = getattr(runtime, mode)(torch.tensor([[0]]), transfer_top_k=4)
    assert output.tolist() == [[0, 1, 1, 1]]
    assert metrics["little_accepted_tokens"] == 0
    assert metrics["draft_accepted_tokens"] == 1
    assert (
        metrics["target_forward_times"] == 2
    )  # outer correction, then target-only tail
    assert generators[0] is not None
    assert all(generator is generators[0] for generator in generators)
    # Sender already owns Q. Honest rejection traffic includes the encoded Q_D
    # verifier row (12B) on inner reject, and the uncompressed target P (16B) on
    # outer reject. Both retain the 6B reject signal even with legacy discount.
    cloud, edge = (94, 82) if mode == "cee_dsd" else (86, 78)
    if not charge_residual:
        cloud -= 16
        edge -= 12
    assert metrics["edge_cloud_data_bytes"] == cloud
    assert metrics["edge_end_data_bytes"] == edge


@pytest.mark.parametrize("mode", ("cee_dsd", "cee_dssd"))
def test_real_context_dependent_cache_matches_fresh_after_repeated_rejections(mode):
    from src.model_gpu import KVCacheModel

    runtime = _fixture.CeeRefactorTests()._make_instance(mode)
    runtime.args.max_tokens = 7
    runtime.args.gamma1, runtime.args.gamma2 = 1, 2
    runtime.args.transfer_top_k_cap = 1
    runtime.args.disable_eos_stop = True
    runtime.color_print = lambda *args, **kwargs: None
    model_rows = {
        "little": [0.75, 0.125, 0.0625, 0.0625],
        "draft": [0.5, 0.25, 0.125, 0.125],
        "target": [0.25] * 4,
    }
    models, caches = {}, {}
    with torch.random.fork_rng(devices=[]), torch.no_grad(), ExitStack() as stack:
        torch.manual_seed(20261111)
        for role, row in model_rows.items():
            model = LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=4,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    max_position_embeddings=32,
                    output_hidden_states=True,
                )
            ).eval()
            model.model.embed_tokens.weight.normal_(1.0, 0.2)
            for parameter in model.model.layers[0].self_attn.parameters():
                parameter.mul_(5)
            scale = model.model(torch.tensor([[0]])).last_hidden_state[0, -1].sum()
            model.lm_head.weight.copy_(
                torch.tensor(row).log()[:, None].expand(4, 8) / scale
                + 0.03 * torch.randn_like(model.lm_head.weight)
            )
            models[role] = model
            setattr(runtime, f"{role}_model", model)

        def cache_factory(model, *args, **kwargs):
            cache = KVCacheModel(model, *args, **kwargs)
            caches[next(role for role, value in models.items() if value is model)] = (
                cache
            )
            return cache

        def categorical(weights, num_samples=1, **kwargs):
            assert kwargs.get("generator") is not None
            return weights.argmax(dim=-1, keepdim=True)

        def uniform(shape, device=None, **kwargs):
            assert kwargs.get("generator") is not None
            return torch.full(shape, 0.99, device=device)

        for destination, replacement in (
            ("src.baselines.runtime.KVCacheModel", cache_factory),
            ("torch.cuda.Event", _fixture._FakeCudaEvent),
            ("torch.cuda.current_stream", lambda *args, **kwargs: None),
            ("torch.cuda.synchronize", lambda *args, **kwargs: None),
            ("torch.multinomial", categorical),
            ("torch.rand", uniform),
        ):
            stack.enter_context(patch(destination, replacement))
        output, metrics = getattr(runtime, mode)(torch.tensor([[0]]), transfer_top_k=4)
        assert output.shape == (1, 8)
        assert metrics["little_accepted_tokens"] < metrics["little_generated_tokens"]
        assert metrics["draft_accepted_tokens"] < metrics["draft_generated_tokens"]
        assert metrics["target_forward_times"] > 1
        committed = output[:, :-1]
        for role, cache in caches.items():
            assert cache.current_length <= committed.shape[1]
            cache.forward_context(committed)
            fresh = KVCacheModel(models[role], 1.0, 0 if role == "target" else 4, 0.0)
            fresh.vocab_size = 4
            fresh.forward_context(committed)
            # Real incremental-vs-bulk GEMMs need floating tolerance; this checks
            # content at every retained context row, not just cache lengths.
            torch.testing.assert_close(
                cache.prob_history, fresh.prob_history, rtol=1e-5, atol=1e-6
            )
