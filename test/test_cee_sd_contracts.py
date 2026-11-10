"""Independent toy-law contracts at the public CEE-SD entry point.

Reuse only standalone CPU cache/event/communication support, never another
TestCase's lifecycle. Probability row i is the prediction after consuming token
position i: sampling the next token must NOT append a forwarded cache row.
Expected tokens and probabilities below follow toy distributions, not an old
implementation or a trajectory oracle.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src import communication
from src.baselines import Baselines, cee, runtime
from src.communication import CUHLM

ROOT = Path(__file__).resolve().parents[1]
TWO = (0.0, 0.0, 1.0, 0.0)
THREE = (0.0, 0.0, 0.0, 1.0)


def _support():
    path = ROOT / "test/test_adaptive_tridecoding_rounds.py"
    spec = importlib.util.spec_from_file_location("cee_sd_round_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support = _support()


@pytest.fixture
def run_cee(monkeypatch):
    """Fresh decoder/cache/real communication ledger per public-entry run."""
    real_rand = torch.rand
    real_multinomial = torch.multinomial
    for key in (
        "CACHE_CHECK_TRACE",
        "ARP_CALIB_TRACE",
        "TOPK_TRACE",
        "COMM_TRACE",
        "APPEND_TRACE",
        "RL_VERIFY_TRACE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("src.decoding_ops.VERIFY_TRACE_PATH", None)
    monkeypatch.setattr(
        communication, "_NTT_TRACE_STATE", {"data": [], "scale": 1.0, "src": ""}
    )
    monkeypatch.setattr(torch.cuda, "Event", support._FakeCudaEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def run(
        *,
        rows=None,
        prompt=(0,),
        budget=5,
        gamma1=2,
        gamma2=1,
        topk=0,
        uniform=None,
        opportunistic=False,
        eos=None,
        early_stop=False,
        mode="per_round",
        residual=False,
        bits=16,
    ):
        draws, samples = [], []

        class Cache(support._RowScriptedCache):
            instances = []
            scripts = rows or {}
            metadata_calls = 0

            def forward_context(self, prefix):
                self.hidden_states = torch.zeros((1, 1, 1))
                return super().forward_context(prefix)

            def generate(self, prefix, gamma):
                # The real KV cache does no forward at all for gamma=0.
                return prefix.clone() if gamma == 0 else super().generate(prefix, gamma)

            def generate_with_rebuilt_topk(self, prefix, gamma, topk):
                return (
                    (prefix.clone(), None)
                    if gamma == 0
                    else super().generate_with_rebuilt_topk(prefix, gamma, topk)
                )

            def generate_with_rebuilt_topk_metadata(self, prefix, gamma, topk):
                return (
                    (prefix.clone(), None, None)
                    if gamma == 0
                    else super().generate_with_rebuilt_topk_metadata(
                        prefix, gamma, topk
                    )
                )

            @property
            def logits_history(self):
                return self.prob_history.clamp_min(1e-12).log()

        class Simulator(support._RecordingCommunicationSimulator):
            instances = []

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.sent_probabilities = []

            def transfer(self, tokens, probs, link_type, *args, **kwargs):
                if probs is not None:
                    self.sent_probabilities.append(
                        (
                            link_type,
                            None if tokens is None else tokens.clone(),
                            probs.clone(),
                        )
                    )
                return super().transfer(tokens, probs, link_type, *args, **kwargs)

        class SkipLittleVerification(CUHLM):
            def calculate_uncertainty(self, *args, **kwargs):
                return 0.0

        def rand(shape, **kwargs):
            value = (
                real_rand(shape, **kwargs)
                if uniform is None
                else torch.full(
                    (shape,) if isinstance(shape, int) else shape,
                    uniform,
                    device=kwargs.get("device"),
                    dtype=torch.float32,
                )
            )
            draws.append(value.clone())
            return value

        def multinomial(probs, *args, **kwargs):
            samples.append(probs.clone())
            return real_multinomial(probs, *args, **kwargs)

        monkeypatch.setattr(runtime, "KVCacheModel", Cache)
        monkeypatch.setattr(
            "src.baselines.simulators.CommunicationSimulator", Simulator
        )
        monkeypatch.setattr(cee, "CUHLM", SkipLittleVerification)
        monkeypatch.setattr(torch, "rand", rand)
        monkeypatch.setattr(torch, "multinomial", multinomial)
        decoder = object.__new__(support._TestBaselines)
        decoder.args = SimpleNamespace(
            gamma1=gamma1,
            gamma2=gamma2,
            max_tokens=budget,
            temp=1.0,
            top_k=0,
            top_p=0.0,
            use_cuda_graph=False,
            edge_cloud_bandwidth=10.0,
            edge_end_bandwidth=10.0,
            cloud_end_bandwidth=10.0,
            batch_delay=0.0,
            comm_round_trip_mode=mode,
            charge_residual_payload=residual,
            prob_payload_bits=bits,
            disable_rl_update=True,
            disable_eos_stop=eos is None,
        )
        decoder.vocab_size = 4
        decoder.num_acc_tokens = []
        decoder.accelerator = SimpleNamespace(is_main_process=True)
        decoder.draft_forward_times = decoder.target_forward_times = 0
        for role in ("little", "draft", "target"):
            setattr(
                decoder,
                f"{role}_model",
                SimpleNamespace(kind=role, device=torch.device("cpu")),
            )
        decoder.small_draft_adapter = decoder.draft_target_adapter = None
        decoder.rl_adapter = decoder.little_rl_adapter = None
        decoder.tokenizer = SimpleNamespace(
            eos_token_id=-1 if eos is None else eos,
            decode=lambda ids, **kwargs: "STOP" if early_stop else "continue",
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(8128)
            output, metrics = Baselines.adaptive_tridecoding(
                decoder,
                torch.tensor([prompt], dtype=torch.long),
                transfer_top_k=topk,
                _ablation="static",
                _opportunistic_first_stage=opportunistic,
                use_early_stopping=early_stop,
                stop_sequences=["STOP"] if early_stop else None,
            )
        return SimpleNamespace(
            output=output,
            metrics=metrics,
            caches={c.role: c for c in Cache.instances},
            simulator=Simulator.instances[0],
            draws=draws,
            samples=samples,
        )

    return run


@pytest.mark.parametrize("gamma1,gamma2", [(2, 1), (0, 1), (2, 0), (0, 0)])
def test_verification_and_zero_candidate_context_do_not_sample(run_cee, gamma1, gamma2):
    # One round: little/draft proposals plus one genuine correction/bonus at each
    # stage. Neither verifier may secretly draw an extra token, even at gamma=0.
    result = run_cee(gamma1=gamma1, gamma2=gamma2, budget=1)
    assert len(result.samples) == gamma1 + gamma2 + 2
    assert result.metrics["target_forward_times"] == 1
    assert result.caches["target"].current_length == 1 + gamma2 + 1 + gamma1


def test_opportunistic_skip_keeps_only_real_proposal_and_correction_samples(run_cee):
    result = run_cee(gamma1=2, gamma2=1, budget=1, opportunistic=True)
    assert len(result.samples) == 1 + 2 + 1  # no inner correction on skipped stage
    assert result.metrics["target_forward_times"] == 1


def test_full_accept_bonus_has_not_been_forwarded(run_cee):
    result = run_cee(budget=5)
    assert result.output.tolist() == [[0, 1, 1, 1, 1, 1]]
    # Prompt + four verified tokens were forwarded; the fifth token is bonus.
    assert result.caches["target"].current_length == 5
    assert result.caches["target"].current_length == result.output.shape[1] - 1
    assert result.metrics["little_accepted_tokens"] == 1
    assert result.metrics["draft_accepted_tokens"] == 4
    assert result.simulator._pending_bytes == {}


@pytest.mark.parametrize("gamma1", [0, 2])
def test_long_prompt_bonus_is_aligned_and_stays_outside_forwarded_cache(
    run_cee, gamma1
):
    forwarded = 5 + gamma1  # three prompt + two inner-stage + gamma1 draft tokens
    result = run_cee(
        prompt=(0, 0, 0),
        gamma1=gamma1,
        budget=3 + gamma1,
        rows={"target": {forwarded - 1: THREE}},
    )
    assert result.output.tolist() == [[0, 0, 0] + [1] * (2 + gamma1) + [3]]
    assert result.metrics["draft_accepted_tokens"] == 2 + gamma1
    assert result.caches["target"].current_length == forwarded


def test_target_confirmed_eos_truncates_generated_span_only(run_cee):
    result = run_cee(rows={"target": {0: TWO}}, eos=2, budget=5)
    assert result.output.tolist() == [[0, 2]]
    assert result.metrics["generated_tokens"] == 1
    assert result.simulator._pending_bytes == {}
    # An EOS in the prompt is context, not a generated terminal token.
    prompt_eos = run_cee(prompt=(2, 0, 0), eos=2, budget=5)
    assert prompt_eos.output.tolist() == [[2, 0, 0, 1, 1, 1, 1, 1]]


def test_zero_draft_extension_still_verifies_first_stage_tokens(run_cee):
    # gamma1=0 does NOT mean the outer verifier has no candidates: stage 1
    # contributed a little token and draft bonus. Neither is legal under target.
    result = run_cee(rows={"target": {0: TWO}}, gamma1=0, budget=5, early_stop=True)
    assert result.output.tolist() == [[0, 2]]
    assert result.metrics["draft_accepted_tokens"] == 0


def test_zero_draft_extension_long_prompt_uses_absolute_probability_row(run_cee):
    result = run_cee(
        rows={"target": {2: THREE}},
        prompt=(0, 0, 0),
        gamma1=0,
        budget=5,
        early_stop=True,
    )
    assert result.output.tolist() == [[0, 0, 0, 3]]


@pytest.mark.parametrize("gamma1", [0, 2])
def test_empty_little_stage_still_checks_draft_bonus_against_target(run_cee, gamma1):
    result = run_cee(
        rows={"target": {2: THREE}},
        prompt=(0, 0, 0),
        gamma2=0,
        gamma1=gamma1,
        budget=5,
        early_stop=True,
    )
    assert result.output.tolist() == [[0, 0, 0, 3]]
    assert result.metrics["little_generated_tokens"] == 0
    assert result.metrics["draft_accepted_tokens"] == 0


def test_zero_uniform_draw_never_accepts_zero_target_mass(run_cee):
    result = run_cee(
        rows={"target": {0: TWO}},
        budget=5,
        uniform=0.0,
        early_stop=True,
    )
    assert result.output.tolist() == [[0, 2]]
    assert result.metrics["draft_accepted_tokens"] == 0


def test_zero_uniform_draw_never_accepts_zero_draft_mass_in_inner_stage(run_cee):
    result = run_cee(
        rows={"draft": {0: TWO}, "target": {0: TWO}},
        budget=5,
        uniform=0.0,
        early_stop=True,
    )
    assert result.output[0, 1].item() == 2
    assert result.metrics["little_accepted_tokens"] == 0


def test_inner_eos_is_not_committed_without_target_acceptance(run_cee):
    result = run_cee(
        rows={"target": {0: TWO}},
        eos=1,
        early_stop=True,
        budget=5,
    )
    assert result.output.tolist() == [[0, 2]]
    assert result.metrics["target_forward_times"] == 1


@pytest.mark.parametrize("gamma1", [0, 2])
def test_opportunistic_prefix_retains_little_sampling_law_at_target(run_cee, gamma1):
    # The skipped token was sampled from little delta(1), NOT draft's p.
    # Target gives token 1 probability .1, so r=.5 rejects it. Positive
    # residual norm((p-delta(1))+) is delta(2), hence the output is exact.
    target = (0.0, 0.1, 0.9, 0.0)
    result = run_cee(
        rows={"draft": {0: target}, "target": {0: target}},
        opportunistic=True,
        gamma1=gamma1,
        uniform=0.5,
        early_stop=True,
        budget=5,
    )
    assert result.output.tolist() == [[0, 2]]
    assert result.metrics["draft_accepted_tokens"] == 0
    assert any(
        torch.equal(weights, torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
        for weights in result.samples
    )


@pytest.mark.parametrize("residual", [False, True])
def test_corrected_inner_prefix_is_not_resampled_or_rebuilt_with_live_topk(
    run_cee, residual
):
    # Inner rejection correction has marginal law D=(0,.512,.488,0).
    # Rebuilding D at k=1 would incorrectly make D(2)=.488/3; target .2
    # must reject against the real .488, not accept against that smaller tail.
    result = run_cee(
        rows={
            "draft": {0: (0.0, 0.512, 0.488, 0.0)},
            "target": {0: (0.0, 0.5, 0.2, 0.3)},
        },
        topk=1,
        residual=residual,
        uniform=0.75,
        early_stop=True,
        budget=5,
    )
    assert result.output.tolist() == [[0, 3]]
    assert result.metrics["draft_accepted_tokens"] == 0


@pytest.mark.parametrize(
    "bits, expected_first, expected_accepts, wire_selected",
    [
        (16, 2, 3, 0.488),
        (8, 3, 0, 0.5),
    ],
)
@pytest.mark.parametrize("residual", [False, True])
def test_lossy_wire_q_is_shared_by_acceptance_and_residual_independent_of_billing(
    run_cee,
    bits,
    expected_first,
    expected_accepts,
    wire_selected,
    residual,
):
    # Inner delta(1) rejects against draft D, so its corrected token is 2.
    # At target, D(2)=.488 but quantize8(D)(2)=.5. For r=.75:
    # .37/.488 > .75 accepts in pass-through mode; .37/.5 < .75 rejects
    # on the deliberately approximate wire protocol. Residual then has ONLY
    # token 3 positive. Charging residual bytes must not change either law.
    result = run_cee(
        rows={
            "draft": {0: (0.0, 0.512, 0.488, 0.0)},
            "target": {0: (0.0, 0.5, 0.37, 0.13)},
        },
        bits=bits,
        topk=2,
        residual=residual,
        uniform=0.75,
        early_stop=True,
        budget=5,
    )
    assert result.output[0, 1].item() == expected_first
    assert result.metrics["draft_accepted_tokens"] == expected_accepts
    outer = next(
        probs
        for link, tokens, probs in result.simulator.sent_probabilities
        if link == "edge_cloud"
    )
    assert outer[0, 0].item() == pytest.approx(wire_selected)
    if bits == 8:
        assert result.output.tolist() == [[0, 3]]


@pytest.mark.parametrize("residual", [False, True])
def test_lossy_rejection_sampler_uses_wire_residual_not_sampled_q(run_cee, residual):
    result = run_cee(
        rows={
            "draft": {0: (0.0, 0.512, 0.488, 0.0)},
            "target": {0: (0.0, 0.506, 0.37, 0.124)},
        },
        bits=8,
        topk=2,
        residual=residual,
        uniform=0.75,
        early_stop=True,
        budget=5,
    )
    # qhat=(0,.5,.5,0), so norm((P-qhat)+)=(0,.006,0,.124)/.13.
    # Using the exact sampled D instead would incorrectly produce delta(3).
    expected = torch.tensor([[0.0, 3 / 65, 0.0, 62 / 65]])
    assert result.metrics["draft_accepted_tokens"] == 0
    assert result.output[0, 1].item() in (1, 3)
    assert any(
        weights.shape == expected.shape
        and torch.allclose(weights, expected, atol=1e-6, rtol=1e-6)
        for weights in result.samples
    )


@pytest.mark.parametrize("mode", ["per_transfer", "per_round"])
@pytest.mark.parametrize("rejected, cloud_bytes", [(False, 84), (True, 106)])
def test_terminal_target_response_and_downlink_are_fully_billed(
    run_cee, mode, rejected, cloud_bytes
):
    rows = {"target": {0 if rejected else 4: TWO}}
    result = run_cee(rows=rows, eos=2, uniform=0.5, mode=mode)
    assert result.output.tolist() == ([[0, 2]] if rejected else [[0, 1, 1, 1, 1, 2]])
    # edge_end: 8 prompt + 12 inner proposal + 12 inner response + 12 final
    # downlink. WAN: 24 prefix + 48 proposals + response (12 accepted,
    # or 4 index + 8 token + 16 full probabilities + 6 rejection = 34).
    assert result.metrics["edge_end_data_bytes"] == 44
    assert result.metrics["edge_cloud_data_bytes"] == cloud_bytes
    assert result.simulator._pending_bytes == {}
    assert result.metrics["wall_time"] == pytest.approx(
        result.metrics["communication_time"]
    )


@pytest.mark.parametrize("budget", [0, 1, 2, 4])
def test_generation_budget_bounds_final_output(run_cee, budget):
    prompt = (0, 0, 0)
    result = run_cee(prompt=prompt, budget=budget)
    assert result.output.tolist() == [list(prompt) + [1] * budget]
    assert result.metrics["generated_tokens"] == budget
    assert result.simulator._pending_bytes == {}
    if budget == 0:
        assert all(cache.current_length == 0 for cache in result.caches.values())
        assert result.metrics["target_forward_times"] == 0


@pytest.mark.parametrize("mode", ["per_transfer", "per_round"])
def test_final_flush_cost_is_in_wall_time_and_per_model_partition(run_cee, mode):
    result = run_cee(mode=mode)
    metrics = result.metrics
    assert result.simulator._pending_bytes == {}
    assert metrics["communication_time"] > 0
    # CPU CUDA-event replacement reports zero compute and queue is zero.
    assert metrics["wall_time"] == pytest.approx(metrics["communication_time"])
    assert metrics["wall_time"] == pytest.approx(
        metrics["little_wall_time"]
        + metrics["draft_wall_time"]
        + metrics["target_wall_time"]
        + metrics["communication_time"]
        + metrics["queuing_time"]
    )
