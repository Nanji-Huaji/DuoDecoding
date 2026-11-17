"""Registered CEE-CUHLM behavior with real CPU Llama models and KV caches.

This characterizes the CU-HLM-aligned protocol, NOT target-law correctness:
every carried and fresh candidate passes its own outer gate (skip is free,
trigger is one CU-HLM round with a prefix-only target forward), the inner
Little->Draft stage uses the strict verification kernel, and a reject discards
the whole unconsumed suffix.  Only CUDA clocks and (where specified) the
uncertainty gate are substituted; communication charges, probability
histories, rollback and sampling stay real.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.baselines import Baselines
from src.communication import CUHLM, PreciseCUHLM
from src.model_gpu import KVCacheModel


class _Runtime(Baselines):
    def eval(self):
        raise AssertionError("invoke the registered decoder only")

    def load_data(self):
        raise AssertionError("no dataset loading")

    def preprocess(self, *args):
        raise AssertionError("already tokenized")

    def postprocess(self, *args):
        raise AssertionError("inspect returned tokens")


class _CpuClock:
    def __init__(self, **kwargs):
        pass

    def record(self, **kwargs):
        pass

    def elapsed_time(self, other):
        return 1.0


def _model(token, role):
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            output_hidden_states=True,
        )
    ).eval()
    model.contract_role = role
    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(1.0)
        for layer in model.model.layers:
            for parameter in layer.self_attn.parameters():
                parameter.zero_()
            for parameter in layer.mlp.parameters():
                parameter.zero_()
        model.lm_head.weight.zero_()
        model.lm_head.weight[token].fill_(1.0)
    return model


def _runtime(budget=5, *, tokens=(1, 1, 3), eos=None, gamma1=2, gamma2=2):
    obj = object.__new__(_Runtime)
    obj.args = SimpleNamespace(
        eval_mode="cee_cuhlm",
        max_tokens=budget,
        gamma=1,
        gamma1=gamma1,
        gamma2=gamma2,
        temp=0.0,
        top_k=4,
        top_p=0.7,
        use_cuda_graph=False,
        edge_cloud_bandwidth=10,
        edge_end_bandwidth=10,
        cloud_end_bandwidth=10,
        batch_delay=0.25,
        comm_round_trip_mode="per_transfer",
        uncertainty_threshold=0.8,
        small_draft_threshold=0.8,
        draft_target_threshold=0.8,
        dump_network_stats=False,
        disable_rl_update=True,
        disable_eos_stop=eos is None,
    )
    obj.vocab_size = 8
    obj.tokenizer = SimpleNamespace(eos_token_id=eos)
    obj.accelerator = SimpleNamespace(is_main_process=False)
    for role, token in zip(("little", "draft", "target"), tokens, strict=True):
        setattr(obj, role + "_model", _model(token, role))
    return obj


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", _CpuClock)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2718)
        yield


def _observe(monkeypatch, flags=(False,), modules=("cee_cuhlm",)):
    """Observe real dependencies at the registered entry's resource boundary."""
    observations = SimpleNamespace(caches={}, simulators=[], draws=[], uniforms=[])
    patched = [
        importlib.import_module(
            f"src.baselines.{name}" if "." not in name else name
        )
        for name in modules
    ]

    class RecordingCache(KVCacheModel):
        def __init__(self, model, temperature, top_k, top_p, **kwargs):
            super().__init__(model, temperature, top_k, top_p, **kwargs)
            self.configuration = (temperature, top_k, top_p)
            self.generations = []
            self.contexts = []
            observations.caches[model.contract_role] = self

        def generate(self, prefix, gamma):
            result = super().generate(prefix, gamma)
            self.generations.append((prefix.shape[1], gamma, result.shape[1]))
            return result

        def forward_context(self, prefix):
            self.contexts.append(prefix.shape[1])
            return super().forward_context(prefix)

    class RecordingBilling:
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.messages = []
            self.gates = []
            self.probe_logits = []
            self.reconstructions = []
            observations.simulators.append(self)

        def simulate_transfer(self, size, link_type="edge_cloud", **kwargs):
            self.messages.append((link_type, float(size)))
            return super().simulate_transfer(size, link_type, **kwargs)

        def compress_rebuild_probs(self, probs, k):
            self.reconstructions.append((probs.detach().clone(), int(k)))
            return super().compress_rebuild_probs(probs, k)

        def calculate_uncertainty(self, logits, *args, **kwargs):
            self.probe_logits.append(logits.detach().clone())
            if flags is None:
                return super().calculate_uncertainty(logits, *args, **kwargs)
            return 1.0

        def determine_transfer_strategy(
            self, uncertainty, current_probs, draft_token=None
        ):
            index = len(self.gates)
            self.gates.append((draft_token, current_probs.detach().clone()))
            if flags is None:
                return super().determine_transfer_strategy(
                    uncertainty,
                    current_probs,
                    draft_token=draft_token,
                )
            return flags[min(index, len(flags) - 1)], 2

    class Plain(RecordingBilling, CUHLM):
        pass

    class Precise(RecordingBilling, PreciseCUHLM):
        pass

    categorical, uniform = torch.multinomial, torch.rand

    def record_categorical(probabilities, *args, **kwargs):
        result = categorical(probabilities, *args, **kwargs)
        observations.draws.append(
            (probabilities.detach().clone(), result.clone(), kwargs)
        )
        return result

    def record_uniform(*args, **kwargs):
        result = uniform(*args, **kwargs)
        observations.uniforms.append((result.clone(), kwargs))
        return result

    for module in patched:
        monkeypatch.setattr(module, "KVCacheModel", RecordingCache)
        monkeypatch.setattr(module, "CUHLM", Plain)
        monkeypatch.setattr(module, "PreciseCUHLM", Precise)
    monkeypatch.setattr(torch, "multinomial", record_categorical)
    monkeypatch.setattr(torch, "rand", record_uniform)
    return observations


def _run(obj, *, precise=False, transfer_top_k=0, method="cee_cuhlm", **kwargs):
    if method == "cee_cuhlm":
        function = obj.get_decoding_method()
        assert function.__func__ is obj.cee_cuhlm.__func__
    else:
        function = getattr(obj, method)
    output, metrics = function(
        torch.tensor([[6, 7]]),
        transfer_top_k=transfer_top_k,
        use_precise_comm_sim=precise,
        **kwargs,
    )
    assert output[0, :2].tolist() == [6, 7]
    assert metrics["generated_tokens"] == output.shape[1] - 2
    return output, metrics


def _gate_tokens(seen):
    return [token for token, _ in seen.simulators[0].gates]


def _cloud_messages(messages):
    return [message for message in messages if message[0] == "edge_cloud"]


def _edge_end_messages(messages):
    return [message for message in messages if message[0] == "edge_end"]


def test_all_skip_finalizes_every_candidate_without_cloud_verification(
    monkeypatch,
):
    obj = _runtime()
    seen = _observe(monkeypatch)
    before = torch.random.get_rng_state().clone()
    output, metrics = _run(obj)
    # 2 prompt + (gamma2=2 accepted Little) + 1 continuation + gamma1=2 fresh.
    assert output.tolist() == [[6, 7, 1, 1, 1, 1, 1]]
    assert metrics["little_accepted_tokens"] == 2
    # Every sampled candidate is finalized: no bonus token, no dropped last one.
    assert metrics["draft_accepted_tokens"] == 5
    assert metrics["outer_reject_times"] == 0
    assert metrics["generated_tokens"] == 5
    assert metrics["target_forward_times"] == 0
    assert metrics["queuing_time"] == 0
    # One gate per candidate, and one candidate per finalized token.
    assert len(seen.simulators[0].gates) == 5 == metrics["generated_tokens"]
    # Skips add no cloud traffic beyond CUHLM's one original-prompt upload.
    assert seen.simulators[0].messages == [
        ("edge_end", 16.0),
        ("edge_end", 22.0),
        ("edge_end", 0.0),
        ("edge_cloud", 16.0),
        ("edge_end", 0.0),
    ]
    assert metrics["edge_cloud_data_bytes"] == 16.0
    # Draws: 2 Little samples, 1 necessary continuation, 2 fresh Draft samples.
    assert len(seen.draws) == 5
    # Only the inner acceptance uniform; skips draw no outer uniform.
    assert len(seen.uniforms) == 1 and seen.uniforms[0][0].numel() == 2
    assert all(draw[2].get("generator") is None for draw in seen.draws)
    assert not torch.equal(before, torch.random.get_rng_state())
    assert {role: cache.current_length for role, cache in seen.caches.items()} == {
        "little": 3,
        "draft": 6,
        "target": 0,
    }
    configurations = {role: cache.configuration for role, cache in seen.caches.items()}
    assert configurations["little"] == (0.0, 0, 0)
    assert configurations["target"] == (0.0, 0, 0)
    # Draft uses standalone CU-HLM's filtering (transfer_top_k>0 else args.top_k).
    assert configurations["draft"] == (0.0, 4, 0.7)


def test_gated_candidates_keep_target_cache_at_or_below_finalized_prefix(
    monkeypatch,
):
    obj = _runtime()
    seen = _observe(monkeypatch)
    output, _ = _run(obj)
    # The target never ran; no cache may be left pointing past the prefix.
    for cache in seen.caches.values():
        assert cache.current_length <= output.shape[1] - 1


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 5, 7])
def test_budget_truncation_also_bounds_real_cache_endpoints(monkeypatch, budget):
    seen = _observe(monkeypatch)
    output, metrics = _run(_runtime(budget=budget))
    assert output.tolist() == [[6, 7] + [1] * budget]
    assert metrics["generated_tokens"] == budget
    assert len(seen.simulators[0].gates) == budget
    for cache in seen.caches.values():
        assert cache.current_length <= output.shape[1] - 1
        if cache.current_length:
            assert cache.prob_history.shape[1] == cache.current_length
            assert cache.logits_history.shape[1] == cache.current_length


def test_zero_budget_still_constructs_caches_and_charges_prompt(monkeypatch):
    seen = _observe(monkeypatch)
    output, metrics = _run(_runtime(budget=0))
    assert output.tolist() == [[6, 7]]
    assert metrics["generated_tokens"] == 0
    assert set(seen.caches) == {"little", "draft", "target"}
    assert seen.simulators[0].messages == [("edge_end", 16.0)]
    assert not seen.draws and not seen.uniforms and not seen.simulators[0].gates


@pytest.mark.parametrize("budget", [1, 2, 5, 6, 8])
@pytest.mark.parametrize("trigger", [False, True])
def test_confirmed_eos_stops_before_gating_any_remaining_candidates(
    monkeypatch, budget, trigger
):
    obj = _runtime(budget=budget, tokens=(1, 1, 1), eos=1)
    seen = _observe(monkeypatch, (trigger,))
    output, metrics = _run(obj)
    assert output.tolist() == [[6, 7, 1]]
    assert metrics["generated_tokens"] == len(seen.simulators[0].gates) == 1
    assert metrics["draft_accepted_tokens"] == 1
    assert metrics["outer_reject_times"] == 0
    assert metrics["target_forward_times"] == int(trigger)
    # The final token is unforwarded; no speculative suffix survives truncation.
    assert {role: cache.current_length for role, cache in seen.caches.items()} == {
        "little": 2,
        "draft": 2,
        "target": 2 if trigger else 0,
    }


@pytest.mark.parametrize("eos", [1, 6])
def test_rejected_proposal_eos_and_prompt_eos_do_not_stop_output(monkeypatch, eos):
    # Little emits EOS=1 but Draft rejects it; EOS=6 occurs only in the prompt.
    seen = _observe(monkeypatch)
    output, metrics = _run(_runtime(budget=3, tokens=(1, 2, 3), eos=eos))
    # Inner reject at the first Little token corrects to Draft's own law (2).
    assert output.tolist() == [[6, 7, 2, 2, 2]]
    assert metrics["little_accepted_tokens"] == 0
    assert metrics["draft_accepted_tokens"] == 3
    assert metrics["target_forward_times"] == 0
    assert _gate_tokens(seen) == [2, 2, 2]


def test_trigger_accepts_when_target_law_equals_the_proposal(monkeypatch):
    # P == q every candidate: ratio is 1, so a trigger must keep the candidate
    # instead of forcing a zero-residual correction (the old defect).
    obj = _runtime(budget=1, tokens=(1, 1, 1))
    seen = _observe(monkeypatch, (True,))
    output, metrics = _run(obj)
    assert output.tolist() == [[6, 7, 1]]
    assert metrics["draft_accepted_tokens"] == 1
    assert metrics["target_forward_times"] == 1
    assert metrics["queuing_time"] == 0.25
    assert _gate_tokens(seen) == [1]
    assert _cloud_messages(seen.simulators[0].messages) == [
        ("edge_cloud", 16.0),
        ("edge_cloud", 2.75),
        ("edge_cloud", 0.0),
    ]
    # Exactly one acceptance uniform, drawn after the whole window was prepared.
    assert [entry[0].shape for entry in seen.uniforms] == [(1, 2), (1, 1)]
    assert all(cache.current_length == 2 for cache in seen.caches.values())


def test_trigger_rejects_with_reconstructed_residual_and_prefix_only_target(
    monkeypatch,
):
    obj = _runtime(budget=1)
    seen = _observe(monkeypatch, (True,))
    output, metrics = _run(obj)
    # Target emits 3, the candidate is 1, so P(1) = 0 rejects at any uniform.
    assert output.tolist() == [[6, 7, 3]]
    assert metrics["draft_accepted_tokens"] == 0
    assert metrics["target_forward_times"] == 1
    # Target only forward_context()s the finalized prefix: never the candidate,
    # never a sampled-and-discarded row.
    assert seen.caches["target"].contexts == [2]
    assert seen.caches["target"].generations == []
    # One residual draw after one acceptance uniform; 0B downlink last.
    assert [entry[0].shape for entry in seen.uniforms] == [(1, 2), (1, 1)]
    assert seen.simulators[0].messages[-3:] == [
        ("edge_cloud", 2.75),
        ("edge_cloud", 0.0),
        ("edge_end", 0.0),
    ]
    # A rejected block discards its unconsumed suffix: one gate, one token.
    assert len(seen.simulators[0].gates) == 1
    assert all(cache.current_length == 2 for cache in seen.caches.values())


@pytest.mark.parametrize(
    "rejected_index,expected",
    [
        # Carried index 0/1 are accepted Little tokens; index 2 is the Draft
        # continuation (correction) token.  Both are carried candidates.
        (
            1,
            {
                "output": [6, 7, 1, 3, 1, 1],
                "target_context": [3],
                "target_end": 3,
                "reproposal_len": 4,
            },
        ),
        (
            2,
            {
                "output": [6, 7, 1, 1, 3, 1],
                "target_context": [4],
                "target_end": 4,
                "reproposal_len": 5,
            },
        ),
    ],
)
def test_carried_candidate_is_gated_and_can_be_rejected(
    monkeypatch, rejected_index, expected
):
    obj = _runtime(budget=4)
    flags = (False,) * rejected_index + (True, False)
    seen = _observe(monkeypatch, flags)
    output, metrics = _run(obj)
    # Gate 0 skips, the carried candidate at rejected_index is rejected with a
    # residual 3, and the whole remaining suffix is thrown away.
    assert output.tolist() == [expected["output"]]
    assert metrics["target_forward_times"] == 1
    # One gate per finalized token: the discarded suffix was never gated.
    assert len(seen.simulators[0].gates) == metrics["generated_tokens"] == 4
    assert _cloud_messages(seen.simulators[0].messages) == [
        ("edge_cloud", 16.0),
        ("edge_cloud", 2.75),
        ("edge_cloud", 0.0),
    ]
    # The target forwarded exactly the prefix finalized before the candidate.
    assert seen.caches["target"].contexts == [expected["target_context"][0]]
    # Both proposer caches stopped at the finalized prefix and re-proposed from
    # the residual-extended prefix (round 2 input length, never the suffix).
    assert seen.caches["little"].generations[1][0] == expected["reproposal_len"]
    assert {role: cache.current_length for role, cache in seen.caches.items()} == {
        "little": 5,
        "draft": 5,
        "target": expected["target_end"],
    }
    # Discarded prefetch is legitimate overhead, not accepted output.
    assert metrics["draft_generated_tokens"] == 2 * 3
    assert metrics["draft_accepted_tokens"] == 3
    assert metrics["outer_reject_times"] == 1
    assert (
        metrics["draft_accepted_tokens"] + metrics["outer_reject_times"]
        == metrics["generated_tokens"]
    )


def test_trigger_passes_original_q_and_reconstructed_qhat(monkeypatch):
    obj = _runtime(budget=1)
    obj.args.temp = 1.0
    obj.args.top_k = 0
    obj.args.top_p = 0.0
    weights = torch.tensor([0.2, 0.15, 0.1, 0.05, 0.0, -0.05, -0.1, 1.0])
    with torch.no_grad():
        for model in (obj.little_model, obj.draft_model):
            model.lm_head.weight.copy_(weights[:, None].expand(8, 8))
    # P == q at every position, so acceptance must not depend on the rebuilt law.
    obj.target_model.load_state_dict(obj.draft_model.state_dict())
    seen = _observe(monkeypatch, (True,))
    output, metrics = _run(obj)
    assert output.shape[1] == 3
    assert metrics["draft_accepted_tokens"] == 1
    simulator = seen.simulators[0]
    gate_token, gate_row = simulator.gates[0]
    assert gate_token == int(output[0, 2].item())
    # The codec reconstruction is built from that same original q row and k*.
    assert len(simulator.reconstructions) == 1
    reconstructed_input, compressed_k = simulator.reconstructions[0]
    assert compressed_k == 2
    torch.testing.assert_close(reconstructed_input[0, 0], gate_row)
    expected = CUHLM.compress_rebuild_probs(gate_row.unsqueeze(0).unsqueeze(0), 2)
    # Reconstructing compresses the tail, so acceptance cannot have used qhat.
    assert not torch.equal(expected[0, 0], gate_row)
    assert metrics["target_forward_times"] == 1


def test_target_catches_up_over_the_whole_finalized_prefix_after_skips(monkeypatch):
    obj = _runtime(budget=6)
    seen = _observe(monkeypatch, (True, False, False, False, False, True))
    output, metrics = _run(obj)
    # Trigger at gate 0 forwards the prompt only; the later trigger forwards the
    # entire prefix finalized in between (the target has no speculative rows).
    assert seen.caches["target"].contexts == [2, 7]
    assert metrics["target_forward_times"] == 2
    assert output.tolist() == [[6, 7, 3, 1, 1, 1, 1, 3]]
    assert len(seen.simulators[0].gates) == metrics["generated_tokens"] == 6


def test_zero_depth_still_gates_one_draft_candidate_per_round(monkeypatch):
    obj = _runtime(budget=3, gamma1=0, gamma2=0)
    seen = _observe(monkeypatch, (True,))
    output, metrics = _run(obj)
    # No Little proposal and no fresh prefetch, but the protocol still samples
    # exactly one Draft candidate per round and gates it (never a target bypass).
    assert output.tolist() == [[6, 7, 3, 3, 3]]
    assert [token for token, _ in seen.simulators[0].gates] == [1, 1, 1]
    assert metrics["little_forward_times"] == 0
    assert metrics["draft_forward_times"] == 0
    assert metrics["target_forward_times"] == 3
    assert metrics["draft_generated_tokens"] == 3
    # Every round rejects (P(1) = 0), so generated = accepts + outer rejects.
    assert metrics["draft_accepted_tokens"] == 0
    assert metrics["outer_reject_times"] == 3
    assert (
        metrics["draft_accepted_tokens"] + metrics["outer_reject_times"]
        == metrics["generated_tokens"]
    )
    assert {role: cache.current_length for role, cache in seen.caches.items()} == {
        "little": 0,
        "draft": 4,
        "target": 4,
    }


def test_zero_depth_matches_standalone_cuhlm_on_same_models_and_seed(monkeypatch):
    obj = _runtime(budget=4, gamma1=0, gamma2=0)
    # A zero outer threshold keeps the real uncertainty gate in play, so the
    # comparison covers skip, accept and reject rounds (seed 99 triggers all).
    obj.args.uncertainty_threshold = 0.0
    obj.args.draft_target_threshold = 0.0
    _observe(monkeypatch, None, modules=("cee_cuhlm", "cuhlm"))
    torch.manual_seed(99)
    cee_output, cee_metrics = obj.cee_cuhlm(
        torch.tensor([[6, 7]]), transfer_top_k=0, use_precise_comm_sim=False
    )
    torch.manual_seed(99)
    cuhlm_output, cuhlm_metrics = obj.uncertainty_decoding(
        torch.tensor([[6, 7]]), transfer_top_k=0, use_precise_comm_sim=False
    )
    # Same models, same filtering, same RNG order: identical token stream and
    # identical gate outcomes.  The inner stage draws no extra outer bonus.
    assert cee_output.tolist() == cuhlm_output.tolist()
    assert cee_metrics["generated_tokens"] == cuhlm_metrics["generated_tokens"]
    assert (
        cee_metrics["target_forward_times"] == cuhlm_metrics["target_forward_times"]
    )
    assert (
        cee_metrics["draft_accepted_tokens"] == cuhlm_metrics["draft_accepted_tokens"]
    )
    assert cee_metrics["draft_generated_tokens"] == cuhlm_metrics[
        "draft_generated_tokens"
    ]


def test_small_draft_threshold_is_not_an_inner_gate(monkeypatch):
    streams = []
    for threshold in (0.0, 1.0):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2718)
            obj = _runtime(budget=4)
            obj.args.small_draft_threshold = threshold
            seen = _observe(monkeypatch, None)
            output, metrics = _run(obj)
            streams.append(
                (
                    output.tolist(),
                    metrics["little_accepted_tokens"],
                    metrics["target_forward_times"],
                    list(seen.simulators[0].messages),
                )
            )
    assert streams[0] == streams[1]


def test_fresh_rows_come_from_the_draft_filtered_law_not_a_uniform_tail(
    monkeypatch,
):
    obj = _runtime()
    obj.args.temp = 1.0
    weights = torch.tensor([0.2, 0.15, 0.1, 0.05, 0.0, -0.05, -0.1, 1.0])
    with torch.no_grad():
        for model in (obj.little_model, obj.draft_model):
            model.lm_head.weight.copy_(weights[:, None].expand(8, 8))
        raw = obj.draft_model(torch.tensor([[6, 7]])).logits[0, -1].softmax(-1)
    # transfer_top_k=1 makes Draft's sampler law a one-hot row; the old
    # uniform-tail rebuild of the raw row is a different law.
    rebuilt = torch.full_like(raw, (1.0 - raw[7]).item() / 7)
    rebuilt[7] = raw[7]
    assert not torch.allclose(raw, rebuilt)
    seen = _observe(monkeypatch)
    torch.manual_seed(0)
    output, metrics = _run(obj, transfer_top_k=1)
    assert output.tolist() == [[6, 7, 7, 7, 7, 7, 7]]
    assert metrics["little_accepted_tokens"] == 2
    gates = seen.simulators[0].gates
    assert len(gates) == 5
    one_hot = torch.zeros(8)
    one_hot[7] = 1.0
    for token, row in gates:
        # Carried and fresh candidates both carry the actual filtered law.
        assert token == 7
        torch.testing.assert_close(row, one_hot)
    assert {role: cache.configuration for role, cache in seen.caches.items()} == {
        "little": (1.0, 0, 0),
        "draft": (1.0, 1, 0.7),
        "target": (1.0, 0, 0),
    }


@pytest.mark.parametrize("precise", [False, True])
def test_ntt_charges_follow_the_real_message_counts(monkeypatch, precise):
    obj = _runtime()
    seen = _observe(monkeypatch)
    _, default_metrics = _run(obj, precise=precise)
    messages = list(seen.simulators[0].messages)
    _, changed_metrics = _run(
        obj,
        precise=precise,
        ntt_ms_edge_cloud=13,
        ntt_ms_edge_end=7,
    )
    expected = 0.013 * len(_cloud_messages(messages)) + 0.007 * len(
        _edge_end_messages(messages)
    )
    assert (
        changed_metrics["communication_time"] - default_metrics["communication_time"]
    ) == pytest.approx(expected)
    # Public defaults are zero NTT, not PreciseCUHLM's 200/20ms defaults.
    assert default_metrics["communication_time"] < 0.001


@pytest.mark.parametrize(
    "mode,expected_factor", [("per_transfer", 2), ("per_round", 1)]
)
@pytest.mark.parametrize("first_triggers", [True, False])
def test_round_trip_accounting_never_merges_two_triggers(
    monkeypatch, mode, expected_factor, first_triggers
):
    obj = _runtime(budget=4)
    obj.args.comm_round_trip_mode = mode
    # The original prompt joins the first candidate's round. If that gate skips,
    # its prompt-only round must not be merged into a later trigger either.
    flags = (True, False, True, False) if first_triggers else (False, True, False, True)
    seen = _observe(monkeypatch, flags)
    _, metrics = _run(obj, ntt_ms_edge_cloud=50, ntt_ms_edge_end=5)
    assert metrics["target_forward_times"] == 2
    prompt_rounds = int(mode == "per_transfer" or not first_triggers)
    assert metrics["connect_times"]["edge_cloud"] == expected_factor * 2 + prompt_rounds
    cloud_bytes = sum(
        size for _, size in _cloud_messages(seen.simulators[0].messages)
    )
    assert metrics["edge_cloud_data_bytes"] == pytest.approx(cloud_bytes)


def test_candidate_laws_and_probe_logits_match_the_finalized_prefix(monkeypatch):
    # Position-dependent models expose row-offset bugs hidden by the one-hot
    # fixtures. Recompute without KV caches from the actual finalized prefix.
    obj = _runtime(budget=6)
    obj.args.temp, obj.args.top_k, obj.args.top_p = 1.0, 0, 0.0
    for role, seed in (("little", 11), ("draft", 22), ("target", 33)):
        torch.manual_seed(seed)
        model = LlamaForCausalLM(getattr(obj, f"{role}_model").config).eval()
        model.contract_role = role
        with torch.no_grad():
            model.lm_head.weight.mul_(4.0)
        setattr(obj, f"{role}_model", model)
    seen = _observe(monkeypatch, (True,))
    torch.manual_seed(7)
    output, metrics = _run(obj)
    simulator = seen.simulators[0]
    assert len(simulator.gates) == metrics["generated_tokens"] == 6
    assert metrics["outer_reject_times"] > 0
    for index, (_, q) in enumerate(simulator.gates):
        with torch.no_grad():
            logits = obj.draft_model(output[:, : 2 + index]).logits[:, -1]
        torch.testing.assert_close(
            q, logits.float().softmax(-1)[0], atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            simulator.probe_logits[index], logits, atol=1e-6, rtol=1e-5
        )
    assert not torch.allclose(simulator.gates[0][1], simulator.gates[-1][1])


def test_negative_speculation_depths_are_rejected(monkeypatch):
    _observe(monkeypatch)
    with pytest.raises(ValueError, match="gamma"):
        _run(_runtime(gamma1=-1))
    with pytest.raises(ValueError, match="gamma"):
        _run(_runtime(gamma2=-1))


def test_protocol_version_is_recorded_in_protocol_deviations(monkeypatch):
    obj = _runtime()
    _observe(monkeypatch)
    _run(obj)
    deviations = " ".join(obj.args.protocol_deviations)
    assert "cee_cuhlm_aligned_20261116" in deviations
    assert "1xNTT" in deviations


def test_explicit_budget_is_an_absolute_endpoint_not_args_new_tokens(monkeypatch):
    seen = _observe(monkeypatch)
    output, metrics = _run(_runtime(budget=100), max_tokens=5)
    assert output.tolist() == [[6, 7] + [1] * 3]
    assert metrics["generated_tokens"] == 3
    assert all(cache.current_length < 5 for cache in seen.caches.values())


def test_disabling_eos_stop_also_applies_to_the_final_block(monkeypatch):
    obj = _runtime(budget=2, eos=1)
    obj.args.disable_eos_stop = True
    seen = _observe(monkeypatch)
    output, metrics = _run(obj)
    assert output.tolist() == [[6, 7, 1, 1]]
    assert metrics["generated_tokens"] == 2
    assert {role: cache.current_length for role, cache in seen.caches.items()} == {
        "little": 3,
        "draft": 3,
        "target": 0,
    }
