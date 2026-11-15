"""Registered CEE-CUHLM behavior with real CPU Llama models and KV caches.

This characterizes the retained approximate protocol, NOT target-law correctness.
Only CUDA clocks and (where specified) the uncertainty gate are substituted;
communication charges, probability histories, rollback and sampling stay real.
"""

import importlib
import inspect
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
        dump_network_stats=False,
        disable_rl_update=True,
        disable_eos_stop=eos is None,
    )
    obj.vocab_size = 8
    obj.tokenizer = SimpleNamespace(eos_token_id=eos)
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


def _observe(monkeypatch, flags=(False,)):
    """Observe real dependencies at the registered entry's resource boundary."""
    observations = SimpleNamespace(caches={}, simulators=[], draws=[], uniforms=[])
    module = importlib.import_module(inspect.unwrap(Baselines.cee_cuhlm).__module__)

    class RecordingCache(KVCacheModel):
        def __init__(self, model, temperature, top_k, top_p, **kwargs):
            super().__init__(model, temperature, top_k, top_p, **kwargs)
            self.configuration = (temperature, top_k, top_p)
            self.generations = []
            observations.caches[model.contract_role] = self

        def generate(self, prefix, gamma):
            result = super().generate(prefix, gamma)
            self.generations.append((prefix.shape[1], gamma, result.shape[1]))
            return result

    class RecordingBilling:
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.messages = []
            self.gates = []
            observations.simulators.append(self)

        def simulate_transfer(self, size, link_type="edge_cloud", **kwargs):
            self.messages.append((link_type, float(size)))
            return super().simulate_transfer(size, link_type, **kwargs)

        def calculate_uncertainty(self, *args, **kwargs):
            if flags is None:
                return super().calculate_uncertainty(*args, **kwargs)
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

    monkeypatch.setattr(module, "KVCacheModel", RecordingCache)
    monkeypatch.setattr(module, "CUHLM", Plain)
    monkeypatch.setattr(module, "PreciseCUHLM", Precise)
    monkeypatch.setattr(torch, "multinomial", record_categorical)
    monkeypatch.setattr(torch, "rand", record_uniform)
    return observations


def _run(obj, *, precise=False, transfer_top_k=0, **kwargs):
    method = obj.get_decoding_method()
    assert method.__func__ is obj.cee_cuhlm.__func__
    output, metrics = method(
        torch.tensor([[6, 7]]),
        transfer_top_k=transfer_top_k,
        use_precise_comm_sim=precise,
        **kwargs,
    )
    assert output[0, :2].tolist() == [6, 7]
    assert metrics["generated_tokens"] == output.shape[1] - 2
    return output, metrics


def test_all_skip_preserves_legacy_last_candidate_resampling_and_paper_messages(
    monkeypatch,
):
    obj = _runtime()
    seen = _observe(monkeypatch)
    before = torch.random.get_rng_state().clone()
    output, metrics = _run(obj)
    assert output.tolist() == [[6, 7, 1, 1, 1, 1, 1]]
    assert metrics["little_accepted_tokens"] == 2
    assert metrics["draft_accepted_tokens"] == 4  # NOT all five sampled candidates.
    assert metrics["target_forward_times"] == 0
    assert metrics["queuing_time"] == 0
    assert len(seen.simulators[0].gates) == 4
    assert seen.simulators[0].messages == [
        ("edge_end", 16.0),
        ("edge_end", 22.0),
        ("edge_end", 0.0),
        ("edge_cloud", 40.0),
        ("edge_end", 0.0),
    ]
    assert len(seen.draws) == 7  # includes discarded draft verifier sample + bonus.
    assert len(seen.uniforms) == 1 and seen.uniforms[0][0].numel() == 2
    assert all(draw[2].get("generator") is None for draw in seen.draws)
    assert not torch.equal(before, torch.random.get_rng_state())
    assert {role: cache.current_length for role, cache in seen.caches.items()} == {
        "little": 3,
        "draft": 6,
        "target": 0,
    }
    assert all(cache.configuration == (0.0, 0, 0) for cache in seen.caches.values())


@pytest.mark.parametrize("offset", [0, 1])
def test_trigger_forces_residual_after_skips_and_forwards_entire_proposal(
    monkeypatch, offset
):
    obj = _runtime(budget=offset + 1)
    seen = _observe(monkeypatch, (False,) * offset + (True,))
    output, metrics = _run(obj)
    assert output.tolist() == [[6, 7] + [1] * offset + [3]]
    assert metrics["draft_accepted_tokens"] == offset
    assert metrics["target_forward_times"] == 1
    assert metrics["queuing_time"] == 0.25
    assert seen.caches["target"].generations == [(7, 1, 8)]
    assert len(seen.draws) == 8  # discarded target sample; no target accept uniform.
    assert len(seen.uniforms) == 1
    assert seen.simulators[0].messages[-3:] == [
        ("edge_cloud", 2.75),
        ("edge_cloud", 0.0),
        ("edge_end", 0.0),
    ]
    assert all(cache.current_length == 2 + offset for cache in seen.caches.values())


def test_inner_rejection_targets_raw_draft_and_does_not_call_inner_gate(monkeypatch):
    obj = _runtime(budget=3, tokens=(1, 2, 3))
    seen = _observe(monkeypatch)
    output, metrics = _run(obj)
    assert output.tolist() == [[6, 7, 2, 2, 2]]
    assert metrics["little_accepted_tokens"] == 0
    assert metrics["draft_accepted_tokens"] == 2
    assert metrics["target_forward_times"] == 0
    assert [token for token, _ in seen.simulators[0].gates] == [2, 2]


def test_skipped_round_leaves_target_stale_until_next_trigger(monkeypatch):
    obj = _runtime(budget=6)
    seen = _observe(monkeypatch, (False,) * 4 + (True,))
    output, metrics = _run(obj)
    assert output.tolist() == [[6, 7] + [1] * 5 + [3]]
    assert metrics["target_forward_times"] == 1
    assert seen.caches["target"].generations == [(12, 1, 13)]
    assert all(cache.current_length == 7 for cache in seen.caches.values())


def test_zero_depth_retains_legacy_ungated_draft_continuation(monkeypatch):
    obj = _runtime(budget=2, gamma1=0, gamma2=0)
    seen = _observe(monkeypatch, (True,))
    output, metrics = _run(obj)
    assert output.tolist() == [[6, 7, 1, 1]]
    assert seen.simulators[0].gates == []
    assert metrics["target_forward_times"] == 0
    assert len(seen.draws) == 6  # three default-generator samples per emitted token.


def test_zero_budget_still_constructs_caches_and_charges_prompt(monkeypatch):
    seen = _observe(monkeypatch)
    output, metrics = _run(_runtime(budget=0))
    assert output.tolist() == [[6, 7]]
    assert metrics["generated_tokens"] == 0
    assert set(seen.caches) == {"little", "draft", "target"}
    assert seen.simulators[0].messages == [("edge_end", 16.0)]
    assert not seen.draws and not seen.uniforms


def test_unscripted_uncertainty_retains_default_generator_consumption(monkeypatch):
    obj = _runtime()
    obj.args.draft_target_threshold = 1.0
    seen = _observe(monkeypatch, flags=None)
    output, metrics = _run(obj)
    assert output.tolist() == [[6, 7, 1, 1, 1, 1, 1]]
    assert metrics["target_forward_times"] == 0
    assert len(seen.draws) == 11  # four real uncertainty probes + seven samples.
    assert [entry[0].numel() for entry in seen.uniforms] == [2, 20, 20, 20, 20]
    assert all(entry[1].get("generator") is None for entry in seen.uniforms)


@pytest.mark.parametrize("precise", [False, True])
def test_explicit_ntt_changes_real_public_billing_by_message_count(
    monkeypatch, precise
):
    obj = _runtime()
    _observe(monkeypatch)
    _, default_metrics = _run(obj, precise=precise)
    _, changed_metrics = _run(
        obj,
        precise=precise,
        ntt_ms_edge_cloud=13,
        ntt_ms_edge_end=7,
    )
    # One cloud prompt and four edge-end messages, including both 0B responses.
    assert changed_metrics["communication_time"] - default_metrics[
        "communication_time"
    ] == pytest.approx(0.041)
    # Public defaults are zero NTT, not PreciseCUHLM's unrelated 200/20ms defaults.
    assert default_metrics["communication_time"] < 0.001


@pytest.mark.parametrize("budget", [1, 2, 5, 6, 8])
def test_confirmed_eos_is_final_even_when_last_block_reaches_budget(
    monkeypatch, budget
):
    obj = _runtime(budget=budget, eos=1)
    seen = _observe(monkeypatch)
    output, metrics = _run(obj)
    assert output.tolist() == [[6, 7, 1]]
    assert metrics["generated_tokens"] == 1
    # The final token is unforwarded; no speculative suffix survives truncation.
    assert {role: cache.current_length for role, cache in seen.caches.items()} == {
        "little": 2,
        "draft": 2,
        "target": 0,
    }


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 5, 7])
def test_budget_truncation_also_bounds_real_cache_endpoints(monkeypatch, budget):
    seen = _observe(monkeypatch)
    output, _ = _run(_runtime(budget=budget))
    assert output.tolist() == [[6, 7] + [1] * budget]
    for cache in seen.caches.values():
        assert cache.current_length <= output.shape[1] - 1
        if cache.current_length:
            assert cache.prob_history.shape[1] == cache.current_length
            assert cache.logits_history.shape[1] == cache.current_length


@pytest.mark.parametrize("eos", [1, 6])
def test_rejected_proposal_eos_and_prompt_eos_do_not_stop_output(monkeypatch, eos):
    # Little emits EOS=1, but Draft rejects it; EOS=6 occurs only in the prompt.
    _observe(monkeypatch)
    output, _ = _run(_runtime(budget=3, tokens=(1, 2, 3), eos=eos))
    assert output.tolist() == [[6, 7, 2, 2, 2]]


def test_carried_rows_stay_raw_draft_while_fresh_rows_are_uniform_tail_codec(
    monkeypatch,
):
    obj = _runtime()
    obj.args.temp = 1.0
    # Constant, nonuniform tails distinguish raw D from R_1(D) at every context.
    weights = torch.tensor([0.2, 0.15, 0.1, 0.05, 0.0, -0.05, -0.1, 1.0])
    with torch.no_grad():
        for model in (obj.little_model, obj.draft_model):
            model.lm_head.weight.copy_(weights[:, None].expand(8, 8))
        raw = obj.draft_model(torch.tensor([[6, 7]])).logits[0, -1].softmax(-1)
    rebuilt = torch.full_like(raw, (1.0 - raw[7]).item() / 7)
    rebuilt[7] = raw[7]
    assert not torch.allclose(raw, rebuilt)
    seen = _observe(monkeypatch)
    torch.manual_seed(0)
    output, metrics = _run(obj, transfer_top_k=1)
    assert output.tolist() == [[6, 7, 7, 7, 7, 7, 7]]
    assert metrics["little_accepted_tokens"] == 2
    gates = seen.simulators[0].gates
    assert len(gates) == 4
    for token, row in gates[:3]:
        assert token == 7
        torch.testing.assert_close(row[0], raw)
    torch.testing.assert_close(gates[3][1][0], rebuilt)


def test_equal_laws_trigger_retains_documented_zero_residual_fallback(monkeypatch):
    # Characterization of a KNOWN legacy defect, not a valid acceptance oracle.
    _observe(monkeypatch, (True,))
    output, _ = _run(_runtime(budget=1, tokens=(1, 1, 1)))
    assert output.tolist() == [[6, 7, 0]]


def test_trigger_then_skip_keeps_target_behind_the_final_prefix(monkeypatch):
    seen = _observe(monkeypatch, (True, False))
    output, _ = _run(_runtime(budget=6))
    assert output.tolist() == [[6, 7, 3, 1, 1, 1, 1, 1]]
    assert seen.caches["target"].generations == [(7, 1, 8)]
    assert {role: cache.current_length for role, cache in seen.caches.items()} == {
        "little": 4,
        "draft": 7,
        "target": 2,
    }
    for cache in seen.caches.values():
        assert cache.prob_history.shape[1] == cache.current_length
        assert cache.logits_history.shape[1] == cache.current_length


@pytest.mark.parametrize("absolute_end", [2, 5])
def test_explicit_budget_is_an_absolute_endpoint_not_args_new_tokens(
    monkeypatch, absolute_end
):
    seen = _observe(monkeypatch)
    output, metrics = _run(_runtime(budget=100), max_tokens=absolute_end)
    assert output.tolist() == [[6, 7] + [1] * (absolute_end - 2)]
    assert metrics["generated_tokens"] == absolute_end - 2
    assert all(cache.current_length < absolute_end for cache in seen.caches.values())


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
