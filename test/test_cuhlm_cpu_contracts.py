"""CU-HLM single-token lifecycle on real tiny CPU Llama + real `KVCacheModel`.

No FakeCache here: the KV cache, prefill/extend routing, probability history
and endpoint lengths are the production `KVCacheModel`.  Only CUDA timing is
replaced and the communication simulator is scripted (trigger/skip flags and
the reconstructed `x_hat`), so the assertions below are about the *cache and
forward* protocol of the public `uncertainty_decoding` entry:

* a trigger forwards the old prefix (never the draft token) with no sampling;
* accept keeps the draft token only -- no full-accept target bonus;
* both caches end a round at the old `prefix_len`; the finalized token is not
  forwarded yet;
* skipped rounds leave the target behind, and its next trigger catches up over
  the whole finalized prefix in one forward (prefill first, extend later);
* `target.generate` is a hard failure (regression guard), and the number of
  sampling draws matches one draft sample per finalized draft (plus one
  residual sample on reject) -- no discarded target sample, no bonus draw.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.baselines import Baselines
from src.communication import CUHLM as _RealCUHLM
from src.model_gpu import KVCacheModel

VOCAB = 8


class _FixtureMethods:
    def eval(self):
        raise AssertionError("these tests invoke the public decoding method only")

    def load_data(self):
        raise AssertionError("no dataset loading in this CPU fixture")

    def preprocess(self, *args, **kwargs):
        raise AssertionError("fixture input is already tokenized")

    def postprocess(self, *args, **kwargs):
        raise AssertionError("assert directly on returned tokens")


class _Runtime(_FixtureMethods, Baselines):
    pass


class _CpuClock:
    def __init__(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass

    def elapsed_time(self, other):
        return 0.0


class _RecordingKVCache(KVCacheModel):
    """Production cache + endpoint/RNG observations (no behavior override)."""

    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_context_input_lens = []
        self.forward_context_rng_stable = []
        _RecordingKVCache.instances.append(self)

    def forward_context(self, input_ids):
        before = torch.random.get_rng_state()
        out = super().forward_context(input_ids)
        after = torch.random.get_rng_state()
        self.forward_context_input_lens.append(int(input_ids.shape[1]))
        self.forward_context_rng_stable.append(bool(torch.equal(before, after)))
        return out


class _ScriptedComm:
    """Scripted trigger/skip policy; cache math comes from production caches."""

    instances = []

    #: per-round should_transfer flags, consumed by strategy call index.
    flags = [True]

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
        self.edge_cloud_trace = []
        self.ntt_edge_cloud_history = []
        self.transfer_calls = []
        self.strategy_calls = []
        _ScriptedComm.instances.append(self)

    def flush_round(self):
        pass

    def set_round(self, round_idx):
        pass

    def transfer(self, tokens, probs, link_type="edge_cloud", **kwargs):
        self.transfer_calls.append(("context_transfer", None))
        return 0.0

    def simulate_transfer(self, data_size_bytes, link_type="edge_cloud", **kwargs):
        kind = "downlink" if int(data_size_bytes) == 0 else "uplink"
        self.transfer_calls.append((kind, int(data_size_bytes)))
        return 0.0

    def calculate_uncertainty(self, logits, M=20, theta_max=2.0, draft_token=None):
        return 1.0

    def determine_transfer_strategy(self, uncertainty, current_probs, draft_token=None):
        idx = len(self.strategy_calls)
        self.strategy_calls.append((uncertainty, draft_token))
        flags = _ScriptedComm.flags
        return (flags[idx] if idx < len(flags) else flags[-1]), 2

    def _get_current_probs(self, prob_history):
        return prob_history[0, -1, :]

    def compress_rebuild_probs(self, probs, k):
        # Real top-k + uniform-tail reconstruction (no 2-D shortcut).
        return _RealCUHLM.compress_rebuild_probs(probs, k)

    def send_accept_message(self, linktype):
        raise AssertionError("CU-HLM has no standalone accept message")

    def send_reject_message(self, linktype):
        raise AssertionError("CU-HLM has no standalone reject message")


def _tiny_model(preferred_token=None):
    """Zero lm_head => exact uniform law; one huge row => near one-hot law.

    Zeroing lm_head alone is enough for the uniform case: whatever the random
    transformer emits, the logits are all zero, so q == p exactly.  The one-hot
    case additionally zeroes attention/MLP so the embedding row becomes a
    constant hidden row before the huge lm_head row.
    """
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        output_hidden_states=True,
    )
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()
        if preferred_token is not None:
            model.model.embed_tokens.weight.fill_(1.0)
            for layer in model.model.layers:
                for parameter in layer.self_attn.parameters():
                    parameter.zero_()
                for parameter in layer.mlp.parameters():
                    parameter.zero_()
            model.lm_head.weight[preferred_token].fill_(1000.0)
    return model


def _runtime(*, budget, flags, draft_preferred=None, target_preferred=None):
    runtime = object.__new__(_Runtime)
    runtime.args = SimpleNamespace(
        eval_mode="uncertainty_decoding",
        edge_cloud_bandwidth=10,
        edge_end_bandwidth=10,
        cloud_end_bandwidth=10,
        max_tokens=budget,
        gamma=1,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        batch_delay=0.0,
        seed=0,
        exp_name="cuhlm-cpu-contract",
        eval_dataset="unit",
        little_model="unused",
        draft_model="tiny-cpu",
        target_model="tiny-cpu",
        dump_network_stats=False,
        use_cuda_graph=False,
        comm_round_trip_mode="per_transfer",
        uncertainty_threshold=0.8,
        disable_rl_update=True,
    )
    runtime.accelerator = SimpleNamespace(is_main_process=False)
    runtime.vocab_size = VOCAB
    runtime.num_acc_tokens = []
    runtime.draft_forward_times = 0
    runtime.target_forward_times = 0
    runtime.color_print = lambda *args, **kwargs: None
    runtime.tokenizer = SimpleNamespace(eos_token_id=None)
    runtime.draft_model = _tiny_model(draft_preferred)
    runtime.target_model = _tiny_model(target_preferred)
    _ScriptedComm.flags = list(flags)
    _ScriptedComm.instances = []
    _RecordingKVCache.instances = []
    return runtime


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    import src.baselines.cuhlm as cuhlm_module

    _ScriptedComm.instances = []
    _RecordingKVCache.instances = []
    monkeypatch.setattr(torch.cuda, "Event", _CpuClock)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(cuhlm_module, "KVCacheModel", _RecordingKVCache)
    monkeypatch.setattr(cuhlm_module, "CUHLM", _ScriptedComm)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261111)
        yield


def _run(runtime, prompt):
    prefix = torch.tensor([prompt], dtype=torch.long)
    output, metrics = runtime.uncertainty_decoding(prefix)
    assert torch.equal(output[:, : prefix.shape[1]], prefix)
    return output, metrics


def test_cuhlm_alias_resolves_to_the_same_registered_entry():
    runtime = _runtime(budget=1, flags=[False])
    runtime.args.eval_mode = "cuhlm"
    method = runtime.get_decoding_method()
    assert method.__func__ is runtime.uncertainty_decoding.__func__
    output, metrics = method(torch.tensor([[1, 2]], dtype=torch.long))
    assert output.shape == (1, 3)
    assert metrics["generated_tokens"] == 1


@pytest.mark.parametrize(
    "flags,draft_preferred,target_preferred",
    [
        ([True], None, None),  # accept
        ([True], 3, 4),  # reject
        ([False], None, None),  # skip
    ],
)
def test_budget_one_finalizes_exactly_one_token(
    flags, draft_preferred, target_preferred
):
    runtime = _runtime(
        budget=1,
        flags=flags,
        draft_preferred=draft_preferred,
        target_preferred=target_preferred,
    )
    output, metrics = _run(runtime, [1, 2])

    assert output.shape == (1, 3)
    assert metrics["generated_tokens"] == 1
    assert metrics["draft_generated_tokens"] == 1
    assert metrics["draft_accepted_tokens"] == (
        0 if draft_preferred == 3 else 1
    )
    assert metrics["target_forward_times"] == (0 if flags == [False] else 1)


def _caches():
    # cuhlm.py's builders are created draft-then-target (eager path).
    assert len(_RecordingKVCache.instances) == 2
    draft, target = _RecordingKVCache.instances
    return draft, target


def _model_forward_lens(model):
    lens = []

    def hook(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        lens.append(int(ids.shape[1]))

    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    return lens, handle


def test_zero_budget_performs_no_model_forward_or_sampling(monkeypatch):
    runtime = _runtime(budget=0, flags=[True])
    draft_lens, draft_handle = _model_forward_lens(runtime.draft_model)
    target_lens, target_handle = _model_forward_lens(runtime.target_model)

    def unexpected(*args, **kwargs):
        raise AssertionError("zero budget must not forward or sample")

    monkeypatch.setattr(torch, "multinomial", unexpected)
    output, metrics = _run(runtime, [1, 2])
    draft_handle.remove()
    target_handle.remove()

    assert output.shape == (1, 2)
    assert draft_lens == [] and target_lens == []
    assert metrics["generated_tokens"] == 0
    assert metrics["draft_forward_times"] == metrics["target_forward_times"] == 0
    assert _ScriptedComm.instances[0].transfer_calls == []


def test_accept_forwards_only_prefix_and_keeps_one_token(monkeypatch):
    runtime = _runtime(budget=1, flags=[True])
    target_lens, target_handle = _model_forward_lens(runtime.target_model)
    samples = []
    original_multinomial = torch.multinomial

    def recording_multinomial(probs, *args, **kwargs):
        samples.append(probs.detach().clone())
        return original_multinomial(probs, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", recording_multinomial)
    output, metrics = _run(runtime, [1, 2])
    target_handle.remove()

    assert output.shape == (1, 3)
    assert metrics["generated_tokens"] == 1
    assert metrics["draft_accepted_tokens"] == 1
    assert metrics["target_forward_times"] == 1
    # target saw the 2-token prefix only -- not prefix+draft and not a bonus
    assert target_lens == [2]
    # exactly one draw: the draft sample. No discarded target sample, no bonus.
    assert len(samples) == 1

    draft_cache, target_cache = _caches()
    assert target_cache.forward_context_input_lens == [2]
    assert target_cache.forward_context_rng_stable == [True]
    assert target_cache.current_length == 2
    assert target_cache._current_seq_len == 2
    assert tuple(target_cache.prob_history.shape) == (1, 2, VOCAB)
    assert tuple(target_cache.logits_history.shape) == (1, 2, VOCAB)
    # draft cache also stopped at the forwarded prefix, not at the output token
    assert draft_cache.current_length == 2
    assert tuple(draft_cache.prob_history.shape) == (1, 2, VOCAB)
    # no rollback is needed by the single-token lifecycle (old prefix endpoint)
    assert draft_cache.current_length == target_cache.current_length == 2


def test_skip_rounds_then_target_catch_up_over_whole_prefix():
    runtime = _runtime(budget=3, flags=[False, False, True])
    target_lens, target_handle = _model_forward_lens(runtime.target_model)
    output, metrics = _run(runtime, [1, 2])
    target_handle.remove()

    assert output.shape == (1, 5)
    assert metrics["generated_tokens"] == 3
    assert metrics["target_forward_times"] == 1
    assert metrics["draft_forward_times"] == 3
    assert metrics["draft_accepted_tokens"] == 3  # two skips + one accepted draft

    draft_cache, target_cache = _caches()
    # Neither cache had seen rounds 1-2; round 3 prefilled the whole 4-token
    # finalized prefix ("catch up"), not merely the last token.
    assert target_cache.forward_context_input_lens == [4]
    assert target_lens == [4]
    assert target_cache.current_length == 4
    assert target_cache._current_seq_len == 4
    assert tuple(target_cache.prob_history.shape) == (1, 4, VOCAB)
    # draft advanced every round (generate), still at the round-3 prefix (4);
    # the finalized round-3 token (output len 5) is not forwarded yet.
    assert draft_cache.current_length == 4
    assert tuple(draft_cache.prob_history.shape) == (1, 4, VOCAB)


def test_later_trigger_extends_target_by_the_skipped_suffix():
    runtime = _runtime(budget=4, flags=[True, False, False, True])
    target_lens, target_handle = _model_forward_lens(runtime.target_model)
    output, metrics = _run(runtime, [1, 2])
    target_handle.remove()

    assert output.shape == (1, 6)
    assert metrics["generated_tokens"] == 4
    assert metrics["target_forward_times"] == 2
    assert metrics["draft_forward_times"] == 4
    assert metrics["draft_accepted_tokens"] == 4

    draft_cache, target_cache = _caches()
    # round 1 prefill len 2; round 4 extends the 3-token suffix (positions
    # 2,3,4) because target was left at 2 while prefix grew to 5.
    assert target_cache.forward_context_input_lens == [2, 5]
    assert target_lens == [2, 3]
    assert target_cache.current_length == 5
    assert target_cache._current_seq_len == 5
    assert tuple(target_cache.prob_history.shape) == (1, 5, VOCAB)
    assert draft_cache.current_length == 5


def test_reject_keeps_one_residual_token_and_prefix_endpoint(monkeypatch):
    runtime = _runtime(
        budget=1, flags=[True], draft_preferred=3, target_preferred=4
    )
    target_lens, target_handle = _model_forward_lens(runtime.target_model)
    samples = []
    original_multinomial = torch.multinomial

    def recording_multinomial(probs, *args, **kwargs):
        samples.append(probs.detach().clone())
        return original_multinomial(probs, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", recording_multinomial)
    output, metrics = _run(runtime, [1, 2])
    target_handle.remove()

    # draft is deterministically 3, target gives it zero support -> reject and
    # residual mass sits entirely on token 4.
    assert int(output[0, -1]) == 4
    assert output.shape == (1, 3)
    assert metrics["generated_tokens"] == 1
    assert metrics["draft_accepted_tokens"] == 0
    # one draft draw + one residual draw only (no bonus sample)
    assert len(samples) == 2

    draft_cache, target_cache = _caches()
    assert target_lens == [2]
    assert target_cache.forward_context_input_lens == [2]
    assert target_cache.forward_context_rng_stable == [True]
    assert target_cache.current_length == 2
    assert tuple(target_cache.prob_history.shape) == (1, 2, VOCAB)
    assert draft_cache.current_length == 2
    # rejected path produced 1 token but only 1 draft sample; no cache advanced
    # beyond the forwarded prefix.
    assert draft_cache.current_length == 2


def test_target_generate_is_a_hard_failure(monkeypatch):
    runtime = _runtime(budget=2, flags=[True, False])
    original_generate = KVCacheModel.generate

    def guarded_generate(cache, input_ids, gamma):
        if cache._model is runtime.target_model:
            raise AssertionError(
                "target.generate is forbidden: CU-HLM uses forward_context(prefix)"
            )
        return original_generate(cache, input_ids, gamma)

    monkeypatch.setattr(KVCacheModel, "generate", guarded_generate)
    output, metrics = _run(runtime, [1, 2])

    assert output.shape == (1, 4)
    assert metrics["target_forward_times"] == 1


def test_rng_budget_matches_one_draft_sample_per_finalized_draft(monkeypatch):
    runtime = _runtime(budget=3, flags=[True, True, True])
    draws = []
    original_multinomial = torch.multinomial

    def recording_multinomial(probs, *args, **kwargs):
        draws.append(tuple(probs.shape))
        return original_multinomial(probs, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", recording_multinomial)
    output, metrics = _run(runtime, [1, 2])

    # 3 rounds, all accepted: exactly 3 draft draws. The old lifecycle would
    # draw a discarded target token AND a full-accept bonus per round (>= 6),
    # and would emit up to 2 output tokens from a budget of 1.
    assert output.shape == (1, 5)
    assert metrics["generated_tokens"] == 3
    assert metrics["draft_accepted_tokens"] == 3
    assert len(draws) == 3
    assert all(shape[-1] == VOCAB for shape in draws)


def test_rejected_draft_eos_does_not_stop_generation():
    runtime = _runtime(
        budget=2, flags=[True, False], draft_preferred=3, target_preferred=4
    )
    runtime.tokenizer = SimpleNamespace(eos_token_id=3)
    runtime.args.disable_eos_stop = False
    output, metrics = _run(runtime, [1, 2])

    # round 1: rejected draft EOS 3 -> residual 4 (must not stop)
    # round 2: skipped draft EOS 3 -> finalized EOS -> stop before round 3
    assert output[0, 2:].tolist() == [4, 3]
    assert metrics["generated_tokens"] == 2
    assert metrics["draft_forward_times"] == 2
    assert metrics["target_forward_times"] == 1


def test_finalized_eos_stops_before_another_forward():
    runtime = _runtime(budget=5, flags=[False], draft_preferred=3)
    runtime.tokenizer = SimpleNamespace(eos_token_id=3)
    runtime.args.disable_eos_stop = False
    output, metrics = _run(runtime, [1, 2])

    assert output[0, 2:].tolist() == [3]
    assert metrics["generated_tokens"] == 1
    assert metrics["draft_forward_times"] == 1


def test_prompt_eos_does_not_stop_generation():
    runtime = _runtime(budget=1, flags=[False], draft_preferred=0)
    runtime.tokenizer = SimpleNamespace(eos_token_id=3)
    runtime.args.disable_eos_stop = False
    output, metrics = _run(runtime, [3, 1])

    assert output[0, 2:].tolist() == [0]
    assert metrics["generated_tokens"] == 1


def test_zero_budget_with_prompt_eos_does_not_sample(monkeypatch):
    runtime = _runtime(budget=0, flags=[True])
    runtime.tokenizer = SimpleNamespace(eos_token_id=3)
    runtime.args.disable_eos_stop = False
    draft_lens, draft_handle = _model_forward_lens(runtime.draft_model)
    target_lens, target_handle = _model_forward_lens(runtime.target_model)

    def unexpected(*args, **kwargs):
        raise AssertionError("zero budget must not sample")

    monkeypatch.setattr(torch, "multinomial", unexpected)
    output, metrics = _run(runtime, [3, 3])
    draft_handle.remove()
    target_handle.remove()

    assert output.shape == (1, 2)
    assert draft_lens == [] and target_lens == []
    assert metrics["generated_tokens"] == 0


def test_incomplete_target_history_fails_before_repeating_a_round(monkeypatch):
    runtime = _runtime(budget=1, flags=[True])
    draft_calls = []

    class IncompleteCache(_RecordingKVCache):
        @property
        def prob_history(self):
            history = super().prob_history
            if history is not None and self._model is runtime.target_model:
                # A broken adapter omits the final verification row.
                return history[:, :-1]
            return history

        def generate(self, prefix, gamma):
            if self._model is runtime.draft_model:
                draft_calls.append(prefix.clone())
                assert len(draft_calls) == 1, "decoding repeated without progress"
            return super().generate(prefix, gamma)

    monkeypatch.setattr("src.baselines.cuhlm.KVCacheModel", IncompleteCache)
    with pytest.raises(ValueError, match="exactly the forwarded prefix"):
        _run(runtime, [1, 2])
    assert len(draft_calls) == 1
