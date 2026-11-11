"""Public TK-SLT contracts on real tiny CPU models and real `KVCacheModel`.

These tests do not script probability history: the draft/target laws come from
tiny Llama models whose lm_head is a token-keyed one-hot selector, and the KV
cache, prefill/extend routing, probability history and rollback are the
production `KVCacheModel`.  Only CUDA timing is replaced and the communication
simulator is a recording fake (as in the CU-HLM / dual CPU contracts).

The assertions cover the public `Baselines.tk_slt` / `tkslt` entry and its
`KVCacheModel.forward_context` seam:

* zero budget forwards nothing; budget one uses the target-only fallback;
* the draft loop advances with `forward_context` only (never `generate`), ends
  at `prefix + gamma - 1`, and samples the reconstructed FP16 wire row;
* all-accept keeps the target bonus, first/middle rejection samples the
  strict wire residual token and stops at the accepted endpoint;
* the finalized bonus/residual token is never forwarded in the same round;
* standalone (AS²) and remaining-budget fallback use `forward_context`, never
  `generate`, so no target RNG draw is discarded;
* EOS truncation still runs after the final sample;
* every target `forward_context` leaves the torch RNG stream untouched.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import src.baselines.tk_slt as tk_slt_module
import src.tk_slt_proposal as tk_slt_proposal_module
import src.verification as verification_module
from src.baselines import Baselines
from src.model_gpu import KVCacheModel
from src.sampling import sampling_weights

VOCAB = 8
K = 1


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


_HADAMARD = torch.tensor([[1.0]])
for _ in range(3):
    _HADAMARD = torch.cat(
        (
            torch.cat((_HADAMARD, _HADAMARD), dim=1),
            torch.cat((_HADAMARD, -_HADAMARD), dim=1),
        ),
        dim=0,
    )


def _tiny_model(preference):
    """One-hot next-token law keyed by the *last* token, without learned noise.

    Zeroing attention and MLP leaves the token embedding as the residual stream;
    the final RMSNorm applies a common positive scale to ±1 Hadamard rows.
    Each preference
    entry adds ``1000 * row[t]`` to lm_head row ``p``, so ``t -> p`` is the
    argmax and the softmax is numerically one-hot.  Unmapped tokens are uniform.
    """
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        output_hidden_states=False,
    )
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.model.embed_tokens.weight.copy_(_HADAMARD)
        for layer in model.model.layers:
            for parameter in layer.self_attn.parameters():
                parameter.zero_()
            for parameter in layer.mlp.parameters():
                parameter.zero_()
        for last_token, preferred in preference.items():
            model.lm_head.weight[preferred] += _HADAMARD[last_token] * 1000.0
    return model


class _RecordingKVCache(KVCacheModel):
    """Production cache plus endpoint/RNG observations (no math override)."""

    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_calls = []
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

    def generate(self, input_ids, gamma):
        self.generate_calls.append((int(input_ids.shape[1]), int(gamma)))
        return super().generate(input_ids, gamma)


class _RecordingComm:
    instances = []

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
        self.edge_cloud_trace = []
        self.ntt_edge_cloud_history = []
        self.transfer_calls = []
        self.stats = {"edge_cloud": [], "edge_end": [], "cloud_end": []}
        self.transfer_top_k = None
        self.coalesce_rounds = False
        _RecordingComm.instances.append(self)

    def flush_round(self):
        pass

    def set_round(self, round_idx):
        pass

    def transfer(self, tokens, probs, link_type="edge_cloud", **kwargs):
        self.transfer_calls.append(("context", link_type, kwargs))
        return 0.0

    def simulate_transfer(self, data_size_bytes, link_type="edge_cloud", **kwargs):
        kind = "downlink" if int(data_size_bytes) == 0 else "uplink"
        self.transfer_calls.append((kind, link_type, kwargs, int(data_size_bytes)))
        if kind == "uplink":
            self.stats["edge_cloud"].append(
                {"data_size_bytes": int(data_size_bytes), "tx_time": 0.0}
            )
        return 0.0

    def uplinks(self):
        return [call for call in self.transfer_calls if call[0] == "uplink"]

    def downlinks(self):
        return [call for call in self.transfer_calls if call[0] == "downlink"]


def _runtime(
    *,
    budget,
    gamma,
    draft_preference,
    target_preference,
    eos_id=None,
    odld=False,
):
    runtime = object.__new__(_Runtime)
    runtime.args = SimpleNamespace(
        eval_mode="tk_slt",
        edge_cloud_bandwidth=10,
        edge_end_bandwidth=10,
        cloud_end_bandwidth=10,
        max_tokens=budget,
        gamma=gamma,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        batch_delay=0.0,
        seed=0,
        exp_name="tk-slt-cpu-contract",
        eval_dataset="unit",
        little_model="unused",
        draft_model="tiny-cpu",
        target_model="tiny-cpu",
        dump_network_stats=False,
        use_cuda_graph=False,
        graph_verify_sizes=None,
        comm_round_trip_mode="per_transfer",
        disable_eos_stop=False,
        disable_rl_update=True,
        protocol_deviations=(),
        tk_slt_odld=odld,
    )
    runtime.accelerator = SimpleNamespace(is_main_process=False)
    runtime.vocab_size = VOCAB
    runtime.num_acc_tokens = []
    runtime.draft_forward_times = 0
    runtime.target_forward_times = 0
    runtime.color_print = lambda *args, **kwargs: None
    runtime.tokenizer = SimpleNamespace(eos_token_id=eos_id)
    runtime.draft_model = _tiny_model(draft_preference)
    runtime.target_model = _tiny_model(target_preference)
    _RecordingComm.instances = []
    _RecordingKVCache.instances = []
    return runtime


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    import src.baselines.simulators as simulators

    monkeypatch.setattr(torch.cuda, "Event", _CpuClock)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(tk_slt_module, "KVCacheModel", _RecordingKVCache)
    monkeypatch.setattr(simulators, "CommunicationSimulator", _RecordingComm)
    monkeypatch.setattr(
        simulators, "PreciseCommunicationSimulator", _RecordingComm
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261111)
        yield


def _run(runtime, prompt=(1, 2)):
    prefix = torch.tensor([list(prompt)], dtype=torch.long)
    output, metrics = runtime.tk_slt(prefix, transfer_top_k=K)
    assert torch.equal(output[:, : prefix.shape[1]], prefix)
    assert metrics["generated_tokens"] == output.shape[1] - prefix.shape[1]
    return output, metrics


def _caches():
    assert len(_RecordingKVCache.instances) == 2
    draft, target = _RecordingKVCache.instances
    return draft, target


def _recording_samples(monkeypatch):
    samples = []
    original = torch.multinomial

    def recording_multinomial(probabilities, *args, **kwargs):
        samples.append(probabilities.detach().clone())
        return original(probabilities, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", recording_multinomial)
    return samples


@pytest.mark.parametrize(
    ("path", "budget", "odld", "expected_draws"),
    [("fallback", 1, False, 1), ("standalone", 3, True, 4),
     ("accept", 4, False, 6), ("reject", 3, False, 7)],
)
def test_all_loop_draws_share_explicit_private_cpu_generator(
    monkeypatch, path, budget, odld, expected_draws
):
    draft = {token: (token + 1) % VOCAB for token in range(VOCAB)}
    target = (
        {token: (token + 2) % VOCAB for token in range(VOCAB)}
        if path == "reject" else draft
    )
    runtime = _runtime(
        budget=budget, gamma=1, odld=odld,
        draft_preference=draft, target_preference=target,
    )
    if odld:
        monkeypatch.setattr(
            tk_slt_module, "tk_slt_select_speculative", lambda *a, **k: (False, 1, 0.5)
        )
    draws = []
    real_sample, real_rand = torch.multinomial, torch.rand

    def sample(*args, **kwargs):
        draws.append(("sample", kwargs.get("generator")))
        return real_sample(*args, **kwargs)

    def rand(*args, **kwargs):
        draws.append(("uniform", kwargs.get("generator")))
        return real_rand(*args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", sample)
    monkeypatch.setattr(torch, "rand", rand)
    output, metrics = _run(runtime)

    assert output.shape[1] == 2 + budget
    assert len(draws) == expected_draws
    # ODLD first calibrates with one speculative round, then switches standalone.
    assert (any(kind == "uniform" for kind, _ in draws)) == (path != "fallback")
    if path == "standalone":
        assert metrics["tk_slt_standalone_rounds"] == 1
    assert all(generator is not None for _, generator in draws)
    assert all(generator is draws[0][1] for _, generator in draws)


def test_tkslt_alias_resolves_to_the_new_defining_module():
    runtime = _runtime(
        budget=1,
        gamma=2,
        draft_preference={2: 3},
        target_preference={2: 3},
    )
    runtime.args.eval_mode = "tkslt"
    method = runtime.get_decoding_method()
    function = inspect.unwrap(method.__func__)
    assert function is inspect.unwrap(Baselines.tk_slt)
    assert function.__module__ == "src.baselines.tk_slt"
    assert function.__globals__ is vars(tk_slt_module)
    output, metrics = method(torch.tensor([[1, 2]], dtype=torch.long))
    assert output.shape == (1, 3)
    assert metrics["generated_tokens"] == 1


def test_zero_budget_forwards_nothing_and_samples_nothing(monkeypatch):
    runtime = _runtime(
        budget=0,
        gamma=2,
        draft_preference={2: 3},
        target_preference={2: 3},
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("zero budget must not forward or sample")

    monkeypatch.setattr(runtime.draft_model, "forward", unexpected)
    monkeypatch.setattr(runtime.target_model, "forward", unexpected)
    monkeypatch.setattr(torch, "multinomial", unexpected)

    rng_state = torch.get_rng_state().clone()
    output, metrics = _run(runtime)
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert output.shape == (1, 2)
    assert metrics["generated_tokens"] == 0
    assert metrics["draft_forward_times"] == metrics["target_forward_times"] == 0
    assert _RecordingComm.instances[0].transfer_calls == []
    assert all(cache.current_length == 0 for cache in _RecordingKVCache.instances)


def test_budget_one_uses_target_only_fallback_and_keeps_the_token_unforwarded(
    monkeypatch,
):
    runtime = _runtime(
        budget=1,
        gamma=2,
        draft_preference={2: 3},
        target_preference={2: 3, 3: 4},
    )
    samples = _recording_samples(monkeypatch)

    output, metrics = _run(runtime)

    assert output[0].tolist() == [1, 2, 3]
    assert metrics["generated_tokens"] == 1
    assert metrics["draft_generated_tokens"] == 0
    assert metrics["target_forward_times"] == 1
    draft, target = _caches()
    assert draft.generate_calls == []
    assert draft.forward_context_input_lens == []
    assert target.generate_calls == []
    assert target.forward_context_input_lens == [2]
    assert target.forward_context_rng_stable == [True]
    # The finalized third token is not forwarded in this round.
    assert target.current_length == 2
    assert tuple(target.prob_history.shape) == (1, 2, VOCAB)
    assert len(samples) == 1
    assert _RecordingComm.instances[0].uplinks() == []


def test_all_accept_samples_one_bonus_without_a_discarded_target_draw(monkeypatch):
    runtime = _runtime(
        budget=3,
        gamma=2,
        draft_preference={2: 3, 3: 4, 4: 5},
        target_preference={2: 3, 3: 4, 4: 5},
    )
    samples = _recording_samples(monkeypatch)

    output, metrics = _run(runtime)

    assert output[0].tolist() == [1, 2, 3, 4, 5]
    assert metrics["generated_tokens"] == 3
    assert metrics["draft_generated_tokens"] == 2
    assert metrics["draft_accepted_tokens"] == 2
    assert metrics["target_forward_times"] == 1
    draft, target = _caches()
    assert target.generate_calls == []
    assert target.forward_context_input_lens == [4]
    assert target.forward_context_rng_stable == [True]
    # Both caches stop at the forwarded window; the bonus token is not forwarded.
    assert target.current_length == 4
    # The draft loop is a per-step forward_context loop, never generate.
    assert draft.generate_calls == []
    assert draft.forward_context_input_lens == [2, 3]
    assert draft.forward_context_rng_stable == [True, True]
    # It ends after the last *verified* draft token (prefix+gamma-1).
    assert draft.current_length == 3
    assert tuple(target.prob_history.shape) == (1, 4, VOCAB)
    # Two draft draws + exactly one retained bonus draw.
    assert len(samples) == 3
    comm = _RecordingComm.instances[0]
    assert len(comm.uplinks()) == 1
    assert len(comm.downlinks()) == 1


def test_first_rejection_samples_residual_and_rolls_back_to_the_prefix(
    monkeypatch,
):
    runtime = _runtime(
        budget=2,
        gamma=2,
        draft_preference={2: 3, 3: 4, 4: 5},
        target_preference={2: 4, 4: 5},
    )
    samples = _recording_samples(monkeypatch)

    output, metrics = _run(runtime)

    # Round 1 rejects the first draft and samples token 4; the one remaining
    # budget token comes from the target-only fallback.
    assert output[0].tolist() == [1, 2, 4, 5]
    assert metrics["generated_tokens"] == 2
    assert metrics["draft_generated_tokens"] == 1
    assert metrics["draft_accepted_tokens"] == 0
    draft, target = _caches()
    assert target.generate_calls == []
    assert target.forward_context_input_lens == [3, 3]
    assert target.forward_context_rng_stable == [True, True]
    # Rejected window rolled back to the old prefix; fallback extended it by one.
    assert target.current_length == 3
    assert draft.generate_calls == []
    assert draft.forward_context_input_lens == [2]
    assert draft.current_length == 2
    assert len(samples) == 3  # draft + residual + fallback


def test_middle_rejection_keeps_first_accept_and_stops_at_reject_offset(
    monkeypatch,
):
    runtime = _runtime(
        budget=3,
        gamma=2,
        draft_preference={2: 3, 3: 4, 4: 5},
        target_preference={2: 3, 3: 5, 4: 5, 5: 5},
    )
    samples = _recording_samples(monkeypatch)

    output, metrics = _run(runtime)

    assert output[0].tolist() == [1, 2, 3, 5, 5]
    assert metrics["generated_tokens"] == 3
    assert metrics["draft_generated_tokens"] == 2
    assert metrics["draft_accepted_tokens"] == 1
    draft, target = _caches()
    assert target.generate_calls == []
    assert target.forward_context_input_lens == [4, 4]
    assert target.forward_context_rng_stable == [True, True]
    assert target.current_length == 4
    assert draft.generate_calls == []
    assert draft.forward_context_input_lens == [2, 3]
    assert draft.current_length == 3
    assert len(samples) == 4  # two drafts + residual + fallback


def test_draft_wire_payload_is_sparse_fp16_and_drives_verification(monkeypatch):
    """A non-one-hot real model row: exact K FP16 slots, one shared wire law.

    The draft model's lm_head is zero, so its next-token law is uniform over the
    vocabulary (all V entries positive, the top-K tie case).  K=1 must still
    transmit exactly one FP16 value at the lowest tied id, the draft must sample
    the reconstructed wire row, and the receiver must use that same row instead
    of the dense model law.
    """
    runtime = _runtime(
        budget=2,
        gamma=1,
        draft_preference={},
        target_preference={token: (token + 1) % VOCAB for token in range(VOCAB)},
    )
    rows, payloads, windows = [], [], []
    real_encode = tk_slt_proposal_module.encode_proposal_row
    real_prepare = verification_module.prepare_verification_window

    def recording_encode(probabilities, top_k, **kwargs):
        payload = real_encode(probabilities, top_k, **kwargs)
        rows.append(probabilities.detach().clone())
        payloads.append(payload)
        return payload

    def recording_prepare(proposal, target_history):
        window = real_prepare(proposal, target_history)
        windows.append(window)
        return window

    monkeypatch.setattr(
        tk_slt_proposal_module, "encode_proposal_row", recording_encode
    )
    monkeypatch.setattr(
        tk_slt_module, "prepare_verification_window", recording_prepare
    )
    samples = _recording_samples(monkeypatch)

    output, metrics = _run(runtime)

    # draft token 0 is rejected, the strict residual emits 3, fallback emits 4
    assert output[0].tolist() == [1, 2, 3, 4]
    assert metrics["generated_tokens"] == 2
    assert metrics["draft_generated_tokens"] == 1
    assert metrics["draft_accepted_tokens"] == 0

    # the real model row is a valid non-one-hot law
    raw = rows[0]
    assert raw.shape == (1, VOCAB)
    assert int((raw > 0).sum().item()) == VOCAB
    assert abs(float(raw.sum().item()) - 1.0) < 1e-6

    # the billed payload is exactly min(K, V) real FP16 values + sorted indices
    payload = payloads[0]
    assert payload.values.dtype == torch.float16
    assert payload.values.shape == (1, K)
    assert payload.indices.tolist() == [[0]]  # uniform row: ties by lowest id
    assert payload.nbytes == 2 * K
    assert torch.equal(
        payload.dense(),
        torch.tensor([[1.0] + [0.0] * (VOCAB - 1)], dtype=torch.float16),
    )

    # the draft sampled that reconstruction; the verifier used the same row
    assert torch.equal(samples[0], sampling_weights(payload.dense()))
    proposal = windows[0].proposal
    assert torch.equal(proposal.probabilities[0, 0], payload.dense()[0])
    assert torch.equal(proposal.sampling_weights[0, 0], samples[0][0])
    assert proposal.selected_probabilities()[0, 0].item() == 1.0
    # the dense raw model law never reached the receiver
    assert not torch.equal(proposal.probabilities[0, 0].float(), raw[0])

    # billing covers exactly the transmitted values (K * 2 B)
    comm = _RecordingComm.instances[0]
    uplink = comm.uplinks()[0]
    assert uplink[2]["topk"] == K
    assert uplink[2]["draft_len"] == 1
    assert uplink[3] == payload.nbytes

    # the draft loop is forward_context only and ends at prefix + gamma - 1
    draft, target = _caches()
    assert draft.generate_calls == []
    assert draft.forward_context_input_lens == [2]
    assert draft.current_length == 2
    assert target.generate_calls == []
    assert len(samples) == 3  # draft + strict residual + fallback


def test_eos_truncates_the_generated_suffix_after_the_round():
    runtime = _runtime(
        budget=3,
        gamma=2,
        draft_preference={2: 3, 3: 3, 4: 3},
        target_preference={2: 3, 3: 3, 4: 3},
        eos_id=3,
    )

    output, metrics = _run(runtime)

    assert output[0].tolist() == [1, 2, 3]
    assert metrics["generated_tokens"] == 1
    # The forward still covered the verified window; EOS truncation is final.
    draft, target = _caches()
    assert draft.generate_calls == []
    assert draft.forward_context_input_lens == [2, 3]
    assert target.generate_calls == []
    assert target.forward_context_input_lens == [4]


def test_odld_standalone_uses_forward_context_and_no_draft_round(monkeypatch):
    runtime = _runtime(
        budget=3,
        gamma=1,
        draft_preference={2: 3, 3: 4, 4: 5},
        target_preference={2: 3, 3: 4, 4: 5, 5: 5},
        odld=True,
    )
    monkeypatch.setattr(
        tk_slt_module,
        "tk_slt_select_speculative",
        lambda *args, **kwargs: (False, 1, 0.5),
    )
    samples = _recording_samples(monkeypatch)

    output, metrics = _run(runtime)

    assert output[0].tolist() == [1, 2, 3, 4, 5]
    assert metrics["generated_tokens"] == 3
    assert metrics["tk_slt_standalone_rounds"] == 1
    assert metrics["tk_slt_gamma_star_final"] == 1
    draft, target = _caches()
    # One DSD draft round only; the standalone round forwards target directly.
    assert draft.generate_calls == []
    assert draft.forward_context_input_lens == [2]
    assert target.generate_calls == []
    assert target.forward_context_input_lens == [3, 4]
    assert target.forward_context_rng_stable == [True, True]
    assert target.current_length == 4
    assert len(samples) == 3  # draft + bonus + standalone
