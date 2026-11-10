"""Public SD/DSD/DSSD contracts using real tiny CPU models and KV caches.

No downloaded weights or fake probability history: only CUDA timing facilities
are replaced. These tests are intentionally red until A4 caller migration; they
are independent of the frozen legacy orchestration oracles.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.baselines import Baselines
from src.engine import Decoding
from src.model_gpu import KVCacheModel

MODES = {"sd": "speculative_decoding", "dsd": "dist_spec", "dssd": "dist_split_spec"}
VOCAB = 8


class _FixtureMethods:
    def eval(self):
        raise AssertionError("these tests invoke public decoding methods only")

    def load_data(self):
        raise AssertionError("no dataset loading in this CPU fixture")

    def preprocess(self, *args, **kwargs):
        raise AssertionError("fixture input is already tokenized")

    def postprocess(self, *args, **kwargs):
        raise AssertionError("assert directly on returned tokens")


class _Runtime(_FixtureMethods, Baselines):
    pass


class _DirectRuntime(_FixtureMethods, Decoding):
    pass


class _CpuClock:
    def __init__(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass

    def elapsed_time(self, other):
        return 1.0


def _model(preferred_token=None):
    config = LlamaConfig(
        vocab_size=VOCAB, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, output_hidden_states=True,
    )
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()
        if preferred_token is not None:
            # Keep RMSNorm's default unit weights; zero only attention/MLP
            # projections. Every context then has the same positive hidden row.
            model.model.embed_tokens.weight.fill_(1.0)
            for layer in model.model.layers:
                for parameter in layer.self_attn.parameters():
                    parameter.zero_()
                for parameter in layer.mlp.parameters():
                    parameter.zero_()
            model.lm_head.weight[preferred_token].fill_(1.0)
    return model


def _runtime(
    mode, gamma, budget, *, preferred_tokens=None, eos_id=None, runtime_class=_Runtime,
):
    runtime = object.__new__(runtime_class)
    runtime.args = SimpleNamespace(
        eval_mode=mode, edge_cloud_bandwidth=10, edge_end_bandwidth=10,
        cloud_end_bandwidth=10, max_tokens=budget, gamma=gamma,
        temp=0.0 if preferred_tokens else 1.0, top_k=0, top_p=0.0,
        batch_delay=0.0, seed=0, exp_name="dual-cpu-contract",
        eval_dataset="unit", little_model="unused", draft_model="tiny-cpu",
        target_model="tiny-cpu", dump_network_stats=False, use_cuda_graph=False,
        disable_eos_stop=eos_id is None, disable_rl_update=True,
    )
    runtime.accelerator = SimpleNamespace(is_main_process=False)
    runtime.vocab_size = VOCAB
    runtime.num_acc_tokens = []
    runtime.draft_forward_times = 0
    runtime.target_forward_times = 0
    runtime.color_print = lambda *args, **kwargs: None
    runtime.tokenizer = SimpleNamespace(eos_token_id=eos_id)
    draft_token, target_token = preferred_tokens or (None, None)
    runtime.draft_model = _model(draft_token)
    runtime.target_model = _model(target_token)
    return runtime


@pytest.fixture(autouse=True)
def _cpu_only_clock_and_rng(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", _CpuClock)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1203)
        yield


def _run(runtime, mode):
    prefix = torch.tensor([[1, 2]], dtype=torch.long)
    method = runtime.get_decoding_method()
    assert method.__func__ is getattr(runtime, MODES[mode]).__func__
    output, metrics = method(prefix, transfer_top_k=0)
    assert torch.equal(output[:, :2], prefix)
    assert metrics["generated_tokens"] == output.shape[1] - 2
    return output, metrics


def test_registered_sd_runs_on_direct_decoding_subclass_without_baselines():
    runtime = _runtime("sd", gamma=2, budget=3, runtime_class=_DirectRuntime)
    assert not isinstance(runtime, Baselines)
    output, metrics = _run(runtime, "sd")
    assert output.shape[1] == 5
    assert metrics["target_forward_times"] == 1
    assert metrics["draft_accepted_tokens"] == 2


@pytest.mark.parametrize("mode", MODES)
def test_gamma_zero_fills_budget_by_target_only_rounds(mode):
    runtime = _runtime(mode, gamma=0, budget=4)
    output, metrics = _run(runtime, mode)
    assert output.shape[1] == 6
    assert metrics["target_forward_times"] == 4
    assert metrics["draft_generated_tokens"] == 0
    assert metrics["draft_accepted_tokens"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_verification_does_not_sample_discarded_tokens(mode, monkeypatch):
    runtime = _runtime(mode, gamma=2, budget=5)
    sample_calls = []
    original = torch.multinomial

    def recording_sample(probabilities, *args, **kwargs):
        sample_calls.append(probabilities.detach().clone())
        return original(probabilities, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", recording_sample)
    output, metrics = _run(runtime, mode)
    assert output.shape[1] == 7
    # q == p == uniform: all draft samples are accepted; every sample must
    # correspond to exactly one returned token, not a thrown-away verifier draw.
    assert metrics["draft_generated_tokens"] == metrics["draft_accepted_tokens"]
    assert len(sample_calls) == 5


@pytest.mark.parametrize("mode", MODES)
def test_zero_target_support_rejects_at_zero_uniform_in_public_loop(mode, monkeypatch):
    runtime = _runtime(mode, gamma=2, budget=3, preferred_tokens=(3, 4))
    monkeypatch.setattr(
        torch, "rand", lambda *shape, **kwargs: torch.zeros(*shape, **kwargs),
    )
    output, _ = _run(runtime, mode)
    assert output[0, 2:].tolist() == [4, 4, 4]


@pytest.mark.parametrize("mode", MODES)
def test_zero_budget_performs_no_model_forward_or_sampling(mode, monkeypatch):
    runtime = _runtime(mode, gamma=2, budget=0)

    def unexpected(*args, **kwargs):
        raise AssertionError("zero budget must not forward or sample")

    monkeypatch.setattr(runtime.draft_model, "forward", unexpected)
    monkeypatch.setattr(runtime.target_model, "forward", unexpected)
    monkeypatch.setattr(torch, "multinomial", unexpected)
    output, metrics = _run(runtime, mode)
    assert output.shape[1] == 2
    assert metrics["draft_forward_times"] == metrics["target_forward_times"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_unconfirmed_draft_eos_cannot_end_generation(mode):
    runtime = _runtime(mode, gamma=2, budget=3, preferred_tokens=(3, 4), eos_id=3)
    output, _ = _run(runtime, mode)
    assert output[0, 2:].tolist() == [4, 4, 4]


@pytest.mark.parametrize("mode", MODES)
def test_confirmed_eos_stops_before_another_target_round(mode):
    runtime = _runtime(mode, gamma=2, budget=5, preferred_tokens=(3, 3), eos_id=3)
    forwards = []
    runtime.target_model.register_forward_pre_hook(
        lambda model, args: forwards.append(1)
    )
    output, metrics = _run(runtime, mode)
    assert output[0, 2:].tolist() == [3]
    assert len(forwards) == metrics["target_forward_times"] == 1


@pytest.mark.parametrize("mode", MODES)
def test_rollback_requests_do_not_include_unforwarded_tokens(mode, monkeypatch):
    runtime = _runtime(mode, gamma=2, budget=3)
    requests = []
    original = KVCacheModel.rollback

    def recording_rollback(cache, end_pos):
        requests.append((end_pos, cache.current_length))
        return original(cache, end_pos)

    monkeypatch.setattr(KVCacheModel, "rollback", recording_rollback)
    _run(runtime, mode)
    assert requests
    assert all(0 <= end <= actually_forwarded for end, actually_forwarded in requests)


@pytest.mark.parametrize("mode", MODES)
def test_real_bfloat16_softmax_uses_actual_sampling_law_on_rejection(mode, monkeypatch):
    runtime = _runtime(mode, gamma=1, budget=2, preferred_tokens=(0, 0))
    runtime.args.temp = 1.0
    p_logits = [
        -0.036865234375, 0.01409912109375, -0.1650390625, 0.046875,
        0.11669921875, -0.126953125, 0.03369140625, -0.05908203125,
    ]
    q_logits = [-0.034912109375, *p_logits[1:]]
    for model, logits in (
        (runtime.draft_model, q_logits), (runtime.target_model, p_logits),
    ):
        model.bfloat16()
        with torch.no_grad():
            weights = torch.tensor(logits, dtype=torch.bfloat16)[:, None] / 8
            model.lm_head.weight.copy_(weights.expand_as(model.lm_head.weight))

    runtime.args.charge_residual_payload = False
    runtime.args.comm_round_trip_mode = "per_transfer"
    payloads = []
    if mode != "sd":
        import src.baselines.dual as dual_module

        factory = dual_module._make_symmetric_comm_simulator

        def recording_factory(*args, **kwargs):
            simulator = factory(*args, **kwargs)
            transfer = simulator.transfer

            def recording_transfer(tokens, probabilities, *args, **kwargs):
                if probabilities is not None:
                    payloads.append(
                        (tokens is not None, probabilities.detach().clone())
                    )
                return transfer(tokens, probabilities, *args, **kwargs)

            simulator.transfer = recording_transfer
            return simulator

        monkeypatch.setattr(
            dual_module, "_make_symmetric_comm_simulator", recording_factory,
        )
    raw_target_rows, sample_rows = [], []
    runtime.target_model.register_forward_hook(
        lambda model, args, output: raw_target_rows.append(
            output.logits.detach().softmax(dim=-1).clone()
        )
    )

    def supported_draw(probabilities, num_samples=1):
        sample_rows.append(probabilities.detach().clone())
        token = 0 if len(sample_rows) == 1 else 1
        assert num_samples == 1 and probabilities[0, token] > 0
        return torch.tensor([[token]], device=probabilities.device)

    monkeypatch.setattr(torch, "multinomial", supported_draw)
    monkeypatch.setattr(
        torch, "rand", lambda size, **kwargs: torch.full(size, 0.998046875, **kwargs),
    )
    output, metrics = _run(runtime, mode)
    assert output[0, 2:].tolist() == [1, 1]
    assert metrics["draft_accepted_tokens"] == 0
    assert metrics["target_forward_times"] == 2
    assert len(sample_rows) == 3  # draft, rejection continuation, target-only tail
    assert raw_target_rows[0].dtype == torch.bfloat16
    assert torch.isfinite(sample_rows[1]).all()
    assert sample_rows[1][0, 0] == 0 and (sample_rows[1][0, 1:] > 0).all()
    # The final bonus/target-only sample must use the same preprocessing as AR.
    target_weights = raw_target_rows[-1][:, -1].float()
    expected_tail = target_weights / target_weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(sample_rows[-1], expected_tail, rtol=0, atol=0)
    if mode == "dsd":
        assert payloads[0][1].dtype == torch.bfloat16  # full raw weights are sufficient
        assert payloads[0][1].numel() * payloads[0][1].element_size() == 16
        assert metrics["edge_cloud_data_bytes"] == 16 + 8 + 16 + 6 + 12 + 12
    elif mode == "dssd":
        actual_q = sample_rows[0].double()
        expected_scalar = (actual_q / actual_q.sum(dim=-1, keepdim=True))[:, :1].float()
        assert payloads[0][0] and payloads[0][1].dtype == torch.float32
        torch.testing.assert_close(payloads[0][1], expected_scalar, rtol=0, atol=0)
        assert payloads[0][1].numel() * payloads[0][1].element_size() == 4
        assert payloads[1][1].dtype == torch.bfloat16  # raw target weights downlink
        assert metrics["edge_cloud_data_bytes"] == 12 + 4 + 16 + 12 + 12
