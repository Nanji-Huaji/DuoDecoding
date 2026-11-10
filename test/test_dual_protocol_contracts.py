"""Independent nonconstant proposal laws and real protocol accounting.

Scripted caches expose physical endpoints and absolute probability rows. No old
loop/oracle supplies expected outputs; the communication simulator is real.
"""

from types import SimpleNamespace

import pytest
import torch

import src.baselines.dual as distributed
import src.dual_decoding as local
import src.engine as engine
from src.baselines import Baselines
from src.communication import CommunicationSimulator
from src.engine import Decoding
from src.utils import rebuild_topk_uniform_probs

MODES = {"sd": "speculative_decoding", "dsd": "dist_spec", "dssd": "dist_split_spec"}
Q = torch.tensor(
    [
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.6, 0.2, 0.1],
        [0.15, 0.1, 0.65, 0.1],
        [0.1, 0.2, 0.3, 0.4],
    ]
)
P = torch.tensor(
    [
        [0.25, 0.25, 0.25, 0.25],
        [0.3, 0.3, 0.2, 0.2],
        [0.2, 0.2, 0.4, 0.2],
        [0.1, 0.1, 0.1, 0.7],
    ]
)


class Runtime(Baselines):
    def eval(self):
        pass

    def load_data(self):
        pass

    def preprocess(self, *args):
        pass

    def postprocess(self, *args):
        pass


class DirectRuntime(Decoding):
    eval = Runtime.eval
    load_data = Runtime.load_data
    preprocess = Runtime.preprocess
    postprocess = Runtime.postprocess


class Clock:
    def __init__(self, **kwargs):
        pass

    def record(self, **kwargs):
        pass

    def elapsed_time(self, other):
        return 1.0


class ScriptCache:
    created = []

    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        self.model = model
        self.device = model.device
        self.prob_history = None
        self.current_length = 0
        self.vocab_size = 4
        self.rollback_requests = []
        self.forward_calls = []
        self.sample_laws = []
        self.configuration = (temperature, top_k, top_p, kwargs)
        self.resets = 0
        self.created.append(self)

    def reset_for_new_sample(self):
        self.resets += 1
        self.current_length = 0
        self.prob_history = None

    def forward_context(self, tokens):
        assert tokens.shape[1] >= self.current_length
        self.forward_calls.append(tokens.clone())
        length = tokens.shape[1]
        rows = self.model.rows
        self.prob_history = torch.stack(
            [rows[min(i, len(rows) - 1)] for i in range(length)]
        )[None]
        self.current_length = length
        return self.prob_history[:, -1]

    def _generate(self, tokens, count, top_k):
        rows = []
        for _ in range(count):
            raw = self.forward_context(tokens)
            q = rebuild_topk_uniform_probs(raw, top_k) if top_k else raw
            rows.append(q.clone())
            self.sample_laws.append(q.clone())
            # A scripted, supported sample: tail samples exercise raw != rebuilt q.
            token = self.model.samples[(tokens.shape[1] - 2) % len(self.model.samples)]
            assert q[0, token] > 0
            tokens = torch.cat((tokens, torch.tensor([[token]])), dim=1)
        return tokens, torch.stack(rows, dim=1)

    def generate(self, prefix, gamma):
        assert self.model.kind == "draft", (
            "verifier must never generate a discarded sample"
        )
        return self._generate(prefix, gamma, None)[0]

    def generate_with_rebuilt_topk(self, prefix, gamma, top_k):
        return self._generate(prefix, gamma, top_k)

    def rollback(self, end):
        self.rollback_requests.append((end, self.current_length))
        assert 0 <= end <= self.current_length
        self.current_length = end
        if self.prob_history is not None:
            self.prob_history = self.prob_history[:, :end]


class RecordingSimulator(CommunicationSimulator):
    def __init__(self):
        super().__init__(
            bandwidth_edge_cloud=10,
            bandwidth_edge_end=10,
            bandwidth_cloud_end=10,
            dimension="Mbps",
            ntt_ms_edge_cloud=50,
        )
        self.events = []

    def transfer(self, tokens, prob, *args, **kwargs):
        self.events.append(
            (
                "transfer",
                None if tokens is None else tokens.clone(),
                None if prob is None else prob.clone(),
            )
        )
        return super().transfer(tokens, prob, *args, **kwargs)

    def simulate_transfer(self, size, link_type="edge_cloud", **kwargs):
        self.events.append(("bytes", size))
        return super().simulate_transfer(size, link_type, **kwargs)

    def send_reject_message(self, link):
        self.events.append(("reject",))
        return super().send_reject_message(link)

    def flush_round(self):
        self.events.append(("flush",))
        return super().flush_round()


def runtime(mode, *, gamma=2, budget=3, direct=False):
    result = object.__new__(DirectRuntime if direct else Runtime)
    result.args = SimpleNamespace(
        gamma=gamma,
        max_tokens=budget,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        use_cuda_graph=False,
        disable_eos_stop=True,
        batch_delay=0.01,
        comm_round_trip_mode="per_round",
        charge_residual_payload=False,
        transfer_top_k_cap=0,
        dump_network_stats=False,
    )
    result.vocab_size = 4
    result.accelerator = SimpleNamespace(is_main_process=True)
    result.num_acc_tokens = []
    result.draft_forward_times = result.target_forward_times = 0
    result.draft_model = SimpleNamespace(
        device=torch.device("cpu"), kind="draft", rows=Q, samples=[1, 2]
    )
    result.target_model = SimpleNamespace(
        device=torch.device("cpu"), kind="target", rows=P
    )
    result.tokenizer = SimpleNamespace(
        eos_token_id=99, decode=lambda ids, **kw: "".join(map(str, ids.tolist()))
    )
    return result


@pytest.fixture
def setup(monkeypatch):
    ScriptCache.created = []
    monkeypatch.setattr(engine, "KVCacheModel", ScriptCache)
    monkeypatch.setattr(distributed, "KVCacheModel", ScriptCache)
    monkeypatch.setattr(torch.cuda, "Event", Clock)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    simulator = RecordingSimulator()
    monkeypatch.setattr(
        distributed, "_make_symmetric_comm_simulator", lambda *a, **kw: simulator
    )
    laws = []

    def sample(row):
        laws.append(row.clone())
        simulator.events.append(("sample",))
        return row.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(local, "sample", sample)
    monkeypatch.setattr(distributed, "sample", sample)
    return simulator, laws


def run(instance, mode, **kwargs):
    return getattr(instance, MODES[mode])(
        torch.tensor([[0, 0]]), transfer_top_k=kwargs.pop("transfer_top_k", 0), **kwargs
    )


@pytest.mark.parametrize("mode", ["dsd", "dssd"])
@pytest.mark.parametrize("per_round", [True, False])
@pytest.mark.parametrize("charge_residual", [True, False])
def test_rejection_accounting_profiles_preserve_law_and_message_totals(
    mode, per_round, charge_residual, setup, monkeypatch
):
    simulator, laws = setup
    obj = runtime(mode)
    obj.args.comm_round_trip_mode = "per_round" if per_round else "per_transfer"
    obj.args.charge_residual_payload = charge_residual
    obj._check_stopping_criteria = lambda *a, **kw: True
    monkeypatch.setattr(torch, "rand", lambda shape, **kw: torch.tensor([[0.1, 0.9]]))
    _, metrics = run(obj, mode, use_early_stopping=True)
    # p2-q2=(.05,.10,-.25,.10); positive mass=.25.
    torch.testing.assert_close(
        laws[0], torch.tensor([[0.2, 0.4, 0.0, 0.4]], dtype=torch.float64),
    )
    if mode == "dsd":
        # prompt + candidate tokens + whole q + reject + optional p + reply
        messages = [16, 16, 32, 6] + ([16] if charge_residual else []) + [12]
    else:
        # candidate tokens/scalars + rejected p (legacy has an extra index) + reply
        messages = [24] + ([16] if charge_residual else [4, 16]) + [12]
    assert [event[1] for event in simulator.events if event[0] == "bytes"] == messages
    assert metrics["edge_cloud_data_bytes"] == sum(messages)
    assert metrics["connect_times"]["edge_cloud"] == (1 if per_round else len(messages))
    assert simulator.events[-1] == ("flush",)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("accept", [True, False])
def test_absolute_q_rows_partial_residual_and_explicit_bonus(
    mode, accept, setup, monkeypatch
):
    simulator, laws = setup
    monkeypatch.setattr(
        torch,
        "rand",
        lambda shape, **kw: torch.tensor([[0.1, 0.1 if accept else 0.9]])[
            :, : shape[1]
        ],
    )
    obj = runtime(mode)
    # Stop after the first round to inspect a genuinely nontrivial partial residual.
    obj._check_stopping_criteria = lambda *a, **kw: True
    output, metrics = run(obj, mode, use_early_stopping=True)
    assert output.tolist() == [[0, 0, 1, 2, 3] if accept else [0, 0, 1, 1]]
    expected = (
        P[3:4] if accept
        else torch.tensor([[0.2, 0.4, 0.0, 0.4]], dtype=torch.float64)
    )
    torch.testing.assert_close(laws[0], expected)
    assert metrics["draft_accepted_tokens"] == (2 if accept else 1)
    draft, target = ScriptCache.created
    assert draft.rollback_requests == [(3, 3)]
    assert target.rollback_requests == [((4 if accept else 3), 4)]
    if mode == "dsd":
        uploads = [
            e for e in simulator.events if e[0] == "transfer" and e[2] is not None
        ]
        torch.testing.assert_close(uploads[0][2], Q[None, 1:3])
        assert sum(e[0] == "reject" for e in simulator.events) == (not accept)
    if mode != "sd":
        # Full q vs scalar q + rejected p/index; response always includes token/index.
        expected_bytes = (
            (16 + 16 + 32 + (0 if accept else 6) + 12)
            if mode == "dsd"
            else (16 + 8 + (0 if accept else 4 + 16) + 12)
        )
        assert metrics["edge_cloud_data_bytes"] == expected_bytes
        assert simulator.events[-1] == ("flush",)
        assert simulator.events[-2] == ("bytes", 12)
        assert metrics["communication_time"] >= 0.05
        if mode == "dssd" and not accept:
            row_idx = next(
                i
                for i, e in enumerate(simulator.events)
                if e[0] == "transfer" and e[1] is None
            )
            sample_idx = next(
                i for i, e in enumerate(simulator.events) if e[0] == "sample"
            )
            assert row_idx < sample_idx


@pytest.mark.parametrize("mode", MODES)
def test_topk_actual_sample_law_not_raw_history_or_later_configuration(
    mode, setup, monkeypatch
):
    simulator, laws = setup
    obj = runtime(mode, gamma=1, budget=2)
    obj.draft_model.samples = [2]  # tail: raw .2, rebuilt 2/15
    monkeypatch.setattr(torch, "rand", lambda shape, **kw: torch.full(shape, 0.99))
    obj.target_model.rows = torch.tensor(
        [[0.25] * 4, [0.4, 0.4, 0.05, 0.15], [0.1, 0.2, 0.3, 0.4]]
    )
    original_forward = ScriptCache.forward_context

    def forward(cache, tokens):
        result = original_forward(cache, tokens)
        if cache.model.kind == "target":
            # Configuration/history changes AFTER the candidates were sampled
            # must not reinterpret their law for verification or residuals.
            obj.args.top_k = 3
            ScriptCache.created[0].prob_history.fill_(0.25)
        return result

    monkeypatch.setattr(ScriptCache, "forward_context", forward)
    output, _ = run(obj, mode, transfer_top_k=1)
    actual_q = torch.tensor([[2 / 15, 0.6, 2 / 15, 2 / 15]])
    residual = (obj.target_model.rows[1:2] - actual_q).clamp_min(0)
    residual /= residual.sum(dim=-1, keepdim=True)
    # Stable residual arithmetic now remains float64 until sampling.
    torch.testing.assert_close(laws[0], residual.double())
    assert output[0, 2].item() == 0
    if mode == "dssd":
        uplink = next(e for e in simulator.events if e[0] == "transfer")
        torch.testing.assert_close(uplink[2], torch.tensor([[2 / 15]]))
    elif mode == "dsd":
        uplink = next(
            e for e in simulator.events if e[0] == "transfer" and e[2] is not None
        )
        torch.testing.assert_close(uplink[2], actual_q[:, None])


@pytest.mark.parametrize("mode", MODES)
def test_gamma_zero_cross_round_stop_and_terminal_reply(mode, setup):
    simulator, laws = setup
    obj = runtime(mode, gamma=0, budget=5, direct=mode == "sd")
    obj.target_model.rows = torch.tensor(
        [[0.25] * 4, [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]]
    )
    observed = []
    original = obj._check_stopping_criteria

    def stop(tokens, sequences, prompt_len):
        observed.append(prompt_len)
        return original(tokens, sequences, prompt_len=prompt_len)

    obj._check_stopping_criteria = stop
    output, metrics = run(obj, mode, use_early_stopping=True, stop_sequences=["12"])
    assert output.tolist() == [[0, 0, 1, 2]]
    assert observed == [2, 2]
    assert metrics["target_forward_times"] == len(laws) == 2
    assert not ScriptCache.created[0].forward_calls
    if mode != "sd":
        assert [e for e in simulator.events if e[0] == "bytes"] == (
            [("bytes", 16)] if mode == "dsd" else []
        ) + [("bytes", 12)] * 2
        assert simulator.events[-1] == ("flush",)
        assert metrics["communication_time"] >= 0.1


@pytest.mark.parametrize("mode", MODES)
def test_confirmed_middle_eos_trims_physical_context_and_replies_before_flush(
    mode, setup, monkeypatch
):
    simulator, _ = setup
    obj = runtime(mode, budget=7)
    obj.args.disable_eos_stop = False
    obj.tokenizer.eos_token_id = 1
    monkeypatch.setattr(torch, "rand", lambda shape, **kw: torch.zeros(shape))
    reply_tokens = []
    original_reply = distributed._send_downlink_token

    def reply(sim, token, link):
        reply_tokens.append(token.clone())
        return original_reply(sim, token, link)

    monkeypatch.setattr(distributed, "_send_downlink_token", reply)
    output, metrics = run(obj, mode)
    assert output.tolist() == [[0, 0, 1]]
    assert metrics["generated_tokens"] == 1
    # Acceptance stats describe verified proposals, not the EOS-truncated output.
    assert metrics["draft_accepted_tokens"] == 2
    assert metrics["target_forward_times"] == 1
    assert all(cache.current_length == 3 for cache in ScriptCache.created)
    if mode != "sd":
        assert len(reply_tokens) == 1 and reply_tokens[0].item() == 1
        assert simulator.events[-2:] == [("bytes", 12), ("flush",)]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "prefix", [torch.zeros(2, 2, dtype=torch.long), torch.empty(1, 0, dtype=torch.long)]
)
def test_invalid_batch_and_empty_prefix_fail_before_model_or_rng(mode, prefix, setup):
    obj = runtime(mode)
    with pytest.raises(ValueError):
        getattr(obj, MODES[mode])(prefix)
    assert ScriptCache.created == []


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("damage", ["missing", "stale", "invalid-q"])
def test_broken_windows_fail_before_acceptance_rng(mode, damage, setup, monkeypatch):
    obj = runtime(mode)
    original_forward = ScriptCache.forward_context
    original_generate = ScriptCache.generate

    def forward(cache, tokens):
        result = original_forward(cache, tokens)
        if cache.model.kind == "target":
            if damage == "missing":
                cache.prob_history = cache.prob_history[:, :-1]
            elif damage == "stale":
                cache.prob_history = torch.cat(
                    (cache.prob_history, cache.prob_history[:, -1:]), dim=1
                )
        return result

    def generate(cache, prefix, count):
        tokens = original_generate(cache, prefix, count)
        if damage == "invalid-q":
            cache.prob_history[0, prefix.shape[1] - 1, 1] = float("nan")
        return tokens

    monkeypatch.setattr(ScriptCache, "forward_context", forward)
    monkeypatch.setattr(ScriptCache, "generate", generate)
    monkeypatch.setattr(
        torch, "rand", lambda *a, **kw: pytest.fail("RNG before validation")
    )
    with pytest.raises(ValueError):
        run(obj, mode)


@pytest.mark.parametrize("mode", ["dsd", "dssd"])
@pytest.mark.parametrize(
    "change", ["draft", "target", "temperature", "top-p", "capacity", "vocab"]
)
def test_distributed_graph_factory_reuses_only_complete_actual_identity(
    mode, change, setup
):
    obj = runtime(mode, gamma=0, budget=1)
    obj.args.use_cuda_graph = True
    run(obj, mode)
    first = ScriptCache.created[:]
    run(obj, mode)
    assert ScriptCache.created == first
    assert all(cache.resets == 1 for cache in first)
    if change in ("draft", "target"):
        model = getattr(obj, change + "_model")
        setattr(obj, change + "_model", SimpleNamespace(**vars(model)))
    elif change == "temperature":
        obj.args.temp = 0.7
    elif change == "top-p":
        obj.args.top_p = 0.8
    elif change == "capacity":
        obj.args.max_tokens += 1
    else:
        obj.vocab_size = 5
        # Exercise just construction: scripted distributions remain vocab-four.
        distributed.acquire_dual_caches(obj, "_" + mode + "_caches", 0, 0, 0)
        assert len(ScriptCache.created) == 4
        return
    run(obj, mode)
    assert len(ScriptCache.created) == 4
