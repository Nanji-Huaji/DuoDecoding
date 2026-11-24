"""Single-request SpecEdge: target-authoritative tokens, cache reuse and wire cost."""

import pytest
import torch
from test_dual_model_cpu_contracts import VOCAB, _runtime

from src.experiments.summaries import run_identity
from src.metrics import accumulate_metrics, get_empty_metrics

PREFIX = torch.tensor([[1, 2]])


def chain_runtime(*, lookahead=None, budget=9, wrong_after=None, eos=None):
    run = _runtime(
        "specedge", gamma=2, budget=budget, preferred_tokens=(1, 1), eos_id=eos
    )
    run.args.specedge_lookahead = lookahead
    # Real transformer/KV path; make the greedy transition i -> i+1 observable.
    # The draft may differ at exactly one context. A stale cached row then uses
    # the wrong last-token embedding even if the model shapes still look valid.
    with torch.no_grad():
        for model in (run.draft_model, run.target_model):
            model.model.embed_tokens.weight.copy_(torch.eye(VOCAB))
            head = model.lm_head.weight
            head.zero_()
            for token in range(VOCAB):
                offset = 2 if model is run.draft_model and token == wrong_after else 1
                head[(token + offset) % VOCAB, token] = 1
    return run


@pytest.mark.parametrize(
    "wrong_after,accepted,reused",
    [(None, 6, 4), (2, 5, 3), (4, 6, 2)],
    ids=["align", "reject", "wrong-bonus"],
)
def test_target_ids_survive_reuse_and_rollback(wrong_after, accepted, reused):
    run = chain_runtime(wrong_after=wrong_after)
    target_calls = []
    run.target_model.register_forward_pre_hook(lambda *args: target_calls.append(1))
    ids, metrics = run.get_decoding_method()(PREFIX)
    assert ids.tolist() == [[1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]]
    assert len(target_calls) == metrics["target_forward_times"] == metrics["loop_times"]
    assert metrics["draft_accepted_tokens"] == accepted
    extra = metrics["specedge"]
    assert extra["reused_tokens"] == reused
    assert (
        metrics["draft_forward_times"]
        == metrics["draft_generated_tokens"] + extra["proactive_tokens"] - reused
    )
    assert 0 < extra["overlap_time"] <= extra["proactive_time"]
    assert (
        extra["overlap_time"]
        <= metrics["communication_time"]
        + metrics["target_computation_time"]
        + metrics["queuing_time"]
    )
    assert metrics["wall_time"] == pytest.approx(
        metrics["computation_time"]
        + metrics["communication_time"]
        + metrics["queuing_time"]
        - extra["overlap_time"]
    )


@pytest.mark.parametrize("round_trip", ["per_transfer", "per_round"])
def test_token_only_wire_cost(round_trip):
    run = chain_runtime(lookahead=0, budget=6)
    run.args.comm_round_trip_mode = round_trip
    _, metrics = run.get_decoding_method()(PREFIX, transfer_top_k=1)
    # Two windows of 2, two authoritative replies of 3, and the prompt once.
    assert metrics["loop_times"] == 2
    assert (
        metrics["edge_cloud_data_bytes"] == (2 + 2 * 2 + 2 * 3) * PREFIX.element_size()
    )
    assert (
        sum(n for n, _ in metrics["comm_trace_edge_cloud"])
        == metrics["edge_cloud_data_bytes"]
    )
    assert metrics["comm_trace_edge_cloud"] == [[32, 1], [24, 1], [16, 2], [24, 2]]
    assert metrics["connect_times"]["edge_cloud"] == (
        2 if round_trip == "per_round" else 4
    )
    assert (
        metrics["specedge"]["proactive_tokens"]
        == metrics["specedge"]["overlap_time"]
        == 0
    )
    total = get_empty_metrics()
    for _ in range(2):
        accumulate_metrics(total, metrics)
    assert total["specedge"]["proposal_rounds"] == 4
    assert total["each_loop_draft_tokens"] == total["avg_draft_len"] == 2


def test_resume_key_uses_effective_lookahead():
    base = {"eval_mode": "specedge", "gamma": 3}
    assert run_identity(base) == run_identity(dict(base, specedge_lookahead=3))
    assert run_identity(base) != run_identity(dict(base, specedge_lookahead=0))


def test_stochastic_server_uses_its_own_draw_instead_of_accepting_by_ratio(monkeypatch):
    run = _runtime("specedge", gamma=2, budget=4, preferred_tokens=(3, 4))
    run.args.temp = 1.0
    run.args.specedge_lookahead = 0

    def draw(probs, num_samples=1, **kw):
        # Draft draws its favorite 3. From the positive target distribution draw
        # 5, deliberately neither its argmax (4) nor the proposed token (3).
        favorite = probs.argmax(dim=-1, keepdim=True)
        return torch.where(favorite == 3, 3, 5).expand(
            *favorite.shape[:-1], num_samples
        )

    monkeypatch.setattr(torch, "multinomial", draw)
    ids, metrics = run.get_decoding_method()(PREFIX)
    assert ids[0, 2:].tolist() == [5] * 4
    assert metrics["draft_accepted_tokens"] == 0


def test_stop_is_applied_to_confirmed_tokens_only():
    # Draft at token 2 proposes EOS=4; target corrects to 3, then confirms 4.
    ids, metrics = chain_runtime(wrong_after=2, eos=4).get_decoding_method()(PREFIX)
    assert ids.tolist() == [[1, 2, 3, 4]]
    assert metrics["generated_tokens"] == 2
    run = chain_runtime()
    pieces = {1: "\n", 2: "print", 3: "\n", 4: "printf"}
    run.tokenizer.decode = lambda tokens, **kw: "".join(
        pieces.get(i, "x") for i in tokens.tolist()
    )
    ids, _ = run.get_decoding_method()(
        PREFIX, use_early_stopping=True, stop_sequences=["\nprint"]
    )
    assert ids.tolist() == [
        [1, 2, 3, 4]
    ]  # stop inside the accepted window, before its bonus
