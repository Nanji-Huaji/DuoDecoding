"""TK-SLT probability-wire codec contracts (pure CPU, no model, no GPU).

The legacy TK-SLT loop drew proposals from the raw top-K model law but the
receiver verified an FP16-roundtripped copy.  With ``Q=[.6,.4]`` and
``P=wire(Q)`` every acceptance ratio was exactly 1, so the first output token
kept the raw Q law while the receiver's law P differs by TVD 9.76502895e-5; a
positive ``1e-8`` tail could also be sampled while its wire support was 0.

These tests lock the replacement law at its own seam:

* exactly ``min(K, V)`` coordinates are selected *before* any draw, ties by
  lowest vocabulary index (``sampling.py`` keeps more than K positives when the
  top-K logits tie, so a K-slot wire must re-select);
* retained mass is renormalized and stored as real FP16 values, and the dense-V
  FP16 row is *the* sampler input on both sides;
* the payload is explicit values+indices (implicit indices for full V), the
  decoder validates it, and an all-zero / invalid payload fails instead of
  being repaired into a one-hot draw;
* the encoder is deterministic, RNG-free and does not mutate the model row.

Small distributions are checked against exact ``Fraction`` arithmetic rather
than Monte-Carlo frequencies.  Everything here is finite-precision float
arithmetic: no bit-exact real-arithmetic claim.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from src.decoding_rng import DecodingRng
from src.probability_laws import validate_probability_rows
from src.sampling import norm_logits, sampling_weights
from src.tk_slt_proposal import (
    TkSltProposal,
    TkSltWireRow,
    build_draft_proposal,
    encode_proposal_row,
)
from src.verification import (
    ProposalWindow,
    decide_verification,
    prepare_verification_window,
)

VOCAB = 4


def row(values, dtype=torch.float64):
    return torch.tensor([list(values)], dtype=dtype)


def curve(row_tensor):
    """Exact categorical law of one row, as Fractions of float64 weights."""
    weights = [Fraction(float(v)) for v in sampling_weights(row_tensor).reshape(-1)]
    total = sum(weights)
    return [w / total for w in weights]


def _nextafter(value, target):
    return float(
        torch.nextafter(
            torch.tensor([value], dtype=torch.float64),
            torch.tensor([target], dtype=torch.float64),
        ).item()
    )


# ---------------------------------------------------------------------------
# selection + payload shape
# ---------------------------------------------------------------------------


def test_selects_exactly_k_coordinates_and_bills_only_them():
    payload = encode_proposal_row(row([0.5, 0.3, 0.15, 0.05]), 2)

    assert payload.values.dtype == torch.float16
    assert payload.values.shape == (1, 2)
    assert payload.indices is not None and payload.indices.dtype == torch.long
    assert payload.indices.tolist() == [[0, 1]]
    # retained mass 0.8 -> [0.625, 0.375], both exact in FP16
    assert payload.values.tolist() == [[0.625, 0.375]]
    assert payload.values.element_size() == 2
    assert payload.nbytes == 4
    assert payload.vocab_size == VOCAB

    dense = payload.dense()
    assert dense.dtype == torch.float16
    assert dense.shape == (1, VOCAB)
    assert dense.tolist() == [[0.625, 0.375, 0.0, 0.0]]


def test_tie_selection_is_lowest_index_and_never_exceeds_k():
    uniform = row([0.25, 0.25, 0.25, 0.25])
    payload = encode_proposal_row(uniform, 2)
    assert payload.indices.tolist() == [[0, 1]]
    assert payload.values.tolist() == [[0.5, 0.5]]

    # clear winner + a threshold tie: the lowest tied id fills the last slot
    payload = encode_proposal_row(row([0.4, 0.2, 0.2, 0.2]), 2)
    assert payload.indices.tolist() == [[0, 1]]
    assert payload.values.tolist() == [[0.66650390625, 0.333251953125]]
    assert int((payload.values > 0).sum().item()) == 2

    # The code fact this re-selection exists for: a K-value wire cannot carry
    # every positive a tied top-K mask keeps.
    tied = norm_logits(torch.zeros(1, 1, VOCAB), 1.0, 2, 0.0)
    assert int((tied > 0).sum().item()) == VOCAB


def test_k_at_or_above_vocab_is_the_implicit_full_vocabulary_payload():
    source = row([0.5, 0.3, 0.15, 0.05])
    expected = source.half()
    for top_k in (VOCAB, VOCAB + 10, 0, -3, None):
        payload = encode_proposal_row(source, top_k)
        assert payload.indices is None
        assert payload.values.shape == (1, VOCAB)
        assert torch.equal(payload.values, expected)
        assert torch.equal(payload.dense(), expected)
        assert payload.nbytes == 2 * VOCAB


def test_declared_vocabulary_projection_conditions_the_wire_law():
    # A valid model row of width 4 that declares V=2: the wire law is the
    # conditional law on the declared vocabulary, not a silently invalid row.
    source = row([0.5, 0.25, 0.2, 0.05])
    validate_probability_rows(source, "source")
    payload = encode_proposal_row(source, 2, vocab_size=2)
    assert payload.vocab_size == 2
    assert payload.indices is None  # k == declared V -> implicit indices
    assert payload.values.tolist() == [[0.66650390625, 0.333251953125]]
    assert payload.dense().shape == (1, 2)


# ---------------------------------------------------------------------------
# codec transform: renormalize, then real FP16
# ---------------------------------------------------------------------------


def test_retained_mass_is_renormalized_before_fp16():
    payload = encode_proposal_row(row([0.5, 0.3, 0.2]), 2)
    # 0.5/0.8 and 0.3/0.8, not 0.5 and 0.3
    assert payload.values.tolist() == [[0.625, 0.375]]
    dense = payload.dense()
    assert float(dense.sum().item()) == 1.0
    # the codec transform is not the model row: no mass outside the support
    assert dense.tolist() == [[0.625, 0.375, 0.0]]


def test_positive_tail_underflow_drops_support_instead_of_inventing_mass():
    source = row([0.6, 0.4, 1e-8, 0.0])
    payload = encode_proposal_row(source, None)
    assert payload.values[0, 2].item() == 0.0
    assert payload.dense()[0, 2].item() == 0.0
    # the FP16 wire law is exactly the law of the reconstructed dense row
    assert curve(payload.dense()) == [
        Fraction(0.60009765625),
        Fraction(0.39990234375),
        Fraction(0),
        Fraction(0),
    ]

    sparse = encode_proposal_row(source, 2)
    assert sparse.indices.tolist() == [[0, 1]]  # the 1e-8 tail is never a slot


# ---------------------------------------------------------------------------
# decoder validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, indices, vocab_size",
    [
        (torch.tensor([[0.5, 0.5]], dtype=torch.float32), [[0, 1]], 4),
        (torch.tensor([0.5, 0.5], dtype=torch.float16), [[0, 1]], 4),
        (torch.tensor([[0.5, -0.5]], dtype=torch.float16), [[0, 1]], 4),
        (torch.tensor([[0.5, float("inf")]], dtype=torch.float16), [[0, 1]], 4),
        (torch.tensor([[0.0, 0.0]], dtype=torch.float16), [[0, 1]], 4),  # all underflow
        (torch.tensor([[0.25, 0.25]], dtype=torch.float16), [[0, 1]], 4),  # lost mass
        (torch.tensor([[1.0, 1.0]], dtype=torch.float16), [[0, 1]], 4),  # excess mass
        (torch.tensor([[0.5, 0.5]], dtype=torch.float16), [[0, 0]], 4),  # duplicate
        (torch.tensor([[0.5, 0.5]], dtype=torch.float16), [[1, 0]], 4),  # unsorted
        (torch.tensor([[0.5, 0.5]], dtype=torch.float16), [[0, 4]], 4),  # range
        (torch.tensor([[0.5, 0.5]], dtype=torch.float16), [[-1, 0]], 4),  # negative
        (torch.tensor([[0.5, 0.5]], dtype=torch.float16), [[0, 1]], 0),  # vocab
    ],
)
def test_decoder_rejects_invalid_payloads(values, indices, vocab_size):
    with pytest.raises(ValueError):
        TkSltWireRow(
            values=values, indices=torch.tensor(indices), vocab_size=vocab_size
        )


def test_decoder_requires_matching_shapes_and_long_indices():
    values = torch.tensor([[0.5, 0.5]], dtype=torch.float16)
    with pytest.raises(ValueError):
        TkSltWireRow(values=values, indices=torch.tensor([[0]]), vocab_size=4)
    with pytest.raises(ValueError):
        TkSltWireRow(
            values=values,
            indices=torch.tensor([[0, 1]], dtype=torch.int32),
            vocab_size=4,
        )
    with pytest.raises(ValueError):
        TkSltWireRow(
            values=values,
            indices=[[0, 1]],
            vocab_size=4,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        TkSltWireRow(
            values=torch.empty(1, 0, dtype=torch.float16),
            indices=None,
            vocab_size=4,
        )
    with pytest.raises(ValueError):
        TkSltWireRow(
            values=torch.tensor([[0.5]], dtype=torch.float16),
            indices=None,
            vocab_size=4,
        )


def test_payload_roundtrips_through_serialized_values_only():
    payload = encode_proposal_row(row([0.5, 0.3, 0.15, 0.05]), 2)
    # simulate the wire: only the billed FP16 values/indices survive
    decoded = TkSltWireRow(
        values=torch.tensor(payload.values.tolist(), dtype=torch.float16),
        indices=torch.tensor(payload.indices.tolist(), dtype=torch.long),
        vocab_size=VOCAB,
    )
    assert torch.equal(decoded.dense(), payload.dense())
    assert torch.equal(
        sampling_weights(decoded.dense()), sampling_weights(payload.dense())
    )


def test_full_vocabulary_payload_roundtrips_without_indices():
    source = row([0.5, 0.3, 0.15, 0.05])
    payload = encode_proposal_row(source, None)
    decoded = TkSltWireRow(
        values=torch.tensor(payload.values.tolist(), dtype=torch.float16),
        indices=None,
        vocab_size=VOCAB,
    )
    assert torch.equal(decoded.dense(), payload.dense())


# ---------------------------------------------------------------------------
# encoder guarantees: validates first, RNG-free, non-mutating, deterministic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([[2.0, 1.0]]),  # not a law
        torch.tensor([[0.9, 0.2]]),  # mass too large
        torch.tensor([[-0.5, 1.5]]),  # negative
        torch.tensor([[float("nan"), 1.0]]),
        torch.tensor([[0, 1]]),  # integer
        torch.tensor([[0.5, 0.5]]).reshape(2, 1),  # wrong shape
        torch.zeros(1, 0),  # empty vocabulary
    ],
)
def test_encoder_validates_the_model_row_before_any_preprocessing(bad):
    with pytest.raises(ValueError):
        encode_proposal_row(bad, 1)


def test_encoder_rejects_bool_top_k_and_empty_declared_mass():
    with pytest.raises(ValueError):
        encode_proposal_row(row([0.5, 0.5]), True)
    with pytest.raises(ValueError):
        # the declared-vocabulary projection lost all retained mass
        encode_proposal_row(row([0.0, 1.0]), 1, vocab_size=1)


def test_encoder_is_rng_free_and_does_not_mutate_the_model_row(monkeypatch):
    source = row([0.5, 0.3, 0.15, 0.05])
    before = source.clone()
    version = source._version
    state = torch.random.get_rng_state()

    def unexpected(*args, **kwargs):
        raise AssertionError("wire encoding must not draw randomness")

    for name in ("rand", "randn", "randint", "randperm", "multinomial"):
        monkeypatch.setattr(torch, name, unexpected)

    first = encode_proposal_row(source, 2)
    second = encode_proposal_row(source, 2)

    assert torch.equal(source, before)
    assert source._version == version
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(first.values, second.values)
    assert torch.equal(first.indices, second.indices)


def test_payload_is_immune_to_later_model_row_writes():
    """Round-local wire rows survive later cache/buffer writes.

    ``KVCacheModel.prob_history`` rows are views into a preallocated buffer that
    later forwards overwrite, so the codec must own its payload storage.
    """
    source = row([0.5, 0.3, 0.15, 0.05])
    payload = encode_proposal_row(source, 2)
    expected_values = payload.values.clone()
    expected_dense = payload.dense()
    source.fill_(0.25)  # simulate a later forward overwriting the model buffer
    assert torch.equal(payload.values, expected_values)
    assert torch.equal(payload.dense(), expected_dense)


def test_dense_is_deterministic_and_does_not_mutate_the_payload():
    payload = encode_proposal_row(row([0.5, 0.3, 0.15, 0.05]), 2)
    before_values = payload.values.clone()
    before_indices = payload.indices.clone()
    first = payload.dense()
    second = payload.dense(device=torch.device("cpu"))
    assert torch.equal(first, second)
    assert torch.equal(payload.values, before_values)
    assert torch.equal(payload.indices, before_indices)
    first.fill_(0.0)  # a receiver-side tensor must not alias the payload storage
    assert torch.equal(payload.dense(), second)


# ---------------------------------------------------------------------------
# the wire law is the verification law (the counterexample that motivated this)
# ---------------------------------------------------------------------------


def test_wire_row_is_the_target_law_in_the_checked_counterexample():
    payload = encode_proposal_row(row([0.6, 0.4]), 2)
    wire = payload.dense()
    assert wire.tolist() == [[0.60009765625, 0.39990234375]]

    raw_weights = sampling_weights(row([0.6, 0.4]))
    wire_weights = sampling_weights(wire)
    assert not torch.equal(raw_weights, wire_weights)  # the old mismatch

    # target P is exactly the transmitted wire row
    # rows for the forwarded context: candidate row + explicit bonus row
    target = torch.stack((wire[0], wire[0], wire[0]), dim=0).unsqueeze(0)
    proposal = ProposalWindow(
        tokens=torch.tensor([[1, 0, 1]]),
        probabilities=wire.unsqueeze(0),
        prefix_len=2,
    )
    window = prepare_verification_window(proposal, target)
    assert torch.equal(window.selected_q, window.selected_p)  # ratio is exactly 1
    decision = decide_verification(window, torch.zeros((1, 1)))
    assert decision.accepted_count == 1
    # first-token law is the wire/target law, not the raw model row
    assert torch.equal(sampling_weights(decision.continuation_probs), wire_weights)
    assert not torch.equal(sampling_weights(decision.continuation_probs), raw_weights)


def test_exact_fraction_enumeration_recovers_the_target_law():
    # Q = [.75,.25] (FP16-exact), P = [.5,.5]. Exact rejection sampling gives
    # output law q(t)*min(1,p/q) + r*normalize((P-Q)+), which must equal P.
    payload = encode_proposal_row(row([0.75, 0.25]), 2)
    wire = payload.dense()
    p_row = row([0.5, 0.5])
    target = torch.stack((p_row[0], p_row[0], p_row[0]), dim=0).unsqueeze(0)
    proposal = ProposalWindow(
        tokens=torch.tensor([[1, 0, 0]]),
        probabilities=wire.unsqueeze(0),
        prefix_len=2,
    )
    window = prepare_verification_window(proposal, target)

    q_exact = Fraction(3, 4)
    p_exact = Fraction(1, 2)
    assert Fraction(float(window.selected_q.item())) == q_exact
    assert Fraction(float(window.selected_p.item())) == p_exact
    ratio = p_exact / q_exact  # 2/3
    ratio_float = ratio.numerator / ratio.denominator

    # strict equality rejects; strictly below accepts
    equal = decide_verification(
        window, torch.tensor([[ratio_float]], dtype=torch.float64)
    )
    assert equal.accepted_count == 0
    below = decide_verification(
        window, torch.tensor([[_nextafter(ratio_float, 0.0)]], dtype=torch.float64)
    )
    assert below.accepted_count == 1
    assert torch.equal(below.continuation_probs, p_row)

    above = decide_verification(
        window, torch.tensor([[_nextafter(ratio_float, 1.0)]], dtype=torch.float64)
    )
    assert above.accepted_count == 0
    residual = above.continuation_probs.reshape(-1)
    # (P-Q)+ = [0, 1/4] -> normalize -> [0, 1]
    assert float(residual[0].item()) == 0.0
    assert abs(float(residual[1].item()) - 1.0) < 1e-15

    # enumerate the one-step output law in exact arithmetic:
    #   o_t = min(q_t, p_t) + (1 - sum_s min(q_s, p_s)) * residual_t
    q_law = [q_exact, 1 - q_exact]
    p_law = [p_exact, 1 - p_exact]
    accepted_per_token = [min(q, p) for q, p in zip(q_law, p_law, strict=True)]
    accepted_mass = sum(accepted_per_token)  # 3/4: the branch the kernel accepts
    residual_mass = 1 - accepted_mass
    residual_law = [Fraction(0), Fraction(1)]
    output_law = [
        accepted_per_token[t] + residual_mass * residual_law[t] for t in (0, 1)
    ]
    assert accepted_per_token[0] == q_exact * ratio  # q0 * min(1, p0/q0)
    assert accepted_mass == Fraction(3, 4)  # total accept probability
    assert output_law == [p_exact, 1 - p_exact]
    assert sum(output_law) == 1


# ---------------------------------------------------------------------------
# the draft forward_context loop
# ---------------------------------------------------------------------------


class _ScriptedCache:
    """Minimal cache: one scripted model row per forward_context call."""

    def __init__(self, rows):
        self.rows = [r.clone() for r in rows]
        self.calls = []

    def forward_context(self, input_ids):
        self.calls.append(input_ids.clone())
        return self.rows[min(len(self.calls) - 1, len(self.rows) - 1)].clone()


def _record_sampler(monkeypatch):
    recorded = []

    def fake_sample(probs):
        recorded.append(probs.detach().clone())
        return probs.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr("src.tk_slt_proposal.sample", fake_sample)
    return recorded


def test_build_draft_proposal_samples_each_reconstructed_wire_row(monkeypatch):
    rows = [
        torch.tensor([[0.6, 0.4, 0.0, 0.0]]),
        torch.tensor([[0.5, 0.5, 0.0, 0.0]]),
        torch.tensor([[0.25, 0.25, 0.25, 0.25]]),
    ]
    cache = _ScriptedCache(rows)
    recorded = _record_sampler(monkeypatch)
    prefix = torch.tensor([[1, 3]], dtype=torch.long)

    proposal = build_draft_proposal(cache, prefix, 3, 2, vocab_size=VOCAB)

    assert isinstance(proposal, TkSltProposal)
    assert proposal.prefix_len == 2 and proposal.count == 3
    assert proposal.tokens.shape == (1, 5) and proposal.tokens.dtype == torch.long
    assert proposal.tokens[:, :2].tolist() == [[1, 3]]
    assert [call.shape[1] for call in cache.calls] == [2, 3, 4]
    assert cache.calls[1].tolist() == [[1, 3, 0]]  # actual prefix contents grow
    assert cache.calls[2].tolist() == [[1, 3, 0, 0]]
    # every draw used exactly the payload reconstruction, not a dense raw row
    assert len(recorded) == 3
    for drawn, payload in zip(recorded, proposal.payloads, strict=True):
        assert torch.equal(drawn, payload.dense())
        assert torch.equal(sampling_weights(drawn), sampling_weights(payload.dense()))
    assert [p.nbytes for p in proposal.payloads] == [4, 4, 4]
    assert proposal.dense_rows().shape == (1, 3, VOCAB)
    assert proposal.dense_rows().dtype == torch.float16
    assert torch.equal(proposal.dense_rows()[0, 0], proposal.payloads[0].dense()[0])


def test_build_draft_proposal_rejects_nonpositive_gamma_without_touching_cache():
    cache = _ScriptedCache([row([0.5, 0.5])])
    prefix = torch.tensor([[0]], dtype=torch.long)
    for gamma in (0, -1):
        with pytest.raises(ValueError):
            build_draft_proposal(cache, prefix, gamma, 1, vocab_size=2)
    assert cache.calls == []


@pytest.mark.parametrize("private_rng", [False, True])
def test_build_draft_proposal_validates_rows_before_drawing(monkeypatch, private_rng):
    cache = _ScriptedCache([torch.tensor([[2.0, 1.0]])])
    rng = DecodingRng() if private_rng else None
    before = torch.get_rng_state().clone()
    monkeypatch.setattr(
        torch, "multinomial", lambda *a, **k: pytest.fail("draw before validation")
    )
    with pytest.raises(ValueError):
        build_draft_proposal(
            cache, torch.tensor([[0]], dtype=torch.long), 1, 1, vocab_size=2, rng=rng
        )
    assert torch.equal(torch.get_rng_state(), before)
    assert len(cache.calls) == 1


def test_build_draft_proposal_rejects_cache_rows_that_are_not_single_rows():
    cache = _ScriptedCache([torch.full((1, 1, 4), 0.25)])
    with pytest.raises(ValueError):
        build_draft_proposal(
            cache, torch.tensor([[0]], dtype=torch.long), 1, 1, vocab_size=VOCAB
        )


@pytest.mark.parametrize(
    "prefix,top_k",
    [
        (torch.tensor([[-1]]), 1),
        (torch.tensor([[4]]), 1),
        (torch.tensor([[0]]), True),
        (torch.tensor([[0]]), 1.5),
    ],
)
def test_builder_validates_arguments_before_forward(prefix, top_k):
    class NoForward:
        def forward_context(self, inputs):
            raise AssertionError("invalid arguments must not advance the cache")

    with pytest.raises(ValueError):
        build_draft_proposal(NoForward(), prefix, 1, top_k, vocab_size=4)


def test_build_draft_proposal_validates_prefix_shape():
    cache = _ScriptedCache([row([0.5, 0.5])])
    bad_prefixes = (
        torch.tensor([7]),
        torch.tensor([[7.0]]),
        torch.tensor([[7, 8], [9, 10]]),
    )
    for bad in bad_prefixes:
        with pytest.raises(ValueError):
            build_draft_proposal(cache, bad, 1, 1, vocab_size=2)
    assert cache.calls == []
