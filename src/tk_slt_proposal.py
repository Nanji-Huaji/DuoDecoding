"""TK-SLT probability-wire codec and draft proposal builder.

The legacy TK-SLT loop drew proposals from the raw top-K model law but sent and
verified an FP16-roundtripped copy, so two categorical laws drove one accepted
token.  With ``Q=[.6,.4]`` and ``P=wire(Q)`` every acceptance ratio was exactly
1: the first output token kept the raw Q law while the receiver's law P differs
by TVD ``9.76502895e-5``, and a positive ``1e-8`` tail (zero on the wire) could
still be sampled while the legacy bridge accepted it.

This module makes the wire law the only proposal law::

    validated model row (full width, unit mass)
      -> declared-vocabulary projection
      -> explicit support of exactly min(K, V) coordinates,
         descending value with the lowest vocabulary index on exact ties
      -> retained-mass renormalization      (a codec transform, not the law)
      -> real FP16 values + sorted indices  (the billed payload)
      -> dense-V FP16 row                   (the one sampler input)

``sampling.py`` masks ``logits < threshold``, so tied top-K logits keep *more*
than K positives; a K-value wire cannot represent those.  Selection therefore
happens here, before any draw, and the receiver never reruns top-k.  The payload
is explicitly sparse: exactly ``min(K, V)`` FP16 values plus negligible sorted
indices (full vocabulary keeps only its V values, implicit indices).  Drafter
and receiver each rebuild the same dense-V FP16 row from that payload and run
the one existing sampler preprocessing, so acceptance, the strict residual and
the emitted token all refer to a single law.

Scope and honesty notes:

* this is an explicit **probability-wire variant**, not a claim of bit-exact
  real arithmetic, GPU validation or a literal reading of the paper's ambiguous
  logits/probabilities terminology;
* the repository's global draft/target temperature stays in force (the paper's
  draft T=1 is *not* silently imposed here);
* an FP16 payload can lose a tiny positive tail.  That is wire lossiness, but
  both sides then use the surviving row: no unbilled dense model Q is needed to
  reconstruct the proposal, and an all-underflow / invalid-mass payload raises
  instead of being repaired into a one-hot draw.

Everything here is finite-precision arithmetic; the encoder is deterministic
and draws no randomness (only :func:`build_draft_proposal` samples tokens).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .decoding_rng import DecodingRng
from .probability_laws import validate_probability_rows
from .sampling import sample

#: The paper's uplink probability width (``b_prob = 16``, §VI-B).
WIRE_DTYPE = torch.float16

__all__ = [
    "TkSltProposal",
    "TkSltWireRow",
    "build_draft_proposal",
    "encode_proposal_row",
]


def _wire_integer(value, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an integer") from exc


@dataclass(frozen=True, eq=False)
class TkSltWireRow:
    """One round-local TK-SLT wire payload: real FP16 values + indices.

    ``values`` has shape ``(1, n)`` and dtype :data:`WIRE_DTYPE`; ``indices`` is
    ``None`` for an implicit full-vocabulary payload (``n == vocab_size``) or a
    ``(1, n)`` ``torch.long`` tensor that is strictly increasing (unique and
    sorted) and in ``[0, vocab_size)``.

    Construction validates the received payload instead of repairing it,
    including approximate unit mass. Zero values are allowed (FP16 tail loss),
    but invalid or materially truncated mass raises. Tensor fields are borrowed
    and must not be mutated after construction; the encoder creates fresh
    payload tensors that do not alias cache rows. ``dense()`` returns a fresh row.
    """

    values: torch.Tensor
    indices: Optional[torch.Tensor]
    vocab_size: int

    def __post_init__(self) -> None:
        vocab = _wire_integer(self.vocab_size, label="wire vocab_size")
        if vocab < 1:
            raise ValueError("wire vocab_size must be at least 1")
        values = self.values
        if not isinstance(values, torch.Tensor):
            raise ValueError("wire values must be a tensor")
        if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] < 1:
            raise ValueError("wire values must have shape (1, n) with n >= 1")
        if values.dtype != WIRE_DTYPE:
            raise ValueError(
                f"wire values must be {WIRE_DTYPE} (b_prob=16), got {values.dtype}"
            )
        # A legal encoder normalizes its selected coordinates before FP16
        # conversion. Positive mass alone would accept a truncated payload and
        # let the sampler silently renormalize it before receiver validation.
        validate_probability_rows(values, "TK-SLT wire payload")

        indices = self.indices
        if indices is None:
            if int(values.shape[1]) != vocab:
                raise ValueError(
                    "an implicit full-vocabulary payload must carry exactly "
                    f"vocab_size={vocab} values, got {int(values.shape[1])}"
                )
        else:
            if not isinstance(indices, torch.Tensor):
                raise ValueError("wire indices must be a tensor or None")
            if indices.ndim != 2 or indices.shape != values.shape:
                raise ValueError(
                    "wire indices must have the same (1, n) shape as values"
                )
            if indices.dtype != torch.long:
                raise ValueError(
                    f"wire indices must be torch.long, got {indices.dtype}"
                )
            if bool((indices < 0).any()) or bool((indices >= vocab).any()):
                raise ValueError(f"wire indices must lie in [0, {vocab})")
            if indices.shape[1] > 1 and not bool(
                (indices[:, 1:] > indices[:, :-1]).all()
            ):
                raise ValueError("wire indices must be unique and strictly increasing")
        object.__setattr__(self, "vocab_size", vocab)

    @property
    def nbytes(self) -> int:
        """Billed bytes: FP16 values only; indices are negligible (§II-B)."""
        return int(self.values.numel() * self.values.element_size())

    def dense(self, *, device: Optional[torch.device] = None) -> torch.Tensor:
        """Rebuild the dense-V FP16 sampler-input row from the payload alone.

        Returns a fresh ``(1, vocab_size)`` tensor, so a receiver-side tensor can
        never alias the immutable payload.  No validation or renormalization
        happens here: the payload is already validated and this is exactly the
        row both sides feed to :func:`src.sampling.sampling_weights`.
        """
        target_device = self.values.device if device is None else torch.device(device)
        if self.indices is None:
            return self.values.to(device=target_device).clone()
        dense = torch.zeros(
            (1, self.vocab_size), dtype=WIRE_DTYPE, device=target_device
        )
        dense.scatter_(
            1,
            self.indices.to(device=target_device),
            self.values.to(device=target_device),
        )
        return dense


def _select_support(values: torch.Tensor, k: int) -> torch.Tensor:
    """Return exactly ``k`` indices: descending value, lowest index on ties.

    Implemented with a plain sort plus integer comparisons so the selection is
    explicit and deterministic (no reliance on unspecified ``topk`` tie
    ordering, and no host-side branch): the threshold is the k-th largest value,
    unambiguously larger values are taken, and the remaining slots go to the
    lowest-id tied entries in ascending index order.

    ``values`` is a validated ``(1, V)`` row with ``0 < k < V``.  Comparisons
    run in the row's own dtype (width-preserving for FP16/BF16/FP32 and exact
    for FP64), so the selection neither pays for a V-sized float64 copy nor
    changes the ordering the model produced.
    """
    ordered = torch.sort(values, dim=-1, descending=True).values
    threshold = ordered[:, k - 1 : k]
    above = values > threshold
    ties = values == threshold
    deficit = k - above.sum(dim=-1, keepdim=True)
    ordinal = ties.to(torch.int64).cumsum(dim=-1) - 1
    chosen = above | (ties & (ordinal < deficit))
    indices = torch.nonzero(chosen, as_tuple=False)[:, 1].reshape(1, k)
    return indices


def encode_proposal_row(
    probabilities: torch.Tensor,
    top_k: Optional[int],
    *,
    vocab_size: Optional[int] = None,
) -> TkSltWireRow:
    """Encode one model probability row into the billed TK-SLT wire payload.

    ``probabilities`` is the ``(1, V_model)`` model row, validated as a raw
    probability law **before** any sampler preprocessing.  ``vocab_size`` may
    declare a narrower shared vocabulary than the row width; the wire law is
    then the conditional law on those leading columns. The receiver also slices
    target rows to that vocabulary, but requires their mass to remain near one:
    it does not silently condition away non-negligible target padding mass.
    The row itself is never mutated.

    ``top_k`` semantics follow the protocol: ``None``/``<=0`` keep the whole
    declared vocabulary (vanilla DSD, implicit indices); otherwise exactly
    ``min(top_k, vocab_size)`` coordinates are selected *before* any draw.
    Selected probabilities are renormalized over the retained mass and only then
    converted to real FP16.  That renormalization is a codec transform; the
    resulting FP16 row — not the raw model row — is the proposal law.

    Raises:
        ValueError: non-row input, a non-law row, an empty declared vocabulary,
            a declared projection with no positive mass, or an encoding whose
            retained mass does not survive FP16.
    """
    if not isinstance(probabilities, torch.Tensor):
        raise ValueError("TK-SLT proposal row must be a tensor")
    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise ValueError("TK-SLT proposal row must have shape (1, V)")
    if probabilities.shape[1] < 1:
        raise ValueError("TK-SLT proposal row must have a nonempty vocabulary")
    if not probabilities.is_floating_point():
        raise ValueError("TK-SLT proposal row must have a floating dtype")

    width = int(probabilities.shape[1])
    declared = (
        width if vocab_size is None else _wire_integer(vocab_size, label="vocab_size")
    )
    if not 1 <= declared <= width:
        raise ValueError(f"declared vocab_size {declared} must lie in [1, {width}]")

    if top_k is None:
        requested = 0
    else:
        requested = _wire_integer(top_k, label="top_k")
    k = declared if requested <= 0 else min(requested, declared)

    # The model law is validated on its real width; only then does the codec
    # condition on the declared vocabulary.
    validate_probability_rows(probabilities, "TK-SLT model proposal row")
    declared_row = probabilities[:, :declared]

    if k >= declared:
        precise = declared_row.to(torch.float64)
        mass = precise.sum(dim=-1, keepdim=True)
        if not bool(torch.isfinite(mass).all()) or not bool((mass > 0).all()):
            raise ValueError(
                "TK-SLT wire has no positive retained mass after the "
                "declared-vocabulary projection"
            )
        values = (precise / mass).to(WIRE_DTYPE)
        indices: Optional[torch.Tensor] = None
    else:
        support = _select_support(declared_row, k)
        retained = torch.gather(declared_row, 1, support).to(torch.float64)
        mass = retained.sum(dim=-1, keepdim=True)
        if not bool(torch.isfinite(mass).all()) or not bool((mass > 0).all()):
            raise ValueError(
                "TK-SLT wire has no positive retained mass after top-K selection"
            )
        values = (retained / mass).to(WIRE_DTYPE)
        indices = support

    return TkSltWireRow(values=values, indices=indices, vocab_size=declared)


@dataclass(frozen=True, eq=False)
class TkSltProposal:
    """A round-local draft proposal: actual context, payloads, start offset.

    ``tokens`` is ``(1, prefix_len + count)``: the old prefix plus the sampled
    draft tokens. Its last token has not yet been forwarded to the draft cache.
    The encoder-created sparse payloads do not alias model probability buffers,
    so later cache writes cannot change them. Callers must not mutate the
    borrowed tensor fields during the round.
    """

    tokens: torch.Tensor
    payloads: Tuple[TkSltWireRow, ...]
    prefix_len: int

    def __post_init__(self) -> None:
        tokens = self.tokens
        if not isinstance(tokens, torch.Tensor):
            raise ValueError("proposal tokens must be a tensor")
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("proposal tokens must have shape (1, seq_len)")
        if tokens.dtype != torch.long:
            raise ValueError("proposal tokens must use torch.long")
        if not isinstance(self.payloads, tuple) or not self.payloads:
            raise ValueError("proposal must carry at least one wire payload")
        if not all(isinstance(p, TkSltWireRow) for p in self.payloads):
            raise ValueError("proposal payloads must be TkSltWireRow instances")
        vocab = self.payloads[0].vocab_size
        if any(p.vocab_size != vocab for p in self.payloads):
            raise ValueError("proposal payloads must share one vocabulary size")
        prefix_len = _wire_integer(self.prefix_len, label="proposal prefix_len")
        if not 1 <= prefix_len <= int(tokens.shape[1]):
            raise ValueError("proposal prefix_len must be a nonempty prefix")
        if int(tokens.shape[1]) != prefix_len + len(self.payloads):
            raise ValueError(
                "proposal tokens must cover the prefix and exactly one row per "
                "wire payload"
            )
        if bool((tokens < 0).any()) or bool((tokens >= vocab).any()):
            raise ValueError("proposal tokens must lie in the wire vocabulary")
        object.__setattr__(self, "prefix_len", prefix_len)

    @property
    def count(self) -> int:
        return len(self.payloads)

    @property
    def nbytes(self) -> int:
        """Billed uplink bytes for this round (FP16 values, negligible indices)."""
        return sum(payload.nbytes for payload in self.payloads)

    def dense_rows(self, *, device: Optional[torch.device] = None) -> torch.Tensor:
        """Receiver-side reconstruction: ``(1, count, vocab)`` FP16 rows."""
        rows = [payload.dense(device=device) for payload in self.payloads]
        return torch.cat(rows, dim=0).unsqueeze(0)


def build_draft_proposal(
    cache,
    prefix: torch.Tensor,
    gamma: int,
    top_k: Optional[int],
    *,
    vocab_size: int,
    device: Optional[torch.device] = None,
    rng: Optional[DecodingRng] = None,
) -> TkSltProposal:
    """Run the draft ``forward_context`` loop and return the wire-law proposal.

    One model row is requested per draft step, encoded into a real FP16 payload,
    and immediately reconstructed as the dense-V FP16 row that is sampled here.
    Because the draw uses that reconstruction, the token always has positive
    wire support; it is written into a preallocated token buffer so the loop
    grows neither the context with ``torch.cat`` nor the cache past
    ``prefix_len + gamma - 1`` (the endpoint the old ``generate`` left behind).

    ``gamma >= 1`` is required: a zero-token step has nothing to propose and must
    not forward, draw or mutate anything.  The cache result is validated as a
    ``(1, V)`` row before any draw. The full TK decoder supplies its shared
    per-invocation ``rng``; standalone helper callers retain the default sampler
    when it is omitted. The supplied context allocates no seed before validation.
    """
    if not isinstance(prefix, torch.Tensor):
        raise ValueError("draft prefix must be a tensor")
    if prefix.ndim != 2 or prefix.shape[0] != 1 or prefix.shape[1] < 1:
        raise ValueError("draft prefix must have shape (1, prefix_len)")
    if prefix.dtype != torch.long:
        raise ValueError("draft prefix must use torch.long")
    count = _wire_integer(gamma, label="gamma")
    if count < 1:
        raise ValueError(
            "draft gamma must be at least 1; a zero-budget step must not draw"
        )
    vocab = _wire_integer(vocab_size, label="vocab_size")
    if vocab < 1:
        raise ValueError("vocab_size must be at least 1")
    if bool((prefix < 0).any()) or bool((prefix >= vocab).any()):
        raise ValueError("draft prefix token is outside the declared vocabulary")
    if top_k is not None:
        top_k = _wire_integer(top_k, label="top_k")
    forward_context = getattr(cache, "forward_context", None)
    if not callable(forward_context):
        raise ValueError("draft cache must expose forward_context(input_ids)")

    target_device = prefix.device if device is None else torch.device(device)
    prefix_len = int(prefix.shape[1])
    tokens = torch.empty(
        (1, prefix_len + count), dtype=torch.long, device=target_device
    )
    tokens[:, :prefix_len] = prefix.to(target_device)

    payloads = []
    for step in range(count):
        model_row = forward_context(tokens[:, : prefix_len + step])
        if (
            not isinstance(model_row, torch.Tensor)
            or model_row.ndim != 2
            or model_row.shape[0] != 1
        ):
            raise ValueError(
                "draft forward_context must return exactly one (1, V) model row"
            )
        payload = encode_proposal_row(model_row, top_k, vocab_size=vocab)
        wire_row = payload.dense()
        token = sample(wire_row) if rng is None else rng.sample(wire_row)
        if not isinstance(token, torch.Tensor) or token.numel() != 1:
            raise ValueError("draft sampling must return exactly one token")
        tokens[0, prefix_len + step] = token.reshape(-1)[0].to(torch.long)
        payloads.append(payload)

    return TkSltProposal(tokens=tokens, payloads=tuple(payloads), prefix_len=prefix_len)
