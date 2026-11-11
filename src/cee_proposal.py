"""CEE uniform-tail probability wire, shared by the producer and both stages.

This is not hard top-k truncation: the unselected coordinates share an explicit
uniform-tail value. Values retain their actual tensor precision on the wire;
indices are int32. No sampler sees the original model row after encoding.
Model filtering and the communication top-k cap remain separate operations.
"""

import operator
from dataclasses import dataclass

import torch

from .probability_laws import validate_probability_rows
from .verification import ProposalWindow


@dataclass(frozen=True, eq=False)
class CeeWireRow:
    """Owned payload; dense() rebuilds without the model/cache or another top-k."""

    values: torch.Tensor
    indices: torch.Tensor | None
    tail: torch.Tensor | None
    vocab_size: int

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.values, self.indices, self.tail)
            if tensor is not None
        )

    def dense(self, *, device=None) -> torch.Tensor:
        device = self.values.device if device is None else device
        if self.indices is None:
            return self.values.to(device).clone()
        row = self.tail.to(device).expand(1, self.vocab_size).clone()
        return row.scatter_(
            1, self.indices.to(device=device, dtype=torch.long), self.values.to(device)
        )


class CeeProbabilityCodec:
    """A fixed per-decode uniform-tail policy, never a retrospective row override."""

    def __init__(self, vocab_size: int, top_k):
        self.vocab_size = operator.index(vocab_size)
        if not 0 < self.vocab_size < 2**31:
            raise ValueError("CEE vocabulary must fit positive int32 indices")
        requested = 0 if top_k is None else operator.index(top_k)
        self.top_k = self.vocab_size if requested <= 0 else min(requested, vocab_size)

    def encode(self, probabilities: torch.Tensor) -> CeeWireRow:
        if (
            not isinstance(probabilities, torch.Tensor)
            or probabilities.ndim != 2
            or probabilities.shape[0] != 1
            or probabilities.shape[1] < self.vocab_size
        ):
            raise ValueError("CEE model row must have shape (1, at least vocab_size)")
        validate_probability_rows(probabilities, "CEE model row")
        row = probabilities[:, : self.vocab_size]
        validate_probability_rows(row, "CEE shared-vocabulary row")
        if self.top_k == self.vocab_size:
            return CeeWireRow(row.clone(), None, None, self.vocab_size)
        # Stable ties select the lowest vocabulary indices. Exactly K entries,
        # unlike logit-threshold filtering which can retain more than K ties.
        indices = torch.argsort(row, dim=-1, descending=True, stable=True)[
            :, : self.top_k
        ]
        values = row.gather(1, indices).clone()
        # Sum the actual omitted mass, not 1-sum(topK): this avoids cancellation
        # and never manufactures a tail where the model row has exact zeros.
        omitted = row.clone().scatter_(1, indices, 0)
        tail = (
            omitted.double().sum(-1, keepdim=True) / (self.vocab_size - self.top_k)
        ).to(row.dtype)
        payload = CeeWireRow(values, indices.to(torch.int32), tail, self.vocab_size)
        validate_probability_rows(payload.dense(), "CEE decoded wire row")
        return payload


@dataclass(frozen=True, eq=False)
class CascadeProposal:
    """Actual candidate laws and the payloads from which they were sampled."""

    window: ProposalWindow
    wire_rows: tuple[CeeWireRow, ...]

    @property
    def probability_bytes(self) -> int:
        return sum(row.nbytes for row in self.wire_rows)


def proposal_from_wire(tokens, prefix_len, wire_rows, vocab_size, *, device):
    rows = tuple(wire_rows)
    if int(tokens.shape[1]) - prefix_len != len(rows):
        raise ValueError("CEE wire rows must match the entire candidate window")
    probabilities = (
        torch.stack([row.dense(device=device) for row in rows], dim=1)
        if rows
        else torch.empty((1, 0, vocab_size), device=device)
    )
    return CascadeProposal(ProposalWindow(tokens, probabilities, prefix_len), rows)


def propose_candidates(cache, prefix, count, codec, rng) -> CascadeProposal:
    """Owned producer: forward without RNG, encode, decode, sample that row."""
    count = operator.index(count)
    if count < 0:
        raise ValueError("CEE candidate count must be nonnegative")
    tokens = prefix.to(cache.device).clone()
    prefix_len = int(tokens.shape[1])
    rows = []
    for _ in range(count):
        wire = codec.encode(cache.forward_context(tokens))
        token = rng.sample(wire.dense())
        rows.append(wire)
        tokens = torch.cat((tokens, token.to(tokens.device)), dim=1)
    return proposal_from_wire(
        tokens, prefix_len, rows, codec.vocab_size, device=cache.device
    )


def extend_carried(carried, draft, count, codec, rng) -> CascadeProposal:
    """Append new draft rows without re-encoding or reinterpreting carried rows."""
    fresh = propose_candidates(draft, carried.window.tokens, count, codec, rng)
    return proposal_from_wire(
        fresh.window.tokens,
        carried.window.prefix_len,
        carried.wire_rows + fresh.wire_rows,
        codec.vocab_size,
        device=draft.device,
    )
