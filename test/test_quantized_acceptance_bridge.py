"""TK-SLT's legacy FP16 wire law is not the actual proposal sampling law."""

from types import SimpleNamespace

import pytest
import torch

from src.baselines_helpers import _quantize_probs_fp16
from src.decoding_ops import (
    compute_acceptance_result,
    compute_legacy_tk_slt_fp16_acceptance_result,
    prepare_verification_inputs,
)


def quantized_inputs(raw_selected=1e-8, target_selected=0.25):
    raw = torch.tensor([[[raw_selected, 1.0 - raw_selected]]], dtype=torch.float64)
    wire = _quantize_probs_fp16(raw)
    proposer = SimpleNamespace(device=torch.device("cpu"), prob_history=raw)
    target = SimpleNamespace(
        prob_history=torch.tensor(
            [[[target_selected, 1.0 - target_selected]]], dtype=torch.float64
        )
    )
    prepared = prepare_verification_inputs(
        proposer,
        target,
        torch.tensor([[1, 0]]),
        prefix_len=1,
        gamma=1,
        draft_probs_override=wire,
    )
    return raw, wire, prepared


@pytest.mark.parametrize("target_selected", [0.25, 1e-30])
def test_actual_fp16_quantization_loses_supported_sample_without_aborting(
    target_selected,
):
    raw, wire, prepared = quantized_inputs(target_selected=target_selected)
    assert raw[0, 0, 0] > 0  # The selected token is a possible raw-q sample.
    assert wire[0, 0, 0] == 0  # Real FP16 underflow, not hand-injected zero.
    result = compute_legacy_tk_slt_fp16_acceptance_result(
        prepared, r=torch.tensor([[0.9]])
    )
    assert result.accepted_count.tolist() == [1]
    assert result.selected_draft_p.tolist() == [[0.0]]
    assert result.selected_draft_p is prepared.selected_draft_p
    assert prepared.selected_draft_p.tolist() == [[0.0]]  # No epsilon/mutation.


@pytest.mark.parametrize("draw", [0.0, 0.5, 0.999])
def test_zero_wire_and_zero_target_reject_including_zero_uniform(draw):
    _, _, prepared = quantized_inputs(target_selected=0.0)
    result = compute_legacy_tk_slt_fp16_acceptance_result(
        prepared,
        r=torch.tensor([[draw]]),
    )
    assert result.accepted_count.tolist() == [0]
    assert result.accept_mask.tolist() == [[False]]
    assert result.selected_draft_p.tolist() == [[0.0]]
    assert result.selected_target_p.tolist() == [[0.0]]


@pytest.mark.parametrize(
    "compute",
    [compute_acceptance_result, compute_legacy_tk_slt_fp16_acceptance_result],
)
@pytest.mark.parametrize("draw, count", [(0.499, 1), (0.5, 0), (0.9, 0)])
def test_nonzero_wire_keeps_strict_ratio_boundary(compute, draw, count):
    _, wire, prepared = quantized_inputs(raw_selected=0.5, target_selected=0.25)
    assert wire[0, 0, 0] == 0.5  # Exactly representable, p/qhat=.5.
    result = compute(prepared, r=torch.tensor([[draw]], dtype=torch.float64))
    assert result.accepted_count.tolist() == [count]
    assert result.selected_draft_p.tolist() == [[0.5]]


@pytest.mark.parametrize(
    "compute",
    [compute_acceptance_result, compute_legacy_tk_slt_fp16_acceptance_result],
)
@pytest.mark.parametrize("field", ["proposal", "target"])
@pytest.mark.parametrize("invalid", [-0.1, float("nan"), float("inf")])
def test_invalid_selected_laws_raise_before_rng(compute, field, invalid, monkeypatch):
    _, _, prepared = quantized_inputs(raw_selected=0.5)
    if field == "proposal":
        prepared.selected_draft_p.fill_(invalid)
    else:
        prepared.target_probs_batch[0, 0, 0] = invalid
    monkeypatch.setattr(
        torch, "rand", lambda *a, **kw: pytest.fail("RNG before validation")
    )
    with pytest.raises(ValueError, match="finite"):
        compute(prepared)


def test_exact_acceptance_does_not_silently_fall_back_to_wire_semantics(monkeypatch):
    _, _, prepared = quantized_inputs()
    monkeypatch.setattr(
        torch, "rand", lambda *a, **kw: pytest.fail("RNG before validation")
    )
    with pytest.raises(ValueError, match="positive and finite"):
        compute_acceptance_result(prepared)
