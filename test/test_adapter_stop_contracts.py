"""A1 stop-decision contracts for the real ARP ``DecodingAdapter``.

Expected semantics (see docs/refactor_roadmap_20261110.md, A1):

- ``cumulative``: stop when ``1 - prod(p_i) > threshold`` over this step.
- ``per_token``: stop when ``1 - p_last > threshold`` (latest calibrated prob).
- ``threshold=None``: keep argmax semantics; no threshold comparison at all.
- ``last_acc_prob`` remains the *raw* observation (pre-calibration).
- Calibration maps are consumed by the stop decision.

The final wiring test exercises the real adapter through the existing
``Baselines._generate_with_optional_rebuilt_proposal`` helper (read-only), with
a minimal fake cache so no model or GPU is needed.
"""

import types

import pytest
import torch
from torch import nn

import src.adapter as adapter_module
from src.adapter import DecodingAdapter
from src.baselines.core import Baselines


class _ConstantAcceptanceHead(nn.Module):
    """Acceptance head with ``softmax([log(1-p), log(p)]) == [1-p, p]``.

    Output shape matches the real ``AcceptancePredictionHead`` contract:
    ``(batch, seq_len, 2)``.  ``self._anchor`` supplies ``parameters()`` so the
    adapter's ``device``/``dtype`` properties work.
    """

    def __init__(self, accept_prob: float):
        super().__init__()
        p = min(max(float(accept_prob), 1e-12), 1.0 - 1e-12)
        self.register_buffer(
            "_logits", torch.log(torch.tensor([1.0 - p, p], dtype=torch.float32))
        )
        self._anchor = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self._logits.to(hidden_states.device).to(hidden_states.dtype)
        return logits.view(1, 1, 2).expand(
            hidden_states.shape[0], hidden_states.shape[1], 2
        )


def _hidden(seq_len: int = 2, hidden_size: int = 4):
    return (torch.zeros(1, seq_len, hidden_size),)


def _adapter(accept_prob: float, threshold, stop_mode: str) -> DecodingAdapter:
    return DecodingAdapter(
        _ConstantAcceptanceHead(accept_prob),
        threshold,
        stop_mode=stop_mode,
    )


@pytest.fixture(autouse=True)
def _disable_calibration(monkeypatch):
    monkeypatch.setattr(adapter_module, "_CALIB_GRID", None)
    monkeypatch.setattr(adapter_module, "_CALIB_VALUES", None)


# ---------------------------------------------------------------------------
# multi-step stop decisions
# ---------------------------------------------------------------------------


def test_per_token_two_steps_uses_latest_probability():
    """p=0.8, threshold=0.3 for two steps: never stop (1-0.8 < 0.3)."""
    adapter = _adapter(0.8, 0.3, "per_token")

    assert adapter.predict(_hidden()) is False
    assert adapter.predict(_hidden()) is False
    assert adapter.step_acc_probs == pytest.approx([0.8, 0.8], abs=1e-6)
    assert adapter.last_acc_prob == pytest.approx(0.8, abs=1e-6)


def test_cumulative_two_steps_uses_one_minus_product():
    """The historical cumulative branch must keep 1 - prod(p_i)."""
    adapter = _adapter(0.8, 0.3, "cumulative")

    assert adapter.predict(_hidden()) is False
    assert adapter.predict(_hidden()) is True
    assert adapter.step_acc_probs == pytest.approx([0.8, 0.8], abs=1e-6)


@pytest.mark.parametrize(
    ("stop_mode", "expected"),
    [("cumulative", [False, True]), ("per_token", [False, False])],
)
def test_reset_step_clears_stop_history(stop_mode, expected):
    adapter = _adapter(0.8, 0.3, stop_mode)

    assert [adapter.predict(_hidden()) for _ in range(2)] == expected
    adapter.reset_step()
    assert adapter.step_acc_probs == []

    # A fresh step starts from only its own probability in both modes.
    assert adapter.predict(_hidden()) is False
    assert adapter.step_acc_probs == pytest.approx([0.8], abs=1e-6)


# ---------------------------------------------------------------------------
# strict threshold boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stop_mode", ["cumulative", "per_token"])
def test_threshold_comparison_is_strict_at_exact_boundary(stop_mode):
    """rejection == threshold must not stop; only rejection > threshold does."""
    adapter = _adapter(0.5, 0.5, stop_mode)

    assert adapter.predict(_hidden()) is False  # 1 - 0.5 == 0.5

    adapter.reset_step()
    adapter.threshold = 0.499
    assert adapter.predict(_hidden()) is True  # 0.5 > 0.499


# ---------------------------------------------------------------------------
# threshold=None keeps argmax semantics
# ---------------------------------------------------------------------------


def test_threshold_none_keeps_argmax_semantics(monkeypatch):
    # Calibration may alter the recorded probability, but must not turn the
    # argmax decision into a threshold decision.
    monkeypatch.setattr(adapter_module, "_CALIB_GRID", [0.0, 1.0])
    monkeypatch.setattr(adapter_module, "_CALIB_VALUES", [0.0, 0.0])

    accept_head = _adapter(1.0 - 1e-6, None, "per_token")
    adapter = accept_head
    assert adapter.predict(_hidden()) is False  # argmax class 1 -> continue
    assert adapter.last_acc_prob == 1.0  # raw, pre-calibration
    assert adapter.step_acc_probs == pytest.approx([0.0])  # calibrated record

    reject_head = _adapter(1e-6, None, "per_token")
    assert reject_head.predict(_hidden()) is True  # argmax class 0 -> stop
    assert reject_head.last_acc_prob == 0.0
    assert reject_head.step_acc_probs == pytest.approx([0.0])


# ---------------------------------------------------------------------------
# calibration map is consumed by the stop decision
# ---------------------------------------------------------------------------


def test_calibrated_probability_drives_per_token_stop(monkeypatch):
    # raw 0.8 -> calibrated 0.2, so rejection 0.8 > 0.3 stops immediately;
    # without calibration raw 0.8 would not stop (1 - 0.8 < 0.3).
    monkeypatch.setattr(adapter_module, "_CALIB_GRID", [0.0, 1.0])
    monkeypatch.setattr(adapter_module, "_CALIB_VALUES", [0.0, 0.25])

    adapter = _adapter(0.8, 0.3, "per_token")

    assert adapter.predict(_hidden()) is True
    assert adapter.last_acc_prob == pytest.approx(0.8, abs=1e-6)
    assert adapter.step_acc_probs == pytest.approx([0.2], abs=1e-6)


def test_calibrated_probability_drives_cumulative_stop(monkeypatch):
    # raw 0.8 -> calibrated 0.9, so 1 - 0.9**2 = 0.19 < 0.3: no stop on step 2.
    # With raw probabilities the cumulative branch would stop (1 - 0.64 > 0.3).
    monkeypatch.setattr(adapter_module, "_CALIB_GRID", [0.0, 0.8, 1.0])
    monkeypatch.setattr(adapter_module, "_CALIB_VALUES", [0.0, 0.9, 1.0])

    adapter = _adapter(0.8, 0.3, "cumulative")

    assert adapter.predict(_hidden()) is False
    assert adapter.predict(_hidden()) is False
    assert adapter.step_acc_probs == pytest.approx([0.9, 0.9], abs=1e-6)


# ---------------------------------------------------------------------------
# real adapter wired through the existing generation helper
# ---------------------------------------------------------------------------


class _FakeProposalCache:
    """Just enough of KVCacheModel for the proposal helper under test."""

    def __init__(self, vocab_size: int = 8, hidden_size: int = 4):
        self.hidden_states = None
        self._hidden = (torch.zeros(1, 3, hidden_size),)
        self._probs = torch.full((1, vocab_size), 1.0 / vocab_size)

    def forward_context(self, x: torch.Tensor) -> torch.Tensor:
        self.current_length = x.shape[1]
        self.hidden_states = self._hidden
        return self._probs


def _run_generation_helper(adapter: DecodingAdapter, gamma: int):
    cache = _FakeProposalCache()
    prefix = torch.tensor([[1, 2, 3]])
    x, _, _, q = Baselines._generate_with_optional_rebuilt_proposal(
        types.SimpleNamespace(),
        cache,
        prefix,
        gamma,
        None,
        adapter=adapter,
    )
    return prefix, x, q


def test_real_adapter_in_generation_helper_does_not_stop_early():
    """p=0.8, threshold=0.3, gamma=5: all 5 draft tokens must be generated.

    This is the real call path used by ``_adaptive_decoding_propose``; the old
    implementation stopped at the second token because it accumulated the
    product even in ``per_token`` mode.
    """
    adapter = _adapter(0.8, 0.3, "per_token")

    prefix, x, q = _run_generation_helper(adapter, gamma=5)

    assert q is not None
    assert x.shape == (1, prefix.shape[1] + 5)
    assert len(adapter.step_acc_probs) == 5


def test_real_adapter_in_generation_helper_stops_when_predicted():
    """p=0.2, threshold=0.3: the first token already stops the loop."""
    adapter = _adapter(0.2, 0.3, "per_token")

    prefix, x, q = _run_generation_helper(adapter, gamma=5)

    assert q is not None
    assert x.shape == (1, prefix.shape[1] + 1)
    assert len(adapter.step_acc_probs) == 1
