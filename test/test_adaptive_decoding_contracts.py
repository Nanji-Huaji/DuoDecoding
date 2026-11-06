"""Independent correctness contracts for the two-model adaptive decoder.

These tests are deliberately *not* a differential replay of the frozen
``a9a61cc`` implementation.  They state the sampling contract directly:

* ``proposal.probabilities`` is the proposal distribution ``q`` that produced
  ``proposal.tokens``; it is not the raw draft cache, and the verifier has no
  raw-draft-cache argument.
* The target history must match the full token span of the proposal exactly,
  including the bonus row used after a fully accepted draft; both short and
  extra stale rows are errors.
* Malformed proposals fail with ``ValueError`` *before* any RNG draw or
  communication billing.
* Acceptance rejects ``r >= p/q`` (``torch.rand`` is half-open, so ``p == q``
  still accepts every real draw), rejection resamples from
  ``norm((p - q)+)``, and the all-accepted bonus comes from the target's last
  row.

Toy distributions and captured sampler weights keep the assertions exact and
deterministic; no frequency thresholds or GPU/model boundaries are involved.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.baselines import Baselines
from src.baselines._adaptive_state import AdaptiveProposal
from src.communication import CommunicationSimulator

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 4
DEVICE = torch.device("cpu")
# ``torch.rand`` samples the half-open interval [0, 1); 1.0 is unreachable.
# The largest value it can actually return is the float32 predecessor of 1.
MAX_RAND = float(
    torch.nextafter(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float32),
    )
)


def _load_rounds_fixtures():
    """Load the pure ``_case``/``_run`` fixtures without importing a TestCase.

    The ``test`` directory is not a package and pytest may collect in importlib
    mode, so a file-location load with a private module name is the stable form.
    """
    path = ROOT / "test" / "test_adaptive_decoding_rounds.py"
    spec = importlib.util.spec_from_file_location(
        "adaptive_decoding_rounds_fixture", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rounds = _load_rounds_fixtures()


def _history(rows):
    """(batch=1, positions, vocab) float32 history."""
    return torch.tensor([list(row) for row in rows], dtype=torch.float32).unsqueeze(0)


def _as_history(rows):
    """Accept either row tuples or an already-built (possibly batched) tensor."""
    if isinstance(rows, torch.Tensor):
        return rows.clone()
    return _history(rows)


def _tokens(ids):
    return torch.tensor([list(ids)], dtype=torch.long)


def _proposal(tokens, probabilities, prefix_len, gamma):
    return AdaptiveProposal(
        tokens=torch.as_tensor(tokens, dtype=torch.long),
        probabilities=torch.as_tensor(probabilities, dtype=torch.float32),
        prefix_len=prefix_len,
        gamma=gamma,
    )


class _Decoder(Baselines):
    """Concrete shell: helper calls never touch model loading or evaluation."""

    def load_data(self):
        return None

    def preprocess(self, value):
        return value

    def postprocess(self, input_text, output_text):
        return output_text

    def eval(self):
        return None


def _decoder():
    decoder = object.__new__(_Decoder)
    decoder.vocab_size = VOCAB
    decoder.args = SimpleNamespace(max_tokens=4, gamma=3, top_k=0)
    return decoder


class _Cache:
    """Minimal cache surface used by verify/next-token helpers."""

    def __init__(self, prob_history):
        self.prob_history = prob_history.clone()
        self.vocab_size = prob_history.shape[-1]
        self.device = DEVICE
        self.rollbacks = []

    def rollback(self, end_pos):
        self.rollbacks.append(end_pos)
        self.prob_history = self.prob_history[:, :end_pos, :]


class _RecordingSimulator(CommunicationSimulator):
    """Real CPU accounting with the transferred payloads kept for assertions."""

    def __init__(self):
        super().__init__(
            bandwidth_edge_cloud=10.0,
            bandwidth_edge_end=float("inf"),
            bandwidth_cloud_end=float("inf"),
            min_bandwidth_mbps=0.0,
            bw_model="instant",
            dimension="Mbps",
            ntt_ms_edge_cloud=0.0,
            ntt_ms_edge_end=0.0,
        )
        self.transfers = []
        self.rejects = []
        self.payloads = []

    def transfer(
        self,
        tokens,
        prob,
        link_type,
        is_compressed=False,
        compressed_k=300,
        prob_bits=None,
    ):
        self.transfers.append(
            {
                "tokens": None
                if tokens is None
                else torch.as_tensor(tokens).detach().clone(),
                "prob": None if prob is None else prob.detach().clone(),
                "link_type": link_type,
                "is_compressed": is_compressed,
                "compressed_k": compressed_k,
                "prob_bits": prob_bits,
            }
        )
        return super().transfer(
            tokens, prob, link_type, is_compressed, compressed_k, prob_bits
        )

    def send_reject_message(self, linktype):
        self.rejects.append(linktype)
        return super().send_reject_message(linktype)

    def simulate_transfer(
        self, size, link_type, add_to_stats=True, topk=0, draft_len=0
    ):
        self.payloads.append((link_type, size, topk, draft_len))
        return super().simulate_transfer(size, link_type, add_to_stats, topk, draft_len)


def _prob_values(value):
    if value is None:
        return None
    return value.detach().reshape(-1).tolist()


class _HelperHarness(unittest.TestCase):
    """Shared invocation helpers for the adaptive proposal/verify helpers."""

    def assertProbClose(self, actual, expected):
        """Toy probabilities are float32; compare numerically, not by repr."""
        torch.testing.assert_close(
            torch.tensor(actual, dtype=torch.float32),
            torch.tensor(expected, dtype=torch.float32),
            rtol=0,
            atol=1e-6,
        )

    def _verify(self, proposal, *, approx_rows, target_rows, random_value=0.5):
        """Run verify against proposal q.

        ``approx_rows`` is a decoy fixture only: the verifier has no raw draft
        cache parameter (pinned by ``test_verify_has_no_raw_draft_cache_arg``),
        so these rows cannot influence acceptance.
        """
        target_cache = _Cache(_as_history(target_rows))
        simulator = _RecordingSimulator()
        rand_calls = []

        def fake_rand(*args, **kwargs):
            rand_calls.append((args, kwargs))
            return torch.tensor([random_value], dtype=torch.float32)

        with patch("src.baselines.core.torch.rand", new=fake_rand):
            n = Baselines._adaptive_decoding_verify(
                _decoder(),
                proposal=proposal,
                target_model_cache=target_cache,
                comm_simulator=simulator,
                draft_device=DEVICE,
            )
        return n, simulator, rand_calls, None, target_cache

    def _assert_validation_error(self, proposal, *, approx_rows, target_rows):
        target_cache = _Cache(_as_history(target_rows))
        simulator = _RecordingSimulator()
        rand_calls = []

        def fake_rand(*args, **kwargs):
            rand_calls.append((args, kwargs))
            return torch.tensor([0.5], dtype=torch.float32)

        with patch("src.baselines.core.torch.rand", new=fake_rand):
            try:
                n = Baselines._adaptive_decoding_verify(
                    _decoder(),
                    proposal=proposal,
                    target_model_cache=target_cache,
                    comm_simulator=simulator,
                    draft_device=DEVICE,
                )
            except ValueError:
                pass
            else:
                self.fail(
                    "expected ValueError before effects; got n="
                    f"{n}, rand draws={len(rand_calls)}, "
                    f"transfers={len(simulator.transfers)}, "
                    f"rejects={simulator.rejects}"
                )
        self.assertEqual(rand_calls, [])
        self.assertEqual(simulator.transfers, [])
        self.assertEqual(simulator.rejects, [])

    def _assert_next_token_error(self, proposal, n, *, target_rows):
        """Invalid accepted positions fail before sampling or billing."""
        target_cache = _Cache(_as_history(target_rows))
        simulator = _RecordingSimulator()
        sample_calls = []

        def fake_sample(probs, num_samples=1):
            sample_calls.append(probs)
            return torch.tensor([[0]], dtype=torch.long)

        with patch("src.baselines.core.sample", new=fake_sample):
            try:
                token = Baselines._adaptive_decoding_next_token(
                    _decoder(),
                    proposal=proposal,
                    n=n,
                    target_model_cache=target_cache,
                    comm_simulator=simulator,
                    transfer_top_k=None,
                    charge_residual=False,
                    draft_device=DEVICE,
                )
            except ValueError:
                pass
            else:
                self.fail(
                    f"expected ValueError for n={n}; got token={token.tolist()}, "
                    f"samples={len(sample_calls)}, "
                    f"transfers={len(simulator.transfers)}, "
                    f"rollbacks={target_cache.rollbacks}"
                )
        self.assertEqual(sample_calls, [])
        self.assertEqual(simulator.transfers, [])
        self.assertEqual(simulator.payloads, [])
        self.assertEqual(target_cache.rollbacks, [])

    def _next_token(
        self,
        proposal,
        n,
        *,
        target_rows,
        sampled_token=2,
        transfer_top_k=None,
        charge_residual=False,
    ):
        target_cache = _Cache(_as_history(target_rows))
        simulator = _RecordingSimulator()
        sample_weights = []

        def fake_sample(probs, num_samples=1):
            sample_weights.append(probs.detach().clone())
            return torch.tensor([[sampled_token]], dtype=torch.long)

        with patch("src.baselines.core.sample", new=fake_sample):
            token = Baselines._adaptive_decoding_next_token(
                _decoder(),
                proposal=proposal,
                n=n,
                target_model_cache=target_cache,
                comm_simulator=simulator,
                transfer_top_k=transfer_top_k,
                charge_residual=charge_residual,
                draft_device=DEVICE,
            )
        return token, sample_weights, simulator, target_cache


class AdaptiveProposalContractTests(_HelperHarness):
    def test_verify_has_no_raw_draft_cache_arg(self):
        # The verifier must consume proposal.probabilities; accepting a raw
        # draft cache again would reintroduce the short/rolled-back-row bug.
        parameters = inspect.signature(Baselines._adaptive_decoding_verify).parameters
        self.assertNotIn("approx_model_cache", parameters)
        self.assertIn("proposal", parameters)

    def test_verify_uses_proposal_q_not_a_short_raw_draft_cache(self):
        # q is complete for all three drafted tokens. The raw draft cache was
        # rolled back to a single row (and holds deliberately different values).
        # A correct verifier must neither consult nor reject on that cache.
        q_rows = [
            (0.0, 0.2, 0.5, 0.3),  # token 1 -> q=0.2
            (0.0, 0.4, 0.1, 0.5),  # token 2 -> q=0.1
            (0.0, 0.3, 0.3, 0.4),  # token 3 -> q=0.4
        ]
        target_rows = [
            (0.0, 0.9, 0.05, 0.05),  # token 1 -> p=0.9
            (0.0, 0.0, 0.9, 0.1),  # token 2 -> p=0.9
            (0.0, 0.0, 0.0, 1.0),  # token 3 -> p=1.0
            (0.0, 0.0, 0.0, 1.0),  # bonus row
        ]
        proposal = _proposal(_tokens([0, 1, 2, 3]), _history(q_rows), 1, 3)

        n, simulator, rand_calls, _, _ = self._verify(
            proposal,
            approx_rows=[(0.0, 0.99, 0.01, 0.0)],
            target_rows=target_rows,
        )

        self.assertEqual(n, 3)
        self.assertEqual(len(rand_calls), 3)
        self.assertEqual(simulator.rejects, [])
        self.assertProbClose(
            [_prob_values(item["prob"])[0] for item in simulator.transfers],
            [0.2, 0.1, 0.4],
        )
        self.assertEqual(
            [int(item["tokens"].reshape(-1)[0]) for item in simulator.transfers],
            [1, 2, 3],
        )

    def test_target_history_must_match_proposal_span_exactly(self):
        q_rows = [
            (0.0, 0.5, 0.3, 0.2),
            (0.0, 0.2, 0.5, 0.3),
            (0.0, 0.3, 0.2, 0.5),
        ]
        full_target = [
            (0.0, 0.9, 0.05, 0.05),
            (0.0, 0.05, 0.9, 0.05),
            (0.0, 0.05, 0.05, 0.9),
            (0.0, 0.25, 0.25, 0.5),
        ]
        proposal = _proposal(_tokens([0, 1, 2, 3]), _history(q_rows), 1, 3)

        # Exactly the four rows of proposal.tokens is enough.
        n, simulator, rand_calls, _, _ = self._verify(
            proposal,
            approx_rows=q_rows,
            target_rows=full_target,
            random_value=0.0,
        )
        self.assertEqual(n, 3)
        self.assertEqual(len(rand_calls), 3)
        self.assertEqual(simulator.rejects, [])

        # Exactly one row short means the all-accept bonus row does not exist.
        # This is a contract violation, not a silent all-accepted draft.
        with self.subTest(case="missing_only_bonus_row"):
            self._assert_validation_error(
                proposal,
                approx_rows=q_rows,
                target_rows=full_target[:-1],
            )
        for missing in (2, 3):
            with self.subTest(target_rows=len(full_target) - missing):
                self._assert_validation_error(
                    proposal,
                    approx_rows=q_rows,
                    target_rows=full_target[: len(full_target) - missing],
                )

        # A longer history contains stale rows past this proposal. It must not
        # be silently sliced, because that would hide a stale-cache bug.
        with self.subTest(target_rows="extra_stale_row"):
            self._assert_validation_error(
                proposal,
                approx_rows=q_rows,
                target_rows=full_target + [(0.0, 0.2, 0.3, 0.5)],
            )

    def test_malformed_proposals_are_rejected_before_rand_or_transfer(self):
        q_rows = [
            (0.0, 0.5, 0.3, 0.2),
            (0.0, 0.2, 0.5, 0.3),
            (0.0, 0.3, 0.2, 0.5),
        ]
        target_rows = [
            (0.0, 0.9, 0.05, 0.05),
            (0.0, 0.05, 0.9, 0.05),
            (0.0, 0.05, 0.05, 0.9),
            (0.0, 0.25, 0.25, 0.5),
        ]
        tokens = _tokens([0, 1, 2, 3])
        probabilities = _history(q_rows)
        cases = {
            "probabilities_shorter_than_draft": _proposal(
                tokens, _history(q_rows[:2]), 1, 3
            ),
            "tokens_shorter_than_prefix_plus_gamma": _proposal(
                _tokens([0, 1, 2]), probabilities, 1, 3
            ),
            "tokens_longer_than_prefix_plus_gamma": _proposal(
                _tokens([0, 1, 2, 3, 4]), probabilities, 1, 3
            ),
            "gamma_disagrees_with_tokens": _proposal(tokens, probabilities, 1, 2),
            "gamma_zero": _proposal(_tokens([0]), probabilities, 1, 0),
            "prefix_disagrees_with_tokens": _proposal(tokens, probabilities, 0, 3),
            "tokens_batch_unsupported": _proposal(
                torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
                probabilities,
                1,
                3,
            ),
            "probability_batch_unsupported": _proposal(
                tokens,
                probabilities.repeat(2, 1, 1),
                1,
                3,
            ),
            "target_batch_unsupported": _proposal(tokens, probabilities, 1, 3),
            "sampled_token_has_zero_proposal_support": _proposal(
                tokens, _history([(0.0, 0.0, 0.5, 0.5)] + q_rows[1:]), 1, 3
            ),
            "sampled_token_has_negative_proposal_support": _proposal(
                tokens, _history([(0.0, -0.1, 0.6, 0.5)] + q_rows[1:]), 1, 3
            ),
            "sampled_token_has_nan_proposal_support": _proposal(
                tokens,
                _history([(0.0, float("nan"), 0.5, 0.5)] + q_rows[1:]),
                1,
                3,
            ),
        }
        for name, proposal in cases.items():
            with self.subTest(case=name):
                candidate_target = target_rows
                if name == "tokens_longer_than_prefix_plus_gamma":
                    candidate_target = target_rows + [(0.0, 0.25, 0.25, 0.5)]
                elif name == "target_batch_unsupported":
                    candidate_target = _history(target_rows).repeat(2, 1, 1)
                self._assert_validation_error(
                    proposal,
                    approx_rows=q_rows,
                    target_rows=candidate_target,
                )

    def test_ratio_one_accepts_largest_reachable_rand(self):
        # torch.rand() is half-open: 1.0 is NOT reachable. With p == q the
        # ratio is exactly 1.0 and every real draw r < 1.0 must be accepted
        # (`reject r >= p/q`). The largest reachable float32 below 1 pins the
        # upper edge without asserting behavior on an impossible value.
        self.assertLess(MAX_RAND, 1.0)
        q_rows = [
            (0.0, 0.2, 0.5, 0.3),
            (0.0, 0.1, 0.6, 0.3),
            (0.0, 0.3, 0.3, 0.4),
        ]
        target_rows = [
            (0.0, 0.2, 0.5, 0.3),
            (0.0, 0.1, 0.6, 0.3),
            (0.0, 0.3, 0.3, 0.4),
            (0.0, 0.25, 0.25, 0.5),
        ]
        proposal = _proposal(_tokens([0, 1, 2, 3]), _history(q_rows), 1, 3)
        n, simulator, rand_calls, _, _ = self._verify(
            proposal,
            approx_rows=q_rows,
            target_rows=target_rows,
            random_value=MAX_RAND,
        )
        self.assertEqual(n, 3)
        self.assertEqual(len(rand_calls), 3)
        self.assertEqual(simulator.rejects, [])
        self.assertProbClose(
            [_prob_values(item["prob"])[0] for item in simulator.transfers],
            [0.2, 0.6, 0.4],
        )

    def test_zero_target_probability_rejects_at_zero_rand_with_exact_residual(self):
        # p(token 1) == 0 while q(token 1) == 0.5, and r == 0 (a reachable
        # draw). The rejection rule must be strict enough to reject equality:
        # `r >= p/q`. An `r > p/q` implementation would wrongly accept here.
        q_rows = [
            (0.0, 0.5, 0.3, 0.2),
            (0.0, 0.4, 0.4, 0.2),
            (0.0, 0.2, 0.4, 0.4),
        ]
        target_rows = [
            (0.0, 0.0, 0.5, 0.5),
            (0.0, 0.1, 0.1, 0.8),
            (0.0, 0.2, 0.2, 0.6),
            (0.0, 0.25, 0.25, 0.5),
        ]
        proposal = _proposal(_tokens([0, 1, 2, 3]), _history(q_rows), 1, 3)

        n, simulator, rand_calls, _, _ = self._verify(
            proposal,
            approx_rows=q_rows,
            target_rows=target_rows,
            random_value=0.0,
        )
        self.assertEqual(n, 0)
        self.assertEqual(len(rand_calls), 1)
        self.assertEqual(simulator.rejects, ["edge_cloud"])
        self.assertEqual(len(simulator.transfers), 1)
        self.assertProbClose(_prob_values(simulator.transfers[0]["prob"]), [0.5])

        # Residual is exactly norm(max(p - q, 0)) for this proposal's q:
        # [0, 0, 0.5, 0.5] - [0, 0.5, 0.3, 0.2] -> norm([0, 0, 0.2, 0.3]).
        token, weights, reject_simulator, target_cache = self._next_token(
            proposal,
            0,
            target_rows=target_rows,
            sampled_token=3,
        )
        expected = torch.tensor([[0.0, 0.0, 0.4, 0.6]])
        self.assertEqual(len(weights), 1)
        torch.testing.assert_close(weights[0], expected, rtol=0, atol=1e-7)
        self.assertEqual(token.tolist(), [[3]])
        self.assertEqual(target_cache.rollbacks, [1])
        self.assertEqual(len(reject_simulator.transfers), 1)
        self.assertIsNone(reject_simulator.transfers[0]["tokens"])
        torch.testing.assert_close(
            reject_simulator.transfers[0]["prob"],
            _history(q_rows)[:, 0, :],
            rtol=0,
            atol=0,
        )

    def test_reject_uses_q_for_ratio_payload_and_normalized_residual(self):
        # q says token 1 has mass 0.8; target p is only 0.2 -> reject at r=0.5.
        # The raw draft cache says token 1 has mass 0.1, which would *accept*.
        # Every one of ratio / reject payload / residual must follow q.
        q_rows = [
            (0.0, 0.8, 0.1, 0.1),
            (0.0, 0.4, 0.4, 0.2),
            (0.0, 0.2, 0.4, 0.4),
        ]
        raw_rows = [(0.0, 0.1, 0.8, 0.1)]
        target_rows = [
            (0.0, 0.2, 0.6, 0.2),
            (0.0, 0.1, 0.1, 0.8),
            (0.0, 0.2, 0.2, 0.6),
            (0.0, 0.25, 0.25, 0.5),
        ]
        proposal = _proposal(_tokens([0, 1, 2, 3]), _history(q_rows), 1, 3)

        n, simulator, rand_calls, _, _ = self._verify(
            proposal,
            approx_rows=raw_rows,
            target_rows=target_rows,
            random_value=0.5,
        )
        self.assertEqual(n, 0)  # reject the very first drafted token
        self.assertEqual(len(rand_calls), 1)
        self.assertEqual(simulator.rejects, ["edge_cloud"])
        self.assertEqual(len(simulator.transfers), 1)
        self.assertProbClose(_prob_values(simulator.transfers[0]["prob"]), [0.8])

        token, weights, reject_simulator, target_cache = self._next_token(
            proposal,
            n,
            target_rows=target_rows,
            sampled_token=2,
        )
        # norm(max(p - q, 0)) = norm([0, 0, 0.5, 0.1]) = [0, 0, 5/6, 1/6].
        expected = torch.tensor([[0.0, 0.0, 5.0 / 6.0, 1.0 / 6.0]])
        self.assertEqual(len(weights), 1)
        torch.testing.assert_close(weights[0], expected, rtol=0, atol=1e-7)
        self.assertEqual(token.tolist(), [[2]])
        self.assertEqual(target_cache.rollbacks, [1])

        # The reject payload is the q row at n, not the raw cache row.
        self.assertEqual(len(reject_simulator.transfers), 1)
        self.assertIsNone(reject_simulator.transfers[0]["tokens"])
        torch.testing.assert_close(
            reject_simulator.transfers[0]["prob"],
            _history(q_rows)[:, 0, :],
            rtol=0,
            atol=0,
        )

    def test_bonus_samples_target_last_row_and_rolls_back_to_forwarded_length(self):
        q_rows = [
            (0.0, 0.5, 0.3, 0.2),
            (0.0, 0.2, 0.5, 0.3),
            (0.0, 0.3, 0.2, 0.5),
        ]
        target_rows = [
            (0.0, 0.9, 0.05, 0.05),
            (0.0, 0.05, 0.9, 0.05),
            (0.0, 0.05, 0.05, 0.9),
            (0.05, 0.15, 0.3, 0.5),  # distinct bonus row
        ]
        proposal = _proposal(_tokens([0, 1, 2, 3]), _history(q_rows), 1, 3)

        n, simulator, _, _, _ = self._verify(
            proposal,
            approx_rows=q_rows,
            target_rows=target_rows,
            random_value=0.0,
        )
        self.assertEqual(n, 3)
        self.assertEqual(simulator.rejects, [])

        token, weights, _, target_cache = self._next_token(
            proposal,
            n,
            target_rows=target_rows,
            sampled_token=3,
        )
        self.assertEqual(len(weights), 1)
        torch.testing.assert_close(
            weights[0],
            _history(target_rows)[:, -1, :],
            rtol=0,
            atol=0,
        )
        self.assertEqual(token.tolist(), [[3]])
        # The target was forwarded for L + gamma = 4 rows and the bonus token
        # was never forwarded. The rollback request is exactly n + 1 (not
        # n + 2), so the real cache length stays at the forwarded length.
        forwarded = proposal.prefix_len + proposal.gamma
        self.assertEqual(n + 1, forwarded)
        self.assertEqual(target_cache.rollbacks, [n + 1])
        self.assertEqual(target_cache.prob_history.shape[1], forwarded)

    def test_next_token_rejects_positions_outside_the_proposal_window(self):
        q_rows = [
            (0.0, 0.5, 0.3, 0.2),
            (0.0, 0.2, 0.5, 0.3),
            (0.0, 0.3, 0.2, 0.5),
        ]
        target_rows = [
            (0.0, 0.9, 0.05, 0.05),
            (0.0, 0.05, 0.9, 0.05),
            (0.0, 0.05, 0.05, 0.9),
            (0.0, 0.25, 0.25, 0.5),
        ]
        proposal = _proposal(_tokens([0, 1, 2, 3]), _history(q_rows), 1, 3)
        # Valid window is [prefix_len - 1, prefix_len + gamma - 1] == [0, 3].
        for n in (proposal.prefix_len - 2, proposal.prefix_len + proposal.gamma):
            with self.subTest(n=n):
                self._assert_next_token_error(
                    proposal,
                    n,
                    target_rows=target_rows,
                )

    def test_propose_keeps_the_sampling_q_when_raw_cache_is_empty(self):
        q_rows = [
            (0.1, 0.5, 0.2, 0.2),
            (0.2, 0.1, 0.6, 0.1),
            (0.05, 0.15, 0.3, 0.5),
        ]

        class _ProposalCache:
            def __init__(self):
                self.prob_history = torch.empty((1, 0, VOCAB))
                self.hidden_states = torch.zeros((1, 1, 1))
                self.vocab_size = VOCAB
                self.device = DEVICE
                self.calls = 0

            def _forward_with_kvcache(self, prefix):
                row = _history(q_rows)[:, self.calls, :]
                self.calls += 1
                return row

        class _NeverStopAdapter:
            def reset_step(self):
                pass

            def predict(self, hidden_states):
                return False

        cache = _ProposalCache()
        decoder = _decoder()
        decoder.adapter = _NeverStopAdapter()
        decoder.rl_adapter = None
        sampler_weights = []

        def fake_sample(probs, num_samples=1):
            sampler_weights.append(probs.detach().clone())
            return probs.argmax(dim=-1, keepdim=True)

        with patch("src.baselines.core.sample", new=fake_sample):
            proposal, _ = Baselines._adaptive_decoding_propose(
                decoder,
                prefix=_tokens([0]),
                prefix_len=1,
                current_gamma=3,
                current_proposal_top_k=2,
                transfer_top_k=2,
                topk_cap=0,
                approx_model_cache=cache,
                comm_simulator=_RecordingSimulator(),
                draft_device=DEVICE,
            )

        self.assertEqual(proposal.prefix_len, 1)
        self.assertEqual(proposal.gamma, 3)
        self.assertEqual(proposal.tokens.shape, (1, 4))
        self.assertEqual(len(sampler_weights), 3)
        expected = torch.cat(sampler_weights, dim=0).unsqueeze(0)
        self.assertEqual(proposal.probabilities.shape, expected.shape)
        torch.testing.assert_close(proposal.probabilities, expected, rtol=0, atol=0)
        # The raw cache was empty; these rows can only have come from sampling.
        self.assertEqual(cache.prob_history.shape[1], 0)
        self.assertFalse(
            torch.allclose(
                proposal.probabilities,
                _history(q_rows),
                rtol=0,
                atol=0,
            ),
            "top-k rebuild must differ from the raw model q",
        )
        self.assertEqual(
            proposal.tokens.tolist()[0][1:],
            [int(weight.argmax()) for weight in sampler_weights],
        )


class AdaptivePublicEntryContractTests(unittest.TestCase):
    def test_short_target_history_fails_before_rng_or_communication(self):
        result = rounds._run(
            Baselines.adaptive_decoding,
            rounds._case(history_limit={"target": 2}),
            expect_error=ValueError,
        )
        self.assertIsInstance(result["error"], ValueError)
        actions = result["actions"]
        self.assertFalse(any(action[0] == "rand" for action in actions))
        self.assertFalse(any(action[0] == "transfer" for action in actions))

    def test_cache_length_equal_to_prefix_is_allowed(self):
        # A cache that has prefilled the complete prompt is a valid hit, not
        # stale state; only current_length strictly beyond the prefix is.
        result = rounds._run(
            Baselines.adaptive_decoding,
            rounds._case(cache_start_lengths={"draft": 1, "target": 1}),
        )
        self.assertIsNone(result["error"])
        self.assertEqual(result["accepted"], [3])
        self.assertEqual(result["metrics"]["generated_tokens"], 4)

    def test_cache_ahead_of_prefix_fails_before_rng_transfer_or_forward(self):
        for kind in ("draft", "target"):
            with self.subTest(kind=kind):
                result = rounds._run(
                    Baselines.adaptive_decoding,
                    rounds._case(cache_start_lengths={kind: 2}),
                    expect_error=ValueError,
                )
                self.assertIsInstance(result["error"], ValueError)
                self.assertIn("ahead", str(result["error"]))
                actions = result["actions"]
                self.assertFalse(
                    any(
                        action[0]
                        in (
                            "rand",
                            "transfer",
                            "payload",
                            "sample",
                            "cache.forward",
                            "cache.generate",
                        )
                        for action in actions
                    ),
                    actions,
                )


if __name__ == "__main__":
    unittest.main()
