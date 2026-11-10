"""Local SD and small round stages shared by the two distributed protocols.

This leaf module never imports engine or baselines. Values carry only tokens,
probability laws and numbers; each protocol owns its loop and side effects.
"""

from dataclasses import dataclass

import torch

from .graph_decode import graph_mode_cache_kwargs
from .metrics import get_empty_metrics
from .register import Register
from .utils import sample
from .verification import (
    ProposalWindow,
    VerificationDecision,
    decide_verification,
    prepare_verification_window,
)


def validate_dual_prefix(prefix: torch.Tensor, vocab_size: int) -> None:
    if not isinstance(prefix, torch.Tensor) or prefix.ndim != 2 or prefix.shape[0] != 1:
        raise ValueError("dual decoding supports exactly one token sequence")
    if prefix.dtype != torch.long or prefix.shape[1] == 0:
        raise ValueError("dual decoding requires a nonempty torch.long prefix")
    if (prefix < 0).any() or (prefix >= vocab_size).any():
        raise ValueError("dual decoding prefix token is outside its vocabulary")


def propose_window(cache, prefix, count: int, top_k, vocab_size: int) -> ProposalWindow:
    """Capture the actual sampling law, never a later reconstruction or wire law."""
    prefix_len = int(prefix.shape[1])
    if count == 0:
        return ProposalWindow(
            prefix, torch.empty(1, 0, vocab_size, device=prefix.device), prefix_len
        )
    if top_k is not None and top_k > 0:
        tokens, probabilities = cache.generate_with_rebuilt_topk(prefix, count, top_k)
    else:
        tokens = cache.generate(prefix, count)
        probabilities = cache.prob_history[
            :, prefix_len - 1 : prefix_len + count - 1, :vocab_size
        ]
    if tokens.shape[1] != prefix_len + count:
        raise ValueError(
            "draft generation did not produce the requested candidate window"
        )
    # History is borrowed storage: cache forward/rollback may overwrite it later.
    return ProposalWindow(tokens, probabilities.clone(), prefix_len)


def verify_window(
    proposal: ProposalWindow, target_cache, *, selected_q=None,
) -> VerificationDecision:
    """Validate before RNG; an explicit scalar payload is the split verifier's Q."""
    window = prepare_verification_window(
        proposal, target_cache.prob_history, selected_q=selected_q,
    )
    draws = (
        torch.rand((1, proposal.count), device=proposal.probabilities.device)
        if proposal.count
        else proposal.probabilities.new_empty((1, 0))
    )
    return decide_verification(window, draws)


def commit_window(proposal, decision, continuation, draft_cache, target_cache):
    """Retain only physically forwarded accepted context, not the sampled token."""
    retain_context(draft_cache, target_cache, decision.retained_context_end)
    return torch.cat(
        (
            proposal.tokens[:, : decision.retained_context_end],
            continuation.to(proposal.tokens.device),
        ),
        dim=1,
    )


def retain_context(draft_cache, target_cache, end: int):
    """Commit/truncate never requests an endpoint beyond a cache's real forward."""
    for cache in (draft_cache, target_cache):
        cache.rollback(min(cache.current_length, end))


def truncate_confirmed_eos(tokens, prompt_len: int, eos_id, disabled: bool):
    """Only call after target confirmation/commit, never on draft candidates."""
    if disabled or eos_id is None:
        return tokens, False
    eos_ids = eos_id if isinstance(eos_id, (list, tuple)) else [eos_id]
    hits = torch.zeros_like(tokens[0, prompt_len:], dtype=torch.bool)
    for value in eos_ids:
        hits |= tokens[0, prompt_len:] == value
    positions = hits.nonzero()
    if positions.numel():
        return tokens[:, : prompt_len + int(positions[0, 0]) + 1], True
    return tokens, False


@dataclass
class DualStats:
    """Numeric run accumulators; no caches, simulator, policy or decoder."""

    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    draft_rounds: int = 0
    top_k_sum: int = 0
    draft_seconds: float = 0.0
    target_seconds: float = 0.0

    def proposal(self, count, top_k):
        self.rounds += 1
        self.drafted += count
        if count:
            self.draft_rounds += 1
            self.top_k_sum += max(0, top_k or 0)

    def metrics(self, generated, elapsed, batch_delay, communication=0.0):
        metrics = get_empty_metrics()
        metrics.update(
            draft_forward_times=self.drafted,
            target_forward_times=self.rounds,
            draft_generated_tokens=self.drafted,
            draft_accepted_tokens=self.accepted,
            generated_tokens=generated,
            loop_times=self.rounds,
            draft_computation_time=self.draft_seconds,
            target_computation_time=self.target_seconds,
            queuing_time=self.rounds * batch_delay,
            communication_time=communication,
            wall_time=elapsed + self.rounds * batch_delay + communication,
            each_loop_draft_tokens=self.drafted / self.rounds if self.rounds else 0,
            avg_draft_len=self.drafted / self.draft_rounds if self.draft_rounds else 0,
            avg_top_k=self.top_k_sum / self.draft_rounds if self.draft_rounds else 0,
        )
        metrics["throughput"] = (
            generated / metrics["wall_time"] if metrics["wall_time"] > 0 else 0
        )
        return metrics


class SpeculativeDecodingMixin:
    """Real SD entry available to direct Decoding subclasses as well as Baselines."""

    @Register.register_decoding("sd")
    @torch.inference_mode()
    def speculative_decoding(
        self,
        prefix,
        transfer_top_k: int | None = 300,
        use_early_stopping: bool = False,
        stop_sequences=None,
        **kwargs,
    ):
        validate_dual_prefix(prefix, self.vocab_size)
        prompt_len = int(prefix.shape[1])
        max_tokens = prompt_len + self.args.max_tokens
        if prefix.shape[1] >= max_tokens:
            return prefix, get_empty_metrics()
        graph_kw = graph_mode_cache_kwargs(self.args, cap=int(self.args.gamma) + 4)
        caches = self._acquire_speculative_caches(
            "_speculative_decoding_caches",
            graph_kw,
            self.args.top_k,
        )
        draft, target = caches["draft"], caches["target"]
        draft.vocab_size = target.vocab_size = int(self.vocab_size)
        prefix = prefix.to(draft.device)
        stats = DualStats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream=torch.cuda.current_stream())
        while prefix.shape[1] < max_tokens:
            count = max(0, min(self.args.gamma, max_tokens - prefix.shape[1] - 1))
            proposal = propose_window(
                draft, prefix, count, transfer_top_k, self.vocab_size
            )
            stats.proposal(proposal.count, transfer_top_k)
            target.forward_context(proposal.tokens.to(target.device))
            decision = verify_window(proposal, target)
            stats.accepted += decision.accepted_count
            self.num_acc_tokens.append(decision.accepted_count)
            if self.accelerator.is_main_process:
                self.draft_forward_times += proposal.count
                self.target_forward_times += 1
            continuation = sample(decision.continuation_probs)
            prefix = commit_window(proposal, decision, continuation, draft, target)
            prefix, eos_hit = truncate_confirmed_eos(
                prefix,
                prompt_len,
                getattr(getattr(self, "tokenizer", None), "eos_token_id", None),
                getattr(self.args, "disable_eos_stop", False),
            )
            if eos_hit:
                retain_context(draft, target, int(prefix.shape[1]))
            if eos_hit or (
                use_early_stopping
                and self._check_stopping_criteria(
                    prefix,
                    stop_sequences,
                    prompt_len=prompt_len,
                )
            ):
                break
        end.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        return prefix, stats.metrics(
            prefix.shape[1] - prompt_len,
            start.elapsed_time(end) / 1000.0,
            self.comm_env.batch_delay,
        )
