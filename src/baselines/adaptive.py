"""Two-model adaptive decoding: entry point and five round helpers.

The proposal keeps the draft law it was sampled from; verification, rejection
payload billing and cache rollback remain explicit steps of this family.

RNG policy ``private-device-v1`` (2026-11-11): one lazy per-invocation context
owns all proposal, acceptance, residual, bonus and tail draws. Different actual
devices do not reuse cloned default-generator keys. The adapter proposal loop
lives here so its draws share that context without altering other families or
the model/cache interface. Seeded bitstreams intentionally change; probability
formulas, policy/reward ordering and communication remain unchanged. This is an
RNG repair, not a claim that the legacy raw-ratio verifier is numerically exact.
"""

import json
import math
import os
import time
from typing import List, Optional, Tuple

import torch

from ..baselines_helpers import (
    _add_per_model_wall_time,
    _finalize_comm_metrics,
    _reward_comm_s,
    _send_downlink_token,
)
from ..communication import CommunicationSimulator
from ..decoding_ops import (
    collect_verification_payload,
    compute_residual_distribution,
    reject_tail_scalar_bytes,
)
from ..decoding_rng import DecodingRng
from ..graph_decode import graph_mode_cache_kwargs as _graph_mode_cache_kwargs
from ..metrics import DecodingMetrics, get_empty_metrics
from ..model_gpu import KVCacheModel
from ..proposal_utils import (
    apply_transfer_top_k_cap,
    proposal_top_k,
    stage_prob_history,
)
from ..register import Register
from ..utils import rebuild_topk_uniform_probs, sample, state_entropy
from ._adaptive_state import (
    AdaptiveDecodeStats as _AdaptiveDecodeStats,
)
from ._adaptive_state import (
    AdaptiveProposal as _AdaptiveProposal,
)
from .simulators import _make_symmetric_comm_simulator


class AdaptiveDecodingMixin:
    """The dedicated two-model ``adaptive_decoding`` family."""

    @Register.register_decoding("adaptive_decoding")
    @torch.no_grad()
    def adaptive_decoding(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim: bool = False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 0,
        ntt_ms_edge_end: float = 0,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        comm_simulator: CommunicationSimulator = _make_symmetric_comm_simulator(
            self.comm_env,
            precise=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
        )
        self.color_print(f"Using transfer_top_k: {transfer_top_k}", 2)

        # Keep accounting/depth reads here: protocol introspection visits this entry.
        _comm = self.comm_config
        comm_simulator.coalesce_rounds = _comm.comm_round_trip_mode == "per_round"
        charge_residual = _comm.charge_residual_payload
        _topk_cap = _comm.transfer_top_k_cap
        transfer_top_k = apply_transfer_top_k_cap(transfer_top_k, _topk_cap)
        comm_simulator.transfer_top_k = transfer_top_k
        batch_delay = self.comm_env.batch_delay
        max_tokens = prefix.shape[1] + self.args.max_tokens
        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)
        draft_top_k = (
            transfer_top_k
            if (transfer_top_k is not None and transfer_top_k > 0)
            else self.args.top_k
        )
        _graph_kw = _graph_mode_cache_kwargs(
            self.args, cap=int(getattr(self.args, "gamma", 5)) + 4
        )
        approx_model_cache, target_model_cache = self._acquire_draft_target_caches(
            "_adaptive_decoding_caches", _graph_kw, draft_top_k, 0, 0.0
        )
        stats = _AdaptiveDecodeStats()
        rng = DecodingRng()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream=torch.cuda.current_stream())
        idx = 0
        prompt_len = prefix.shape[1]

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            for role, cache in (
                ("draft", approx_model_cache),
                ("target", target_model_cache),
            ):
                cached_length = getattr(cache, "current_length", None)
                if cached_length is not None and cached_length > prefix_len:
                    raise ValueError(
                        f"adaptive {role} cache is ahead of prefix: "
                        f"cached={cached_length}, prefix_len={prefix_len}"
                    )
            idx += 1
            comm_simulator.set_round(idx)
            current_proposal_top_k = proposal_top_k(transfer_top_k)
            remaining_tokens = max_tokens - prefix_len
            current_gamma = min(self.args.gamma, remaining_tokens - 1)
            if current_gamma <= 0:
                # The target-only tail has no rollback, trace, or downlink.
                stats.queuing_time += batch_delay
                target_model_cache.forward_context(prefix.to(target_device))
                stats.target_forward_times += 1
                if self.accelerator.is_main_process:
                    self.target_forward_times += 1
                t = rng.sample(
                    target_model_cache.prob_history[:, -1, : self.vocab_size]
                ).to(prefix.device)
                prefix = torch.cat((prefix, t), dim=1)
                self.num_acc_tokens.append(1)
                break

            # Price the proposal and serial verification before resampling.
            step_start_time = time.time()
            proposal, transfer_top_k = self._adaptive_decoding_propose(
                prefix=prefix,
                prefix_len=prefix_len,
                current_gamma=current_gamma,
                current_proposal_top_k=current_proposal_top_k,
                transfer_top_k=transfer_top_k,
                topk_cap=_topk_cap,
                approx_model_cache=approx_model_cache,
                comm_simulator=comm_simulator,
                draft_device=draft_device,
                rng=rng,
            )
            current_gamma = proposal.gamma
            stats.draft_steps += 1
            stats.sum_draft_len += current_gamma
            stats.sum_top_k += (
                current_proposal_top_k if current_proposal_top_k is not None else 0
            )
            stats.draft_forward_times += current_gamma
            stats.drafted_tokens += current_gamma
            stats.queuing_time += batch_delay
            target_model_cache.forward_context(proposal.tokens.to(target_device))
            stats.target_forward_times += 1
            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            n = self._adaptive_decoding_verify(
                proposal=proposal,
                target_model_cache=target_model_cache,
                comm_simulator=comm_simulator,
                draft_device=draft_device,
                rng=rng,
            )
            this_step_accepted_tokens = n - prefix_len + 1
            stats.accepted_tokens += this_step_accepted_tokens
            self.num_acc_tokens.append(this_step_accepted_tokens)
            step_end_time = time.time()
            if self.rl_adapter is not None:
                # Reward observes verification, before rollback and reject payloads.
                step_time = step_end_time - step_start_time
                step_comm_time = _reward_comm_s(self.args, comm_simulator, "edge_cloud")
                tps_part = this_step_accepted_tokens / (
                    step_time + step_comm_time + 1e-9
                )
                reward = math.exp(min(tps_part, 100) / 20.0)
                if current_gamma > 1:
                    acc_rate = this_step_accepted_tokens / current_gamma
                    reward *= acc_rate**2
                if not getattr(self.args, "disable_rl_update", False):
                    self.rl_adapter.step(reward)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = proposal.tokens[:, : n + 1]
            approx_model_cache.rollback(n + 1)
            if prefix.shape[1] >= max_tokens:
                break
            t = self._adaptive_decoding_next_token(
                proposal=proposal,
                n=n,
                target_model_cache=target_model_cache,
                comm_simulator=comm_simulator,
                transfer_top_k=transfer_top_k,
                charge_residual=charge_residual,
                draft_device=draft_device,
                rng=rng,
            )
            t = t.to(prefix.device)
            prefix = torch.cat((prefix, t), dim=1)
            prefix, _eos_hit = self._stop_at_eos(prefix, prompt_len)
            if _eos_hit:
                break
            self._trace_adaptive_decoding_append(
                idx=idx,
                proposal=proposal,
                n=n,
                accepted=this_step_accepted_tokens,
                token=t,
                prefix=prefix,
            )
            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break
            _send_downlink_token(comm_simulator, t, "edge_cloud")

        comm_simulator.flush_round()
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0
        prefix, metrics = self._adaptive_decoding_finalize(
            prefix=prefix,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            stats=stats,
            comm_simulator=comm_simulator,
            elapsed_time=elapsed_time,
        )
        return prefix, metrics

    def _adaptive_decoding_propose(
        self,
        *,
        prefix: torch.Tensor,
        prefix_len: int,
        current_gamma: int,
        current_proposal_top_k: Optional[int],
        transfer_top_k: Optional[int],
        topk_cap: int,
        approx_model_cache: KVCacheModel,
        comm_simulator: CommunicationSimulator,
        draft_device: torch.device,
        rng: Optional[DecodingRng] = None,
    ) -> Tuple[_AdaptiveProposal, Optional[int]]:
        """Generate with the old policy, then select the top-k for settlement.

        The proposal keeps its pre-selection probability rows. RL updates the
        next proposal/top-k and adapter threshold, not the simulator's default.
        """
        self.adapter.reset_step()
        # Keep the adapter's forward -> sample -> predict order. This is the
        # adapter-only path formerly owned by SharedOperations; no metadata-only
        # or cache.generate path is needed by this family. Direct helper callers
        # may omit the context for compatibility; the public entry never does.
        x = prefix.to(draft_device).clone()
        rebuilt_rows: List[torch.Tensor] = []
        q: Optional[torch.Tensor] = None
        draw = sample if rng is None else rng.sample
        for _ in range(current_gamma):
            q = approx_model_cache.forward_context(x)
            sample_probs = rebuild_topk_uniform_probs(q, current_proposal_top_k)
            if current_proposal_top_k is not None:
                rebuilt_rows.append(sample_probs.unsqueeze(1))
            next_tok = draw(sample_probs)
            x = torch.cat((x, next_tok), dim=1)
            hidden_states = approx_model_cache.hidden_states
            assert hidden_states is not None
            if self.adapter.predict(hidden_states):
                break
        rebuilt_draft_probs = torch.cat(rebuilt_rows, dim=1) if rebuilt_rows else None
        if self.rl_adapter is not None:
            bandwidth = comm_simulator.bandwidth_edge_cloud_mbps
            latency = comm_simulator.ntt_edge_cloud_ms
            acc_probs = getattr(self.adapter, "step_acc_probs", [])
            entropy = state_entropy(q, approx_model_cache)
            task_name = getattr(self, "task", "unknown")
            next_topk, next_threshold = self.rl_adapter.select_config(
                bandwidth,
                latency,
                acc_probs,
                entropy,
                task_name,
                training=not getattr(self.args, "disable_rl_update", False),
            )
            transfer_top_k = next_topk
            if topk_cap > 0 and transfer_top_k is not None:
                transfer_top_k = min(int(transfer_top_k), topk_cap)
            self.adapter.threshold = next_threshold
        actual_gamma = x.shape[1] - prefix_len
        stage_probs = stage_prob_history(
            approx_model_cache, prefix_len, rebuilt_draft_probs
        )
        return _AdaptiveProposal(
            x, stage_probs, prefix_len, actual_gamma
        ), transfer_top_k

    def _adaptive_decoding_verify(
        self,
        *,
        proposal: _AdaptiveProposal,
        target_model_cache: KVCacheModel,
        comm_simulator: CommunicationSimulator,
        draft_device: torch.device,
        rng: Optional[DecodingRng] = None,
    ) -> int:
        """Verify against the sampled proposal, with complete target context.

        Rebuilt proposal rows are authoritative even when the raw draft history
        is shorter. Incomplete inputs are an error, not an accepted draft. Target
        history must also include the row used to sample an all-accepted bonus.
        """
        prefix_len = proposal.prefix_len
        current_gamma = proposal.gamma
        x = proposal.tokens
        stage_probs = proposal.probabilities
        target_probs = target_model_cache.prob_history
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError("adaptive_decoding requires batch_size=1 token sequences")
        if (
            prefix_len < 1
            or current_gamma <= 0
            or x.shape[1] != prefix_len + current_gamma
        ):
            raise ValueError(
                "adaptive proposal length does not match prefix_len + gamma"
            )
        if (
            stage_probs is None
            or stage_probs.ndim != 3
            or stage_probs.shape[0] != 1
            or stage_probs.shape[1] < prefix_len + current_gamma - 1
            or stage_probs.shape[2] < self.vocab_size
        ):
            raise ValueError("proposal probability history does not cover the draft")
        if (
            target_probs is None
            or target_probs.ndim != 3
            or target_probs.shape[0] != 1
            or target_probs.shape[1] != x.shape[1]
            or target_probs.shape[2] < self.vocab_size
        ):
            raise ValueError(
                "target probability history must cover exactly the draft and bonus "
                "context: history="
                f"{None if target_probs is None else tuple(target_probs.shape)}, "
                f"tokens={tuple(x.shape)}, prefix_len={prefix_len}, "
                f"gamma={current_gamma}"
            )

        draft_tokens, selected_q = collect_verification_payload(
            stage_probs, x, prefix_len, current_gamma
        )
        if not torch.isfinite(selected_q).all() or (selected_q <= 0).any():
            raise ValueError("proposal has non-positive or non-finite selected support")
        # Copy only this proposal's verification rows, once per round.
        target_rows = target_probs[
            :, prefix_len - 1 : prefix_len + current_gamma - 1, : self.vocab_size
        ].to(draft_device)
        n = prefix_len + current_gamma - 1
        for i in range(current_gamma):
            r = (
                torch.rand(1, device=draft_device)
                if rng is None
                else rng.rand((1,), device=draft_device)
            )
            j = draft_tokens[:, i]
            q_token = selected_q[:, i : i + 1]
            comm_simulator.transfer(j, q_token, "edge_cloud")
            if r >= target_rows[:, i, j] / q_token:
                n = prefix_len + i - 1
                comm_simulator.send_reject_message("edge_cloud")
                break
        return n

    def _adaptive_decoding_next_token(
        self,
        *,
        proposal: _AdaptiveProposal,
        n: int,
        target_model_cache: KVCacheModel,
        comm_simulator: CommunicationSimulator,
        transfer_top_k: Optional[int],
        charge_residual: bool,
        draft_device: torch.device,
        rng: Optional[DecodingRng] = None,
    ) -> torch.Tensor:
        """Bill a rejection if needed, sample one token, then roll back target.

        Draft rollback and reward are already complete in the caller. The target
        retains only the accepted prefix: the newly sampled token has not been
        forwarded through either cache yet.
        """
        if not proposal.prefix_len - 1 <= n <= proposal.prefix_len + proposal.gamma - 1:
            raise ValueError("accepted position is outside the adaptive proposal")
        stage_probs = proposal.probabilities
        draw = sample if rng is None else rng.sample
        if n < proposal.prefix_len + proposal.gamma - 1:
            comm_simulator.transfer(
                None,
                stage_probs[:, n, : self.vocab_size],
                "edge_cloud",
                transfer_top_k is not None and transfer_top_k > 0,
                transfer_top_k,
            )
            if charge_residual:
                comm_simulator.simulate_transfer(
                    reject_tail_scalar_bytes(
                        stage_probs[:, n, : self.vocab_size], transfer_top_k
                    ),
                    "edge_cloud",
                )
            t = draw(
                compute_residual_distribution(
                    target_model_cache.prob_history[:, n, : self.vocab_size].to(
                        draft_device
                    ),
                    stage_probs[:, n, : self.vocab_size],
                )
            )
        else:
            bonus_row = proposal.prefix_len + proposal.gamma - 1
            t = draw(
                target_model_cache.prob_history[:, bonus_row, : self.vocab_size]
            ).to(draft_device)
        target_model_cache.rollback(n + 1)
        return t

    def _trace_adaptive_decoding_append(
        self,
        *,
        idx: int,
        proposal: _AdaptiveProposal,
        n: int,
        accepted: int,
        token: torch.Tensor,
        prefix: torch.Tensor,
    ) -> None:
        """Append the existing per-round diagnostic after EOS, before early stop."""
        _append_trace = os.environ.get("APPEND_TRACE")
        if _append_trace:
            try:
                with open(_append_trace, "a") as _fh:
                    _fh.write(
                        json.dumps(
                            {
                                "iter": int(idx),
                                "prefix_len_before": int(proposal.prefix_len),
                                "gamma": int(proposal.gamma),
                                "n": int(n),
                                "accepted": int(accepted),
                                "appended_token": int(token.item()),
                                "prefix_len_after": int(prefix.shape[1]),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass

    def _adaptive_decoding_finalize(
        self,
        *,
        prefix: torch.Tensor,
        prompt_len: int,
        max_tokens: int,
        stats: _AdaptiveDecodeStats,
        comm_simulator: CommunicationSimulator,
        elapsed_time: float,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """Apply final length limits and publish metrics, checkpoint guard, ledger."""
        prefix, _ = self._stop_at_eos(prefix, prompt_len)
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, :max_tokens]
        generated_tokens = prefix.shape[1] - prompt_len
        metrics = get_empty_metrics()
        metrics["avg_top_k"] = (
            stats.sum_top_k / stats.draft_steps if stats.draft_steps > 0 else 0
        )
        metrics["avg_draft_len"] = (
            stats.sum_draft_len / stats.draft_steps if stats.draft_steps > 0 else 0
        )
        metrics["draft_forward_times"] = stats.draft_forward_times
        metrics["target_forward_times"] = stats.target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = stats.drafted_tokens
        metrics["draft_accepted_tokens"] = stats.accepted_tokens
        metrics["queuing_time"] = stats.queuing_time
        metrics["wall_time"] = (
            elapsed_time + comm_simulator.edge_cloud_comm_time + stats.queuing_time
        )
        metrics["throughput"] = (
            generated_tokens / metrics["wall_time"] if metrics["wall_time"] > 0 else 0
        )
        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time,
            queuing_time=stats.queuing_time,
        )
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times
        self._save_adaptive_rl_checkpoints(metrics.get("throughput"))
        _finalize_comm_metrics(metrics, comm_simulator, self.args)
        return prefix, metrics
