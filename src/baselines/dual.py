"""Distributed dual-model protocols: independent DSD and DSSD orchestrations.

DSD uploads whole proposal laws and receives a reject signal; DSSD uploads only
selected q scalars and receives the rejected target row before local resampling.
Neither wire accounting nor a later top-k setting changes the actual proposal q.
"""

import time

import torch

from ..baselines_helpers import (
    _add_per_model_wall_time,
    _finalize_comm_metrics,
    _send_downlink_token,
)
from ..decoding_ops import reject_residual_payload_bytes
from ..dual_decoding import (
    DualStats,
    commit_window,
    propose_window,
    retain_context,
    truncate_confirmed_eos,
    validate_dual_prefix,
    verify_window,
)
from ..graph_decode import (
    acquire_graph_caches,
    graph_mode_cache_kwargs,
    kv_cache_construction_key,
)
from ..metrics import INT_SIZE, get_empty_metrics
from ..model_gpu import KVCacheModel
from ..proposal_utils import apply_transfer_top_k_cap
from ..register import Register
from ..utils import sample
from .simulators import _make_symmetric_comm_simulator


def acquire_dual_caches(owner, attr, draft_top_k, target_top_k, target_top_p):
    """The complete effective constructor identity guards graph reuse."""
    graph_kw = graph_mode_cache_kwargs(owner.args, cap=int(owner.args.gamma) + 4)
    caches = acquire_graph_caches(
        owner,
        attr,
        graph_kw,
        {
            "draft": lambda: KVCacheModel(
                owner.draft_model,
                owner.args.temp,
                draft_top_k,
                owner.args.top_p,
                **graph_kw,
            ),
            "target": lambda: KVCacheModel(
                owner.target_model,
                owner.args.temp,
                target_top_k,
                target_top_p,
            ),
        },
        cache_key={
            "draft": kv_cache_construction_key(
                owner.draft_model,
                temperature=owner.args.temp,
                top_k=draft_top_k,
                top_p=owner.args.top_p,
                vocab_size=owner.vocab_size,
                **graph_kw,
            ),
            "target": kv_cache_construction_key(
                owner.target_model,
                temperature=owner.args.temp,
                top_k=target_top_k,
                top_p=target_top_p,
                vocab_size=owner.vocab_size,
            ),
        }
        if graph_kw
        else None,
    )
    for cache in caches.values():
        cache.vocab_size = owner.vocab_size
    return caches["draft"], caches["target"]


def distributed_metrics(stats, generated, elapsed, batch_delay, comm_simulator, args):
    """Settle communication before calling this numeric reporting stage."""
    metrics = stats.metrics(
        generated, elapsed, batch_delay, comm_simulator.edge_cloud_comm_time
    )
    metrics.update(
        edge_cloud_data_bytes=comm_simulator.edge_cloud_data,
        comm_energy=comm_simulator.total_comm_energy,
        connect_times=comm_simulator.connect_times,
    )
    _add_per_model_wall_time(
        metrics,
        elapsed_time=elapsed,
        comm_time=comm_simulator.edge_cloud_comm_time,
        queuing_time=metrics["queuing_time"],
        draft_comp_time=stats.draft_seconds,
        target_comp_time=stats.target_seconds,
    )
    _finalize_comm_metrics(metrics, comm_simulator, args)
    return metrics


class DualDecodingMixin:
    @Register.register_decoding("dist_spec")
    @Register.register_decoding("dsd")
    @torch.no_grad()
    def dist_spec(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm=False,
        ntt_ms_edge_cloud=200,
        ntt_ms_edge_end=20,
        use_early_stopping=False,
        stop_sequences=None,
        **kwargs,
    ):
        validate_dual_prefix(prefix, self.vocab_size)
        prompt_len = int(prefix.shape[1])
        max_tokens = prompt_len + self.args.max_tokens
        if prefix.shape[1] >= max_tokens:
            return prefix, get_empty_metrics()
        comm_simulator = _make_symmetric_comm_simulator(
            self.comm_env,
            precise=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
        )
        config = self.comm_config
        comm_simulator.coalesce_rounds = config.comm_round_trip_mode == "per_round"
        charge_residual = config.charge_residual_payload
        transfer_top_k = apply_transfer_top_k_cap(
            transfer_top_k, config.transfer_top_k_cap
        )
        comm_simulator.transfer_top_k = transfer_top_k
        draft, target = acquire_dual_caches(
            self,
            "_dsd_caches",
            self.args.top_k,
            self.args.top_k,
            self.args.top_p,
        )
        prefix = prefix.to(draft.device)
        stats = DualStats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream=torch.cuda.current_stream())
        while prefix.shape[1] < max_tokens:
            comm_simulator.set_round(stats.rounds + 1)
            if stats.rounds == 0:
                comm_simulator.transfer(prefix, None, "edge_cloud")
            count = max(0, min(self.args.gamma, max_tokens - prefix.shape[1] - 1))
            t0 = time.time()
            proposal = propose_window(
                draft, prefix, count, transfer_top_k, self.vocab_size
            )
            stats.draft_seconds += time.time() - t0
            stats.proposal(proposal.count, transfer_top_k)
            if proposal.count:
                comm_simulator.transfer(
                    proposal.tokens[:, proposal.prefix_len :], None, "edge_cloud"
                )
                # Upload the exact absolute candidate window, not history[-gamma-1:-1].
                comm_simulator.transfer(
                    None,
                    proposal.probabilities,
                    "edge_cloud",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )
            t0 = time.time()
            target.forward_context(proposal.tokens.to(target.device))
            stats.target_seconds += time.time() - t0
            decision = verify_window(proposal, target)
            stats.accepted += decision.accepted_count
            self.num_acc_tokens.append(decision.accepted_count)
            if self.accelerator.is_main_process:
                self.draft_forward_times += proposal.count
                self.target_forward_times += 1
            if not decision.all_accepted:
                comm_simulator.send_reject_message("edge_cloud")
                if charge_residual:
                    comm_simulator.simulate_transfer(
                        reject_residual_payload_bytes(
                            decision.target_row, transfer_top_k
                        ),
                        "edge_cloud",
                    )
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
            # Reply carries the committed terminal token, including confirmed EOS.
            _send_downlink_token(comm_simulator, prefix[:, -1:], "edge_cloud")
            if eos_hit or (
                use_early_stopping
                and self._check_stopping_criteria(
                    prefix,
                    stop_sequences,
                    prompt_len=prompt_len,
                )
            ):
                break
        comm_simulator.flush_round()
        end.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        return prefix, distributed_metrics(
            stats,
            prefix.shape[1] - prompt_len,
            start.elapsed_time(end) / 1000.0,
            self.comm_env.batch_delay,
            comm_simulator,
            self.args,
        )

    @Register.register_decoding("dist_split_spec")
    @Register.register_decoding("dssd")
    @torch.no_grad()
    def dist_split_spec(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm=False,
        ntt_ms_edge_cloud=200,
        ntt_ms_edge_end=20,
        use_early_stopping=False,
        stop_sequences=None,
        **kwargs,
    ):
        validate_dual_prefix(prefix, self.vocab_size)
        prompt_len = int(prefix.shape[1])
        max_tokens = prompt_len + self.args.max_tokens
        if prefix.shape[1] >= max_tokens:
            return prefix, get_empty_metrics()
        comm_simulator = _make_symmetric_comm_simulator(
            self.comm_env,
            precise=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
        )
        config = self.comm_config
        comm_simulator.coalesce_rounds = config.comm_round_trip_mode == "per_round"
        charge_residual = config.charge_residual_payload
        transfer_top_k = apply_transfer_top_k_cap(
            transfer_top_k, config.transfer_top_k_cap
        )
        comm_simulator.transfer_top_k = transfer_top_k
        # Preserve DSSD's effective target law (temperature, no target top-k/p).
        draft_top_k = (
            transfer_top_k
            if transfer_top_k is not None and transfer_top_k > 0
            else self.args.top_k
        )
        draft, target = acquire_dual_caches(self, "_dssd_caches", draft_top_k, 0, 0)
        prefix = prefix.to(draft.device)
        stats = DualStats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream=torch.cuda.current_stream())
        while prefix.shape[1] < max_tokens:
            comm_simulator.set_round(stats.rounds + 1)
            count = max(0, min(self.args.gamma, max_tokens - prefix.shape[1] - 1))
            t0 = time.time()
            proposal = propose_window(
                draft, prefix, count, transfer_top_k, self.vocab_size
            )
            stats.draft_seconds += time.time() - t0
            stats.proposal(proposal.count, transfer_top_k)
            # One FP32 normalized-Q scalar per token. Raw BF16 weights are not
            # that law; transmit exactly the rounded scalar the receiver uses.
            selected_q = proposal.selected_probabilities().float()
            if proposal.count:
                tokens = proposal.tokens[:, proposal.prefix_len :]
                comm_simulator.transfer(tokens, selected_q, "edge_cloud")
            t0 = time.time()
            target.forward_context(proposal.tokens.to(target.device))
            stats.target_seconds += time.time() - t0
            decision = verify_window(proposal, target, selected_q=selected_q)
            stats.accepted += decision.accepted_count
            self.num_acc_tokens.append(decision.accepted_count)
            if self.accelerator.is_main_process:
                self.draft_forward_times += proposal.count
                self.target_forward_times += 1
            if not decision.all_accepted:
                # Device already owns q. Edge returns rejected p BEFORE
                # local residual sampling.
                if charge_residual:
                    comm_simulator.simulate_transfer(
                        reject_residual_payload_bytes(
                            decision.target_row, transfer_top_k
                        ),
                        "edge_cloud",
                    )
                else:
                    comm_simulator.simulate_transfer(INT_SIZE, "edge_cloud")
                    comm_simulator.transfer(None, decision.target_row, "edge_cloud")
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
            # Reply carries the committed terminal token, including confirmed EOS.
            _send_downlink_token(comm_simulator, prefix[:, -1:], "edge_cloud")
            if eos_hit or (
                use_early_stopping
                and self._check_stopping_criteria(
                    prefix,
                    stop_sequences,
                    prompt_len=prompt_len,
                )
            ):
                break
        comm_simulator.flush_round()
        end.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        return prefix, distributed_metrics(
            stats,
            prefix.shape[1] - prompt_len,
            start.elapsed_time(end) / 1000.0,
            self.comm_env.batch_delay,
            comm_simulator,
            self.args,
        )
