"""CEE DSD block and DSSD scalar protocols under the effective wire-draft law.

Both use the same strict numerical stages, but own their message sequences.
Only DSD uploads the full encoded Q. DSSD uploads normalized float64 selected-Q
scalars (not raw low-precision weights); rejection accounting returns the exact
verifier row/payload. This remains an in-process network simulation, not RPC.
"""

import torch

from ..baselines_helpers import (
    _add_per_model_wall_time,
    _finalize_comm_metrics,
    _send_downlink_token,
)
from ..cee_cascade import (
    CascadeStats,
    carry_inner_output,
    decide_stage,
    finish_stage,
    prepare_inner_stage,
    prepare_target_stage,
    retain_cascade_context,
)
from ..cee_proposal import (
    CeeProbabilityCodec,
    extend_carried,
    proposal_from_wire,
    propose_candidates,
)
from ..decoding_rng import DecodingRng
from ..dual_decoding import truncate_confirmed_eos, validate_dual_prefix
from ..graph_decode import graph_mode_cache_kwargs
from ..metrics import get_empty_metrics
from ..proposal_utils import apply_transfer_top_k_cap
from ..register import Register
from .simulators import _make_comm_simulator


def _acquire_caches(owner, attr, requested_top_k):
    if owner.args.gamma1 < 0 or owner.args.gamma2 < 0:
        raise ValueError("CEE gamma1/gamma2 must be nonnegative")
    # Preserve model filtering separately from the communication cap. Correctness
    # comes from Q_D ownership, not from hiding the cap bug by changing raw D.
    model_top_k = (
        requested_top_k
        if requested_top_k is not None and requested_top_k > 0
        else owner.args.top_k
    )
    graph = graph_mode_cache_kwargs(
        owner.args,
        cap=int(owner.args.gamma1) + int(owner.args.gamma2) + 4,
    )
    return owner._acquire_three_layer_caches(attr, graph, model_top_k)


def _target_tail(prefix, target, codec, rng):
    empty = proposal_from_wire(
        prefix, int(prefix.shape[1]), (), codec.vocab_size, device=prefix.device
    )
    stage = prepare_target_stage(empty, target)
    decision = decide_stage(stage, rng)
    return finish_stage(stage, decision, rng)


def _reject_payload(comm, stage, decision, link, charge_residual):
    if decision.all_accepted:
        return
    comm.send_reject_message(link)
    if charge_residual:
        if stage.verifier_payloads is not None:
            size = stage.verifier_payloads[decision.accepted_count].nbytes
        else:
            row = stage.verifier_rows[:, decision.accepted_count, :]
            size = row.numel() * row.element_size()
        comm.simulate_transfer(size, link)


def _confirmed_output(owner, prefix, prompt_len, caches):
    prefix, stopped = truncate_confirmed_eos(
        prefix,
        prompt_len,
        getattr(getattr(owner, "tokenizer", None), "eos_token_id", None),
        getattr(owner.args, "disable_eos_stop", False),
    )
    # Includes EOS cropping: even a formerly forwarded accepted EOS becomes the
    # final unforwarded emitted token. Never extend a shorter physical cache.
    retain_cascade_context(caches.values(), int(prefix.shape[1]) - 1)
    return prefix, stopped


def _metrics(owner, stats, generated, elapsed, comm):
    communication = comm.edge_cloud_comm_time + comm.edge_end_comm_time
    queuing = stats.target_forwards * owner.comm_env.batch_delay
    metrics = get_empty_metrics()
    metrics.update(
        little_forward_times=stats.little_forwards,
        draft_forward_times=stats.draft_forwards,
        target_forward_times=stats.target_forwards,
        little_generated_tokens=stats.little_candidates,
        draft_generated_tokens=stats.draft_candidates,
        little_accepted_tokens=stats.little_accepted,
        draft_accepted_tokens=stats.draft_accepted,
        generated_tokens=generated,
        loop_times=stats.rounds,
        computation_time=elapsed,
        communication_time=communication,
        edge_end_comm_time=comm.edge_end_comm_time,
        edge_cloud_data_bytes=comm.edge_cloud_data,
        edge_end_data_bytes=comm.edge_end_data,
        cloud_end_data_bytes=comm.cloud_end_data,
        comm_energy=comm.total_comm_energy,
        connect_times=comm.connect_times,
        queuing_time=queuing,
        wall_time=elapsed + communication + queuing,
        cee_probability_policy="wire-draft-v1",
        rng_policy="private-device-v1",
    )
    metrics["throughput"] = (
        generated / metrics["wall_time"] if metrics["wall_time"] > 0 else 0
    )
    _add_per_model_wall_time(
        metrics, elapsed_time=elapsed, comm_time=communication, queuing_time=queuing
    )
    _finalize_comm_metrics(metrics, comm, owner.args)
    if owner.accelerator.is_main_process:
        owner.draft_forward_times += stats.draft_forwards
        owner.target_forward_times += stats.target_forwards
    return metrics


class CeeDistributedMixin:
    @Register.register_decoding("cee_dsd")
    @torch.no_grad()
    def cee_dsd(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm=False,
        ntt_ms_edge_cloud=10,
        ntt_ms_edge_end=1,
        use_early_stopping=False,
        stop_sequences=None,
        **kwargs,
    ):
        validate_dual_prefix(prefix, self.vocab_size)
        prompt_len = int(prefix.shape[1])
        end_pos = prompt_len + self.args.max_tokens
        if prompt_len >= end_pos:
            return prefix, get_empty_metrics()
        caches = _acquire_caches(self, "_cee_dsd_caches", transfer_top_k)
        little, draft, target = (caches[role] for role in ("little", "draft", "target"))
        config = self.comm_config
        comm = _make_comm_simulator(
            self.comm_env,
            precise=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            edge_end=self.comm_env.edge_end_bandwidth,
            cloud_end=self.comm_env.cloud_end_bandwidth,
            transfer_top_k=transfer_top_k,
        )
        comm.coalesce_rounds = config.comm_round_trip_mode == "per_round"
        charge_residual = config.charge_residual_payload
        transfer_top_k = apply_transfer_top_k_cap(
            transfer_top_k, config.transfer_top_k_cap
        )
        comm.transfer_top_k = transfer_top_k
        codec = CeeProbabilityCodec(self.vocab_size, transfer_top_k)
        rng, stats = DecodingRng(), CascadeStats()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record(stream=torch.cuda.current_stream())
        comm.transfer(prefix, None, "edge_end")
        while prefix.shape[1] < end_pos:
            stats.rounds += 1
            comm.set_round(stats.rounds)
            if stats.rounds == 1:
                comm.transfer(prefix, None, "edge_cloud")
            remaining = end_pos - int(prefix.shape[1])
            if remaining == 1:
                prefix = _target_tail(prefix, target, codec, rng)
            else:
                count = min(self.args.gamma2, remaining - 2)
                little_proposal = propose_candidates(little, prefix, count, codec, rng)
                stats.little_forwards += count
                stats.little_candidates += count
                if count:
                    comm.simulate_transfer(
                        count * little_proposal.window.tokens.element_size()
                        + little_proposal.probability_bytes,
                        "edge_end",
                    )
                inner = prepare_inner_stage(little_proposal, draft, codec)
                stats.draft_forwards += 1
                inner_decision = decide_stage(inner, rng)
                _reject_payload(
                    comm, inner, inner_decision, "edge_end", charge_residual
                )
                inner_tokens = finish_stage(inner, inner_decision, rng)
                carried = carry_inner_output(inner, inner_decision, inner_tokens)
                retain_cascade_context(
                    (little, draft), inner_decision.retained_context_end
                )
                stats.little_accepted += inner_decision.accepted_count
                _send_downlink_token(comm, inner_tokens[:, -1:], "edge_end")
                extra = min(self.args.gamma1, remaining - 1 - carried.window.count)
                proposal = extend_carried(carried, draft, extra, codec, rng)
                stats.draft_forwards += extra
                stats.draft_candidates += proposal.window.count
                comm.simulate_transfer(
                    proposal.window.count * proposal.window.tokens.element_size()
                    + proposal.probability_bytes,
                    "edge_cloud",
                    topk=codec.top_k,
                    draft_len=proposal.window.count,
                )
                outer = prepare_target_stage(proposal, target)
                decision = decide_stage(outer, rng)
                stats.draft_accepted += decision.accepted_count
                self.num_acc_tokens.append(decision.accepted_count)
                _reject_payload(comm, outer, decision, "edge_cloud", charge_residual)
                prefix = finish_stage(outer, decision, rng)
            stats.target_forwards += 1
            prefix, stopped = _confirmed_output(self, prefix, prompt_len, caches)
            _send_downlink_token(comm, prefix[:, -1:], "edge_cloud")
            _send_downlink_token(comm, prefix[:, -1:], "edge_end")
            if stopped or (
                use_early_stopping
                and self._check_stopping_criteria(
                    prefix, stop_sequences, prompt_len=prompt_len
                )
            ):
                break
        comm.flush_round()
        end.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        return prefix, _metrics(
            self,
            stats,
            int(prefix.shape[1]) - prompt_len,
            start.elapsed_time(end) / 1000.0,
            comm,
        )

    @Register.register_decoding("cee_dssd")
    @torch.no_grad()
    def cee_dssd(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm=False,
        ntt_ms_edge_cloud=10,
        ntt_ms_edge_end=1,
        use_early_stopping=False,
        stop_sequences=None,
        **kwargs,
    ):
        validate_dual_prefix(prefix, self.vocab_size)
        prompt_len = int(prefix.shape[1])
        end_pos = prompt_len + self.args.max_tokens
        if prompt_len >= end_pos:
            return prefix, get_empty_metrics()
        caches = _acquire_caches(self, "_cee_dssd_caches", transfer_top_k)
        little, draft, target = (caches[role] for role in ("little", "draft", "target"))
        config = self.comm_config
        comm = _make_comm_simulator(
            self.comm_env,
            precise=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            edge_end=self.comm_env.edge_end_bandwidth,
            cloud_end=self.comm_env.cloud_end_bandwidth,
            transfer_top_k=transfer_top_k,
        )
        comm.coalesce_rounds = config.comm_round_trip_mode == "per_round"
        charge_residual = config.charge_residual_payload
        transfer_top_k = apply_transfer_top_k_cap(
            transfer_top_k, config.transfer_top_k_cap
        )
        comm.transfer_top_k = transfer_top_k
        codec = CeeProbabilityCodec(self.vocab_size, transfer_top_k)
        rng, stats = DecodingRng(), CascadeStats()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record(stream=torch.cuda.current_stream())
        comm.transfer(prefix, None, "edge_end")
        while prefix.shape[1] < end_pos:
            stats.rounds += 1
            comm.set_round(stats.rounds)
            if stats.rounds == 1:
                comm.transfer(prefix, None, "edge_cloud")
            remaining = end_pos - int(prefix.shape[1])
            if remaining == 1:
                prefix = _target_tail(prefix, target, codec, rng)
            else:
                count = min(self.args.gamma2, remaining - 2)
                little_proposal = propose_candidates(little, prefix, count, codec, rng)
                stats.little_forwards += count
                stats.little_candidates += count
                selected_little_q = little_proposal.window.selected_probabilities()
                if count:
                    comm.transfer(
                        little_proposal.window.tokens[
                            :, little_proposal.window.prefix_len :
                        ],
                        selected_little_q,
                        "edge_end",
                    )
                inner = prepare_inner_stage(
                    little_proposal, draft, codec, selected_q=selected_little_q
                )
                stats.draft_forwards += 1
                inner_decision = decide_stage(inner, rng)
                _reject_payload(
                    comm, inner, inner_decision, "edge_end", charge_residual
                )
                inner_tokens = finish_stage(inner, inner_decision, rng)
                carried = carry_inner_output(inner, inner_decision, inner_tokens)
                retain_cascade_context(
                    (little, draft), inner_decision.retained_context_end
                )
                stats.little_accepted += inner_decision.accepted_count
                _send_downlink_token(comm, inner_tokens[:, -1:], "edge_end")
                extra = min(self.args.gamma1, remaining - 1 - carried.window.count)
                proposal = extend_carried(carried, draft, extra, codec, rng)
                stats.draft_forwards += extra
                stats.draft_candidates += proposal.window.count
                # The same received scalar payload drives strict verification.
                selected_draft_q = proposal.window.selected_probabilities()
                comm.transfer(
                    proposal.window.tokens[:, proposal.window.prefix_len :],
                    selected_draft_q,
                    "edge_cloud",
                )
                outer = prepare_target_stage(
                    proposal, target, selected_q=selected_draft_q
                )
                decision = decide_stage(outer, rng)
                stats.draft_accepted += decision.accepted_count
                self.num_acc_tokens.append(decision.accepted_count)
                _reject_payload(comm, outer, decision, "edge_cloud", charge_residual)
                prefix = finish_stage(outer, decision, rng)
            stats.target_forwards += 1
            prefix, stopped = _confirmed_output(self, prefix, prompt_len, caches)
            _send_downlink_token(comm, prefix[:, -1:], "edge_cloud")
            _send_downlink_token(comm, prefix[:, -1:], "edge_end")
            if stopped or (
                use_early_stopping
                and self._check_stopping_criteria(
                    prefix, stop_sequences, prompt_len=prompt_len
                )
            ):
                break
        comm.flush_round()
        end.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        return prefix, _metrics(
            self,
            stats,
            int(prefix.shape[1]) - prompt_len,
            start.elapsed_time(end) / 1000.0,
            comm,
        )
