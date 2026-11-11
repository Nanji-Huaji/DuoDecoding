"""CEE-SD orchestration: the tri-model entry, its five stage helpers and
the policy/trace helpers plus current ablation wrappers.

The three wrappers in this module only select an explicit CEE-SD ablation; they
do not carry a second implementation. CEE-DSD/DSSD/CUHLM compositions remain in
``core`` until their own batch.
"""

import json
import os
import time
from typing import List, Optional, Tuple, cast

import torch

from ..baselines_helpers import (
    _add_per_model_wall_time,
    _finalize_comm_metrics,
    _move_token_tensor,
    _reward_comm_s,
    _send_downlink_token,
    _validate_token_range,
)
from ..communication import CUHLM, CommunicationSimulator
from ..decoding_ops import (
    apply_rollback,
    sample_accept_token,
    sample_reject_token,
)
from ..decoding_types import RollbackPlan as _CeeRollbackPlan
from ..graph_decode import graph_mode_cache_kwargs as _graph_mode_cache_kwargs
from ..metrics import INT_SIZE, DecodingMetrics, get_empty_metrics
from ..model_gpu import KVCacheModel
from ..proposal_utils import (
    apply_transfer_top_k_cap,
    proposal_top_k,
)
from ..register import Register
from ..utils import state_entropy
from ._cee_state import (
    CeeDecodeStats as _CeeDecodeStats,
)
from ._cee_state import (
    CeeProposal as _CeeProposal,
)
from ._cee_state import (
    CeeVerification as _CeeVerification,
)
from .operations import _uplink_prob_payload
from .simulators import _make_comm_simulator


class CeeSdMixin:
    """CEE-SD main entry, stage seams, policy/trace hooks and wrappers."""

    @Register.register_decoding("tridecoding")
    @torch.no_grad()
    def tridecoding(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 10,
        ntt_ms_edge_end: float = 1,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """Table VIII Static 消融：CEE-SD 关 ARP/DRA，固定 γ 与 top-k。

        实现上转发到 ``adaptive_tridecoding(_ablation="static")``，不再维护
        第二套三级循环——消融只允许"关模块"，不允许换实现
        （判定标准见 docs/modes_inventory.md §3）。

        与主方法的差异（都在 ``_ablation="static"`` 分支里表达）：
        - 不驱动 ARP head：``MODE_FEATURES["tridecoding"]`` 是 no-RL/no-head，
          草稿按固定 γ1/γ2 生成，没有 accept 预测早停；
        - 不跑 RL/DRA：top-k 由调用方固定传入（Table VIII 用 1024）；
        - 验证/回滚/计费仍与 ``adaptive_tridecoding`` 走完全相同的路径。

        历史提示：2026-10-19 之前这是独立实现（旧 verify/rollback API），
        Table VIII 的旧数字来自那份代码；改成本 wrapper 后数字会变，需要
        重跑该表并重新收割 golden。
        """
        return self.adaptive_tridecoding(
            prefix,
            transfer_top_k=transfer_top_k,
            use_precise_comm_sim=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            use_early_stopping=use_early_stopping,
            stop_sequences=stop_sequences,
            _ablation="static",
            **kwargs,
        )

    @Register.register_decoding("cee_sd_no_arp")
    @torch.no_grad()
    def cee_sd_no_arp(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 10,
        ntt_ms_edge_end: float = 1,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """CEE-SD "无 ARP"消融（当前实现，推荐用于论文消融）。

        与 ``tridecoding``（static）的区别：保留 RL/DRA 的 top-k 选择，只是
        不挂 ARP 验收头、不用接受率提前停；验证/回滚/计费仍走
        ``adaptive_tridecoding`` 同一条路径，因此和 CEE-SD 主结果只差 ARP
        这一个模块。

        ``ceesd_without_arp`` 是旧血缘实现，只用于复现历史 run；新的消融
        实验请用本模式。
        """
        return self.adaptive_tridecoding(
            prefix,
            transfer_top_k=transfer_top_k,
            use_precise_comm_sim=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            use_early_stopping=use_early_stopping,
            stop_sequences=stop_sequences,
            _ablation="no_arp",
            **kwargs,
        )

    @Register.register_decoding("adaptive_tridecoding")
    @Register.register_decoding("cee_sd")
    @torch.no_grad()
    def adaptive_tridecoding(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud=10,
        ntt_ms_edge_end=1,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        _opportunistic_first_stage: bool = False,
        _ablation: str | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """Little→draft correction followed by draft→target verification.

        Only target-verified tokens are committed. Sampling rows are captured
        before policy changes; lossy uplinks explicitly verify against their
        dequantized wire approximation, not an untransmitted exact distribution.
        """
        if prefix.dtype != torch.long:
            prefix = prefix.long()
        if prefix.ndim != 2 or prefix.shape[0] != 1 or prefix.shape[1] < 1:
            raise ValueError(
                "adaptive_tridecoding requires a nonempty batch_size=1 prefix"
            )
        if _ablation not in (None, "static", "no_arp"):
            raise ValueError(
                f"adaptive_tridecoding: 未知 _ablation={_ablation!r}"
                "（只支持 None/static/no_arp）"
            )
        static = _ablation == "static"
        no_arp = _ablation == "no_arp"
        # These are real consumers: protocol introspection visits this method.
        accounting = self.comm_config
        if accounting.force_full_vocab_transfer:
            transfer_top_k = int(getattr(self, "vocab_size", 0) or 32000)
        charge_residual = accounting.charge_residual_payload
        prob_payload_bits = accounting.prob_payload_bits
        topk_cap = accounting.transfer_top_k_cap
        batch_delay = self.comm_env.batch_delay
        prompt_len = prefix.shape[1]
        max_tokens = prompt_len + self.args.max_tokens
        little_device = self.get_model_input_device(self.little_model)
        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)
        draft_top_k = (
            transfer_top_k
            if transfer_top_k is not None and transfer_top_k > 0
            else self.args.top_k
        )
        graph_kw = _graph_mode_cache_kwargs(
            self.args, cap=int(self.args.gamma1) + int(self.args.gamma2) + 4
        )
        # Capacity belongs to cache construction, not the graph reuse switch.
        probe_length = getattr(self.args, "probe_cache_max_length", None)
        cache_kwargs = {"max_length": probe_length} if probe_length is not None else {}
        caches = self._acquire_three_layer_caches(
            "_adaptive_tridecoding_caches",
            graph_kw,
            draft_top_k,
            cache_kwargs=cache_kwargs,
        )
        little_cache, draft_cache, target_cache = (
            caches["little"],
            caches["draft"],
            caches["target"],
        )
        comm = _make_comm_simulator(
            self.comm_env,
            precise=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            edge_end=self.comm_env.edge_end_bandwidth,
            cloud_end=self.comm_env.cloud_end_bandwidth,
            transfer_top_k=transfer_top_k,
            mode=getattr(self.args, "comm_trace_mode", "static"),
        )
        comm.coalesce_rounds = accounting.comm_round_trip_mode == "per_round"
        if not accounting.force_full_vocab_transfer:
            transfer_top_k = apply_transfer_top_k_cap(transfer_top_k, topk_cap)
        if accounting.force_full_vocab_transfer or topk_cap > 0:
            comm.transfer_top_k = transfer_top_k

        uncertainty_sim = None
        if _opportunistic_first_stage:
            _, threshold = self._select_cuhlm_stage_config(
                stage="little_to_draft",
                transfer_top_k=transfer_top_k,
                uncertainty_threshold=getattr(self.args, "uncertainty_threshold", 0.8),
            )
            uncertainty_sim = CUHLM(
                min_bandwidth_mbps=self.comm_env.min_bandwidth_mbps,
                bw_model=self.comm_env.comm_bw_model,
                bandwidth_edge_cloud=self.comm_env.edge_cloud_bandwidth,
                bandwidth_edge_end=self.comm_env.edge_end_bandwidth,
                bandwidth_cloud_end=self.comm_env.cloud_end_bandwidth,
                uncertainty_threshold=threshold,
                vocab_size=self.vocab_size,
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=self.comm_env.stochastic_ntt,
            )
        stats = _CeeDecodeStats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream=torch.cuda.current_stream())
        comm.transfer(prefix, None, "edge_end")
        round_index = 0
        while prefix.shape[1] < max_tokens:
            round_index += 1
            comm.set_round(round_index)
            prefix_len = prefix.shape[1]
            proposal_k = proposal_top_k(transfer_top_k)
            step_start = time.time()
            end_energy_start = comm.total_comm_energy
            gamma2 = getattr(self, "_next_gamma2", None)
            gamma2 = self.args.gamma2 if gamma2 is None else gamma2
            adapter = None
            if not static and not no_arp:
                self.small_draft_adapter.reset_step()
                adapter = self.small_draft_adapter
                assert adapter.device != torch.device("cpu")
            little = self._cee_propose(
                little_cache,
                _move_token_tensor(prefix, little_device),
                1 if _opportunistic_first_stage else gamma2,
                proposal_k,
                adapter,
            )
            stats.little_computation_time += little.computation_time
            stats.little_forward_times += little.count
            stats.little_generated_tokens += little.count
            little_entropy = state_entropy(little.entropy_source, little_cache)
            if self.little_rl_adapter is not None and not static:
                selection_start = time.time()
                self._select_adaptive_little_config(
                    comm_simulator=comm,
                    little_entropy=little_entropy,
                    gamma2_used=gamma2,
                    no_arp=no_arp,
                )
                stats.dra_overhead_time += time.time() - selection_start
                if uncertainty_sim is not None and not no_arp:
                    uncertainty_sim.uncertainty_threshold = float(
                        self.small_draft_adapter.threshold
                    )
            first, draft_verify_time = self._cee_first_stage(
                little,
                little_cache,
                draft_cache,
                comm,
                uncertainty_sim=uncertainty_sim,
                prob_payload_bits=prob_payload_bits,
                charge_residual=charge_residual,
                proposal_top_k=proposal_k,
            )
            stats.draft_computation_time += draft_verify_time
            stats.draft_forward_times += int(not first.skipped)
            stats.little_accepted_tokens += first.accepted_count
            # Reward observes verification and residual payload, before the
            # first-stage correction response, exactly at the policy cutpoint.
            if self.little_rl_adapter is not None and not static:
                reward = self.little_rl_adapter.compute_reward(
                    accepted=first.accepted_count,
                    generated=little.count,
                    comm_s=_reward_comm_s(self.args, comm, "edge_end"),
                    compute_s=time.time() - step_start,
                    energy_j=comm.total_comm_energy - end_energy_start,
                    forward_counts={"little": float(little.count)},
                    opportunistic=_opportunistic_first_stage,
                )
                if getattr(self.args, "rl_team_reward", False):
                    team = getattr(self, "_team_reward_buffer", None)
                    if team is not None:
                        reward = team
                if not getattr(self.args, "disable_rl_update", False):
                    self.little_rl_adapter.step(reward)
            if not first.skipped:
                rejection_bytes = 0.0
                if not first.all_accepted:
                    rejection_bytes = 6.0 + self._adaptive_rejection_prob_bytes(
                        prob_data=first.wire_probabilities[:, first.accepted_count, :],
                        transfer_top_k=transfer_top_k,
                    )
                comm.simulate_transfer(
                    INT_SIZE
                    + first.token.element_size() * first.token.numel()
                    + rejection_bytes,
                    "edge_end",
                )
            _validate_token_range(
                first.token,
                vocab_size=self.vocab_size,
                label="adaptive_tridecoding.edge_end.sampled_token",
            )
            prefix = torch.cat(
                (little.tokens[:, : first.accepted_end + 1], first.token), dim=1
            )
            _validate_token_range(
                prefix,
                vocab_size=self.vocab_size,
                label="adaptive_tridecoding.edge_end.prefix_after_concat",
            )
            # Inner EOS is only a proposal; target may reject it.
            forwarded_tokens = prefix[:, prefix_len:]
            cloud_energy_start = comm.total_comm_energy
            cloud_bytes_start = comm.edge_cloud_data
            step_start = time.time()
            gamma1 = getattr(self, "_next_gamma1", None)
            gamma1 = self.args.gamma1 if gamma1 is None else gamma1
            adapter = None
            if not static and not no_arp:
                self.draft_target_adapter.reset_step()
                adapter = self.draft_target_adapter
                assert adapter.device != torch.device("cpu")
            draft = self._cee_propose(
                draft_cache,
                _move_token_tensor(prefix, draft_device),
                gamma1,
                proposal_k,
                adapter,
            )
            stats.draft_computation_time += draft.computation_time
            proposal = self._cee_target_proposal(first, draft, prefix_len)
            comm.transfer(
                prefix if round_index == 1 else forwarded_tokens, None, "edge_cloud"
            )
            stats.queuing_time += batch_delay
            target_start = time.time()
            target_cache.forward_context(
                _move_token_tensor(proposal.tokens, target_device)
            )
            stats.target_computation_time += time.time() - target_start
            # Preserve CPU observation AFTER WAN transfer and target launch.
            draft_entropy = state_entropy(draft.entropy_source, draft_cache)
            if self.rl_adapter is not None and not static:
                selection_start = time.time()
                transfer_top_k = self._select_adaptive_main_config(
                    comm_simulator=comm,
                    draft_entropy=draft_entropy,
                    gamma1_used=gamma1,
                    topk_cap=topk_cap,
                    no_arp=no_arp,
                )
                stats.dra_overhead_time += time.time() - selection_start
            stats.draft_forward_times += draft.count
            stats.target_forward_times += 1
            stats.draft_generated_tokens += proposal.count
            # Even gamma1=0 has the first stage's candidate window to verify.
            wire_probs, wire_bits = _uplink_prob_payload(
                proposal.probabilities, prob_payload_bits
            )
            wire_probs = cast(torch.Tensor, wire_probs)
            tokens = proposal.tokens[:, proposal.prefix_len :]
            selected_wire = torch.gather(wire_probs, 2, tokens.unsqueeze(-1)).squeeze(
                -1
            )
            comm.transfer(tokens, selected_wire, "edge_cloud", prob_bits=wire_bits)
            self._trace_adaptive_cache_preverify(
                x=proposal.tokens,
                prefix_len=prefix_len,
                draft_model_cache=draft_cache,
                target_model_cache=target_cache,
                actual_gamma1=draft.count,
            )
            second = self._cee_verify_proposal(
                proposal,
                draft_cache,
                target_cache,
                wire_probabilities=wire_probs,
                output_device=draft_device,
            )
            if charge_residual and not second.all_accepted:
                comm.simulate_transfer(
                    self._residual_payload_bytes(wire_probs, proposal_k), "edge_cloud"
                )
            self._trace_adaptive_calibration(
                x=proposal.tokens,
                prefix_len=prefix_len,
                draft_model_cache=draft_cache,
                target_model_cache=target_cache,
                draft_accepted_this_iter=second.accepted_count,
            )
            stats.draft_accepted_tokens += second.accepted_count
            if self.rl_adapter is not None and not static:
                reward = self.rl_adapter.compute_reward(
                    accepted=second.accepted_count,
                    generated=draft.count,
                    comm_s=_reward_comm_s(self.args, comm, "edge_cloud"),
                    compute_s=time.time() - step_start,
                    energy_j=comm.total_comm_energy - cloud_energy_start,
                    transferred_bytes=(
                        comm.round_bytes("edge_cloud")
                        if getattr(self.args, "rl_comm_cost_mode", "nominal")
                        == "nominal"
                        else comm.edge_cloud_data - cloud_bytes_start
                    ),
                    forward_counts={
                        "little": float(little.count),
                        "draft": float(draft.count),
                        "target": 1.0,
                    },
                    queue_s=float(self.comm_env.batch_delay),
                )
                if getattr(self.args, "rl_team_reward", False):
                    self._team_reward_buffer = reward
                if not getattr(self.args, "disable_rl_update", False):
                    self.rl_adapter.step(reward)
            prefix = proposal.tokens[:, : second.accepted_end + 1]
            if second.accepted_end <= little_cache.current_length:
                little_cache.rollback(second.accepted_end + 1)
            rejection_bytes = 0.0
            if not second.all_accepted:
                rejection_bytes = 6.0 + self._adaptive_rejection_prob_bytes(
                    prob_data=wire_probs[:, second.accepted_count, :],
                    transfer_top_k=transfer_top_k,
                )
            prefix = torch.cat((prefix, second.token), dim=1)
            comm.simulate_transfer(
                INT_SIZE
                + second.token.element_size() * second.token.numel()
                + rejection_bytes,
                "edge_cloud",
                topk=transfer_top_k,
                draft_len=proposal.count,
            )
            _send_downlink_token(comm, second.token, "edge_end")
            prefix, eos_hit = self._stop_at_eos(prefix, prompt_len)
            if eos_hit:
                break
            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break
        # Pending per-round bytes/NTT must exist before every derived time metric.
        comm.flush_round()
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0
        return self._cee_finalize(
            prefix, prompt_len, max_tokens, stats, comm, elapsed_time
        )

    def _cee_propose(
        self,
        cache,
        prefix,
        gamma,
        top_k,
        adapter=None,
    ) -> _CeeProposal:
        """Snapshot just the generated positions, before policy or wire changes.

        A zero extension must not forward or sample a token. Entropy remains a
        tensor reference here: its CPU observation belongs to the caller's stage
        ordering, after target launch for the outer stage.
        """
        if gamma < 0:
            raise ValueError("CEE proposal gamma must be nonnegative")
        if gamma == 0:
            return _CeeProposal(
                prefix,
                torch.empty((1, 0, self.vocab_size), device=prefix.device),
                prefix.shape[1],
            )
        started = time.time()
        tokens, rebuilt, _, entropy_source = (
            self._generate_with_optional_rebuilt_proposal(
                cache,
                prefix,
                gamma,
                top_k,
                adapter=adapter,
            )
        )
        elapsed = time.time() - started
        start = prefix.shape[1] - 1
        count = tokens.shape[1] - prefix.shape[1]
        if rebuilt is None:
            probabilities = cache.prob_history[
                :, start : start + count, : self.vocab_size
            ]
        else:
            probabilities = rebuilt[:, :count, : self.vocab_size]
        if count < 0 or probabilities.shape != (1, count, self.vocab_size):
            raise ValueError("CEE sampled proposal rows do not cover generated tokens")
        return _CeeProposal(
            tokens,
            probabilities.clone(),
            prefix.shape[1],
            entropy_source,
            elapsed,
        )

    def _cee_first_stage(
        self,
        proposal,
        little_cache,
        draft_cache,
        comm,
        *,
        uncertainty_sim,
        prob_payload_bits,
        charge_residual,
        proposal_top_k,
    ) -> Tuple[_CeeVerification, float]:
        """Decide opportunistic forwarding or correct little's proposal at draft.

        Skipping preserves little's actual proposal rows for outer verification;
        it never fabricates draft probability history. Normal correction carries
        raw verifier rows, without rebuilding them at a new policy top-k.
        """
        little_device = proposal.tokens.device
        if uncertainty_sim is not None and proposal.count > 0:
            logits = little_cache.logits_history
            if logits is None:
                raise ValueError(
                    "Little model logits history is required for opportunistic CEE-SD"
                )
            proposal_token = int(proposal.tokens[:, proposal.prefix_len].item())
            uncertainty = uncertainty_sim.calculate_uncertainty(
                logits[:, proposal.prefix_len - 1, : self.vocab_size],
                M=20,
                theta_max=2.0,
                draft_token=proposal_token,
            )
            should_transfer, _ = uncertainty_sim.determine_transfer_strategy(
                uncertainty,
                proposal.probabilities[:, 0, :],
                draft_token=proposal_token,
            )
            if not should_transfer:
                comm.simulate_transfer(8, "edge_end")
                comm.send_accept_message("edge_end")
                return _CeeVerification(
                    proposal.count,
                    proposal.tokens.shape[1] - 1,
                    torch.empty((1, 0), dtype=torch.long, device=little_device),
                    True,
                    proposal.probabilities,
                    proposal.probabilities,
                    True,
                ), 0.0
        started = time.time()
        draft_cache.forward_context(
            _move_token_tensor(proposal.tokens, draft_cache.device)
        )
        elapsed = time.time() - started
        wire_probs, wire_bits = _uplink_prob_payload(
            proposal.probabilities, prob_payload_bits
        )
        wire_probs = cast(torch.Tensor, wire_probs)
        if proposal.count > 0:
            tokens = proposal.tokens[:, proposal.prefix_len :]
            selected_wire = torch.gather(wire_probs, 2, tokens.unsqueeze(-1)).squeeze(
                -1
            )
            comm.transfer(tokens, selected_wire, "edge_end", prob_bits=wire_bits)
        result = self._cee_verify_proposal(
            proposal,
            little_cache,
            draft_cache,
            wire_probabilities=wire_probs,
            output_device=little_device,
        )
        if charge_residual and not result.all_accepted:
            comm.simulate_transfer(
                self._residual_payload_bytes(wire_probs, proposal_top_k), "edge_end"
            )
        return result, elapsed

    def _cee_target_proposal(
        self,
        first: _CeeVerification,
        draft: _CeeProposal,
        prefix_len: int,
    ) -> _CeeProposal:
        """Join the corrected/forwarded inner law with newly sampled draft rows.

        Prefix rows are raw draft verifier rows after normal correction, or
        little proposal rows on the skip path. They are NOT reinterpreted using
        the new top-k, and remain present when the draft extension is empty.
        """
        carried = first.carried_probabilities.to(draft.probabilities.device)
        if carried.shape != (1, draft.prefix_len - prefix_len, self.vocab_size):
            raise ValueError(
                "CEE forwarded probability rows do not cover the inner output"
            )
        probabilities = (
            torch.cat((carried, draft.probabilities), dim=1) if draft.count else carried
        )
        return _CeeProposal(
            draft.tokens,
            probabilities,
            prefix_len,
            draft.entropy_source,
            draft.computation_time,
        )

    def _cee_verify_proposal(
        self,
        proposal: _CeeProposal,
        proposer_cache,
        verifier_cache,
        *,
        wire_probabilities: torch.Tensor,
        output_device,
    ) -> _CeeVerification:
        """Verify a complete candidate window and settle its real cache endpoints.

        Wire probabilities are the SAME law for acceptance and residual sampling,
        independent of the residual *billing* flag. At <16 bits this is the
        explicitly lossy wire approximation, not the exact sampled proposal law.
        The bonus row has an absolute context index; sampled bonus/correction
        tokens themselves have not been forwarded through either cache.
        """
        tokens = proposal.tokens
        gamma = proposal.count
        history = verifier_cache.prob_history
        if (
            tokens.ndim != 2
            or tokens.shape[0] != 1
            or proposal.prefix_len < 1
            or gamma < 0
            or wire_probabilities.shape != (1, gamma, self.vocab_size)
        ):
            raise ValueError("CEE proposal token/probability window is inconsistent")
        if (
            history is None
            or history.ndim != 3
            or history.shape[0] != 1
            or history.shape[1] != tokens.shape[1]
            or history.shape[2] < self.vocab_size
        ):
            raise ValueError(
                "CEE verifier history must cover exactly the candidate "
                "and bonus context"
            )
        if (
            not torch.isfinite(wire_probabilities).all()
            or (wire_probabilities < 0).any()
        ):
            raise ValueError("CEE wire probabilities must be finite and nonnegative")
        rows = history[
            :, proposal.prefix_len - 1 : tokens.shape[1], : self.vocab_size
        ].to(output_device)
        if not torch.isfinite(rows).all() or (rows < 0).any():
            raise ValueError(
                "CEE verifier probabilities must be finite and nonnegative"
            )
        wire = wire_probabilities.to(output_device)
        candidate_tokens = tokens[:, proposal.prefix_len :].to(output_device)
        selected_q = torch.gather(wire, 2, candidate_tokens.unsqueeze(-1)).squeeze(-1)
        if (selected_q <= 0).any():
            raise ValueError("CEE wire proposal has non-positive selected support")
        selected_p = torch.gather(
            rows[:, :gamma, :], 2, candidate_tokens.unsqueeze(-1)
        ).squeeze(-1)
        draws = torch.rand((1, gamma), device=output_device)
        continuous = (
            (draws < selected_p / selected_q).to(torch.int8).cummin(dim=1).values
        )
        accepted = int(continuous.sum().item())
        accepted_end = proposal.prefix_len + accepted - 1
        all_accepted = accepted == gamma
        if all_accepted:
            token = sample_accept_token(rows[:, gamma, :], output_device=output_device)
        else:
            token = sample_reject_token(
                rows[:, accepted, :],
                wire[:, accepted, :],
                output_device=output_device,
            )
        carried = rows[:, : accepted + 1, :].clone()
        # Both caches retain only context that has really been forwarded. Cache
        # implementations clamp a proposer endpoint beyond its last sampled row.
        apply_rollback(
            proposer_cache,
            verifier_cache,
            _CeeRollbackPlan(
                draft_end_pos=accepted_end + 1,
                target_end_pos_reject=accepted_end + 1,
                target_end_pos_accept=accepted_end + 1,
                all_accepted=all_accepted,
            ),
        )
        return _CeeVerification(
            accepted,
            accepted_end,
            token,
            all_accepted,
            carried,
            wire_probabilities,
        )

    def _cee_finalize(
        self,
        prefix,
        prompt_len,
        max_tokens,
        stats: _CeeDecodeStats,
        comm,
        elapsed_time,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """Publish run aggregates after the final communication round is settled."""
        prefix, _ = self._stop_at_eos(prefix, prompt_len)
        prefix = prefix[:, :max_tokens]
        metrics = get_empty_metrics()
        for key in (
            "little_forward_times",
            "draft_forward_times",
            "target_forward_times",
            "little_computation_time",
            "draft_computation_time",
            "target_computation_time",
            "little_generated_tokens",
            "draft_generated_tokens",
            "little_accepted_tokens",
            "draft_accepted_tokens",
            "queuing_time",
            "dra_overhead_time",
        ):
            metrics[key] = getattr(stats, key)
        comm_time = comm.edge_cloud_comm_time + comm.edge_end_comm_time
        metrics["generated_tokens"] = prefix.shape[1] - prompt_len
        metrics["wall_time"] = elapsed_time + comm_time + stats.queuing_time
        metrics["throughput"] = (
            metrics["generated_tokens"] / metrics["wall_time"]
            if metrics["wall_time"] > 0
            else 0
        )
        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_time,
            queuing_time=stats.queuing_time,
            little_comp_time=stats.little_computation_time,
            draft_comp_time=stats.draft_computation_time,
            target_comp_time=stats.target_computation_time,
        )
        metrics["communication_time"] = comm_time
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm.edge_end_data
        metrics["cloud_end_data_bytes"] = comm.cloud_end_data
        metrics["comm_energy"] = comm.total_comm_energy
        metrics["connect_times"] = comm.connect_times
        metrics["arp_overhead_time"] = 0.0
        checkpoint_start = time.time()
        self._save_adaptive_rl_checkpoints(metrics["throughput"])
        metrics["dra_overhead_time"] += time.time() - checkpoint_start
        _finalize_comm_metrics(metrics, comm, self.args)
        return prefix, metrics

    def _adaptive_rejection_prob_bytes(
        self,
        *,
        prob_data: torch.Tensor,
        transfer_top_k: Optional[int],
    ) -> float:
        """两层共用的拒绝概率载荷字节；不包含索引、token 或拒绝报文开销。

        保留先按元素宽度/top-k 估算、再做原子量化并按位宽覆盖的顺序。
        量化张量仅局部用于计数、不回传；验证/上行载荷仍由调用方处理。
        调用方选择概率行并负责计费；这里不移动任何通信或 EOS 点位。
        """
        prob_bytes = prob_data.element_size() * prob_data.numel()
        if transfer_top_k is not None and transfer_top_k > 0:
            prob_bytes = transfer_top_k * prob_data.element_size()
        # B15：实际量化与位宽计费保持原子决策，不只改变计费数字。
        prob_data, _pb_bits = _uplink_prob_payload(
            prob_data, self.comm_config.prob_payload_bits
        )
        if _pb_bits is not None:
            _n = (
                transfer_top_k
                if (transfer_top_k is not None and transfer_top_k > 0)
                else prob_data.numel()
            )
            prob_bytes = int(_n) * _pb_bits / 8.0
        return prob_bytes

    def _select_adaptive_little_config(
        self,
        *,
        comm_simulator: CommunicationSimulator,
        little_entropy: float,
        gamma2_used: int,
        no_arp: bool,
    ) -> None:
        """生成后选择 little 动作；只应用阈值/下一轮 gamma，不应用 top-k。

        调用方保留 RL/static 门控与计时，机会门控在本次阈值写回之后运行。
        固定深度及通信口径不在此读取；训练开关仍随 select_config 原点读取。
        """
        bandwidth = comm_simulator.bandwidth_edge_end_mbps
        latency = comm_simulator.ntt_edge_end_ms
        _little_adapter = getattr(self, "small_draft_adapter", None)
        acc_probs = getattr(_little_adapter, "step_acc_probs", [])

        task_name = getattr(self, "task", "unknown")
        next_topk, next_threshold = self.little_rl_adapter.select_config(
            bandwidth,
            latency,
            acc_probs,
            little_entropy,
            task_name,
            training=not getattr(self.args, "disable_rl_update", False),
        )
        # 小模型层面的 top-k 压缩（如果需要）和 ARP 阈值
        # transfer_top_k = next_topk  # edge-end 通常不压缩
        if not no_arp and _little_adapter is not None:
            _little_adapter.threshold = next_threshold
        # 同主适配器：只有动作空间含 gamma 时才覆盖（本路径下第一级被
        # opportunistic 接管，gamma2 实际恒为 1，见诊断文档 F7）。
        if getattr(self.little_rl_adapter, "gamma_dim", 1) > 1:
            self._next_gamma2 = int(
                getattr(self.little_rl_adapter, "last_gamma", None) or gamma2_used
            )

    def _select_adaptive_main_config(
        self,
        *,
        comm_simulator: CommunicationSimulator,
        draft_entropy: float,
        gamma1_used: int,
        topk_cap: int,
        no_arp: bool,
    ) -> Optional[int]:
        """生成后选择主级动作：top-k 本轮结算用，gamma/ARP 留待后续生成。

        调用方持有 RL/static 门控与计时；这里不推进缓存、不做 token 采样或计费。
        仅训练开关 disable_rl_update 随 select_config 在原点读取；通信口径与
        固定深度留在调用方，勿在此新增这些读取（协议内省不跟随普通 helper）。
        """
        bandwidth = comm_simulator.bandwidth_edge_cloud_mbps
        latency = comm_simulator.ntt_edge_cloud_ms
        _main_adapter = getattr(self, "draft_target_adapter", None)
        acc_probs = getattr(_main_adapter, "step_acc_probs", [])

        # q 可能为 None（未开 top-k 压缩）；RL 的熵特征已在上面由
        # state_entropy(q, draft_model_cache) 取到，这里不需要 q。
        task_name = getattr(self, "task", "unknown")
        next_topk, next_threshold = self.rl_adapter.select_config(
            bandwidth,
            latency,
            acc_probs,
            draft_entropy,
            task_name,
            training=not getattr(self.args, "disable_rl_update", False),
        )
        # 更新 top-k 压缩参数和 ARP 阈值
        transfer_top_k = next_topk
        if topk_cap > 0 and transfer_top_k is not None:
            transfer_top_k = min(int(transfer_top_k), topk_cap)
        if not no_arp and _main_adapter is not None:
            _main_adapter.threshold = next_threshold
        # 只有动作空间真的包含 gamma 时才覆盖固定超参。legacy 适配器
        # （gamma_dim == 1）的 last_gamma 只是 gamma_candidates[0]（默认 2），
        # 无条件赋值会把 --gamma1 悄悄改掉 —— 这个 bug 曾让 A/B 的基线全部
        # 跑在 gamma=2 上，从而虚增了扩展动作空间的收益（诊断文档 F14）。
        if getattr(self.rl_adapter, "gamma_dim", 1) > 1:
            self._next_gamma1 = int(
                getattr(self.rl_adapter, "last_gamma", None) or gamma1_used
            )
        return transfer_top_k

    def _trace_adaptive_cache_preverify(
        self,
        *,
        x: torch.Tensor,
        prefix_len: int,
        draft_model_cache: KVCacheModel,
        target_model_cache: KVCacheModel,
        actual_gamma1: int,
    ) -> None:
        """CACHE_CHECK_TRACE 启用时追加验证前缓存长度诊断。"""
        # 缓存一致性自检（CACHE_CHECK_TRACE）：验证者 cache 里应当恰好有
        # prefix_len 个 token（= 已确认的上下文）。若实际长度与之不符，
        # 说明此前的 rollback 截断位置错了，后续前向会在错误的上下文上
        # 计算 logits —— 恒等检验里"模型把 prompt 又写一遍"的分歧就是
        # 这种失配的典型症状。
        _cache_trace = os.environ.get("CACHE_CHECK_TRACE")
        if _cache_trace:
            try:
                with open(_cache_trace, "a") as _fh:
                    _fh.write(
                        json.dumps(
                            {
                                "where": "pre_verify",
                                "x_len": int(x.shape[1]),
                                "prefix_len": int(prefix_len),
                                "target_cache_len": int(
                                    target_model_cache.current_length
                                ),
                                "draft_cache_len": int(
                                    draft_model_cache.current_length
                                ),
                                "actual_gamma1": int(actual_gamma1),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass

    def _trace_adaptive_calibration(
        self,
        *,
        x: torch.Tensor,
        prefix_len: int,
        draft_model_cache: KVCacheModel,
        target_model_cache: KVCacheModel,
        draft_accepted_this_iter: int,
    ) -> None:
        """按各自环境开关追加 ARP 校准及同索引的 top-k 分布诊断。"""
        # ARP_CALIB_TRACE: 逐位配对 (ARP预测, 真接受概率) —— 校准诊断专用。
        # 真值 = min(1, p_t(x_i)/p_d(x_i))（采样域）与 argmax 匹配（贪心域），
        # 两个都dump，分析端按温度取用。arp 行只取 draft_target 头（主级）。
        _calib_path = os.environ.get("ARP_CALIB_TRACE")
        _calib_adapter = getattr(self, "draft_target_adapter", None)
        _acc_probs_calib = list(
            getattr(_calib_adapter, "step_acc_probs", []) or []
        )
        if _calib_path and _acc_probs_calib:
            try:
                _g_calib = len(_acc_probs_calib)
                _toks_calib = x[0, prefix_len : prefix_len + _g_calib].tolist()
                _rows_calib = []
                for _i, _ap in enumerate(_acc_probs_calib):
                    _tok = int(_toks_calib[_i])
                    _row_d = draft_model_cache.prob_history[
                        0, prefix_len - 1 + _i
                    ]
                    _row_t = target_model_cache.prob_history[
                        0, prefix_len - 1 + _i
                    ]
                    _pd = float(_row_d[_tok])
                    _pt = float(_row_t[_tok])
                    _rows_calib.append(
                        json.dumps(
                            {
                                "arp": float(_ap),
                                "tok": _tok,
                                "pd": _pd,
                                "pt": _pt,
                                "ratio": (
                                    min(1.0, _pt / _pd) if _pd > 0 else 1.0
                                ),
                                "gmatch": int(
                                    int(_row_t.argmax().item()) == _tok
                                ),
                            }
                        )
                    )
                with open(_calib_path, "a") as _fh:
                    _fh.write("\n".join(_rows_calib) + "\n")
            except Exception:
                pass
        # TOPK_TRACE: 逐位 dump 草稿分布形状（排序质量/熵/目标token秩）——
        # top-k 集中度诊断用。与 ARP_CALIB_TRACE 同点位同索引, 独立 env 门控。
        _topk_path = os.environ.get("TOPK_TRACE")
        if _topk_path and _acc_probs_calib:
            try:
                _kg = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                _rows_tk = []
                for _i in range(len(_acc_probs_calib)):
                    _tok = int(x[0, prefix_len + _i].item())
                    # 优先取真实分布（logits→softmax）；prob_history 在该
                    # 路径可能已被重建/截断对象覆盖（传输侧视图），降级时标记。
                    _lh_tk = draft_model_cache.logits_history
                    if _lh_tk is not None:
                        _rd = torch.softmax(
                            _lh_tk[0, prefix_len - 1 + _i, : self.vocab_size]
                            .float(),
                            dim=-1,
                        )
                        _obj_tk = "true"
                    else:
                        _rd = draft_model_cache.prob_history[
                            0, prefix_len - 1 + _i
                        ]
                        _obj_tk = "rebuilt"
                    _rt = target_model_cache.prob_history[
                        0, prefix_len - 1 + _i
                    ]
                    _srt, _ord = _rd.sort(descending=True)
                    _cum = torch.cumsum(_srt, 0)[
                        [k - 1 for k in _kg if k <= _srt.numel()]
                    ]
                    _tok_rank = int((_rd > _rd[_tok]).sum().item()) + 1
                    _tam = int(_rt.argmax().item())
                    _tam_rank = int((_rd > _rd[_tam]).sum().item()) + 1
                    _ent = float(
                        -(_rd[_rd > 0] * _rd[_rd > 0].log()).sum().item()
                    )
                    _rows_tk.append(
                        json.dumps(
                            {
                                "k_grid": _kg[: len(_cum)],
                                "cum_mass": [round(float(v), 5) for v in _cum],
                                "tok": _tok,
                                "tok_rank": _tok_rank,
                                "tok_pd": float(_rd[_tok]),
                                "t_argmax": _tam,
                                "t_argmax_rank": _tam_rank,
                                "gmatch": int(_tam == _tok),
                                "acc": int(_i < draft_accepted_this_iter),
                                "obj": _obj_tk,
                                "entropy": round(_ent, 4),
                                "top20": [
                                    round(float(v), 5) for v in _srt[:20]
                                ],
                            }
                        )
                    )
                with open(_topk_path, "a") as _fh:
                    _fh.write("\n".join(_rows_tk) + "\n")
            except Exception:
                pass

    @Register.register_decoding("cee_sd_opportunistic")
    @torch.no_grad()
    def cee_sd_opportunistic(
        self,
        prefix: torch.Tensor,
        transfer_top_k: int | None = 300,
        use_precise_comm_sim: bool = False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 10,
        ntt_ms_edge_end: float = 1,
        use_early_stopping: bool = False,
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, DecodingMetrics]:
        return self.adaptive_tridecoding(
            prefix,
            transfer_top_k=transfer_top_k,
            use_precise_comm_sim=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            use_early_stopping=use_early_stopping,
            stop_sequences=stop_sequences,
            _opportunistic_first_stage=True,
            **kwargs,
        )
