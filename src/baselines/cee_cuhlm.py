"""CEE-CUHLM: three-tier speculative decoding under CU-HLM's outer protocol.

Standalone CU-HLM (:mod:`src.baselines.cuhlm`) is the protocol reference for the
outer stage.  CEE keeps its Little -> Draft speculation (``gamma2`` Little
proposals, ``gamma1`` fresh Draft prefetch) but **every** carried and fresh
candidate is gated by one independent CU-HLM round:

* **skip** finalizes the candidate token with no cloud message, no target
  forward and no queuing cost; the still-unconsumed prefetched suffix stays
  valid and is gated candidate by candidate.
* **trigger** bills one Eq. (5) uplink, forwards only the already finalized
  prefix with the target (no candidate forward, no sampled-and-discarded row),
  then accepts the candidate on ``u < P(d) / q(d)`` where ``q`` is the original
  proposal law that sampled it; on reject it samples one token from
  ``normalize((P - qhat)+)`` where ``qhat`` is the codec reconstruction of that
  same row, and finally bills one 0-byte cloud downlink.

A reject also discards the whole unconsumed suffix, rewinds Little/Draft to the
pre-candidate prefix endpoint and re-proposes from the residual-extended prefix.
There is no outer bonus token, no forced reject and no zero-residual fallback.

The inner Little -> Draft stage uses the strict multi-token kernel in
:mod:`src.verification`: Little's proposal is a ``ProposalWindow`` over the rows
the rebuilt-proposal generator actually sampled from, Draft verifies it with a
sample-free ``forward_context``, and the kernel's accepted tokens plus its one
necessary continuation token become the carried candidates.  The strict
accept/correct kernel's output marginal law is Draft's sampler law ``Q_D``, so
every carried candidate (including the continuation/correction) carries the
corresponding ``Q_D`` row, never a conditional residual.  Those rows and their
probe logits are cloned before any rollback or continuation forward, because
row ``r`` describes token ``r + 1`` and the fresh prefetch overwrites them.
Fresh candidates come from Draft's own filtered sampler rows under standalone
CU-HLM's rule (``transfer_top_k`` if positive else ``args.top_k``, with
``args.top_p``), never from a uniform-tail rebuild; the target keeps ``k=p=0``.

This remains the paper's intentionally lossy protocol (original-``q``
acceptance against reconstructed-``qhat`` residuals), carried on top of CEE's
extra edge cost, so it is not claimed to be target-distribution exact.
``small_draft_threshold`` stays inert (it is not an inner gate);
``uncertainty_threshold``/``draft_target_threshold`` drive the outer gate.  The
legacy ``_finalize_cuhlm_verification`` export is retained for compatibility
only and is no longer on any active path.
"""

import time
from dataclasses import dataclass
from typing import Optional, cast

import torch

from ..baselines_helpers import (
    _add_per_model_wall_time,
    _finalize_comm_metrics,
    _move_token_tensor,
    _send_downlink_index_only,
    _validate_token_range,
)
from ..communication import CUHLM, PreciseCUHLM, cuhlm_uplink_payload_bytes
from ..cuhlm_verification import (
    decide_cuhlm_verification,
    prepare_cuhlm_verification,
)
from ..decoding_ops import (
    apply_rollback,
    build_rollback_plan,
    sample_accept_token,
    sample_reject_token,
)
from ..graph_decode import (
    acquire_graph_caches,
)
from ..graph_decode import (
    graph_mode_cache_kwargs as _graph_mode_cache_kwargs,
)
from ..graph_decode import (
    kv_cache_construction_key as _kv_cache_construction_key,
)
from ..metrics import DecodingMetrics, get_empty_metrics
from ..model_gpu import KVCacheModel
from ..proposal_utils import proposal_top_k
from ..register import Register
from ..sampling import sample
from ..verification import (
    ProposalWindow,
    decide_verification,
    prepare_verification_window,
)


@dataclass
class _CeeRunStats:
    """Per-run loop counters; owns no cache, simulator, model or policy."""

    little_forward_times: int = 0
    draft_forward_times: int = 0
    target_forward_times: int = 0
    little_generated_tokens: int = 0
    draft_generated_tokens: int = 0
    little_accepted_tokens: int = 0
    draft_accepted_tokens: int = 0
    outer_rejections: int = 0
    pending_resync: int = 0
    little_comp_time: float = 0.0
    draft_comp_time: float = 0.0
    target_comp_time: float = 0.0
    queuing_time: float = 0.0
    arp_overhead_time: float = 0.0
    dra_overhead_time: float = 0.0


@dataclass(frozen=True, eq=False)
class _CeeBlock:
    """One stage's candidate tokens and the per-candidate laws/logits.

    ``tokens`` is ``(1, count)``, ``probabilities`` is ``(1, count, vocab)`` and
    ``logits`` is ``(1, count, vocab)``.  Row ``i`` belongs to ``tokens[:, i]``.
    Both carried and fresh rows represent the marginal Draft law Q_D; a carried
    correction does not carry its conditional residual law. Tokens stay on the
    producer device until materialized as windows on the finalized prefix device.
    """

    tokens: torch.Tensor
    probabilities: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True, eq=False)
class _CeeCandidate:
    """One outer-gate candidate: its single-token window and probe logits.

    ``proposal`` is built eagerly (and therefore validated) so a skipped
    candidate cannot hide an inconsistent law.  ``logits`` is the Draft's
    uncompressed row for the candidate's position and only the uncertainty
    probe reads it; ``token`` is cached as an int because the probe and the
    strategy call both need it.
    """

    proposal: ProposalWindow
    logits: torch.Tensor
    token: int


def _shrink_cache(cache: KVCacheModel, end_pos: int) -> None:
    """Roll a proposer cache back to ``end_pos``; never extend a shorter one."""
    if cache.current_length > end_pos:
        cache.rollback(end_pos)


def _finalize_cuhlm_verification(
    *,
    proposer_cache: KVCacheModel,
    verifier_cache: KVCacheModel,
    verification_inputs,
    x: torch.Tensor,
    prefix_len: int,
    accepted_count: int,
    reject_offset: Optional[int],
    output_device: torch.device,
) -> tuple[int, torch.Tensor, bool]:
    actual_gamma = verification_inputs.actual_gamma
    all_accepted = reject_offset is None
    if all_accepted:
        n = prefix_len + actual_gamma - 1
    else:
        n = prefix_len + cast(int, reject_offset) - 1

    rollback_plan = build_rollback_plan(prefix_len, actual_gamma, n)
    if rollback_plan.all_accepted:
        t = sample_accept_token(
            verifier_cache.prob_history[:, -1, : verifier_cache.vocab_size],
            output_device=output_device,
        )
    else:
        assert reject_offset is not None
        vocab_limit = min(proposer_cache.vocab_size, verifier_cache.vocab_size)
        t = sample_reject_token(
            verification_inputs.target_probs_batch[:, reject_offset, :vocab_limit],
            verification_inputs.draft_probs_batch[:, reject_offset, :vocab_limit],
            output_device=output_device,
        )

    apply_rollback(
        proposer_cache,
        verifier_cache,
        rollback_plan,
    )
    return n, t, rollback_plan.all_accepted


class CeeCuhlmMixin:
    """Own the aligned protocol and its three cache lifecycles."""

    @Register.register_decoding("cee_cuhlm")
    @torch.no_grad()
    def cee_cuhlm(
        self,
        prefix: torch.Tensor,
        max_tokens: int | None = None,
        # --- 传输控制 ---
        transfer_top_k: int | None = 300,
        use_precise_comm_sim: bool = True,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float | int = 0,
        ntt_ms_edge_end: float | int = 0,
        use_early_stopping: bool = False,
        stop_sequences: list[str] | None = None,
    ) -> tuple[torch.Tensor, DecodingMetrics]:
        # 深度键与往返口径必须在本方法体内直接读取：src.protocols 的运行时
        # 内省只看注册方法的 code object，不跟随私有 helper。
        gamma1 = int(getattr(self.args, "gamma1", 1))
        gamma2 = int(getattr(self.args, "gamma2", 1))
        if gamma1 < 0 or gamma2 < 0:
            raise ValueError(
                "cee_cuhlm 的投机深度必须非负："
                f"gamma1={gamma1}, gamma2={gamma2}"
            )
        round_trip_mode = self.comm_config.comm_round_trip_mode
        # 独立 CU-HLM 的草稿过滤规则：transfer_top_k>0 时用它，否则
        # args.top_k；top_p 恒为 args.top_p。Target 保持 k=p=0。
        draft_top_k = (
            int(transfer_top_k)
            if (transfer_top_k is not None and int(transfer_top_k) > 0)
            else int(self.args.top_k)
        )

        if max_tokens is None:
            # 不放大预算：每个外层候选只定稿一个 token，不会整块越界。
            max_tokens = prefix.shape[1] + self.args.max_tokens

        comm_simulator = self._build_cuhlm_comm_simulator(
            transfer_top_k=transfer_top_k,
            use_precise_comm_sim=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            round_trip_mode=round_trip_mode,
        )
        batch_delay = self.comm_env.batch_delay

        # 图容量对 gamma1/gamma2 的读取同样保留在本方法体内。
        little_cache, draft_cache, target_cache = self._acquire_cuhlm_caches(
            gamma1 + gamma2 + 4, draft_top_k
        )

        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)
        little_device = self.get_model_input_device(self.little_model)

        stats = _CeeRunStats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream=torch.cuda.current_stream())

        comm_simulator.transfer(prefix, None, "edge_end")  # 首轮 prompt 上传

        prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点
        finalized = prefix
        billing_round = 0
        stopped = False

        while finalized.shape[1] < max_tokens and not stopped:
            # B10：循环内 EOS 早停（与 dssd/adaptive_* 已接线方法语义对齐）
            finalized, eos_hit = self._stop_at_eos(finalized, prompt_len)
            if eos_hit:
                break
            _, outer_threshold = self._select_cuhlm_stage_config(
                stage="draft_to_target",
                transfer_top_k=transfer_top_k,
                uncertainty_threshold=comm_simulator.uncertainty_threshold,
            )

            # Stage 1: Little -> Draft 严格内层验证（含一个必要 continuation）
            billing_round += 1
            comm_simulator.set_round(billing_round)
            carried = self._cee_inner_stage(
                prefix=finalized,
                little_cache=little_cache,
                draft_cache=draft_cache,
                little_device=little_device,
                draft_device=draft_device,
                comm_simulator=comm_simulator,
                transfer_top_k=transfer_top_k,
                gamma2=gamma2,
                stats=stats,
            )
            # Stage-1 轮末响应（edge→device）：索引 negligible，0B 报文。
            _send_downlink_index_only(comm_simulator, "edge_end")
            comm_simulator.flush_round()

            block_prefix = torch.cat(
                (finalized, _move_token_tensor(carried.tokens, finalized.device)), dim=1
            )
            candidates = self._cee_block_candidates(
                prefix=finalized, block=carried
            )

            # Stage 2: fresh Draft 预取（独立 CU-HLM 的过滤后实际采样行）
            fresh = self._cee_fresh_stage(
                prefix=block_prefix,
                draft_cache=draft_cache,
                draft_device=draft_device,
                gamma1=gamma1,
                stats=stats,
            )
            if fresh is not None:
                candidates += self._cee_block_candidates(
                    prefix=block_prefix, block=fresh
                )

            rejected = False
            for candidate in candidates:
                if finalized.shape[1] >= max_tokens:
                    break
                # 每个外层候选独立划轮：同块多 trigger 不得合成一次 RTT。
                billing_round += 1
                comm_simulator.set_round(billing_round)
                if finalized.shape[1] == prompt_len:
                    # Match standalone CUHLM: upload the original prompt once,
                    # in the first candidate's round (even if that gate skips).
                    comm_simulator.transfer(prefix, None, "edge_cloud")
                finalized, rejected = self._cee_gate_candidate(
                    prefix=finalized,
                    candidate=candidate,
                    comm_simulator=comm_simulator,
                    target_cache=target_cache,
                    target_device=target_device,
                    threshold=outer_threshold,
                    batch_delay=batch_delay,
                    stats=stats,
                )
                comm_simulator.flush_round()
                finalized, eos_hit = self._stop_at_eos(finalized, prompt_len)
                if eos_hit:
                    stopped = True
                    break
                if use_early_stopping and self._check_stopping_criteria(
                    finalized, stop_sequences
                ):
                    stopped = True
                    break
                if rejected:
                    break

            # 轮末响应回传设备（edge→end）：索引 negligible，0B 报文。
            billing_round += 1
            comm_simulator.set_round(billing_round)
            _send_downlink_index_only(comm_simulator, "edge_end")
            comm_simulator.flush_round()

            if rejected:
                # 丢弃全部未消费 suffix：两位提议者回滚到拒绝前 prefix 长度；
                # 已定稿的残差 token 尚未前向。
                context_end = finalized.shape[1] - 1
                _shrink_cache(little_cache, context_end)
                _shrink_cache(draft_cache, context_end)

            _validate_token_range(
                finalized,
                vocab_size=self.vocab_size,
                label="cee_cuhlm.round.prefix",
            )

        # 最后一轮的字节可能仍在合并桶里，结算必须在读数之前。
        comm_simulator.flush_round()
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        # 只有定稿输出能触发 EOS 截断，被拒绝提议的 EOS 不算。
        finalized, _ = self._stop_at_eos(finalized, prompt_len)
        if finalized.shape[1] > prompt_len:
            # 最后一个 token 不再前向，投机后缀全部丢弃；短缓存不延长。
            context_end = finalized.shape[1] - 1
            for cache in (little_cache, draft_cache, target_cache):
                _shrink_cache(cache, context_end)

        generated_tokens = finalized.shape[1] - prompt_len
        wall_time = elapsed_time + (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )
        metrics = self._finalize_cuhlm_metrics(
            stats=stats,
            comm_simulator=comm_simulator,
            elapsed_time=elapsed_time,
            wall_time=wall_time,
            generated_tokens=generated_tokens,
        )

        return finalized, metrics

    def _build_cuhlm_comm_simulator(
        self,
        *,
        transfer_top_k: Optional[int],
        use_precise_comm_sim: bool,
        use_stochastic_comm: bool,
        ntt_ms_edge_cloud: float | int,
        ntt_ms_edge_end: float | int,
        round_trip_mode: str,
    ) -> CUHLM:
        """Construct the per-run CU-HLM billing simulator (precise or plain).

        The NTT edge delays are passed through explicitly: the simulator's own
        defaults (200/20 ms) must not silently apply to cee_cuhlm.  Unified
        round trips (docs/protocol.md §3.4) come from the run's
        ``comm_round_trip_mode``: the caller marks one round per outer
        candidate, so ``per_round`` bills exactly one cloud round trip per
        trigger and never merges two triggers of the same prefetch block.
        """
        if use_precise_comm_sim:
            # B45：channel_gain/noise_power_watt 为必填参数。
            comm_simulator: CUHLM = PreciseCUHLM(
                min_bandwidth_mbps=self.comm_env.min_bandwidth_mbps,
                bw_model=self.comm_env.comm_bw_model,
                bandwidth_hz=self.comm_env.edge_cloud_bandwidth * 1e6,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                uncertainty_threshold=getattr(self.args, "uncertainty_threshold", 0.8),
                vocab_size=self.vocab_size,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CUHLM(
                min_bandwidth_mbps=self.comm_env.min_bandwidth_mbps,
                bw_model=self.comm_env.comm_bw_model,
                bandwidth_edge_cloud=self.comm_env.edge_cloud_bandwidth,
                bandwidth_edge_end=self.comm_env.edge_end_bandwidth,
                bandwidth_cloud_end=self.comm_env.cloud_end_bandwidth,
                uncertainty_threshold=getattr(self.args, "uncertainty_threshold", 0.8),
                vocab_size=self.vocab_size,
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=self.comm_env.stochastic_ntt,
            )

        # 如果不使用精准通信模拟器，则应用推荐的初始阈值
        if not use_precise_comm_sim:
            _, uncertainty_threshold = self._select_cuhlm_stage_config(
                stage="shared",
                transfer_top_k=transfer_top_k,
                uncertainty_threshold=comm_simulator.uncertainty_threshold,
            )
            comm_simulator.uncertainty_threshold = uncertainty_threshold

        # 口径自述随 metrics 落盘（eval/utils.py 读 args.protocol_deviations），
        # 保证本轮协议版本不被误当成旧 CEE-CUHLM 结果比较。
        comm_simulator.coalesce_rounds = round_trip_mode == "per_round"
        _dev = (
            "cee_cuhlm_aligned_20261116: 每个 carried/fresh 候选独立走一次"
            "独立 CU-HLM 门（原 q 接受、重建 qhat 残差）；target=prefix-only"
            " 前向，无外层 bonus、无强制 reject；"
            "round_trip=unified(1xNTT/外层候选)"
        )
        _devs = list(getattr(self.args, "protocol_deviations", ()) or ())
        if _dev not in _devs:
            self.args.protocol_deviations = tuple(_devs + [_dev])
        return comm_simulator

    def _acquire_cuhlm_caches(
        self, graph_cap: int, draft_top_k: int
    ) -> tuple[KVCacheModel, KVCacheModel, KVCacheModel]:
        """Acquire the three caches with graph-reuse identity.

        Draft uses standalone CU-HLM's sampler configuration
        (``draft_top_k``/``args.top_p``), because fresh candidates must be drawn
        from that actual filtered law.  Little keeps the raw ``k=p=0`` cache so
        its rebuilt-proposal rows stay the true sampler law of the inner stage,
        and the target stays ``k=p=0`` (the paper's full-vocabulary P).
        """
        _graph_kw = _graph_mode_cache_kwargs(self.args, cap=graph_cap)
        draft_top_p = self.args.top_p
        _caches = acquire_graph_caches(
            self,
            "_cee_cuhlm_caches",
            _graph_kw,
            {
                "little": lambda: KVCacheModel(
                    self.little_model,
                    self.args.temp,
                    0,
                    0,
                    **_graph_kw,
                ),
                "draft": lambda: KVCacheModel(
                    self.draft_model,
                    self.args.temp,
                    draft_top_k,
                    draft_top_p,
                    **_graph_kw,
                ),
                "target": lambda: KVCacheModel(
                    self.target_model,
                    self.args.temp,
                    0,
                    0,
                    **_graph_kw,
                ),
            },
            cache_key={
                "little": _kv_cache_construction_key(
                    self.little_model,
                    temperature=self.args.temp,
                    top_k=0,
                    top_p=0,
                    vocab_size=self.vocab_size,
                    **_graph_kw,
                ),
                "draft": _kv_cache_construction_key(
                    self.draft_model,
                    temperature=self.args.temp,
                    top_k=draft_top_k,
                    top_p=draft_top_p,
                    vocab_size=self.vocab_size,
                    **_graph_kw,
                ),
                "target": _kv_cache_construction_key(
                    self.target_model,
                    temperature=self.args.temp,
                    top_k=0,
                    top_p=0,
                    vocab_size=self.vocab_size,
                    **_graph_kw,
                ),
            }
            if _graph_kw
            else None,
        )
        little_model_cache = _caches["little"]
        draft_model_cache, target_model_cache = _caches["draft"], _caches["target"]
        little_model_cache.vocab_size = self.vocab_size
        draft_model_cache.vocab_size = self.vocab_size
        target_model_cache.vocab_size = self.vocab_size
        return little_model_cache, draft_model_cache, target_model_cache

    def _cee_inner_stage(
        self,
        *,
        prefix: torch.Tensor,
        little_cache: KVCacheModel,
        draft_cache: KVCacheModel,
        little_device: torch.device,
        draft_device: torch.device,
        comm_simulator: CUHLM,
        transfer_top_k: Optional[int],
        gamma2: int,
        stats: _CeeRunStats,
    ) -> _CeeBlock:
        """Stage 1 (Little -> Draft): propose, bill the paper uplink, verify.

        Returns the carried candidates — the strictly accepted Little tokens
        plus the one necessary Draft continuation — with the ``Q_D`` row and
        probe logits that belong to each of them.
        """
        prefix_len = prefix.shape[1]
        little_transfer_top_k, _ = self._select_cuhlm_stage_config(
            stage="little_to_draft",
            transfer_top_k=transfer_top_k,
            uncertainty_threshold=comm_simulator.uncertainty_threshold,
        )
        # 内层阈值（small_draft_threshold）在本协议里不是 gate：内层严格
        # accept/correct 已由 Q_D 决定，这里只保留接口与 k₁。
        little_k = proposal_top_k(little_transfer_top_k)

        t0 = time.time()
        x, rebuilt, _, _ = self._generate_with_optional_rebuilt_proposal(
            little_cache,
            _move_token_tensor(prefix, little_device),
            gamma2,
            little_k,
            adapter=None,
            need_topk_metadata=False,
        )
        stats.little_comp_time += time.time() - t0
        actual_gamma2 = x.shape[1] - prefix_len
        _validate_token_range(
            x, vocab_size=self.vocab_size, label="cee_cuhlm.little_stage.proposal_x"
        )
        stats.little_forward_times += actual_gamma2
        stats.little_generated_tokens += actual_gamma2

        if actual_gamma2 > 0:
            # 论文式 (5) 的 Stage-1 上行：逐位置 k₁·(b_prob+b_index) bits。
            comm_simulator.simulate_transfer(
                actual_gamma2 * cuhlm_uplink_payload_bytes(little_k, self.vocab_size),
                "edge_end",
                topk=(int(little_k) if little_k else 0),
                draft_len=int(actual_gamma2),
            )

        # Little 的真实采样行：重建提议 generator 直接返回的就是它；未启用
        # 重建（k₁ 为空）时 cache 的过滤后行就是采样律本身。事后不得用 raw
        # 行替换。
        if actual_gamma2 <= 0:
            sampling_rows = x.new_empty(
                (1, 0, self.vocab_size), dtype=torch.float32
            )
        elif rebuilt is not None:
            sampling_rows = rebuilt.clone()
        else:
            sampling_rows = little_cache.prob_history[
                :, prefix_len - 1 : prefix_len + actual_gamma2 - 1, :
            ].clone()

        # Draft 只做上下文前向：不采样、不产生被丢弃的样本行。
        t0 = time.time()
        draft_cache.forward_context(_move_token_tensor(x, draft_device))
        stats.draft_comp_time += time.time() - t0
        stats.draft_forward_times += actual_gamma2

        proposal = ProposalWindow(
            tokens=x,
            probabilities=sampling_rows,
            prefix_len=prefix_len,
        )
        window = prepare_verification_window(
            proposal, draft_cache.prob_history[:, :, : self.vocab_size]
        )
        # 先验证整窗、再取一次 acceptance uniform（空窗不消耗 RNG）。
        uniforms = (
            torch.rand((1, proposal.count), device=sampling_rows.device)
            if proposal.count
            else sampling_rows.new_empty((1, 0))
        )
        decision = decide_verification(window, uniforms)
        accepted = decision.accepted_count
        stats.little_accepted_tokens += accepted

        continuation = sample(decision.continuation_probs)
        if continuation.dtype != torch.long:
            continuation = continuation.to(torch.long)
        continuation = continuation.to(x.device)

        # 内层严格 kernel 的输出边际律是 Q_D：carried correction 也带对应
        # Q_D row（不是条件残差）。row r 描述 token r+1，切片必须完整覆盖
        # 最后一个候选，并且在 rollback / fresh 续前向前 clone。
        carried_probs = window.target_rows[:, : accepted + 1, :].clone()
        carried_logits = draft_cache.logits_history[
            :, prefix_len - 1 : prefix_len + accepted, :
        ].clone()
        carried_tokens = torch.cat(
            (x[:, prefix_len : prefix_len + accepted], continuation), dim=1
        )

        # 已接受前缀之后的缓存行只属于被拒绝的候选：提议者不得越过已定稿
        # 前缀；continuation 尚未前向。短缓存不延长。
        _shrink_cache(little_cache, prefix_len + accepted)
        _shrink_cache(draft_cache, prefix_len + accepted)

        stats.draft_generated_tokens += 1  # 这个必要 continuation 由 Draft 采样
        return _CeeBlock(
            tokens=carried_tokens,
            probabilities=carried_probs,
            logits=carried_logits,
        )

    def _cee_fresh_stage(
        self,
        *,
        prefix: torch.Tensor,
        draft_cache: KVCacheModel,
        draft_device: torch.device,
        gamma1: int,
        stats: _CeeRunStats,
    ) -> Optional[_CeeBlock]:
        """Draft's fresh prefetch from its own filtered sampler rows."""
        if gamma1 <= 0:
            return None
        prefix_len = prefix.shape[1]
        t0 = time.time()
        x = draft_cache.generate(_move_token_tensor(prefix, draft_device), gamma1)
        stats.draft_comp_time += time.time() - t0
        _validate_token_range(
            x[:, prefix_len:],
            vocab_size=self.vocab_size,
            label="cee_cuhlm.fresh_stage.proposal_x",
        )
        stats.draft_forward_times += gamma1
        stats.draft_generated_tokens += gamma1
        return _CeeBlock(
            tokens=x[:, prefix_len:],
            probabilities=draft_cache.prob_history[
                :, prefix_len - 1 : prefix_len + gamma1 - 1, :
            ].clone(),
            logits=draft_cache.logits_history[
                :, prefix_len - 1 : prefix_len + gamma1 - 1, :
            ].clone(),
        )

    def _cee_block_candidates(
        self, *, prefix: torch.Tensor, block: _CeeBlock
    ) -> list[_CeeCandidate]:
        """Zip one block's tokens/laws into per-candidate single-token windows.

        Candidate ``i`` is gated over ``prefix + block.tokens[:i]``; earlier
        candidates of the same block finalize exactly one token each, so the
        eager window prefix is the prefix that candidate will actually see.
        """
        candidates: list[_CeeCandidate] = []
        tokens = _move_token_tensor(block.tokens, prefix.device)
        for index in range(tokens.shape[1]):
            context = torch.cat((prefix, tokens[:, :index]), dim=1)
            candidates.append(
                _CeeCandidate(
                    proposal=ProposalWindow(
                        tokens=torch.cat(
                            (context, tokens[:, index : index + 1]), dim=1
                        ),
                        probabilities=block.probabilities[:, index : index + 1, :],
                        prefix_len=context.shape[1],
                    ),
                    logits=block.logits[:, index, :],
                    token=int(tokens[0, index].item()),
                )
            )
        return candidates

    def _cee_gate_candidate(
        self,
        *,
        prefix: torch.Tensor,
        candidate: _CeeCandidate,
        comm_simulator: CUHLM,
        target_cache: KVCacheModel,
        target_device: torch.device,
        threshold: float,
        batch_delay: float,
        stats: _CeeRunStats,
    ) -> tuple[torch.Tensor, bool]:
        """Gate one candidate; return the extended prefix and whether it rejected.

        Skip is zero-communication.  Trigger is one standalone CU-HLM round:
        Eq. (5) uplink, prefix-only target forward, prepare-then-one-uniform
        decision, one residual sample on reject, then the 0-byte downlink.
        """
        proposal = candidate.proposal
        assert proposal.prefix_len == prefix.shape[1]
        token = proposal.tokens[:, proposal.prefix_len :].to(prefix.device)

        original_threshold = comm_simulator.uncertainty_threshold
        comm_simulator.uncertainty_threshold = threshold
        try:
            uncertainty = comm_simulator.calculate_uncertainty(
                candidate.logits, M=20, theta_max=2.0, draft_token=candidate.token
            )
            should_transfer, vocab_size = comm_simulator.determine_transfer_strategy(
                uncertainty,
                proposal.probabilities[0, 0, :],
                draft_token=candidate.token,
            )
        finally:
            comm_simulator.uncertainty_threshold = original_threshold

        if not should_transfer:
            # 机会跳过：不新增云端报文、Target 前向或排队；该 token 计入
            # 待重同步缓冲，下次触发随上行捎带（索引 negligible）。
            stats.draft_accepted_tokens += 1
            stats.pending_resync += 1
            return torch.cat((prefix, token), dim=1), False

        # 触发 = 一次合并上行（一次 NTT）；载荷按论文式 (5) 的压缩形式计。
        stats.queuing_time += batch_delay
        comm_simulator.simulate_transfer(
            cuhlm_uplink_payload_bytes(int(vocab_size), self.vocab_size),
            "edge_cloud",
            topk=int(vocab_size),
            draft_len=stats.pending_resync + 1,
        )
        stats.pending_resync = 0

        t0 = time.time()
        target_cache.forward_context(_move_token_tensor(prefix, target_device))
        stats.target_comp_time += time.time() - t0
        stats.target_forward_times += 1

        q = proposal.probabilities
        window = prepare_cuhlm_verification(
            proposal,
            target_cache.prob_history[:, :, : self.vocab_size],
            comm_simulator.compress_rebuild_probs(q, int(vocab_size)),
        )
        uniform = torch.rand((1, 1), device=q.device)
        decision = decide_cuhlm_verification(window, uniform)

        # 顺序是协议的一部分：接受用原 q；拒绝先采样残差，再计 0B 下行。
        residual = None
        if not decision.accepted:
            residual = sample(decision.residual_probs)
            if residual.dtype != torch.long:
                residual = residual.to(torch.long)
            residual = residual.to(prefix.device)
        _send_downlink_index_only(comm_simulator, "edge_cloud")
        if residual is None:
            stats.draft_accepted_tokens += 1
            return torch.cat((prefix, token), dim=1), False

        stats.outer_rejections += 1
        return torch.cat((prefix, residual), dim=1), True

    def _finalize_cuhlm_metrics(
        self,
        *,
        stats: _CeeRunStats,
        comm_simulator: CUHLM,
        elapsed_time: float,
        wall_time: float,
        generated_tokens: int,
    ) -> DecodingMetrics:
        """Assemble the public metrics surface from the shared run tallies."""
        metrics = get_empty_metrics()
        metrics["little_forward_times"] = stats.little_forward_times
        metrics["draft_forward_times"] = stats.draft_forward_times
        metrics["target_forward_times"] = stats.target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["little_generated_tokens"] = stats.little_generated_tokens
        metrics["draft_generated_tokens"] = stats.draft_generated_tokens
        metrics["little_accepted_tokens"] = stats.little_accepted_tokens
        # draft_accepted_tokens = 最终被 skip/accept 定稿的候选数；
        # generated_tokens 与它的差就是外层 reject 数（见 outer_reject_times，
        # 与 CUHLM 同口径的生成守恒，额外键不进 get_empty_metrics）。
        metrics["draft_accepted_tokens"] = stats.draft_accepted_tokens
        metrics["outer_reject_times"] = stats.outer_rejections
        metrics["queuing_time"] = stats.queuing_time
        metrics["wall_time"] = wall_time + stats.queuing_time
        metrics["throughput"] = (
            metrics["generated_tokens"] / metrics["wall_time"]
            if metrics["wall_time"] > 0
            else 0
        )
        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time
            + comm_simulator.edge_end_comm_time,
            queuing_time=stats.queuing_time,
            little_comp_time=stats.little_comp_time,
            draft_comp_time=stats.draft_comp_time,
            target_comp_time=stats.target_comp_time,
        )
        metrics["communication_time"] = (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times
        metrics["arp_overhead_time"] = stats.arp_overhead_time
        metrics["dra_overhead_time"] = stats.dra_overhead_time

        # 评测（--disable_rl_update）不得写 checkpoint；统一走带守卫的入口
        dra_start = time.time()
        self._save_adaptive_rl_checkpoints(metrics.get("throughput"))
        metrics["dra_overhead_time"] += time.time() - dra_start

        # 带宽/top-k/草稿历史 + 逐消息账 + 口径标签（与其余方法同一 helper）
        _finalize_comm_metrics(metrics, comm_simulator, self.args)

        return metrics
