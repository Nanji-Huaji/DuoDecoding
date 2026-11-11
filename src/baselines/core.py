import json
import os
import math
import time
import warnings
from typing import Callable, Dict as Dict, List, Optional, Tuple, cast

import torch
import transformers

from ..decoding_types import RollbackPlan as _CeeRollbackPlan
from ._cee_state import (
    CeeDecodeStats as _CeeDecodeStats,
    CeeProposal as _CeeProposal,
    CeeVerification as _CeeVerification,
)

from ..SpecDec_pp.specdec_pp.wrap_model import AcceptancePredictionHead

transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")

from ..adapter import DecodingAdapter
from ..comm_accounting import CommEnv
from ..communication import (
    CUHLM,
    CommunicationSimulator,
    PreciseCommunicationSimulator as PreciseCommunicationSimulator,
    PreciseCUHLM,
    PROB_QUANT_PASS_THROUGH_BITS,
    cuhlm_uplink_payload_bytes,
    tk_slt_uplink_payload_bytes,  # noqa: F401
)
from ..cee_prefix import CEETriPrefix
from ..checkpoint_provenance import (
    describe_dir,
    describe_path,
    record as record_checkpoint,
    summary_line,
)
from ..decoding_ops import (
    _prepare_legacy_verification_inputs,
    apply_rollback,
    build_rollback_plan,
    collect_verification_payload,
    compute_acceptance_result,
    compute_residual_distribution,
    materialize_acceptance,
    prepare_verification_inputs,
    reject_residual_payload_bytes,
    reject_tail_scalar_bytes,
    resolve_stage_verification,
    sample_accept_token,
    sample_reject_token,
    sample_reject_token_from_topk_proposal,
    verify_draft_sequence_result,
)
from ..acc_head_registry import anchor_repo_path, is_usable_acc_head_dir
from ..engine import Decoding
from ..graph_decode import (  # B19：CUDA Graph 接线的单一入口
    acquire_graph_caches,
    graph_mode_cache_kwargs as _graph_mode_cache_kwargs,
    kv_cache_construction_key as _kv_cache_construction_key,
)
from ..metrics import INT_SIZE, DecodingMetrics, get_empty_metrics
from ..model_gpu import KVCacheModel
from ..mode_features import MODE_FEATURES
from ..proposal_utils import (
    apply_transfer_top_k_cap,
    build_stage_prefix_topk_history,
    build_draft_probs_override,  # noqa: F401
    build_topk_proposal_history_step,
    concat_topk_proposal_history,
    merge_stage_topk_histories,
    proposal_top_k,
    stage_topk_proposal_history,
    stage_prob_history,
)
from ..register import Register
from ..rl_agent_registry import (
    ROLE_LITTLE,
    ROLE_MAIN,
    get_rl_agent_spec,
    resolve_legacy_rl_agent_load_path,
)
from ..rl_adapter import RLNetworkAdapter
from ..utils import (
    max_fn,
    norm_logits,
    rebuild_topk_uniform_probs,
    sample,
    skip_token_validation,
    state_entropy,
)


from ..baselines_helpers import (
    _add_comm_accounting_metrics,
    _add_per_model_wall_time,
    _compute_token_vocab_rank,
    _compute_transfer_topk_rank,
    _ensure_token_shape,
    _finalize_comm_metrics,
    _lambert_w_minus_one,  # noqa: F401
    _last_edge_cloud_tx_seconds,  # noqa: F401
    _move_token_tensor,
    _quantize_probs_fp16,  # noqa: F401
    _quantize_probs_logspace,
    _record_accepted_token_ranks as _record_accepted_token_ranks,
    _reward_comm_s,
    _rl_train_paths,
    _sample_token_from_probs,  # noqa: F401
    _send_downlink_index_only,
    _send_downlink_token,
    _simulate_topk_prob_transfer,
    _validate_token_range,
    compute_stage_reward,
    get_decoding_fn,
    load_acceptance_prediction_head,
    tk_slt_optimal_draft_length,  # noqa: F401
    tk_slt_select_speculative,  # noqa: F401
    tk_slt_speedup_ratio,  # noqa: F401
)


from ._adaptive_state import (
    AdaptiveDecodeStats as _AdaptiveDecodeStats,
    AdaptiveProposal as _AdaptiveProposal,
)

# 私有 re-export：实现已移到 simulators.py；保留 core 全局名，
# 让既有 src.baselines 导入和调用方的方法体继续解析到同一个函数。
from .simulators import (
    _make_comm_simulator,
    _make_symmetric_comm_simulator,
)


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


# A2：算法族实现已迁到真实定义模块；这里只做组合与有限兼容 re-export。
# import 别名保持 core 的星号导出面不变，避免把 mixin 名称变成新公共 API。
from .adaptive import AdaptiveDecodingMixin as _AdaptiveDecodingMixin
from .cee import CeeSdMixin as _CeeSdMixin
from .cee_distributed import CeeDistributedMixin as _CeeDistributedMixin
from .cuhlm import CuhlmMixin as _CuhlmMixin
from .dual import DualDecodingMixin as _DualDecodingMixin
from .operations import SharedOperations as _SharedOperations
from .operations import _uplink_prob_payload
from .runtime import RuntimeResources as _RuntimeResources
from .runtime import _build_cache
from .tk_slt import (
    _TK_SLT_ODLD_GAMMA_CAP,  # noqa: F401
    TkSltMixin as _TkSltMixin,
)


class Baselines(
    _RuntimeResources,
    _AdaptiveDecodingMixin,
    _CuhlmMixin,
    _TkSltMixin,
    _CeeSdMixin,
    _CeeDistributedMixin,
    _DualDecodingMixin,
    _SharedOperations,
    Decoding,
):
    """
    用于实验的方法。
    包含：
    - dssd
    - Uncertainty Decoding
    - dsd
    - Tridecoding
    """


    @staticmethod
    def _collect_dssd_uplink_payload(
        prob_history: torch.Tensor,
        x: torch.Tensor,
        prefix_len: int,
        gamma: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return collect_verification_payload(prob_history, x, prefix_len, gamma)


    def _select_cuhlm_stage_config(
        self,
        *,
        stage: str,
        transfer_top_k: Optional[int],
        uncertainty_threshold: float,
    ) -> tuple[Optional[int], float]:
        """
        Central hook for future dynamic policies.
        `cee_cuhlm` currently uses static CUHLM settings and leaves RL disabled.
        """
        default_threshold = getattr(
            self.args,
            "uncertainty_threshold",
            uncertainty_threshold,
        )
        stage_threshold = default_threshold
        if stage == "little_to_draft":
            stage_threshold = getattr(
                self.args,
                "small_draft_threshold",
                default_threshold,
            )
        elif stage == "draft_to_target":
            stage_threshold = getattr(
                self.args,
                "draft_target_threshold",
                default_threshold,
            )

        return transfer_top_k, float(stage_threshold)


    @Register.register_decoding("ceesd_w/o_arp")
    @Register.register_decoding("ceesd_without_arp")
    def ceesd_without_arp(
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
        """Legacy 实现（旧验证/回滚血缘），已移到 ``src/legacy/ceesd_without_arp.py``。

        仅用于复现历史 run；新的无 ARP 消融请用 ``cee_sd_no_arp``。
        """
        from ..legacy.ceesd_without_arp import run as _legacy_run

        # 兼容协议内省：实现已移到 legacy 模块，但 protocols.mode_consumption
        # 通过 co_names 判断本模式消费哪些计费/深度开关。这里保留只读引用，
        # 让声明与运行时观测一致（不改变行为）。
        _wiring = (
            self.comm_config.comm_round_trip_mode,
            self.comm_config.charge_residual_payload,
            self.comm_config.transfer_top_k_cap,
            self.args.gamma1,
            self.args.gamma2,
        )
        del _wiring

        return _legacy_run(
            self,
            prefix,
            transfer_top_k=transfer_top_k,
            use_precise_comm_sim=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            use_early_stopping=use_early_stopping,
            stop_sequences=stop_sequences,
            **kwargs,
        )


    @Register.register_decoding("target_only")
    @torch.no_grad()
    def target_only(
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
        """只用目标模型解码（无草稿、无投机、无 ARP、无通信）。

        这个模式承担两个不可替代的作用：
        1. **论文缺失的对照行**：加速比的分母只能是"不投机"的目标模型本身，
           而不是另一个投机配置（审稿人 R5 质疑的正是后者）。
        2. **无损性检验的基准**：在 temp=0（贪心）下，任何正确的投机解码
           都必须逐 token 复现目标模型的输出。把它与三级流水线的输出逐 token
           比对，就能把"加速是否以输出分布为代价"变成零噪声的二值判定
           （见 docs/rl_controller_diagnosis.md 的 E-A 实验）。
        """
        if prefix.dtype != torch.long:
            prefix = prefix.long()

        target_device = self.get_model_input_device(self.target_model)
        if use_precise_comm_sim or use_stochastic_comm:
            raise ValueError(
                "target_only has no communication stage; "
                "use_precise_comm_sim/use_stochastic_comm are not applicable."
            )

        # CUDA Graph **故意不接**（有据的口径决定，不是漏接）：
        # target_only 是两个不可替代作用的载体（见上面 docstring）——尤其是
        # "temp=0 下逐 token 复现目标模型"的无损性基准。图回放与 eager 有约 1%
        # 的 bf16 logits 漂移，实测会变成 40 样本里 1 个不同（docs/
        # graph_integration_status.md 旧 A/B）；且 13B 的图回放实测只快 1.03×
        # （13B 是 kernel/带宽瓶颈，不是 launch 瓶颈——同文件"对原调查结论的
        # 重要修正"）。用 1.03× 换掉基准的逐位可复现性不划算。
        # 因此 `--use_cuda_graph` 对本模式是**无操作**，这是设计而非遗漏。
        target_model_cache = KVCacheModel(
            self.target_model,
            self.args.temp,
            self.args.top_k,
            self.args.top_p,
        )
        target_model_cache.vocab_size = self.vocab_size

        max_new_tokens = int(self.args.max_tokens)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream=torch.cuda.current_stream())
        prompt_len = prefix.shape[1]
        generated = target_model_cache.generate(
            prefix.to(target_device), max_new_tokens
        )
        generated, _ = self._stop_at_eos(generated, prompt_len)
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        generated_tokens = generated.shape[1] - prefix.shape[1]
        metrics = get_empty_metrics()
        metrics["generated_tokens"] = generated_tokens
        metrics["target_forward_times"] = max(generated_tokens, 0)
        metrics["wall_time"] = elapsed_time
        metrics["throughput"] = (
            generated_tokens / elapsed_time if elapsed_time > 0 else 0.0
        )
        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=0.0,
            queuing_time=0.0,
        )
        metrics["communication_time"] = 0.0
        metrics["edge_cloud_data_bytes"] = 0
        metrics["comm_energy"] = 0.0

        return generated, metrics


    # Two-stage CUHLM variant using uncertainty-gated acceptance on both layers.


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
        if max_tokens is None:
            # 不放大预算：γ1+γ2+1 的放大会让本方法比基线多生成 token、
            # 高估吞吐（F33 契约）。越界由函数末尾的显式截断兜底。
            max_tokens = prefix.shape[1] + self.args.max_tokens
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream=torch.cuda.current_stream())

        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)
        little_device = self.get_model_input_device(self.little_model)

        # --- Communication Simulator ---
        if use_precise_comm_sim:
            # B45：channel_gain/noise_power_watt 为必填参数，此前缺省导致
            # cee_cuhlm 的精确仿真分支一进就 TypeError（对齐 1544 处
            # uncertainty_decoding 的完整调用）
            comm_simulator: CUHLM = PreciseCUHLM(
                min_bandwidth_mbps=self.comm_env.min_bandwidth_mbps,
                bw_model=self.comm_env.comm_bw_model,
                bandwidth_hz=self.comm_env.edge_cloud_bandwidth * 1e6,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                uncertainty_threshold=getattr(
                    self.args, "uncertainty_threshold", 0.8
                ),
                vocab_size=self.vocab_size,
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

        batch_delay = self.comm_env.batch_delay
        queuing_time = 0.0

        # 如果不使用精准通信模拟器，则应用推荐的初始阈值
        if not use_precise_comm_sim:
            _, uncertainty_threshold = self._select_cuhlm_stage_config(
                stage="shared",
                transfer_top_k=transfer_top_k,
                uncertainty_threshold=comm_simulator.uncertainty_threshold,
            )
            comm_simulator.uncertainty_threshold = uncertainty_threshold

        # --- KVCache Models ---
        # CUDA Graph：little/draft/target 都做多 token 验证前向（调度开销主体），
        # 三个都开图；图开启时挂 self 跨样本复用（与 adaptive_tridecoding 一致）。
        _graph_kw = _graph_mode_cache_kwargs(
            self.args,
            cap=int(getattr(self.args, "gamma1", 1))
            + int(getattr(self.args, "gamma2", 1))
            + 4,
        )
        _caches = acquire_graph_caches(
            self, "_cee_cuhlm_caches", _graph_kw,
            {
                "little": lambda: KVCacheModel(
                    self.little_model, self.args.temp, 0, 0, **_graph_kw,
                ),
                "draft": lambda: KVCacheModel(
                    self.draft_model, self.args.temp, 0, 0, **_graph_kw,
                ),
                "target": lambda: KVCacheModel(
                    self.target_model, self.args.temp, 0, 0, **_graph_kw,
                ),
            },
            cache_key={
                "little": _kv_cache_construction_key(
                    self.little_model, temperature=self.args.temp, top_k=0, top_p=0,
                    vocab_size=self.vocab_size, **_graph_kw,
                ),
                "draft": _kv_cache_construction_key(
                    self.draft_model, temperature=self.args.temp, top_k=0, top_p=0,
                    vocab_size=self.vocab_size, **_graph_kw,
                ),
                "target": _kv_cache_construction_key(
                    self.target_model, temperature=self.args.temp, top_k=0, top_p=0,
                    vocab_size=self.vocab_size, **_graph_kw,
                ),
            } if _graph_kw else None,
        )
        little_model_cache = _caches["little"]
        draft_model_cache, target_model_cache = _caches["draft"], _caches["target"]
        little_model_cache.vocab_size = self.vocab_size
        draft_model_cache.vocab_size = self.vocab_size
        target_model_cache.vocab_size = self.vocab_size

        # Metrics tracking
        little_model_forward_times = 0
        draft_model_forward_times = 0
        target_model_forward_times = 0
        little_comp_time = 0.0
        draft_comp_time = 0.0
        target_comp_time = 0.0
        total_little_model_generated_tokens = 0
        total_draft_model_generated_tokens = 0
        total_little_model_accepted_tokens = 0
        total_draft_model_accepted_tokens = 0
        wall_time = 0
        arp_overhead_time = 0.0
        dra_overhead_time = 0.0

        idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        start_event.record(stream=torch.cuda.current_stream())

        comm_simulator.transfer(prefix, None, "edge_end")  # 将 prompt 传输到 edge

        _tri_prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点

        while prefix.shape[1] < max_tokens:

            # B10：循环内 EOS 早停（与 dssd/adaptive_* 已接线方法语义对齐——
            # 计算与通信在 EOS 后立即停，时延口径跨方法可比）
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            idx += 1
            step_start_time = time.time()
            prefix_len = prefix.shape[1]
            little_transfer_top_k, little_uncertainty_threshold = (
                self._select_cuhlm_stage_config(
                    stage="little_to_draft",
                    transfer_top_k=transfer_top_k,
                    uncertainty_threshold=comm_simulator.uncertainty_threshold,
                )
            )
            current_proposal_top_k = proposal_top_k(little_transfer_top_k)
            little_stage_probs: Optional[torch.Tensor] = None
            draft_stage_probs: Optional[torch.Tensor] = None

            # Stage 1 speculative (Little -> Draft) with rejection sampling
            # Replaces CUHLM so Draft's distribution governs what enters Stage 2,
            # improving distribution consistency for the Stage 2 CUHLM check.
            t0 = time.time()
            x, little_rebuilt_probs, little_rebuilt_meta, _ = self._generate_with_optional_rebuilt_proposal(
                little_model_cache,
                _move_token_tensor(prefix, little_device),
                self.args.gamma2,
                current_proposal_top_k,
                adapter=None,
                need_topk_metadata=False,
            )
            little_comp_time += time.time() - t0

            actual_gamma2 = x.shape[1] - prefix_len

            t0 = time.time()
            _ = draft_model_cache.generate(_move_token_tensor(x, draft_device), 1)
            draft_comp_time += time.time() - t0

            little_model_forward_times += actual_gamma2
            draft_model_forward_times += 1
            total_little_model_generated_tokens += actual_gamma2

            n1: int = prefix_len + actual_gamma2 - 1
            little_accepted_this_iter = 0
            little_all_accepted = True

            if actual_gamma2 > 0:
                little_stage_probs = stage_prob_history(
                    little_model_cache, prefix_len, little_rebuilt_probs,
                )
                # F-CUHLM（论文口径）：Stage-1 上行 = 论文式 (5) 的逐位置
                # 压缩分布——每个起草位置 k₁·(b_prob+b_index) bits
                # （b_prob=8、b_index=⌈log₂V⌉）；k₁ 未配置时按全词表计
                # （论文 vanilla HLM 的上行）。token 索引 negligible
                # （§II-B）。旧口径：gamma2 × V×4B 整行 fp32 + token 字节。
                comm_simulator.simulate_transfer(
                    actual_gamma2
                    * cuhlm_uplink_payload_bytes(
                        current_proposal_top_k, self.vocab_size
                    ),
                    "edge_end",
                    topk=(
                        int(current_proposal_top_k)
                        if current_proposal_top_k
                        else 0
                    ),
                    draft_len=int(actual_gamma2),
                )

                # Standard rejection sampling: Draft verifies Little's tokens
                verification_inputs = _prepare_legacy_verification_inputs(
                    draft_model_cache=little_model_cache,
                    target_model_cache=draft_model_cache,
                    x=x, prefix_len=prefix_len, gamma=actual_gamma2,
                    draft_probs_override=little_stage_probs,
                )
                acceptance_result = compute_acceptance_result(verification_inputs)
                little_accepted_this_iter, n1, little_all_accepted = materialize_acceptance(
                    verification_inputs, acceptance_result
                )

                # F-CUHLM（论文口径）：accept/reject 判定不单独计费——验证
                # 结论随轮末响应回传（索引 negligible）；reject 在验证方
                # 基于已持有的分布重采样（论文式 17），无额外上行。
                # 旧口径：每接受 token 8B + accept 消息（6B+NTT）；reject
                # 再 8B + 二次计费的概率行 + reject 消息（6B+NTT）。
                if not little_all_accepted:
                    reject_pos = little_accepted_this_iter

                # Rollback and sample bonus/replacement token
                rollback_plan = build_rollback_plan(
                    prefix_len, verification_inputs.actual_gamma, n1
                )
                apply_rollback(little_model_cache, draft_model_cache, rollback_plan)

                if rollback_plan.all_accepted:
                    t = sample_accept_token(
                        draft_model_cache.prob_history[:, -1, : self.vocab_size],
                        output_device=little_device,
                    )
                else:
                    t = sample_reject_token(
                        verification_inputs.target_probs_batch[:, reject_pos, :],
                        verification_inputs.draft_probs_batch[:, reject_pos, :],
                        output_device=little_device,
                    )
            else:
                t = sample_accept_token(
                    draft_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=little_device,
                )

            total_little_model_accepted_tokens += little_accepted_this_iter

            assert n1 >= prefix_len - 1
            prefix = x[:, : n1 + 1]

            # F-CUHLM（论文口径）：Stage-1 轮末响应（edge→device）的 token
            # 索引 negligible（§II-B）——0 字节报文，保留一次 NTT。
            # 旧口径：INT_SIZE + token 字节 +（reject 时）二次计费的
            # prob_bytes + 魔数 reject_overhead=6B。
            _send_downlink_index_only(comm_simulator, "edge_end")

            prefix = torch.cat((prefix, t), dim=1)
            new_generated_token = prefix[:, prefix_len:]

            # 第二层 speculative (Draft -> Target) with CUHLM Uncertainty
            draft_transfer_top_k, draft_uncertainty_threshold = (
                self._select_cuhlm_stage_config(
                    stage="draft_to_target",
                    transfer_top_k=transfer_top_k,
                    uncertainty_threshold=comm_simulator.uncertainty_threshold,
                )
            )

            # F-CUHLM（论文口径）：首轮 prompt 上传保留（全仓一次性约定，
            # 论文不建模 prompt）；此后逐轮的 token 同步**不再单独上行**——
            # 新 token 索引随触发轮的上行捎带（negligible，§II-B），全跳过
            # 轮对云端零通信（旧口径：每轮一次 token 上行 = 每轮一个 NTT）。
            if idx == 1:
                comm_simulator.transfer(prefix, None, "edge_cloud")

            t0 = time.time()
            x, draft_rebuilt_probs, draft_rebuilt_meta, _ = self._generate_with_optional_rebuilt_proposal(
                draft_model_cache,
                _move_token_tensor(prefix, draft_device),
                self.args.gamma1,
                proposal_top_k(draft_transfer_top_k),
                adapter=None,
                need_topk_metadata=False,
            )
            draft_comp_time += time.time() - t0
            _validate_token_range(
                x,
                vocab_size=self.vocab_size,
                label="cee_cuhlm.draft_stage.proposal_x",
            )

            actual_gamma1 = x.shape[1] - prefix.shape[1]
            total_gamma = new_generated_token.shape[1] + actual_gamma1

            draft_model_forward_times += actual_gamma1
            total_draft_model_generated_tokens += total_gamma

            # Compute draft_stage_probs BEFORE target call (only needs draft_model_cache)
            draft_accepted_this_iter = 0
            draft_stage_probs = stage_prob_history(
                draft_model_cache,
                prefix_len + new_generated_token.shape[1],
                draft_rebuilt_probs,
            )

            # CUHLM Uncertainty Check Loop (BEFORE target call — as in the paper)
            original_threshold = comm_simulator.uncertainty_threshold
            comm_simulator.uncertainty_threshold = draft_uncertainty_threshold
            reject_offset: Optional[int] = None
            effective_gamma = min(
                total_gamma,
                draft_stage_probs.shape[1] - prefix_len,
            )
            n2 = prefix_len + effective_gamma - 1
            for i in range(effective_gamma):
                logit_idx = prefix_len + i - 1
                assert draft_model_cache.logits_history is not None, (
                    "Draft model logits history is None"
                )
                current_logit = draft_model_cache.logits_history[
                    :, logit_idx, : self.vocab_size
                ]
                current_token_id = int(x[:, prefix_len + i].item())

                uncertainty = comm_simulator.calculate_uncertainty(
                    current_logit, M=20, theta_max=2.0, draft_token=current_token_id
                )

                should_transfer, vocab_size = (
                    comm_simulator.determine_transfer_strategy(
                        uncertainty,
                        draft_stage_probs[:, prefix_len - 1 + i, : self.vocab_size],
                        draft_token=current_token_id,
                    )
                )

                if should_transfer:
                    # F-CUHLM（论文口径）：触发 = 一次上行，载荷 k(t)·(b_prob+
                    # b_index) bits（cuhlm_uplink_payload_bytes）；draft token
                    # 索引 negligible。旧口径：vocab_size×4+8 字节 + 独立
                    # reject 消息（6B + 一次 NTT）。
                    comm_simulator.simulate_transfer(
                        cuhlm_uplink_payload_bytes(int(vocab_size), self.vocab_size),
                        "edge_cloud",
                        topk=int(vocab_size),
                        draft_len=1,
                    )
                    reject_offset = i
                    n2 = prefix_len + i - 1
                    break
                else:
                    # F-CUHLM（论文口径）：跳过 = 零通信（旧口径：8B 上行 +
                    # accept 消息 6B+NTT，把论文的"跳过=免费"变成 2 RTT/token，
                    # 在 NTT 主导场景下系统性压死该基线）。
                    draft_accepted_this_iter += 1
            comm_simulator.uncertainty_threshold = original_threshold

            total_draft_model_accepted_tokens += draft_accepted_this_iter

            assert n2 >= prefix_len - 1
            prefix = x[:, : n2 + 1]

            if reject_offset is not None:
                # High uncertainty: run target model for verification
                queuing_time += batch_delay
                t0 = time.time()
                _ = target_model_cache.generate(
                    _move_token_tensor(x, target_device), 1
                )
                target_comp_time += time.time() - t0
                target_model_forward_times += 1

                verification_inputs = _prepare_legacy_verification_inputs(
                    draft_model_cache=draft_model_cache,
                    target_model_cache=target_model_cache,
                    x=x,
                    prefix_len=prefix_len,
                    gamma=total_gamma,
                    draft_probs_override=draft_stage_probs,
                )

                # F-CUHLM（论文口径）：触发分布已在上面的一次上行里计费
                # （k(t)·(b_prob+b_index)），这里不再二次计费。旧口径：
                # prob_bytes 按全行/压缩行再计一次 + reject_overhead 魔数 6B。
                new_generated_token = prefix[:, prefix_len:]

                n2, t, _ = _finalize_cuhlm_verification(
                    proposer_cache=draft_model_cache,
                    verifier_cache=target_model_cache,
                    verification_inputs=verification_inputs,
                    x=x,
                    prefix_len=prefix_len,
                    accepted_count=draft_accepted_this_iter,
                    reject_offset=reject_offset,
                    output_device=draft_device,
                )
            else:
                # All tokens pass CUHLM check: truly skip target model
                # target_model_forward_times NOT incremented (target genuinely not called)
                new_generated_token = prefix[:, prefix_len:]

                # Sample bonus token from draft model (target not available)
                bonus_probs = draft_stage_probs[:, n2, : self.vocab_size]
                t = sample_accept_token(bonus_probs, output_device=draft_device)

                # Rollback draft model to n2+1 for proper KVCache continuation
                draft_model_cache.rollback(n2 + 1)

            prefix = x[:, : n2 + 1]
            if n2 <= little_model_cache.current_length:
                little_model_cache.rollback(n2 + 1)

            prefix = torch.cat((prefix, t), dim=1)
            _validate_token_range(
                prefix,
                vocab_size=self.vocab_size,
                label="cee_cuhlm.edge_cloud.prefix_after_concat",
            )

            if reject_offset is not None:
                # F-CUHLM（论文口径）：触发轮的云端响应（cloud→edge）是
                # negligible 的 token 索引——0 字节报文，保留一次 NTT。
                # 全跳过轮对云端零通信（旧口径：无条件 INT_SIZE+token+
                # 二次计费的 prob_bytes + 魔数 6B，且每轮一次 NTT）。
                _send_downlink_index_only(comm_simulator, "edge_cloud")
            # 轮末响应回传设备（edge→end）：token 索引 negligible（§II-B），
            # 0 字节报文，保留一次 NTT（旧口径：INT_SIZE + token 字节）。
            _send_downlink_index_only(comm_simulator, "edge_end")

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        wall_time += elapsed_time
        # 遵守 max_tokens：投机解码按整块追加，最后一轮可能多出若干 token
        # （F33 契约）。参照 dist_spec 的 remaining-1 截断，这里显式截断——
        # 否则会多拿 token，使配对质量比较与时延/吞吐统计都不公平。
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, :max_tokens]
        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        wall_time += (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )

        metrics = get_empty_metrics()
        metrics["little_forward_times"] = little_model_forward_times
        metrics["draft_forward_times"] = draft_model_forward_times
        metrics["target_forward_times"] = target_model_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["little_generated_tokens"] = total_little_model_generated_tokens
        metrics["draft_generated_tokens"] = total_draft_model_generated_tokens
        metrics["little_accepted_tokens"] = total_little_model_accepted_tokens
        metrics["draft_accepted_tokens"] = total_draft_model_accepted_tokens
        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = wall_time + queuing_time
        metrics["throughput"] = (
            metrics["generated_tokens"] / metrics["wall_time"]
            if metrics["wall_time"] > 0
            else 0
        )
        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time,
            queuing_time=queuing_time,
            little_comp_time=little_comp_time,
            draft_comp_time=draft_comp_time,
            target_comp_time=target_comp_time,
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
        metrics["arp_overhead_time"] = arp_overhead_time
        metrics["dra_overhead_time"] = dra_overhead_time

        # 评测（--disable_rl_update）不得写 checkpoint；统一走带守卫的入口
        dra_start = time.time()
        self._save_adaptive_rl_checkpoints(metrics.get("throughput"))
        metrics["dra_overhead_time"] += time.time() - dra_start

        # 带宽/top-k/草稿历史 + 逐消息账 + 口径标签（与其余方法同一 helper）
        _finalize_comm_metrics(metrics, comm_simulator, self.args)

        return prefix, metrics
