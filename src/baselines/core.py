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
    tk_slt_uplink_payload_bytes,
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
    compute_legacy_tk_slt_fp16_acceptance_result as _compute_legacy_tk_slt_fp16_acceptance_result,
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
    build_draft_probs_override,
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
    _lambert_w_minus_one,
    _last_edge_cloud_tx_seconds,
    _move_token_tensor,
    _quantize_probs_fp16,
    _quantize_probs_logspace,
    _record_accepted_token_ranks as _record_accepted_token_ranks,
    _reward_comm_s,
    _rl_train_paths,
    _sample_token_from_probs,
    _send_downlink_index_only,
    _send_downlink_token,
    _simulate_topk_prob_transfer,
    _validate_token_range,
    compute_stage_reward,
    get_decoding_fn,
    load_acceptance_prediction_head,
    tk_slt_optimal_draft_length,
    tk_slt_select_speculative,
    tk_slt_speedup_ratio,
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


#: TK-SLT ODLD（--tk_slt_odld）的 γ* 上限：估计噪声下防病态值；
#: 论文 Table I 的最大 γ* 是 14（α=0.8、L=0.01），64 只在极端估计下触及。
_TK_SLT_ODLD_GAMMA_CAP = 64


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
from .cuhlm import CuhlmMixin as _CuhlmMixin
from .dual import DualDecodingMixin as _DualDecodingMixin
from .operations import SharedOperations as _SharedOperations
from .operations import _uplink_prob_payload
from .runtime import RuntimeResources as _RuntimeResources
from .runtime import _build_cache


class Baselines(
    _RuntimeResources,
    _AdaptiveDecodingMixin,
    _CuhlmMixin,
    _CeeSdMixin,
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


    @Register.register_decoding("tk_slt")
    @Register.register_decoding("tkslt")
    @torch.no_grad()
    def tk_slt(
        self,
        prefix,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim: bool = False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 200,
        ntt_ms_edge_end: float = 20,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """Top-K Sparse Logits Transmission（TK-SLT）基线。

        论文 "Communication-Efficient Collaborative LLM Inference via
        Distributed Speculative Decoding"（WCSP'25, Zheng & Yang）：DSD 框架
        下，端侧 SLM 只对 top-K logits 做 softmax——草稿分布 Y_i 的支撑集
        就是 top-K（§III Solution 1），采样自该稀疏分布；上行每个草稿位置
        只传 K 个概率值 + 词表索引，BS 用重建的稀疏 Q 做标准投机验证，
        拒绝时从 norm(max(0, P−Q)) 重采样（稀疏 Q ⇒ 非 top-K 位置拿到
        完整的 P，正是论文的残差定义）。

        通信口径 = 论文自身（与 CUHLM 系同一处理方式，见 docs/protocol.md
        §3.1/§3.3），不接仓库统一计费开关：
        - 上行：γ·K·b_prob bits，b_prob=16（FP16，§VI-B）。概率值**真实**
          量化到 fp16 再按行重归一化——验证判据与计费看到同一个 q̂
          （B15/B16 的"计费与数据一致"原则）；
        - 草稿 token 索引、下行响应（结果 token + 位置 j）按 §II-B 判为
          negligible：0 字节报文，链路模型仍按报文计一次 NTT；
        - K<=0/None ⇒ 整词表载荷 + 全量 softmax 提议（vanilla DSD，
          该论文自己的不压缩基线）。

        可选 --tk_slt_odld：按论文 §V Algorithm 1/2（ODLD/AS²）在线估计
        (α, b, c)（接受率、T_V/T_LLM、T_SLM/T_LLM 的运行均值）并逐轮选
        γ*；S*<=1 的轮次退回 standalone LLM（target 直出、无上行分布）。
        默认关闭（固定 γ，与 dsd/dssd/cuhlm 同口径可比）；首轮无估计，
        用 --gamma 兜底。
        """
        comm_simulator: CommunicationSimulator = _make_symmetric_comm_simulator(
            self.comm_env,
            precise=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
        )
        comm_simulator.transfer_top_k = transfer_top_k
        self.color_print(f"TK-SLT using top-K: {transfer_top_k}", 2)
        # 统一往返口径（2026-10-09，docs/protocol.md §3.4）：per_round 下
        # 一轮的上行载荷与 0B 下行合并成一次云请求往返（1×NTT）。
        # 此前逐报文各付 50ms（每轮 2×NTT），比 dsd/dssd/ours 每轮多付
        # 50ms 纯记账差异。载荷字节仍是论文口径（tk_slt_uplink_payload_bytes）。
        comm_simulator.coalesce_rounds = (
            self.comm_config.comm_round_trip_mode == "per_round"
        )

        # 口径自述随 metrics 落盘（F-CUHLM 同款），保证论文口径跑出的工件
        # 可辨识、不与仓库统一口径的结果混排。
        _dev = (
            "tk_slt_paper_accounting: uplink=gamma*K*b_prob(FP16 16b, "
            "real-quantized), indices negligible, "
            "reject-resample=server-side sparse fp16 Q; "
            "round_trip=unified(1xNTT/round since 2026-10-09)"
        )
        _devs = list(getattr(self.args, "protocol_deviations", ()) or ())
        if _dev not in _devs:
            self.args.protocol_deviations = tuple(_devs + [_dev])

        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        # K 同时定义采样稀疏性（softmax 只作用于 top-K logits）与传输载荷。
        # K<=0/None ⇒ 不压缩：全量 softmax 提议 + 整词表载荷（vanilla DSD）。
        topk_enabled = transfer_top_k is not None and int(transfer_top_k) > 0
        draft_sampling_top_k = int(transfer_top_k) if topk_enabled else 0
        topk_history_val = int(transfer_top_k) if topk_enabled else 0

        odld_enabled = bool(getattr(self.args, "tk_slt_odld", False))

        # CUDA Graph：草稿缓存走 γ 步循环是图收益点；target 整段前向不图化
        # （与 dist_spec 同一取舍）。ODLD 开启时 γ* 可超过 --gamma，cap 按
        # ODLD 上限取，保证验证图档位覆盖。
        default_gamma_cap = int(getattr(self.args, "gamma", 5))
        _odld_cap = _TK_SLT_ODLD_GAMMA_CAP if odld_enabled else default_gamma_cap
        _graph_kw = _graph_mode_cache_kwargs(self.args, cap=_odld_cap + 4)
        _caches = acquire_graph_caches(
            self, "_tk_slt_caches", _graph_kw,
            {
                "draft": lambda: KVCacheModel(
                    self.draft_model, self.args.temp, draft_sampling_top_k,
                    self.args.top_p, **_graph_kw,
                ),
                "target": lambda: KVCacheModel(self.target_model, self.args.temp, 0, 0),
            },
            cache_key={
                "draft": _kv_cache_construction_key(
                    self.draft_model, temperature=self.args.temp,
                    top_k=draft_sampling_top_k, top_p=self.args.top_p,
                    vocab_size=self.vocab_size, **_graph_kw,
                ),
                "target": _kv_cache_construction_key(
                    self.target_model, temperature=self.args.temp, top_k=0, top_p=0,
                    vocab_size=self.vocab_size,
                ),
            } if _graph_kw else None,
        )
        approx_model_cache, target_model_cache = _caches["draft"], _caches["target"]
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache.vocab_size = self.vocab_size

        draft_forward_times = 0
        target_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0
        queuing_time = 0
        batch_delay = self.comm_env.batch_delay

        # 追踪 top-k 和 draft length（B18：avg_top_k 只统计传输压缩 top-k，
        # 未压缩记 0，与 dsd/dssd/tridecoding 同名列口径一致）
        total_draft_steps = 0
        sum_draft_len = 0.0
        sum_top_k = 0.0

        # ODLD/AS² 的在线估计（论文式 (5)：b=T_V/T_LLM、c=T_SLM/T_LLM；
        # α=期望接受率）。运行均值跨轮更新；standalone 轮不产生新估计
        # （无草稿/验证），只有 DSD 轮喂入。
        alpha_num = 0.0  # Σ 接受 token 数
        alpha_den = 0.0  # Σ 草稿 token 数
        t_slm_seconds = 0.0  # Σ 草稿前向耗时
        t_slm_tokens = 0
        t_llm_seconds = 0.0  # Σ 验证前向耗时
        t_llm_runs = 0
        t_v_seconds = 0.0  # Σ 上行纯发射耗时（不含 NTT，同论文 T_V 口径）
        t_v_dists = 0  # Σ 已传分布数
        standalone_rounds = 0
        last_gamma_star = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        start_event.record(stream=torch.cuda.current_stream())

        idx: int = 0

        draft_comp_time = 0.0
        target_comp_time = 0.0

        _tri_prompt_len = prefix.shape[1]  # EOS 早停的生成段起点（B10）

        while prefix.shape[1] < max_tokens:
            # 循环内 EOS 早停（B10）：计算与通信在 EOS 后立即停，
            # 时延口径与 dsd/dssd/adaptive_* 对齐
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            idx += 1
            prefix_len = prefix.shape[1]
            # 轮号标注（离线重放的分组依据；per_round 下轮首 bucket 已空，
            # set_round 的防御性 flush 是 no-op，不影响轮末结算语义）
            comm_simulator.set_round(idx)
            prefix = _ensure_token_shape(prefix, label="tk_slt.prefix")
            _validate_token_range(
                prefix, vocab_size=self.vocab_size, label="tk_slt.prefix"
            )

            remaining_tokens = max_tokens - prefix_len
            if remaining_tokens <= 0:
                break

            if idx == 1:
                # 初始上下文上传（全仓一次性约定：论文不建模 prompt）
                comm_simulator.transfer(prefix, None, "edge_cloud")

            # AS²（论文 Algorithm 2）：有估计后才启用；S*<=1 的轮次退回
            # standalone LLM。首轮（无估计）用 --gamma。
            base_gamma = int(self.args.gamma)
            if odld_enabled and t_llm_runs >= 1 and alpha_den > 0:
                alpha_hat = alpha_num / alpha_den
                llm_per_run = t_llm_seconds / t_llm_runs
                b_hat = (
                    (t_v_seconds / t_v_dists) / llm_per_run if t_v_dists > 0 else 0.0
                )
                c_hat = (
                    (t_slm_seconds / t_slm_tokens) / llm_per_run
                    if t_slm_tokens > 0
                    else 0.0
                )
                use_dsd, gamma_star, _s_star = tk_slt_select_speculative(
                    alpha_hat, b_hat, c_hat, _TK_SLT_ODLD_GAMMA_CAP
                )
                last_gamma_star = gamma_star
                self.color_print(
                    f"ODLD: alpha={alpha_hat:.3f} b={b_hat:.3f} c={c_hat:.3f} "
                    f"-> gamma*={gamma_star} S*={_s_star:.3f}",
                    3,
                )
                if not use_dsd:
                    # standalone LLM 轮（AS² 分支）：target 直出 1 token。
                    # 无草稿、无上行分布；下行响应 negligible（0 字节报文）。
                    # 1-token 前向 ≠ 验证前向，不进 T_LLM 估计。
                    standalone_rounds += 1
                    queuing_time += batch_delay
                    t0 = time.time()
                    _ = target_model_cache.generate(
                        _move_token_tensor(prefix, target_device), 1
                    )
                    target_comp_time += time.time() - t0
                    target_forward_times += 1
                    if self.accelerator.is_main_process:
                        self.target_forward_times += 1

                    t = _sample_token_from_probs(
                        target_model_cache.prob_history[:, -1, : self.vocab_size],
                        output_device=prefix.device,
                        vocab_size=self.vocab_size,
                        label="tk_slt.standalone_t",
                    )
                    prefix = torch.cat((prefix, t), dim=1)
                    self.num_acc_tokens.append(1)

                    if use_early_stopping and self._check_stopping_criteria(
                        prefix, stop_sequences
                    ):
                        break
                    _send_downlink_index_only(comm_simulator, "edge_cloud")
                    continue
                base_gamma = gamma_star

            # 调整 gamma 以不超过剩余的 token 数（减 1 留给最后的采样 token）
            current_gamma = min(base_gamma, remaining_tokens - 1)
            if current_gamma <= 0:
                # 只剩 1 个 token：target 直出（与 dist_spec 的边界分支同口径，
                # 不计通信）
                queuing_time += batch_delay
                t0 = time.time()
                _ = target_model_cache.generate(
                    _move_token_tensor(prefix, target_device), 1
                )
                target_comp_time += time.time() - t0
                target_forward_times += 1
                if self.accelerator.is_main_process:
                    self.target_forward_times += 1

                t = _sample_token_from_probs(
                    target_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=prefix.device,
                    vocab_size=self.vocab_size,
                    label="tk_slt.fallback_t",
                )
                prefix = torch.cat((prefix, t), dim=1)
                self.num_acc_tokens.append(1)
                break

            # ---- Draft：top-K 稀疏采样（softmax 只作用于 top-K logits）----
            t0 = time.time()
            x = approx_model_cache.generate(
                _move_token_tensor(prefix, draft_device), current_gamma
            )
            draft_comp_time += time.time() - t0
            x = _ensure_token_shape(x, label="tk_slt.generated_x")
            _validate_token_range(
                x, vocab_size=self.vocab_size, label="tk_slt.generated_x"
            )
            draft_forward_times += current_gamma
            total_drafted_tokens += current_gamma
            t_slm_seconds += time.time() - t0
            t_slm_tokens += current_gamma

            total_draft_steps += 1
            sum_draft_len += current_gamma
            sum_top_k += topk_history_val

            # ---- 上行：K 个概率/草稿位置，FP16 量化后传输 ----
            # prob_history 行 = 稀疏分布（temp>0 时非零项 ≤ K；temp=0 时
            # one-hot）。窗口按"本轮输入前缀之后"切片，与计费的 γ 一致。
            window_end = min(
                prefix_len + current_gamma - 1, approx_model_cache.prob_history.shape[1]
            )
            draft_prob_window = approx_model_cache.prob_history[
                :, prefix_len - 1 : window_end, :
            ]
            window_rows = int(draft_prob_window.shape[1])
            quantized_window = _quantize_probs_fp16(draft_prob_window)

            comm_simulator.simulate_transfer(
                tk_slt_uplink_payload_bytes(
                    transfer_top_k, window_rows, self.vocab_size
                ),
                "edge_cloud",
                topk=topk_history_val,
                draft_len=current_gamma,
            )
            # ODLD 的 b̂ 估计读"本轮上行"的纯发射时长（TransferUnit.tx_time）。
            # per_transfer（直充）模式下上行报文此刻已在 stats 里；per_round
            # 合并模式下字节推迟到轮末 flush 才落账，改在 flush 后读。
            if not comm_simulator.coalesce_rounds:
                _tx_seconds = _last_edge_cloud_tx_seconds(comm_simulator)
                if _tx_seconds is not None:
                    t_v_seconds += _tx_seconds
                    t_v_dists += window_rows

            # ---- Verify：BS 用收到的稀疏 FP16 分布做标准投机验证 ----
            queuing_time += batch_delay
            t0 = time.time()
            _ = target_model_cache.generate(_move_token_tensor(x, target_device), 1)
            target_comp_time += time.time() - t0
            t_llm_seconds += time.time() - t0
            t_llm_runs += 1

            target_forward_times += 1
            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            verification_inputs = _prepare_legacy_verification_inputs(
                draft_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=current_gamma,
                draft_probs_override=build_draft_probs_override(
                    approx_model_cache,
                    prefix_len,
                    quantized_window,
                ),
            )
            # FP16 wire q may lose support present in the actual sampling law.
            acceptance_result = _compute_legacy_tk_slt_fp16_acceptance_result(
                verification_inputs
            )
            (
                this_step_accepted_tokens,
                n,
                _,
            ) = materialize_acceptance(verification_inputs, acceptance_result)
            total_accepted_tokens += this_step_accepted_tokens
            alpha_num += this_step_accepted_tokens
            alpha_den += verification_inputs.actual_gamma

            self.num_acc_tokens.append(this_step_accepted_tokens)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, : n + 1]
            rollback_plan = build_rollback_plan(
                prefix_len,
                verification_inputs.actual_gamma,
                n,
            )

            # 检查是否还有空间添加一个 token
            if prefix.shape[1] >= max_tokens:
                apply_rollback(
                    approx_model_cache,
                    target_model_cache,
                    rollback_plan,
                )
                break

            if not rollback_plan.all_accepted:
                # 拒绝路径：BS 基于已持有的稀疏 FP16 分布 Q̂ 从
                # norm(max(0, P−Q̂)) 重采样（论文 §II-A 3b；稀疏 Q ⇒
                # 非 top-K 位置拿到完整的 P）。无额外传输。
                rejection_offset = n - (prefix_len - 1)
                t = sample_reject_token(
                    verification_inputs.target_probs_batch[
                        :, rejection_offset, : self.vocab_size
                    ],
                    verification_inputs.draft_probs_batch[
                        :, rejection_offset, : self.vocab_size
                    ],
                    output_device=prefix.device,
                )
            else:
                # 全接受路径：从 P_{γ+1} 采样 bonus token
                t = sample_accept_token(
                    target_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=prefix.device,
                )
            t = _ensure_token_shape(t, label="tk_slt.sampled_t")
            _validate_token_range(
                t, vocab_size=self.vocab_size, label="tk_slt.sampled_t"
            )

            apply_rollback(
                approx_model_cache,
                target_model_cache,
                rollback_plan,
            )

            if prefix.shape[1] < max_tokens:
                prefix = torch.cat((prefix, t), dim=1)
                prefix = _ensure_token_shape(prefix, label="tk_slt.prefix_after_concat")
                _validate_token_range(
                    prefix,
                    vocab_size=self.vocab_size,
                    label="tk_slt.prefix_after_concat",
                )

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

            # ---- 下行：结果 token + 位置 j（§II-B negligible ⇒ 0 字节报文）----
            _send_downlink_index_only(comm_simulator, "edge_cloud")

            # ---- 轮末结算：per_round 下本轮字节在此落账（1×NTT/轮）----
            # per_transfer 模式下 flush_round 是 no-op（直充已落账）。
            comm_simulator.flush_round()
            if comm_simulator.coalesce_rounds:
                _tx_seconds = _last_edge_cloud_tx_seconds(comm_simulator)
                if _tx_seconds is not None:
                    t_v_seconds += _tx_seconds
                    t_v_dists += window_rows

        # 中途 break（max_tokens/EOS/fallback）时最后一轮字节可能仍在合并桶里，
        # 结算必须在 wall_time/communication_time 读数之前。
        comm_simulator.flush_round()
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        # 遵守 max_tokens（整块追加可能越界）+ EOS 截断必须在 metrics 结算前
        # （B9：被截掉的 token 不计入吞吐；最后一轮越过 EOS 的情况由这里兜底）
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, :max_tokens]
        prefix, _ = self._stop_at_eos(prefix, _tri_prompt_len)

        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        throughput = (
            generated_tokens / (elapsed_time + comm_simulator.edge_cloud_comm_time)
            if (elapsed_time + comm_simulator.edge_cloud_comm_time) > 0
            else 0
        )

        metrics = get_empty_metrics()
        metrics["avg_top_k"] = (
            sum_top_k / total_draft_steps if total_draft_steps > 0 else 0
        )
        metrics["avg_draft_len"] = (
            sum_draft_len / total_draft_steps if total_draft_steps > 0 else 0
        )
        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["draft_computation_time"] = draft_comp_time
        metrics["target_computation_time"] = target_comp_time
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = total_drafted_tokens
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["wall_time"] = elapsed_time + comm_simulator.edge_cloud_comm_time
        metrics["throughput"] = throughput
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times

        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = (
            elapsed_time + queuing_time + comm_simulator.edge_cloud_comm_time
        )
        if metrics["wall_time"] > 0:
            metrics["throughput"] = metrics["generated_tokens"] / metrics["wall_time"]

        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time,
            queuing_time=queuing_time,
            draft_comp_time=draft_comp_time,
            target_comp_time=target_comp_time,
        )

        # ODLD 观测：落盘最终估计与 standalone 轮数（分析用）
        metrics["tk_slt_odld"] = odld_enabled
        metrics["tk_slt_standalone_rounds"] = standalone_rounds
        if odld_enabled:
            metrics["tk_slt_alpha_hat"] = (
                alpha_num / alpha_den if alpha_den > 0 else 0.0
            )
            llm_per_run = t_llm_seconds / t_llm_runs if t_llm_runs > 0 else 0.0
            metrics["tk_slt_b_hat"] = (
                (t_v_seconds / t_v_dists) / llm_per_run
                if t_v_dists > 0 and llm_per_run > 0
                else 0.0
            )
            metrics["tk_slt_c_hat"] = (
                (t_slm_seconds / t_slm_tokens) / llm_per_run
                if t_slm_tokens > 0 and llm_per_run > 0
                else 0.0
            )
            metrics["tk_slt_gamma_star_final"] = last_gamma_star

        # 带宽/top-k/草稿历史 + 逐消息账 + 口径标签（与其余方法同一 helper）
        _finalize_comm_metrics(metrics, comm_simulator, self.args)

        return prefix, metrics


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
    def _cee_first_stage_dssd(
        self,
        prefix,
        *,
        prefix_ops,
        prefix_len: int,
        current_proposal_top_k,
        little_model_cache,
        draft_model_cache,
        comm_simulator,
        little_device,
        draft_device,
        transfer_top_k,
        charge_residual: bool,
        little_model_forward_times: int,
        draft_model_forward_times: int,
        total_little_model_generated_tokens: int,
        total_little_model_accepted_tokens: int,
    ):
        """CEE 第一层（little proposal + verify + reject）；与内联块逐语句等价。"""
        little_stage_probs: Optional[torch.Tensor] = None

        # --- Layer 1: Little -> Draft (Serial) ---
        x, little_rebuilt_probs, little_rebuilt_meta, _ = self._generate_with_optional_rebuilt_proposal(
            little_model_cache,
            prefix.to(little_device),
            self.args.gamma2,
            current_proposal_top_k,
            need_topk_metadata=True,
        )
        # Sync with Draft Model
        _ = draft_model_cache.generate(x.to(draft_device), 1)

        little_model_forward_times += self.args.gamma2
        draft_model_forward_times += 1
        total_little_model_generated_tokens += self.args.gamma2

        n1: int = prefix_len + self.args.gamma2 - 1

        if self.args.gamma2 > 0:
            little_stage_probs = stage_prob_history(
                little_model_cache,
                prefix_len,
                little_rebuilt_probs,
            )
            draft_tokens, draft_probs = collect_verification_payload(
                little_stage_probs,
                x,
                prefix_len,
                self.args.gamma2,
            )
            comm_simulator.transfer(draft_tokens, draft_probs, "edge_end")
            little_stage_kwargs = {
                "proposer_cache": little_model_cache,
                "verifier_cache": draft_model_cache,
                "x": x,
                "prefix_len": prefix_len,
                "gamma": self.args.gamma2,
                "output_device": little_device,
                "draft_probs_override": cast(torch.Tensor, little_stage_probs),
            }
            little_topk_history = stage_topk_proposal_history(
                little_rebuilt_meta,
                self.args.gamma2,
            )
            if little_topk_history is not None:
                little_stage_kwargs["draft_topk_history"] = little_topk_history
            (
                little_accepted_this_iter,
                n1,
                t,
                little_all_accepted,
            ) = resolve_stage_verification(**little_stage_kwargs)
            if not little_all_accepted:
                comm_simulator.send_reject_message("edge_end")
        else:
            little_accepted_this_iter = 0
            t = sample_accept_token(
                draft_model_cache.prob_history[:, -1, : self.vocab_size],
                output_device=little_device,
            )
            little_all_accepted = True

        total_little_model_accepted_tokens += little_accepted_this_iter

        assert n1 >= prefix_len - 1
        prefix = prefix_ops.assign(x, n1 + 1)

        if not little_all_accepted:
            # Reject, resample from Draft
            comm_simulator.transfer(
                None,
                cast(torch.Tensor, little_stage_probs)[:, n1, : self.vocab_size],
                "edge_end",
                transfer_top_k is not None and transfer_top_k > 0,
                transfer_top_k,
            )
            if charge_residual:
                # 统一口径（§3.4）：压缩行已含 k×(4+元素)，补尾部标量。
                comm_simulator.simulate_transfer(
                    reject_tail_scalar_bytes(
                        cast(torch.Tensor, little_stage_probs)[
                            :, n1, : self.vocab_size
                        ],
                        transfer_top_k,
                    ),
                    "edge_end",
                )

        # Transfer sampled token index back
        _send_downlink_token(comm_simulator, t, "edge_end")

        prefix = prefix_ops.append(t)

        # New tokens generated in Layer 1
        new_generated_token_layer1 = prefix[:, prefix_len:]
        return (
            prefix,
            new_generated_token_layer1,
            little_model_forward_times,
            draft_model_forward_times,
            total_little_model_generated_tokens,
            total_little_model_accepted_tokens,
        )

    def _cee_first_stage_dsd(
        self,
        prefix,
        *,
        prefix_ops,
        prefix_len: int,
        current_proposal_top_k,
        little_model_cache,
        draft_model_cache,
        comm_simulator,
        little_device,
        draft_device,
        transfer_top_k,
        charge_residual: bool,
        little_model_forward_times: int,
        draft_model_forward_times: int,
        total_little_model_generated_tokens: int,
        total_little_model_accepted_tokens: int,
    ):
        """CEE 第一层（little proposal + verify + reject）；与内联块逐语句等价。"""

        # --- Layer 1: Little -> Draft (Parallel) ---
        if current_proposal_top_k is not None and hasattr(
            little_model_cache, "generate_with_topk_metadata_only"
        ):
            x, little_topk_history = little_model_cache.generate_with_topk_metadata_only(
                prefix.to(little_device),
                self.args.gamma2,
                current_proposal_top_k,
            )
        else:
            x, little_rebuilt_probs = self._generate_with_optional_rebuilt_proposal(
                little_model_cache,
                prefix.to(little_device),
                self.args.gamma2,
                current_proposal_top_k,
                need_topk_metadata=False,
            )[:2]
            little_topk_history = None

        # Launch draft verification on GPU immediately (overlaps with CPU comm below)
        _ = draft_model_cache.generate(x.to(draft_device), 1)

        # Communication simulation (pure CPU): overlaps with draft.generate GPU above
        comm_simulator.transfer(x, None, "edge_end")

        little_model_forward_times += self.args.gamma2
        draft_model_forward_times += 1
        total_little_model_generated_tokens += self.args.gamma2

        n1: int = prefix_len + self.args.gamma2 - 1

        little_accepted_this_iter = 0
        if self.args.gamma2 > 0:
            l1_actual_gamma = min(self.args.gamma2, x.shape[1] - prefix_len)
            if l1_actual_gamma <= 0:
                little_accepted_this_iter = 0
                n1 = prefix_len - 1
                t = sample(
                    draft_model_cache.prob_history[
                        :, -1, : draft_model_cache.vocab_size
                    ]
                ).to(little_device)
                little_all_accepted = True
                little_model_cache.rollback(n1 + 1)
                draft_model_cache.rollback(n1 + 2)
            else:
                _simulate_topk_prob_transfer(
                    comm_simulator,
                    link_type="edge_end",
                    draft_len=l1_actual_gamma,
                    transfer_top_k=transfer_top_k,
                    prob_dtype=draft_model_cache.prob_history.dtype,
                )
                little_stage_kwargs = {
                    "proposer_cache": little_model_cache,
                    "verifier_cache": draft_model_cache,
                    "x": x,
                    "prefix_len": prefix_len,
                    "gamma": self.args.gamma2,
                    "output_device": little_device,
                }
                if little_topk_history is not None:
                    little_stage_kwargs["draft_topk_history"] = little_topk_history
                (
                    little_accepted_this_iter,
                    n1,
                    t,
                    little_all_accepted,
                ) = resolve_stage_verification(**little_stage_kwargs)
                if not little_all_accepted:
                    comm_simulator.send_reject_message("edge_end")
                    if charge_residual:
                        # 统一口径（§3.4）：legacy 只发 6B 拒绝信号，拒绝
                        # 位置的残差分布被无声省掉；honest 补
                        # k*(4+元素)+元素（行取验证方 draft 在 n1 的分布）。
                        comm_simulator.simulate_transfer(
                            reject_residual_payload_bytes(
                                draft_model_cache.prob_history[
                                    :, n1, : self.vocab_size
                                ],
                                transfer_top_k,
                            ),
                            "edge_end",
                        )
        else:
            t = sample_accept_token(
                draft_model_cache.prob_history[:, -1, : self.vocab_size],
                output_device=little_device,
            )
            little_all_accepted = True

        total_little_model_accepted_tokens += little_accepted_this_iter

        assert n1 >= prefix_len - 1
        n1_plus1 = n1 + 1
        prefix = prefix_ops.assign(x, n1_plus1)

        # Transfer sampled token index + token back (merged)
        _send_downlink_token(comm_simulator, t, "edge_end")

        prefix = prefix_ops.append(t)

        new_generated_token_layer1 = prefix[:, prefix_len:]
        return (
            prefix,
            new_generated_token_layer1,
            little_model_forward_times,
            draft_model_forward_times,
            total_little_model_generated_tokens,
            total_little_model_accepted_tokens,
        )

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

    def _cee_second_stage_dssd(
        self,
        prefix,
        *,
        prefix_ops,
        prefix_len: int,
        new_generated_token_layer1,
        current_proposal_top_k,
        draft_model_cache,
        target_model_cache,
        little_model_cache,
        comm_simulator,
        draft_device,
        target_device,
        transfer_top_k,
        charge_residual: bool,
        idx: int,
        batch_delay: float,
        draft_model_forward_times: int,
        target_model_forward_times: int,
        total_draft_model_generated_tokens: int,
        total_draft_model_accepted_tokens: int,
        queuing_time: float,
    ):
        """CEE 第二层（draft proposal + verify + reject + prefix 更新）；与内联块逐语句等价。"""
        # --- Layer 2: Draft -> Target (Serial) ---
        # Pre-launch GPU work (draft + target) to overlap with CPU comm simulation

        x, draft_rebuilt_probs, draft_rebuilt_meta, _ = self._generate_with_optional_rebuilt_proposal(
            draft_model_cache,
            prefix.to(draft_device),
            self.args.gamma1,
            current_proposal_top_k,
            need_topk_metadata=True,
        )

        queuing_time += batch_delay
        # Sync with Target Model (runs on GPU, overlaps with CPU comm below)
        _ = target_model_cache.generate(x.to(target_device), 1)

        # Communication simulation (pure CPU): overlaps with target GPU forward above
        if idx == 1:
            comm_simulator.transfer(prefix, None, "edge_cloud")
        else:
            comm_simulator.transfer(new_generated_token_layer1, None, "edge_cloud")

        draft_model_forward_times += self.args.gamma1
        target_model_forward_times += 1
        total_draft_model_generated_tokens += (
            new_generated_token_layer1.shape[1] + self.args.gamma1
        )

        n2: int = (
            prefix_len + new_generated_token_layer1.shape[1] + self.args.gamma1 - 1
        )
        draft_accepted_this_iter = 0

        # Iterate over both new tokens from Layer 1 and Draft speculation
        total_gamma_layer2 = new_generated_token_layer1.shape[1] + self.args.gamma1

        if total_gamma_layer2 > 0:
            draft_stage_probs = stage_prob_history(
                draft_model_cache,
                prefix_len + new_generated_token_layer1.shape[1],
                draft_rebuilt_probs,
            )
            draft_tokens_second, draft_probs_second = collect_verification_payload(
                draft_stage_probs,
                x,
                prefix_len,
                total_gamma_layer2,
            )
            comm_simulator.transfer(
                draft_tokens_second,
                draft_probs_second,
                "edge_cloud",
            )
            draft_stage_kwargs = {
                "proposer_cache": draft_model_cache,
                "verifier_cache": target_model_cache,
                "x": x,
                "prefix_len": prefix_len,
                "gamma": total_gamma_layer2,
                "output_device": draft_device,
                "draft_probs_override": cast(torch.Tensor, draft_stage_probs),
            }
            prefix_topk_history_dssd = None
            if new_generated_token_layer1.shape[1] > 0:
                prefix_prob_rows = draft_model_cache.prob_history[
                    :,
                    prefix_len - 1 : prefix_len - 1 + new_generated_token_layer1.shape[1],
                    :,
                ]
                prefix_topk_history_dssd = build_stage_prefix_topk_history(
                    prefix_prob_rows,
                    current_proposal_top_k,
                )
            draft_stage_topk_history = merge_stage_topk_histories(
                prefix_topk_history_dssd,
                stage_topk_proposal_history(
                    draft_rebuilt_meta,
                    self.args.gamma1,
                ),
            )
            draft_stage_topk_history = stage_topk_proposal_history(
                draft_stage_topk_history,
                total_gamma_layer2,
            )
            if draft_stage_topk_history is not None:
                draft_stage_kwargs["draft_topk_history"] = draft_stage_topk_history
            (
                draft_accepted_this_iter,
                n2,
                t,
                draft_all_accepted,
            ) = resolve_stage_verification(**draft_stage_kwargs)
            if not draft_all_accepted:
                comm_simulator.send_reject_message("edge_cloud")
        else:
            draft_accepted_this_iter = 0
            t = sample_accept_token(
                target_model_cache.prob_history[:, -1, : self.vocab_size],
                output_device=draft_device,
            )
            draft_all_accepted = True

        total_draft_model_accepted_tokens += draft_accepted_this_iter

        assert n2 >= prefix_len - 1
        prefix = prefix_ops.assign(x, n2 + 1)
        if n2 <= little_model_cache.current_length:
            little_model_cache.rollback(n2 + 1)

        if not draft_all_accepted:
            # Reject, resample from Target
            comm_simulator.transfer(
                None,
                cast(torch.Tensor, draft_stage_probs)[:, n2, : self.vocab_size],
                "edge_cloud",
                transfer_top_k is not None and transfer_top_k > 0,
                transfer_top_k,
            )
            if charge_residual:
                # 统一口径（§3.4）：同阶段一，补尾部标量。
                comm_simulator.simulate_transfer(
                    reject_tail_scalar_bytes(
                        cast(torch.Tensor, draft_stage_probs)[
                            :, n2, : self.vocab_size
                        ],
                        transfer_top_k,
                    ),
                    "edge_cloud",
                )

        prefix = prefix_ops.append(t)

        # Transfer index back
        _send_downlink_token(comm_simulator, t, "edge_cloud")
        _send_downlink_token(comm_simulator, t, "edge_end")

        return (
            prefix,
            draft_model_forward_times,
            target_model_forward_times,
            total_draft_model_generated_tokens,
            total_draft_model_accepted_tokens,
            queuing_time,
        )

    def _cee_second_stage_dsd(
        self,
        prefix,
        *,
        prefix_ops,
        prefix_len: int,
        new_generated_token_layer1,
        current_proposal_top_k,
        draft_model_cache,
        target_model_cache,
        little_model_cache,
        comm_simulator,
        draft_device,
        target_device,
        transfer_top_k,
        charge_residual: bool,
        idx: int,
        batch_delay: float,
        draft_model_forward_times: int,
        target_model_forward_times: int,
        total_draft_model_generated_tokens: int,
        total_draft_model_accepted_tokens: int,
        queuing_time: float,
    ):
        """CEE 第二层（draft proposal + verify + reject + prefix 更新）；与内联块逐语句等价。"""
        # --- Layer 2: Draft -> Target (Parallel) ---
        # Pre-launch GPU work (draft) to overlap with CPU comm simulation

        if current_proposal_top_k is not None and hasattr(
            draft_model_cache, "generate_with_topk_metadata_only"
        ):
            x, draft_topk_history = draft_model_cache.generate_with_topk_metadata_only(
                prefix.to(draft_device),
                self.args.gamma1,
                current_proposal_top_k,
            )
        else:
            x, draft_rebuilt_probs = self._generate_with_optional_rebuilt_proposal(
                draft_model_cache,
                prefix.to(draft_device),
                self.args.gamma1,
                current_proposal_top_k,
                need_topk_metadata=False,
            )[:2]
            draft_topk_history = None

        # Transfer generated tokens by draft model (CPU: overlaps with GPU above)
        speculated_tokens = x[:, -self.args.gamma1 :]

        # Communication simulation (pure CPU): overlaps with draft.generate GPU above
        if idx == 1:
            comm_simulator.transfer(prefix, None, "edge_cloud")
        else:
            comm_simulator.transfer(new_generated_token_layer1, None, "edge_cloud")
        comm_simulator.transfer(speculated_tokens, None, "edge_cloud")

        queuing_time += batch_delay
        # Sync with Target Model (runs on GPU while CPU does verification prep below)
        _ = target_model_cache.generate(x.to(target_device), 1)

        draft_model_forward_times += self.args.gamma1
        target_model_forward_times += 1
        total_draft_model_generated_tokens += (
            new_generated_token_layer1.shape[1] + self.args.gamma1
        )

        total_gamma_layer2 = new_generated_token_layer1.shape[1] + self.args.gamma1
        n2: int = prefix_len + total_gamma_layer2 - 1

        draft_accepted_this_iter = 0

        if total_gamma_layer2 > 0:
            l2_actual_gamma = min(total_gamma_layer2, x.shape[1] - prefix_len)
            if l2_actual_gamma <= 0:
                draft_accepted_this_iter = 0
                n2 = prefix_len - 1
                t = sample(
                    target_model_cache.prob_history[
                        :, -1, : target_model_cache.vocab_size
                    ]
                ).to(draft_device)
                draft_all_accepted = True
                draft_model_cache.rollback(n2 + 1)
                target_model_cache.rollback(n2 + 2)
            else:
                _simulate_topk_prob_transfer(
                    comm_simulator,
                    link_type="edge_cloud",
                    draft_len=l2_actual_gamma,
                    transfer_top_k=transfer_top_k,
                    prob_dtype=target_model_cache.prob_history.dtype,
                )
                prefix_topk_history = None
                if new_generated_token_layer1.shape[1] > 0:
                    prefix_prob_rows = draft_model_cache.prob_history[
                        :,
                        prefix_len - 1 : prefix_len - 1 + new_generated_token_layer1.shape[1],
                        :,
                    ]
                    prefix_topk_history = build_stage_prefix_topk_history(
                        prefix_prob_rows,
                        current_proposal_top_k,
                    )
                draft_stage_topk_history = merge_stage_topk_histories(
                    prefix_topk_history,
                    draft_topk_history,
                )
                draft_stage_topk_history = stage_topk_proposal_history(
                    draft_stage_topk_history,
                    total_gamma_layer2,
                )
                draft_stage_kwargs = {
                    "proposer_cache": draft_model_cache,
                    "verifier_cache": target_model_cache,
                    "x": x,
                    "prefix_len": prefix_len,
                    "gamma": total_gamma_layer2,
                    "output_device": draft_device,
                }
                if draft_stage_topk_history is not None:
                    draft_stage_kwargs["draft_topk_history"] = draft_stage_topk_history
                (
                    draft_accepted_this_iter,
                    n2,
                    t,
                    draft_all_accepted,
                ) = resolve_stage_verification(**draft_stage_kwargs)
                if not draft_all_accepted:
                    comm_simulator.send_reject_message("edge_cloud")
                    if charge_residual:
                        # 统一口径（§3.4）：同阶段一，残差取验证方 target
                        # 在 n2 的分布。
                        comm_simulator.simulate_transfer(
                            reject_residual_payload_bytes(
                                target_model_cache.prob_history[
                                    :, n2, : self.vocab_size
                                ],
                                transfer_top_k,
                            ),
                            "edge_cloud",
                        )
        else:
            t = sample_accept_token(
                target_model_cache.prob_history[:, -1, : self.vocab_size],
                output_device=draft_device,
            )
            draft_all_accepted = True

        total_draft_model_accepted_tokens += draft_accepted_this_iter

        assert n2 >= prefix_len - 1
        n2_plus1 = n2 + 1
        prefix = prefix_ops.assign(x, n2_plus1)
        if n2 <= little_model_cache.current_length:
            little_model_cache.rollback(n2_plus1)

        prefix = prefix_ops.append(t)

        # Transfer index + token back to both links (merged)
        _send_downlink_token(comm_simulator, t, "edge_cloud")
        _send_downlink_token(comm_simulator, t, "edge_end")

        return (
            prefix,
            draft_model_forward_times,
            target_model_forward_times,
            total_draft_model_generated_tokens,
            total_draft_model_accepted_tokens,
            queuing_time,
        )

    @Register.register_decoding("cee_dssd")
    @torch.no_grad()
    def cee_dssd(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud=10,
        ntt_ms_edge_end=1,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """
        CEE-DSSD: Combined Edge-End Distributed Split Speculative Decoding (3-layer).
        Uses serial verification similar to DSSD but in a 3-layer architecture.
        """
        max_tokens = prefix.shape[1] + self.args.max_tokens
        little_device = self.get_model_input_device(self.little_model)
        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        # 使用 transfer_top_k 作为草稿模型的 top-k 压缩参数
        draft_top_k = (
            transfer_top_k
            if (transfer_top_k is not None and transfer_top_k > 0)
            else self.args.top_k
        )

        # B19：little/draft/target 都做多 token 验证前向 ⇒ 三个都开图并跨样本
        # 复用；此前只给 little/draft 单步图、无验证档位、且每样本重捕获。
        graph_kw = _graph_mode_cache_kwargs(
            self.args,
            cap=int(getattr(self.args, "gamma1", 1))
            + int(getattr(self.args, "gamma2", 1))
            + 4,
        )
        caches = self._acquire_three_layer_caches(
            "_cee_dssd_caches", graph_kw, draft_top_k
        )
        little_model_cache = caches["little"]
        draft_model_cache = caches["draft"]
        target_model_cache = caches["target"]

        comm_simulator: CommunicationSimulator = _make_comm_simulator(
            self.comm_env,
            precise=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            edge_end=self.comm_env.edge_end_bandwidth,
            cloud_end=self.comm_env.cloud_end_bandwidth,
            transfer_top_k=transfer_top_k,
        )

        # 统一计费口径（docs/protocol.md §3）：与 adaptive_tridecoding 同一套
        # 开关。CUHLM 系（uncertainty_decoding / cee_cuhlm）按决策暂不接。
        # 口径在 self.comm_config 构造一次；属性名与开关名逐字一致，
        # protocols.py 的运行时内省才看得见（docs/comm_call_sites.md §4.1）。
        _comm = self.comm_config
        comm_simulator.coalesce_rounds = _comm.comm_round_trip_mode == "per_round"
        charge_residual = _comm.charge_residual_payload
        transfer_top_k = apply_transfer_top_k_cap(
            transfer_top_k, _comm.transfer_top_k_cap
        )
        comm_simulator.transfer_top_k = transfer_top_k

        # Metrics tracking
        little_model_forward_times = 0
        draft_model_forward_times = 0
        target_model_forward_times = 0
        total_little_model_generated_tokens = 0
        total_draft_model_generated_tokens = 0
        total_little_model_accepted_tokens = 0
        total_draft_model_accepted_tokens = 0
        queuing_time = 0
        batch_delay = self.comm_env.batch_delay
        wall_time = 0

        idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()
        prefix_ops = CEETriPrefix(prefix)
        prefix = prefix_ops.tensor

        start_event.record(stream=torch.cuda.current_stream())

        comm_simulator.transfer(prefix, None, "edge_end")

        _tri_prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点

        while prefix.shape[1] < max_tokens:

            # B10：循环内 EOS 早停（与 dssd/adaptive_* 已接线方法语义对齐——
            # 计算与通信在 EOS 后立即停，时延口径跨方法可比）
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            idx += 1
            comm_simulator.set_round(idx)
            prefix_len = prefix.shape[1]
            current_proposal_top_k = proposal_top_k(transfer_top_k)
            draft_stage_probs: Optional[torch.Tensor] = None
            (
                prefix,
                new_generated_token_layer1,
                little_model_forward_times,
                draft_model_forward_times,
                total_little_model_generated_tokens,
                total_little_model_accepted_tokens,
            ) = self._cee_first_stage_dssd(
                prefix,
                prefix_ops=prefix_ops,
                prefix_len=prefix_len,
                current_proposal_top_k=current_proposal_top_k,
                little_model_cache=little_model_cache,
                draft_model_cache=draft_model_cache,
                comm_simulator=comm_simulator,
                little_device=little_device,
                draft_device=draft_device,
                transfer_top_k=transfer_top_k,
                charge_residual=charge_residual,
                little_model_forward_times=little_model_forward_times,
                draft_model_forward_times=draft_model_forward_times,
                total_little_model_generated_tokens=total_little_model_generated_tokens,
                total_little_model_accepted_tokens=total_little_model_accepted_tokens,
            )

            (
                prefix,
                draft_model_forward_times,
                target_model_forward_times,
                total_draft_model_generated_tokens,
                total_draft_model_accepted_tokens,
                queuing_time,
            ) = self._cee_second_stage_dssd(
                prefix,
                prefix_ops=prefix_ops,
                prefix_len=prefix_len,
                new_generated_token_layer1=new_generated_token_layer1,
                current_proposal_top_k=current_proposal_top_k,
                draft_model_cache=draft_model_cache,
                target_model_cache=target_model_cache,
                little_model_cache=little_model_cache,
                comm_simulator=comm_simulator,
                draft_device=draft_device,
                target_device=target_device,
                transfer_top_k=transfer_top_k,
                charge_residual=charge_residual,
                idx=idx,
                batch_delay=batch_delay,
                draft_model_forward_times=draft_model_forward_times,
                target_model_forward_times=target_model_forward_times,
                total_draft_model_generated_tokens=total_draft_model_generated_tokens,
                total_draft_model_accepted_tokens=total_draft_model_accepted_tokens,
                queuing_time=queuing_time,
            )
            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        comm_simulator.flush_round()  # 结算最后一轮（按轮合并模式下必须）
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
        metrics["wall_time"] = wall_time
        metrics["throughput"] = (
            metrics["generated_tokens"] / wall_time if wall_time > 0 else 0
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

        metrics["queuing_time"] = target_model_forward_times * batch_delay
        metrics["wall_time"] += metrics["queuing_time"]
        if metrics["wall_time"] > 0:
            metrics["throughput"] = metrics["generated_tokens"] / metrics["wall_time"]

        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time,
            queuing_time=queuing_time,
        )

        # 带宽/top-k/草稿历史 + 逐消息账 + 口径标签（与其余方法同一 helper）
        _finalize_comm_metrics(metrics, comm_simulator, self.args)

        return prefix, metrics

    @Register.register_decoding("cee_dsd")
    @torch.no_grad()
    def cee_dsd(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud=10,
        ntt_ms_edge_end=1,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """
        CEE-DSD: Combined Edge-End Distributed Speculative Decoding (3-layer).
        Uses parallel/batch verification (send all probs at once) similar to DSD.
        """
        max_tokens = prefix.shape[1] + self.args.max_tokens
        little_device = self.get_model_input_device(self.little_model)
        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        draft_top_k = (
            transfer_top_k
            if (transfer_top_k is not None and transfer_top_k > 0)
            else self.args.top_k
        )

        # B19：little/draft/target 都做多 token 验证前向 ⇒ 三个都开图并跨样本
        # 复用；此前只给 little/draft 单步图、无验证档位、且每样本重捕获。
        graph_kw = _graph_mode_cache_kwargs(
            self.args,
            cap=int(getattr(self.args, "gamma1", 1))
            + int(getattr(self.args, "gamma2", 1))
            + 4,
        )
        caches = self._acquire_three_layer_caches(
            "_cee_dsd_caches", graph_kw, draft_top_k
        )
        little_model_cache = caches["little"]
        draft_model_cache = caches["draft"]
        target_model_cache = caches["target"]

        comm_simulator: CommunicationSimulator = _make_comm_simulator(
            self.comm_env,
            precise=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            edge_end=self.comm_env.edge_end_bandwidth,
            cloud_end=self.comm_env.cloud_end_bandwidth,
            transfer_top_k=transfer_top_k,
        )

        # 统一计费口径（docs/protocol.md §3）：与 adaptive_tridecoding 同一套
        # 开关。CUHLM 系（uncertainty_decoding / cee_cuhlm）按决策暂不接。
        # 口径在 self.comm_config 构造一次；属性名与开关名逐字一致，
        # protocols.py 的运行时内省才看得见（docs/comm_call_sites.md §4.1）。
        _comm = self.comm_config
        comm_simulator.coalesce_rounds = _comm.comm_round_trip_mode == "per_round"
        charge_residual = _comm.charge_residual_payload
        transfer_top_k = apply_transfer_top_k_cap(
            transfer_top_k, _comm.transfer_top_k_cap
        )
        comm_simulator.transfer_top_k = transfer_top_k

        # Metrics tracking
        little_model_forward_times = 0
        draft_model_forward_times = 0
        target_model_forward_times = 0
        total_little_model_generated_tokens = 0
        total_draft_model_generated_tokens = 0
        total_little_model_accepted_tokens = 0
        total_draft_model_accepted_tokens = 0
        queuing_time = 0
        batch_delay = self.comm_env.batch_delay
        wall_time = 0

        idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        # Pre-allocate prefix buffer to avoid O(n²) torch.cat during generation
        # x can grow up to max_tokens + gamma2 + gamma1 + 1 inside the loop
        prefix_ops = CEETriPrefix(
            prefix, capacity=max_tokens + self.args.gamma1 + self.args.gamma2 + 1
        )
        prefix = prefix_ops.tensor

        start_event.record(stream=torch.cuda.current_stream())

        comm_simulator.transfer(prefix, None, "edge_end")

        _tri_prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点

        while prefix.shape[1] < max_tokens:

            # B10：循环内 EOS 早停（与 dssd/adaptive_* 已接线方法语义对齐——
            # 计算与通信在 EOS 后立即停，时延口径跨方法可比）
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            idx += 1
            comm_simulator.set_round(idx)
            prefix_len = prefix.shape[1]
            current_proposal_top_k = proposal_top_k(transfer_top_k)
            (
                prefix,
                new_generated_token_layer1,
                little_model_forward_times,
                draft_model_forward_times,
                total_little_model_generated_tokens,
                total_little_model_accepted_tokens,
            ) = self._cee_first_stage_dsd(
                prefix,
                prefix_ops=prefix_ops,
                prefix_len=prefix_len,
                current_proposal_top_k=current_proposal_top_k,
                little_model_cache=little_model_cache,
                draft_model_cache=draft_model_cache,
                comm_simulator=comm_simulator,
                little_device=little_device,
                draft_device=draft_device,
                transfer_top_k=transfer_top_k,
                charge_residual=charge_residual,
                little_model_forward_times=little_model_forward_times,
                draft_model_forward_times=draft_model_forward_times,
                total_little_model_generated_tokens=total_little_model_generated_tokens,
                total_little_model_accepted_tokens=total_little_model_accepted_tokens,
            )

            (
                prefix,
                draft_model_forward_times,
                target_model_forward_times,
                total_draft_model_generated_tokens,
                total_draft_model_accepted_tokens,
                queuing_time,
            ) = self._cee_second_stage_dsd(
                prefix,
                prefix_ops=prefix_ops,
                prefix_len=prefix_len,
                new_generated_token_layer1=new_generated_token_layer1,
                current_proposal_top_k=current_proposal_top_k,
                draft_model_cache=draft_model_cache,
                target_model_cache=target_model_cache,
                little_model_cache=little_model_cache,
                comm_simulator=comm_simulator,
                draft_device=draft_device,
                target_device=target_device,
                transfer_top_k=transfer_top_k,
                charge_residual=charge_residual,
                idx=idx,
                batch_delay=batch_delay,
                draft_model_forward_times=draft_model_forward_times,
                target_model_forward_times=target_model_forward_times,
                total_draft_model_generated_tokens=total_draft_model_generated_tokens,
                total_draft_model_accepted_tokens=total_draft_model_accepted_tokens,
                queuing_time=queuing_time,
            )
            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        comm_simulator.flush_round()  # 结算最后一轮（按轮合并模式下必须）
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
        metrics["wall_time"] = wall_time
        metrics["throughput"] = (
            metrics["generated_tokens"] / wall_time if wall_time > 0 else 0
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

        metrics["queuing_time"] = target_model_forward_times * batch_delay
        metrics["wall_time"] += metrics["queuing_time"]
        if metrics["wall_time"] > 0:
            metrics["throughput"] = metrics["generated_tokens"] / metrics["wall_time"]

        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time,
            queuing_time=queuing_time,
        )

        # 带宽/top-k/草稿历史 + 逐消息账 + 口径标签（与其余方法同一 helper）
        _finalize_comm_metrics(metrics, comm_simulator, self.args)

        return prefix, metrics
