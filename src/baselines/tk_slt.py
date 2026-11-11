"""TK-SLT baseline: Top-K Sparse Logits Transmission.

The complete ``tk_slt`` / ``tkslt`` loop lives here as an independent family
mixin.  It keeps the paper-specific protocol rather than reusing the SD/DSD
verification skeleton: the draft law is the sparse softmax over top-K logits,
the uplink carries FP16-quantized sparse probabilities, and rejects resample
from the server-side sparse wire law.

Batch scope (2026-11-11): module migration plus the target-forward cleanup --
standalone, remaining-budget fallback and verification now call the public
``KVCacheModel.forward_context`` instead of ``generate`` when the extra sampled
token was always discarded.  Forward endpoints, cache rollback, wire bytes,
sampling laws and communication ordering are unchanged.

This remains an explicitly approximate legacy protocol: proposals are sampled
from raw top-K Q while the receiver verifies with the FP16-roundtripped Qhat.
Positive raw support can therefore be zero on the wire; the dedicated
``compute_legacy_tk_slt_fp16_acceptance_result`` bridge is retained and this
module does not claim target-distribution exactness.
"""

import time
from typing import List, Optional, Tuple

import torch

from ..baselines_helpers import (
    _add_per_model_wall_time,
    _ensure_token_shape,
    _finalize_comm_metrics,
    _last_edge_cloud_tx_seconds,
    _move_token_tensor,
    _quantize_probs_fp16,
    _sample_token_from_probs,
    _send_downlink_index_only,
    _validate_token_range,
)
from ..communication import CommunicationSimulator, tk_slt_uplink_payload_bytes
from ..decoding_ops import (
    _prepare_legacy_verification_inputs,
    apply_rollback,
    build_rollback_plan,
    compute_legacy_tk_slt_fp16_acceptance_result as _compute_legacy_tk_slt_fp16_acceptance_result,
    materialize_acceptance,
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
from ..proposal_utils import build_draft_probs_override
from ..register import Register
from ..tk_slt_policy import tk_slt_select_speculative
from .simulators import _make_symmetric_comm_simulator


#: TK-SLT ODLD（--tk_slt_odld）的 γ* 上限：估计噪声下防病态值；
#: 论文 Table I 的最大 γ* 是 14（α=0.8、L=0.01），64 只在极端估计下触及。
_TK_SLT_ODLD_GAMMA_CAP = 64


class TkSltMixin:
    """The TK-SLT baseline loop, registered as ``tk_slt``/``tkslt``."""

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

        Target forwards only advance context here: standalone/fallback and
        verification call ``forward_context`` and sample exactly the tokens
        they keep.  The pre-migration ``generate(..., 1)`` calls drew an extra
        target token that was always discarded, so this intentionally changes
        the RNG stream without changing the sampled-token laws.
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
                    target_model_cache.forward_context(
                        _move_token_tensor(prefix, target_device)
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
                target_model_cache.forward_context(
                    _move_token_tensor(prefix, target_device)
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
            # 只推进 target 上下文，不采丢弃 token：候选窗口与 prefill/extend
            # 端点都和旧 generate(..., 1) 的首步完全相同。
            target_model_cache.forward_context(
                _move_token_tensor(x, target_device)
            )
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
