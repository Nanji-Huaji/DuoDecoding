"""TK-SLT baseline: Top-K Sparse Logits Transmission.

The complete ``tk_slt`` / ``tkslt`` loop lives here as an independent family
mixin.  It keeps the paper-specific protocol rather than reusing the SD/DSD
verification skeleton: the draft law is the sparse top-K model law, the uplink
carries its explicit FP16 probability payload, and rejection uses the positive
residual of the target law against that same wire proposal.

Probability-wire variant (2026-11-11): a proposal is now *defined* by the row
that is actually sent and verified.  :mod:`src.tk_slt_proposal` validates the
model row before sampler preprocessing, selects exactly ``min(K, V)``
coordinates (``sampling.py`` masks ``logits < threshold``, so tied top-K logits
keep more than K positives), renormalizes the retained mass, writes real FP16
values with negligible sorted indices, and the draft loop samples the dense-V
FP16 row rebuilt from that payload.  The receiver rebuilds the same row from the
same payload on the target device and runs the strict ``ProposalWindow`` /
``prepare_verification_window`` / ``decide_verification`` contract, so
acceptance, the stable residual and the emitted token all refer to one
categorical law.  With ``Q=[.6,.4]`` the old loop emitted the raw Q law while
verifying ``P=wire(Q)`` (TVD 9.76502895e-5) and could sample a token whose wire
support was 0.  Both are gone: no unbilled dense model row reaches the receiver,
and an invalid wire payload fails closed instead of being repaired.

This remains an explicit probability-wire variant: it does not claim bit-exact
real arithmetic, GPU validation or a literal reproduction of the paper's
ambiguous logits terminology.  The repository's global draft/target temperature
stays in force (the paper's draft T=1 is not silently imposed). Every draw uses
one per-invocation ``DecodingRng`` context with distinct device keys, avoiding
cloned default-CUDA streams after ``torch.manual_seed``. This intentionally
changes RNG consumption; real multi-GPU execution remains unvalidated.
The dedicated
``compute_legacy_tk_slt_fp16_acceptance_result`` bridge in ``decoding_ops`` is
retained only as a historical callable; this loop no longer uses it.

Batch scope (2026-11-11): module migration plus the target-forward cleanup --
standalone, remaining-budget fallback and verification call the public
``KVCacheModel.forward_context`` instead of ``generate`` when the extra sampled
token was always discarded, and the draft loop advances the same way one step at
a time.  Forward endpoints (``prefix + gamma - 1``), cache construction keys,
graph-cache identity/reset, wire bytes and communication ordering are unchanged.
The intended sampling change is the proposal law itself (now the FP16 wire law).
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
    _send_downlink_index_only,
    _validate_token_range,
)
from ..communication import CommunicationSimulator, tk_slt_uplink_payload_bytes
from ..decoding_ops import apply_rollback, build_rollback_plan
from ..decoding_rng import DecodingRng
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
from ..probability_laws import validate_probability_rows
from ..register import Register
from ..tk_slt_policy import tk_slt_select_speculative
from ..tk_slt_proposal import build_draft_proposal
from ..verification import (
    ProposalWindow,
    decide_verification,
    prepare_verification_window,
)
from .simulators import _make_symmetric_comm_simulator

#: TK-SLT ODLD（--tk_slt_odld）的 γ* 上限：估计噪声下防病态值；
#: 论文 Table I 的最大 γ* 是 14（α=0.8、L=0.01），64 只在极端估计下触及。
_TK_SLT_ODLD_GAMMA_CAP = 64


def _sample_tk_token(
    rng: DecodingRng,
    probs: torch.Tensor,
    *,
    output_device: torch.device,
    vocab_size: int,
    label: str,
) -> torch.Tensor:
    """Keep token-boundary checks while drawing from this decode's private RNG."""
    token = _ensure_token_shape(rng.sample(probs), label=label)
    _validate_token_range(token, vocab_size=vocab_size, label=label)
    return _move_token_tensor(token, output_device)


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
        就是 top-K（§III Solution 1）；上行每个草稿位置只传 K 个概率值 +
        词表索引，BS 用重建的稀疏 Q 做标准投机验证，拒绝时从
        norm(max(0, P−Q̂)) 重采样（稀疏 Q̂ ⇒ 非 top-K 位置拿到完整的 P，
        正是论文的残差定义）。

        本实现是**显式概率-wire 变体**：提案律就是上行/验证实际使用的
        那个律。`src/tk_slt_proposal.py` 先校验模型概率行（采样器预处理
        之前），显式选正好 min(K,V) 个坐标（top-K 平局会让 mask 保留多于
        K 个正项），按保留质量重归一化后写成真实 FP16 值 + negligible
        排序索引；草稿从该 payload 重建的 dense-V FP16 行采样，接收端只
        从同一 payload 重建并走 strict `ProposalWindow` /
        `prepare_verification_window` / `decide_verification`。因此接受
        判据、稳定残差和输出 token 都指向同一个分类律——不再出现
        "raw Q 采样 + FP16 Q̂ 验证"（Q=[.6,.4] 时首 token 与原 P 的 TVD
        9.76502895e-5，且正的 1e-8 尾部可在 wire 上为 0 却仍被采到）。

        通信口径 = 论文自身（与 CUHLM 系同一处理方式，见 docs/protocol.md
        §3.1/§3.3），不接仓库统一计费开关：
        - 上行：γ·K·b_prob bits，b_prob=16（FP16，§VI-B）。载荷就是
          min(K,V) 个真实 FP16 值/位置（K<=0/None ⇒ 整词表 V 值），
          验证判据与计费看到同一个 payload（B15/B16 的"计费与数据一致"
          原则）；索引按 §II-B negligible；
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
        the RNG stream without changing the sampled-token laws.  The draft loop
        likewise advances one ``forward_context`` step per draft token and
        samples the reconstructed FP16 wire row; it ends at the same
        ``prefix + gamma - 1`` endpoint as the old ``generate`` did.
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
            "tk_slt_paper_accounting: uplink=gamma*min(K,V)*b_prob(FP16 16b, "
            "real wire values+negligible indices), "
            "reject-resample=strict server-side wire law; "
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

        # One lazy context for every draw in this invocation, across all rounds.
        # Equal default CUDA seeds must not clone proposal/acceptance randomness.
        rng = DecodingRng()
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

                    target_row = target_model_cache.prob_history[
                        :, -1, : self.vocab_size
                    ]
                    # 采样前显式校验目标行：不依赖 sampling_weights 对无效行的
                    # one-hot 兜底（strict 契约不允许"发明"一个分布）。
                    validate_probability_rows(target_row, "tk_slt.standalone_t")
                    t = _sample_tk_token(
                        rng,
                        target_row,
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

                target_row = target_model_cache.prob_history[
                    :, -1, : self.vocab_size
                ]
                # 采样前显式校验目标行（同上：无无效行兜底）。
                validate_probability_rows(target_row, "tk_slt.fallback_t")
                t = _sample_tk_token(
                    rng,
                    target_row,
                    output_device=prefix.device,
                    vocab_size=self.vocab_size,
                    label="tk_slt.fallback_t",
                )
                prefix = torch.cat((prefix, t), dim=1)
                self.num_acc_tokens.append(1)
                break

            # ---- Draft：概率-wire 提议（先选支撑/FP16 编码，再从 wire 行采样）----
            # 每步 forward_context 推进一格，编码后立即从重建的 dense-V FP16
            # 行采样；缓存端点仍是 prefix+γ-1（与旧 generate 相同），token 写入
            # 预分配缓冲，不做逐步增长的 torch.cat。
            t0 = time.time()
            draft = build_draft_proposal(
                approx_model_cache,
                _move_token_tensor(prefix, draft_device),
                current_gamma,
                draft_sampling_top_k,
                vocab_size=self.vocab_size,
                device=draft_device,
                rng=rng,
            )
            draft_comp_time += time.time() - t0
            x = _ensure_token_shape(draft.tokens, label="tk_slt.generated_x")
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

            # ---- 上行：真实稀疏 wire 值（每值 2B）+ negligible 索引 ----
            # 载荷 = γ·min(K,V) 个 FP16 值（K<=0/None ⇒ γ·V），与
            # tk_slt_uplink_payload_bytes 的论文口径逐值一致；索引与下行按
            # §II-B negligible。没有额外的 dense Q 旁路。
            window_rows = draft.count
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

            # ---- Verify：BS 只从已计费 payload 重建，再走 strict 契约 ----
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

            # 接收端在 target 设备上从 payload 重建 dense-V FP16 行：target
            # 的完整 P 不会伪装成发给草稿的未计费数据；round-local wire 行是
            # 不可变的稀疏值+索引，后续缓存写入不会改动它。
            proposal = ProposalWindow(
                tokens=x,
                probabilities=draft.dense_rows(device=target_device),
                prefix_len=prefix_len,
            )
            # 先验完整个窗口（维度/词表/概率质量），再取接受随机数：非法
            # target 或 wire 行在这里响亮失败，不落到任何 fallback。
            window = prepare_verification_window(
                proposal,
                target_model_cache.prob_history[:, :, : self.vocab_size],
            )
            acceptance_uniforms = rng.rand(
                (1, proposal.count), device=proposal.probabilities.device
            )
            decision = decide_verification(window, acceptance_uniforms)

            this_step_accepted_tokens = decision.accepted_count
            n = decision.accepted_end
            total_accepted_tokens += this_step_accepted_tokens
            alpha_num += this_step_accepted_tokens
            alpha_den += current_gamma

            self.num_acc_tokens.append(this_step_accepted_tokens)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, : n + 1]
            rollback_plan = build_rollback_plan(
                prefix_len,
                current_gamma,
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

            # 续采：接受分支 = P_{γ+1}（target bonus 行）；拒绝分支 = strict
            # norm((P−Q̂)+)（论文 §II-A 3b；Q̂ 即已计费的 wire 行）。两者都由
            # prepare/decide 给出：无 legacy max_fn 兜底、无额外抽样。
            t = _sample_tk_token(
                rng,
                decision.continuation_probs,
                output_device=prefix.device,
                vocab_size=self.vocab_size,
                label="tk_slt.sampled_t",
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
