"""CU-HLM uncertainty-aware opportunistic transmission baseline.

This module owns the complete ``uncertainty_decoding`` / ``cuhlm`` loop. It was
moved out of ``core`` as an independent mixin rather than folded into the SD or
CEE-SD verification skeleton: CU-HLM may skip the target model entirely and,
when it does call the target, resamples rejects from the compressed ``x_hat``
that the server actually holds.

Batch scope (2026-11-11): the migration plus three directed fixes only --
constructor vocabulary propagation, real draft-token use in the Eq. (26) k*
solver, and the Eq. (10) strict ``u > threshold`` trigger.

Single-token lifecycle (2026-11-11, second batch): paper II-A/IV-B and
Algorithm 1 finalize exactly one token per round and have no full-accept
target bonus.  A trigger now calls ``target.forward_context(prefix)`` (no
sampling, no draft-token forward, no discarded target sample) and then either
keeps the draft token (accept) or appends one residual token (reject).  Both
caches end the round at the **old** prefix endpoint; the finalized token is
not forwarded until a later draft/trigger round.  After skipped rounds the
target catches up over the whole finalized prefix on its next trigger.
Rollback is no longer needed because no cache is advanced past the prefix.

Verification now goes through the independent single-token interface in
``src.cuhlm_verification``: acceptance compares the target law with the
normalized selected *original* q(d) (the sampler law actually used), rejection
returns ``normalize((P - qhat)+)`` from the reconstructed law the server holds,
and a rejected proposal with zero positive residual mass fails closed instead of
falling back to argmax or the original q.  This remains the paper's intentional
lossy protocol (original-history q(d) acceptance, reconstructed qhat residual),
so the path is still not claimed target-distribution exact.  The Eq. (26) k
solver's fallback is unchanged and still not proven conservative.
"""

import time
from typing import List, Optional, Tuple

import torch

from ..baselines_helpers import (
    _add_per_model_wall_time,
    _finalize_comm_metrics,
    _send_downlink_index_only,
)
from ..communication import CUHLM, PreciseCUHLM, cuhlm_uplink_payload_bytes
from ..cuhlm_verification import (
    decide_cuhlm_verification,
    prepare_cuhlm_verification,
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
from ..register import Register
from ..sampling import sample
from ..verification import ProposalWindow


class CuhlmMixin:
    """The CU-HLM baseline loop, registered as ``uncertainty_decoding``/``cuhlm``."""

    @Register.register_decoding("uncertainty_decoding")
    @Register.register_decoding("cuhlm")
    @torch.no_grad()
    def uncertainty_decoding(
        self,
        prefix,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim=False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 200,
        ntt_ms_edge_end: float = 20,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """
        CU-HLM baseline loop: "Communication-Efficient Hybrid Language Model
        via Uncertainty-Aware Opportunistic and Compressed Transmission".
        """
        if use_precise_comm_sim:
            comm_simulator: CUHLM = PreciseCUHLM(
                min_bandwidth_mbps=self.comm_env.min_bandwidth_mbps,
                bw_model=self.comm_env.comm_bw_model,
                vocab_size=self.vocab_size,
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                uncertainty_threshold=getattr(
                    self.args, "uncertainty_threshold", 0.8
                ),
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            threshold = getattr(self.args, "uncertainty_threshold", 0.8)
            comm_simulator = CUHLM(
                min_bandwidth_mbps=self.comm_env.min_bandwidth_mbps,
                bw_model=self.comm_env.comm_bw_model,
                bandwidth_edge_cloud=self.comm_env.edge_cloud_bandwidth,
                vocab_size=self.vocab_size,
                uncertainty_threshold=threshold,
                dimension="Mbps",
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=self.comm_env.stochastic_ntt,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )

        # F-CUHLM 口径自述：随 metrics 落盘（eval/utils.py get_save_dict
        # 读取 args.protocol_deviations），保证新口径跑出的工件可辨识、
        # 不与 2026-10-05 前的旧口径结果混排。
        # 2026-10-09 统一往返（docs/protocol.md §3.4）：per_round 下每次
        # 触发的上行+下行合并成一次云请求往返（1×NTT）。载荷字节仍是
        # 论文式 (5) 口径（cuhlm_uplink_payload_bytes）。
        comm_simulator.coalesce_rounds = (
            self.comm_config.comm_round_trip_mode == "per_round"
        )
        _dev = (
            "cuhlm_fair_accounting: skip=0RTT(buffered resync), "
            "trigger=1 uplink(resync+draft+topk)+1 downlink, "
            "queue=per-trigger, reject-resample=x_hat; "
            "round_trip=unified(1xNTT/trigger since 2026-10-09)"
        )
        _devs = list(getattr(self.args, "protocol_deviations", ()) or ())
        if _dev not in _devs:
            self.args.protocol_deviations = tuple(_devs + [_dev])

        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        # 使用 transfer_top_k 作为草稿模型的 top-k 压缩参数
        draft_top_k = (
            transfer_top_k
            if (transfer_top_k is not None and transfer_top_k > 0)
            else self.args.top_k
        )

        # CUDA Graph：只有草稿缓存跑 γ 步单 token 循环，是图的收益点；
        # target 是整段前向，开图无收益反而多占显存（与 engine.py 的取舍一致）。
        # 图开启时缓存挂 self 跨样本复用（StaticCache 原地 reset，不重捕获）。
        _graph_kw = _graph_mode_cache_kwargs(
            self.args, cap=int(getattr(self.args, "gamma", 5)) + 4
        )
        _caches = acquire_graph_caches(
            self, "_uncertainty_decoding_caches", _graph_kw,
            {
                "draft": lambda: KVCacheModel(
                    self.draft_model, self.args.temp, draft_top_k, self.args.top_p,
                    **_graph_kw,
                ),
                "target": lambda: KVCacheModel(self.target_model, self.args.temp, 0, 0),
            },
            cache_key={
                "draft": _kv_cache_construction_key(
                    self.draft_model, temperature=self.args.temp, top_k=draft_top_k,
                    top_p=self.args.top_p, vocab_size=self.vocab_size, **_graph_kw,
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

        # Metrics Tracking
        target_forward_times = 0
        draft_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0
        queuing_time = 0
        batch_delay = self.comm_env.batch_delay

        loop_idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record(stream=torch.cuda.current_stream())

        input_len = prefix.shape[1]

        draft_comp_time = 0.0
        target_comp_time = 0.0

        _tri_prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点

        # F-CUHLM 口径（2026-10-05 对齐论文 Algorithm 1 的通信模式；
        # 2026-04 起字节按论文式 (5) 计，见 docs/protocol.md §3 CUHLM 小节）：
        # - 跳过分支：零通信、零排队。被跳过的 token 缓存在端侧
        #   （_pending_resync 计数），下次触发时随上行消息捎带——论文 III-B
        #   Step 5 "lightweight index-based resynchronization...negligible and
        #   therefore omitted from the cost analysis"：索引不计字节、不单独
        #   付出 RTT。
        # - 触发分支：一次上行（一次 NTT），载荷 = 论文式 (5) 的压缩形式
        #   k(t)·(b_prob+b_index) bits（b_prob=8、b_index=⌈log₂V⌉，即
        #   cuhlm_uplink_payload_bytes）；draft token 索引 negligible 不计
        #   字节。旧口径：token 按 element_size、概率按 fp32+int32（k×8B），
        #   比论文口径高约 2.8×。
        #   + 一次 0 字节下行（一次 NTT，_send_downlink_index_only）。
        #   batch_delay 只在触发时计（与 DSD/DSSD/CEE-SD 的排队口径一致，
        #   它们均为 target_forward_times × batch_delay）。
        # - reject 重采样改用压缩重构分布 x̂（论文式 17）；旧实现用全量分布，
        #   压缩包只计费不上场（精度口径偏向该基线）。reject 在服务器端
        #   基于已持有的分布重采样（论文式 17），不产生额外传输——这与
        #   统一口径的 reject 残差计费不同，是论文自身的协议设计。
        _pending_resync = 0

        while prefix.shape[1] < max_tokens:

            # B10：循环内 EOS 早停（与 dssd/adaptive_* 已接线方法语义对齐——
            # 计算与通信在 EOS 后立即停，时延口径跨方法可比）
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            loop_idx += 1
            prefix_len = prefix.shape[1]
            # 统一往返（§3.4）：标记新一轮，上一轮累积的字节在此结算
            # （per_round = 每次云端交互 1×NTT；per_transfer 下是 no-op）
            comm_simulator.set_round(loop_idx)

            if loop_idx == 1:
                comm_simulator.transfer(prefix, None, link_type="edge_cloud")

            # Draft generates 1 token
            t0 = time.time()
            x = approx_model_cache.generate(prefix.to(draft_device), 1)
            draft_comp_time += time.time() - t0
            if approx_model_cache.logits_history is not None:
                current_logit = approx_model_cache.logits_history[
                    :, -1, : self.vocab_size
                ]
            else:
                raise ValueError("Approx model logits history is None")

            # 本轮候选的原始 q 行就是 draft 在本轮 prefix 上实际采样用到的行。
            # 草稿生成后立即验证 proposal：skip 分支也不能把非法 proposal 藏
            # 起来。这一步只做校验与规范化，不采样、不消耗 RNG。
            proposal = ProposalWindow(
                tokens=x,
                probabilities=approx_model_cache.prob_history[
                    :, prefix_len - 1 : prefix_len, : self.vocab_size
                ],
                prefix_len=prefix_len,
            )

            # Calculate uncertainty BEFORE target call (as in the paper)
            draft_token = int(x[0, -1].item())
            uncertainty = comm_simulator.calculate_uncertainty(
                current_logit, M=20, theta_max=2.0, draft_token=draft_token
            )
            # 压缩词表公式（论文式 (16)/(23)/(26)）定义在概率分布 x(t) 上：
            # 不确定度用原始 logits 算（温度扰动采样，设计如此），
            # 但 k* 的求解必须吃 softmax 后的概率分布。
            current_probs = comm_simulator._get_current_probs(
                approx_model_cache.prob_history
            )
            should_transfer, vocab_size = comm_simulator.determine_transfer_strategy(
                uncertainty, current_probs, draft_token=draft_token
            )

            draft_forward_times += 1
            total_drafted_tokens += 1

            if not should_transfer:
                # Low uncertainty: truly skip target model (as in the paper)
                # target_forward_times NOT incremented (target genuinely not called)
                # target_comp_time NOT incremented

                accepted_token = x[:, -1:]
                prefix = torch.cat(
                    (prefix.to(accepted_token.device), accepted_token), dim=1
                )

                # 跳过即直接定稿该 draft token：计入 draft_accepted_tokens
                # （draft 定稿数，不是目标准确率）。目标模型未调用。
                total_accepted_tokens += 1

                # F-CUHLM：跳过即零通信——不发 accept 消息、不即时上行；
                # token 计入待同步缓冲，下次触发时随上行捎带（见循环上方口径说明）。
                _pending_resync += 1

                # No bonus token from target (target was not called)
                # No KVCache rollback needed (target cache was not advanced)

                if use_early_stopping and self._check_stopping_criteria(
                    prefix, stop_sequences
                ):
                    break

                continue

            # High uncertainty: run target model for verification
            # F-CUHLM：排队只在真正发起云端调用时计（对齐 DSD/DSSD/CEE-SD）。
            queuing_time += batch_delay

            # F-CUHLM：触发 = 一次合并上行（一次 NTT）。载荷按论文式 (5) 的
            # 压缩形式计：k(t)·(b_prob+b_index) bits（b_prob=8、b_index=
            # ⌈log₂V⌉）。捎带的重同步 token 与当前 draft token 的索引按论文
            # §II-B/III-B Step 5 的 negligible 假设**不计字节**（旧口径按
            # element_size 逐 token 计）；topk/draft_len 仍进历史记录，
            # avg_top_k 反映的是自适应 k(t)。
            comm_simulator.simulate_transfer(
                cuhlm_uplink_payload_bytes(int(vocab_size), self.vocab_size),
                "edge_cloud",
                topk=int(vocab_size),
                draft_len=_pending_resync + 1,
            )
            _pending_resync = 0

            # 单 token 协议（论文 II-A/IV-B、Algorithm 1）：一轮只定稿一个
            # token。target 只做一次**不采样**的 prefix 前向，取“旧 prefix
            # 最后一个位置之后”的分布 P 来验证本轮 draft；不 forward draft
            # token、不产生被丢弃的 target 样本、不做 full-accept bonus。
            # 两缓存本轮端点都停在旧 prefix_len（其余历史行也到 prefix_len），
            # 最终定稿的 token 尚未 forward：下一轮由 draft generate 追一格，
            # 跳过若干轮后由 target 下次触发一次追平整个已定稿 prefix。
            t0 = time.time()
            _ = target_model_cache.forward_context(prefix.to(target_device))
            target_comp_time += time.time() - t0
            target_forward_times += 1

            # 单 token 验证（论文 II-A/IV-B、Algorithm 1）：
            # - target 历史必须恰好覆盖本轮 forward 的旧 prefix（无 bonus 行）；
            # - qhat 是旧 codec 返回的原始重建概率（不是已处理的 sampler
            #   weights，也不再走 legacy clipped 适配器）。先验证整窗，再取一次
            #   acceptance uniform，避免非法 law 先消费 RNG；
            # - 接受用原 proposal 的实际 sampler law 对 P；拒绝用 qhat 残差。
            reconstructed = comm_simulator.compress_rebuild_probs(
                proposal.probabilities, int(vocab_size)
            )
            window = prepare_cuhlm_verification(
                proposal,
                target_model_cache.prob_history[:, :, : self.vocab_size],
                reconstructed,
            )
            uniform = torch.rand(
                (1, 1), device=proposal.probabilities.device
            )
            decision = decide_cuhlm_verification(window, uniform)

            self.color_print(
                f"Uncertainty: {uncertainty:.4f}, Vocab size: {vocab_size}", 3
            )

            # F-CUHLM：压缩分布已在触发时的合并上行中计费（每次触发必传，
            # 与论文一致）；reject 不再有独立的 reject 消息与二次压缩包传输，
            # 下行只回传最终 token（见下方 _send_downlink_index_only）。
            total_accepted_tokens += 1 if decision.accepted else 0

            # 接受只保留 x 的最后一个 token（draft 本身），不额外采 bonus；
            # 拒绝只保留旧 prefix，下面追加一个由重建残差采出的 token。
            prefix = x if decision.accepted else x[:, :prefix_len]

            # 拒绝采样仍发生在下行之前（与上一批的 RNG/通信交错顺序一致：
            # acceptance 随机数 -> residual 采样 -> 下行报文）。模拟器在随机
            # 通信模式下也会消耗 RNG，不能把这次采样挪到下行之后。
            residual_token = None
            if decision.residual_probs is not None:
                residual_token = sample(decision.residual_probs)
                if residual_token.device != prefix.device:
                    residual_token = residual_token.to(prefix.device)

            # F-CUHLM（论文口径）：响应 token 索引 negligible（§II-B）——
            # 0 字节下行报文，保留一次 NTT（见 _send_downlink_index_only）。
            # 旧口径（B17）：INT_SIZE + token 字节。
            _send_downlink_index_only(comm_simulator, "edge_cloud")
            if residual_token is not None:
                prefix = torch.cat((prefix, residual_token), dim=1)

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        # 最后一轮（或中途 break 时）的字节可能仍在合并桶里，结算必须在
        # wall_time/communication_time 读数之前。
        comm_simulator.flush_round()
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        metrics = get_empty_metrics()

        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["draft_computation_time"] = draft_comp_time
        metrics["target_computation_time"] = target_comp_time
        metrics["generated_tokens"] = prefix.shape[1] - input_len
        metrics["draft_generated_tokens"] = draft_forward_times
        # draft_accepted_tokens 口径：被**直接定稿的 draft token 数**——既含
        # 触发轮里被 target 接受的 draft，也含机会跳过时直接追加的 draft；
        # 拒绝轮不计（该轮没有 draft token 成为输出）。它不是目标准确率：
        # 生成守恒为 generated_tokens = draft_accepted_tokens + 拒绝轮数。
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = (
            elapsed_time + queuing_time + comm_simulator.edge_cloud_comm_time
        )
        metrics["throughput"] = (
            (prefix.shape[1] - input_len) / metrics["wall_time"]
            if metrics["wall_time"] > 0
            else 0
        )
        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time,
            queuing_time=queuing_time,
            draft_comp_time=draft_comp_time,
            target_comp_time=target_comp_time,
        )
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times

        # 带宽/top-k/草稿历史 + 逐消息账 + 口径标签（与其余方法同一 helper）
        _finalize_comm_metrics(metrics, comm_simulator, self.args)

        return prefix, metrics
