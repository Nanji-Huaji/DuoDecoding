"""Historical bandwidth-aware speculative decoding baseline and its accounting."""

from typing import Optional, Tuple

import torch

from ..communication import CommunicationSimulator, PreciseCommunicationSimulator
from ..decoding_ops import (
    finalize_verification,
    reject_tail_scalar_bytes,
    verify_draft_sequence,
)
from ..graph_decode import graph_mode_cache_kwargs
from ..metrics import INT_SIZE, DecodingMetrics, get_empty_metrics
from ..proposal_utils import apply_transfer_top_k_cap, stage_topk_proposal_history
from ..register import Register
from ..utils import rebuild_topk_uniform_probs, sample


@Register.register_decoding("speculative_decoding_with_bandwidth")
@torch.inference_mode()
def speculative_decoding_with_bandwidth(
    self,
    prefix,
    transfer_top_k: Optional[int] = 300,
    use_precise_comm_sim: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, DecodingMetrics]:
    # **kwargs：评测 harness 统一以 partial(decoding, use_stochastic_comm=...,
    # ntt_ms_*=..., use_early_stopping=...) 装配全部方法；本方法自建仿真器，
    # 接受并忽略这些参数（与其余 11 个注册方法的兜底模式一致）。
    if use_precise_comm_sim:
        comm_simulator = PreciseCommunicationSimulator(
            min_bandwidth_mbps=self.comm_env.min_bandwidth_mbps,
            bandwidth_hz=1e6,
            channel_gain=1e-8,
            send_power_watt=0.5,
            noise_power_watt=1e-10,
        )
    else:
        comm_simulator = CommunicationSimulator(
            min_bandwidth_mbps=self.comm_env.min_bandwidth_mbps,
            bw_model=self.comm_env.comm_bw_model,
            bandwidth_edge_cloud=self.comm_env.edge_cloud_bandwidth,
            bandwidth_edge_end=float("inf"),
            bandwidth_cloud_end=float("inf"),
            dimension="Mbps",
        )
    # 统一计费口径（docs/protocol.md §3）：与 adaptive_tridecoding 同一套
    # 开关，经 Decoding.comm_config 单点构造。开关名必须出现在本方法
    # 代码里（protocols.py 运行时内省的依据）——属性名会留在 co_names。
    _comm = self.comm_config
    comm_simulator.coalesce_rounds = _comm.comm_round_trip_mode == "per_round"
    charge_residual = bool(_comm.charge_residual_payload)
    transfer_top_k = apply_transfer_top_k_cap(
        transfer_top_k, int(_comm.transfer_top_k_cap)
    )
    comm_simulator.transfer_top_k = transfer_top_k
    self.color_print(f"Using transfer_top_k: {transfer_top_k}", 2)

    max_tokens = prefix.shape[1] + self.args.max_tokens

    draft_device = self.draft_model.device
    target_device = self.target_model.device

    # CUDA Graph：只有草稿缓存跑 γ 步多 token 前向，是图的收益点；target 每步
    # 只做单 token 前向，开图无收益反而多占显存（与 speculative_decoding 一致）。
    # B19：补上验证图档位/桶长与跨样本复用。
    graph_kw = graph_mode_cache_kwargs(
        self.args, cap=int(getattr(self.args, "gamma", 5)) + 4
    )
    caches = self._acquire_speculative_caches(
        "_speculative_decoding_bw_caches", graph_kw, self.args.top_k
    )
    approx_model_cache = caches["draft"]
    target_model_cache = caches["target"]
    approx_model_cache.vocab_size = self.vocab_size
    target_model_cache.vocab_size = self.vocab_size

    draft_forward_times = 0
    target_forward_times = 0
    total_accepted_tokens = 0
    total_drafted_tokens = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    current_tokens = prefix.clone()

    start_event.record(stream=torch.cuda.current_stream())

    idx: int = 0

    while prefix.shape[1] < max_tokens:
        prefix_len = prefix.shape[1]

        idx += 1
        comm_simulator.set_round(idx)

        # 确保不会生成超过max_tokens的token
        remaining_tokens = max_tokens - prefix_len
        if remaining_tokens <= 0:
            break

        # 调整gamma以不超过剩余的token数量
        current_gamma = min(
            self.args.gamma, remaining_tokens - 1
        )  # 减1是为了留给最后的采样token
        if current_gamma <= 0:
            # 如果只剩1个token，直接用target model生成
            _ = target_model_cache.generate(prefix.to(target_device), 1)
            target_forward_times += 1
            if self.accelerator.is_main_process:
                self.target_forward_times += 1

            t = sample(
                target_model_cache.prob_history[:, -1, : self.vocab_size]
            ).to(draft_device)
            prefix = torch.cat((prefix, t), dim=1)
            self.num_acc_tokens.append(1)
            break

        proposal_top_k = (
            transfer_top_k
            if transfer_top_k is not None and transfer_top_k > 0
            else None
        )
        rebuilt_draft_probs = None
        rebuilt_draft_meta = None
        if proposal_top_k is not None:
            x, rebuilt_draft_probs = approx_model_cache.generate_with_rebuilt_topk(
                prefix.to(draft_device),
                current_gamma,
                proposal_top_k,
            )
        else:
            x = approx_model_cache.generate(prefix.to(draft_device), current_gamma)
        draft_forward_times += current_gamma
        total_drafted_tokens += current_gamma

        _ = target_model_cache.generate(x.to(target_device), 1)

        target_forward_times += 1

        if self.accelerator.is_main_process:
            self.draft_forward_times += current_gamma
            self.target_forward_times += 1

        verify_kwargs = {
            "draft_model_cache": approx_model_cache,
            "target_model_cache": target_model_cache,
            "x": x,
            "prefix_len": prefix_len,
            "gamma": current_gamma,
            "comm_simulator": comm_simulator,
            "comm_link": "edge_cloud",
            "transfer_mode": "serial",
            "send_reject_message": True,
            "draft_probs_override": (
                None
                if rebuilt_draft_probs is None
                else torch.cat(
                    (
                        approx_model_cache.prob_history[:, : prefix_len - 1, :],
                        rebuilt_draft_probs,
                    ),
                    dim=1,
                )
            ),
        }
        draft_topk_history = stage_topk_proposal_history(
            rebuilt_draft_meta,
            current_gamma,
        )
        if draft_topk_history is not None:
            verify_kwargs["draft_topk_history"] = draft_topk_history
        this_step_accepted_tokens, n = verify_draft_sequence(**verify_kwargs)

        total_accepted_tokens += this_step_accepted_tokens

        self.num_acc_tokens.append(this_step_accepted_tokens)

        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"

        # 检查是否还有空间添加一个token
        if n + 1 >= max_tokens:
            prefix = x[:, :max_tokens]
            break

        # 针对使用了通信压缩的传输逻辑
        if n < prefix_len + current_gamma - 1:
            if transfer_top_k is not None and transfer_top_k > 0:
                rebuild_probs = rebuild_topk_uniform_probs(
                    approx_model_cache.prob_history[:, n, : self.vocab_size],
                    transfer_top_k,
                )
                approx_model_cache.prob_history[:, n, : self.vocab_size] = (
                    rebuild_probs
                )

            comm_simulator.transfer(
                None,
                approx_model_cache.prob_history[:, n, : self.vocab_size],
                "edge_cloud",
                transfer_top_k is not None and transfer_top_k > 0,
                transfer_top_k,
            )
            if charge_residual:
                # 统一口径（§3.4）：压缩行已含 k×(4+元素)，补尾部标量。
                comm_simulator.simulate_transfer(
                    reject_tail_scalar_bytes(
                        approx_model_cache.prob_history[
                            :, n, : self.vocab_size
                        ],
                        transfer_top_k,
                    ),
                    "edge_cloud",
                )

        # finalize_verification自动回流状态与重新采样（或简单采样下一个字符）
        prefix = finalize_verification(
            approx_model_cache=approx_model_cache,
            target_model_cache=target_model_cache,
            x=x,
            prefix_len=prefix_len,
            gamma=current_gamma,
            n=n,
            draft_probs_override=(
                None
                if rebuilt_draft_probs is None
                else torch.cat(
                    (
                        approx_model_cache.prob_history[:, : prefix_len - 1, :],
                        rebuilt_draft_probs,
                    ),
                    dim=1,
                )
            ),
            draft_topk_history=stage_topk_proposal_history(
                rebuilt_draft_meta,
                current_gamma,
            ),
        )

        # 传输新生成的 token id
        comm_simulator.simulate_transfer(INT_SIZE, "edge_cloud")

    comm_simulator.flush_round()  # 结算最后一轮（按轮合并模式下必须）
    end_event.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000.0

    generated_tokens = prefix.shape[1] - current_tokens.shape[1]

    batch_delay = self.comm_env.batch_delay
    queuing_time = target_forward_times * batch_delay
    wall_time = elapsed_time + comm_simulator.edge_cloud_comm_time + queuing_time

    throughput = generated_tokens / wall_time if wall_time > 0 else 0

    metrics = get_empty_metrics()
    metrics["draft_forward_times"] = draft_forward_times
    metrics["target_forward_times"] = target_forward_times
    metrics["generated_tokens"] = generated_tokens
    metrics["draft_generated_tokens"] = total_drafted_tokens
    metrics["draft_accepted_tokens"] = total_accepted_tokens
    metrics["wall_time"] = wall_time
    metrics["throughput"] = throughput
    metrics["queuing_time"] = queuing_time
    metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
    metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data

    metrics["comm_energy"] = comm_simulator.total_comm_energy

    return prefix, metrics
