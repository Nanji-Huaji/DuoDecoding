import torch
from typing import Optional, List
from src.baselines import Baselines
from src.register import Register
from src.communication import PreciseCommunicationSimulator, CommunicationSimulator
from src.decoding_ops import (
    build_rollback_plan,
    collect_verification_payload,
    verify_draft_sequence_result,
    sample_accept_token,
    sample_reject_token,
)
from src.metrics import INT_SIZE, get_empty_metrics
from src.proposal_utils import proposal_top_k, build_draft_probs_override


class ProfiledBaselines(Baselines):
    @Register.register_decoding("cee_dsd")
    @torch.no_grad()
    def cee_dsd(
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
    ):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/profiler"),
        ) as prof:
            max_tokens = prefix.shape[1] + self.args.max_tokens
            little_device = self.little_model.device
            draft_device = self.draft_model.device
            target_device = self.target_model.device

            caches = self.build_adaptive_tridecoding_caches(transfer_top_k)
            little_model_cache = caches["little"]
            draft_model_cache = caches["draft"]
            target_model_cache = caches["target"]

            if use_precise_comm_sim:
                comm_simulator = PreciseCommunicationSimulator(
                    bandwidth_hz=1e7,
                    channel_gain=1e-8,
                    send_power_watt=0.5,
                    noise_power_watt=1e-10,
                    ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                    ntt_ms_edge_end=ntt_ms_edge_end,
                )
            else:
                comm_simulator = CommunicationSimulator(
                    bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                    bandwidth_edge_end=self.args.edge_end_bandwidth,
                    bandwidth_cloud_end=self.args.cloud_end_bandwidth,
                    transfer_top_k=transfer_top_k,
                    dimension="Mbps",
                    ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                    ntt_ms_edge_end=ntt_ms_edge_end,
                    use_stochastic=use_stochastic_comm,
                )

            # Metrics tracking
            little_model_forward_times = 0
            draft_model_forward_times = 0
            target_model_forward_times = 0
            total_little_model_generated_tokens = 0
            total_draft_model_generated_tokens = 0
            total_little_model_accepted_tokens = 0
            total_draft_model_accepted_tokens = 0
            queuing_time = 0
            batch_delay = getattr(self.args, "batch_delay", 0)
            wall_time = 0
            total_draft_steps = 0
            sum_draft_len = 0
            sum_top_k = 0

            append_buffers: dict[torch.device, torch.Tensor] = {}

            def append_token(
                prefix_tensor: torch.Tensor, token: torch.Tensor
            ) -> torch.Tensor:
                target_len = prefix_tensor.shape[1] + token.shape[1]
                buffer = append_buffers.get(prefix_tensor.device)
                required_len = max(max_tokens, target_len)
                if buffer is None or buffer.shape[1] < required_len:
                    buffer = torch.empty(
                        (prefix_tensor.shape[0], required_len),
                        dtype=prefix_tensor.dtype,
                        device=prefix_tensor.device,
                    )
                    append_buffers[prefix_tensor.device] = buffer
                buffer[:, : prefix_tensor.shape[1]].copy_(prefix_tensor)
                buffer[:, prefix_tensor.shape[1] : target_len].copy_(token)
                return buffer[:, :target_len]

            idx = 0

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            current_tokens = prefix.clone()  # 用于计算生成token数

            start_event.record(stream=torch.cuda.current_stream())

            comm_simulator.transfer(prefix, None, "edge_end")  # 将 prompt 传输到 edge

            while prefix.shape[1] < max_tokens:
                idx += 1

                prefix_len = prefix.shape[1]

                # 第一层 speculative

                current_proposal_top_k = proposal_top_k(transfer_top_k)
                little_rebuilt_probs = None

                with torch.profiler.record_function("Little Draft Generation"):
                    if current_proposal_top_k is not None:
                        x, little_rebuilt_probs = (
                            little_model_cache.generate_with_rebuilt_topk(
                                prefix.to(little_device),
                                self.args.gamma2,
                                current_proposal_top_k,
                            )
                        )
                    else:
                        x = little_model_cache.generate(
                            prefix.to(little_device), self.args.gamma2
                        )
                    _ = draft_model_cache.generate(x.to(draft_device), 1)

                little_model_forward_times += self.args.gamma2
                draft_model_forward_times += 1
                total_little_model_generated_tokens += self.args.gamma2

                # 累积追踪指标 - 记录第一层 draft
                total_draft_steps += 1
                sum_draft_len += self.args.gamma2
                sum_top_k += (
                    current_proposal_top_k if current_proposal_top_k is not None else 0
                )

                n1: int = prefix_len + self.args.gamma2 - 1

                little_accepted_this_iter = 0
                # 批量传输 draft tokens 和对应的 probabilities 以节省 RTT
                if self.args.gamma2 > 0:
                    little_stage_probs = build_draft_probs_override(
                        little_model_cache,
                        prefix_len,
                        little_rebuilt_probs,
                    )
                    if little_stage_probs is None:
                        little_stage_probs = little_model_cache.prob_history
                    draft_tokens, draft_probs = collect_verification_payload(
                        little_stage_probs,
                        x,
                        prefix_len,
                        self.args.gamma2,
                    )
                    with torch.profiler.record_function("Communication Delay 1"):
                        comm_simulator.transfer(draft_tokens, draft_probs, "edge_end")

                with torch.profiler.record_function("Draft Verification"):
                    first_stage_inputs, first_stage_acceptance = (
                        verify_draft_sequence_result(
                            draft_model_cache=little_model_cache,
                            target_model_cache=draft_model_cache,
                            x=x,
                            prefix_len=prefix_len,
                            gamma=self.args.gamma2,
                            draft_probs_override=build_draft_probs_override(
                                little_model_cache,
                                prefix_len,
                                little_rebuilt_probs,
                            ),
                        )
                    )
                n1 = first_stage_acceptance.n
                little_accepted_this_iter = first_stage_acceptance.accepted_count

                total_little_model_accepted_tokens += little_accepted_this_iter

                assert n1 >= prefix_len - 1, f"n {n1}, prefix_len {prefix_len}"
                prefix = x[:, : n1 + 1]

                first_stage_rollback_plan = build_rollback_plan(
                    prefix_len,
                    first_stage_inputs.actual_gamma,
                    n1,
                )

                with torch.profiler.record_function("KV Cache Rollback 1"):
                    little_model_cache.rollback(first_stage_rollback_plan.draft_end_pos)

                    if not first_stage_rollback_plan.all_accepted:
                        comm_simulator.transfer(
                            None,
                            first_stage_inputs.draft_probs_batch[
                                :, n1 - (prefix_len - 1), : self.vocab_size
                            ],
                            "edge_end",
                            transfer_top_k is not None and transfer_top_k > 0,
                            transfer_top_k,
                        )

                        t = sample_reject_token(
                            draft_model_cache.prob_history[:, n1, : self.vocab_size],
                            first_stage_inputs.draft_probs_batch[
                                :, n1 - (prefix_len - 1), : self.vocab_size
                            ],
                            output_device=little_device,
                        )

                        draft_model_cache.rollback(
                            first_stage_rollback_plan.target_end_pos_reject
                        )

                    else:
                        t = sample_accept_token(
                            draft_model_cache.prob_history[:, -1, : self.vocab_size],
                            output_device=little_device,
                        )

                        draft_model_cache.rollback(
                            first_stage_rollback_plan.target_end_pos_accept
                        )

                # 传输索引
                with torch.profiler.record_function("Communication Delay 2"):
                    comm_simulator.simulate_transfer(INT_SIZE, "edge_end")
                    comm_simulator.transfer(t, None, "edge_end")

                prefix = append_token(prefix, t)
                new_generated_token = prefix[:, prefix_len:]

                # 第二层 speculative
                with torch.profiler.record_function("Communication Delay 3"):
                    if idx == 1:
                        comm_simulator.transfer(prefix, None, "edge_cloud")
                    else:
                        comm_simulator.transfer(new_generated_token, None, "edge_cloud")

                with torch.profiler.record_function("Middle Draft Generation"):
                    x, draft_rebuilt_probs, _ = (
                        self._generate_with_optional_rebuilt_proposal(
                            draft_model_cache,
                            prefix.to(draft_device),
                            self.args.gamma1,
                            current_proposal_top_k,
                        )
                    )

                with torch.profiler.record_function(
                    "Target Generation Delay / Queuing"
                ):
                    queuing_time += batch_delay
                    _ = target_model_cache.generate(x.to(target_device), 1)

                draft_model_forward_times += self.args.gamma1
                target_model_forward_times += 1
                total_draft_model_generated_tokens += (
                    new_generated_token.shape[1] + self.args.gamma1
                )

                total_gamma = new_generated_token.shape[1] + self.args.gamma1
                n2: int = prefix_len + total_gamma - 1

                # 批量传输 draft tokens 和对应的 probabilities 以节省 RTT
                if total_gamma > 0:
                    draft_stage_probs = build_draft_probs_override(
                        draft_model_cache,
                        prefix_len,
                        draft_rebuilt_probs,
                    )
                    if draft_stage_probs is None:
                        draft_stage_probs = draft_model_cache.prob_history
                    draft_tokens_second, draft_probs_second = (
                        collect_verification_payload(
                            draft_stage_probs,
                            x,
                            prefix_len,
                            total_gamma,
                        )
                    )
                    with torch.profiler.record_function("Communication Delay 4"):
                        comm_simulator.transfer(
                            draft_tokens_second, draft_probs_second, "edge_cloud"
                        )

                with torch.profiler.record_function("Target Verification"):
                    second_stage_inputs, second_stage_acceptance = (
                        verify_draft_sequence_result(
                            draft_model_cache=draft_model_cache,
                            target_model_cache=target_model_cache,
                            x=x,
                            prefix_len=prefix_len,
                            gamma=total_gamma,
                            draft_probs_override=build_draft_probs_override(
                                draft_model_cache,
                                prefix_len,
                                draft_rebuilt_probs,
                            ),
                        )
                    )
                n2 = second_stage_acceptance.n
                draft_accepted_this_iter = second_stage_acceptance.accepted_count
                total_draft_model_accepted_tokens += draft_accepted_this_iter

                assert n2 >= prefix_len - 1, (
                    f"n {n2} should be greater or equal than prefix_len {prefix_len}"
                )
                prefix = x[:, : n2 + 1]
                second_stage_rollback_plan = build_rollback_plan(
                    prefix_len,
                    second_stage_inputs.actual_gamma,
                    n2,
                )

                with torch.profiler.record_function("KV Cache Rollback 2"):
                    draft_model_cache.rollback(second_stage_rollback_plan.draft_end_pos)
                    if n2 <= little_model_cache.current_length:
                        little_model_cache.rollback(
                            second_stage_rollback_plan.draft_end_pos
                        )
                    if not second_stage_rollback_plan.all_accepted:
                        comm_simulator.transfer(
                            None,
                            second_stage_inputs.draft_probs_batch[
                                :, n2 - (prefix_len - 1), : self.vocab_size
                            ],
                            "edge_cloud",
                            transfer_top_k is not None and transfer_top_k > 0,
                            transfer_top_k,
                        )
                        t = sample_reject_token(
                            target_model_cache.prob_history[:, n2, : self.vocab_size],
                            second_stage_inputs.draft_probs_batch[
                                :, n2 - (prefix_len - 1), : self.vocab_size
                            ],
                            output_device=draft_device,
                        )
                        new_generated_token = prefix[:, prefix_len:]

                        target_model_cache.rollback(
                            second_stage_rollback_plan.target_end_pos_reject
                        )

                    else:
                        t = sample_accept_token(
                            target_model_cache.prob_history[:, -1, : self.vocab_size],
                            output_device=draft_device,
                        )
                        new_generated_token = prefix[:, prefix_len:]

                        target_model_cache.rollback(
                            second_stage_rollback_plan.target_end_pos_accept
                        )

                prefix = append_token(prefix, t)
                # 传输索引
                with torch.profiler.record_function("Communication Delay 5"):
                    comm_simulator.simulate_transfer(INT_SIZE, "edge_cloud")
                    comm_simulator.transfer(t, None, "edge_cloud")
                    comm_simulator.simulate_transfer(INT_SIZE, "edge_end")
                    comm_simulator.transfer(t, None, "edge_end")
                # 同步

                if use_early_stopping and self._check_stopping_criteria(
                    prefix, stop_sequences
                ):
                    break

                prof.step()

            end_event.record(stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0

            wall_time += elapsed_time
            generated_tokens = prefix.shape[1] - current_tokens.shape[1]
            wall_time += (
                comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
            )

            metrics = get_empty_metrics()
            metrics["avg_top_k"] = (
                sum_top_k / total_draft_steps if total_draft_steps > 0 else 0
            )
            metrics["avg_draft_len"] = (
                sum_draft_len / total_draft_steps if total_draft_steps > 0 else 0
            )
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

            batch_delay = getattr(self.args, "batch_delay", 0)
            queuing_time = target_model_forward_times * batch_delay
            metrics["queuing_time"] = queuing_time
            metrics["wall_time"] += queuing_time
            if metrics["wall_time"] > 0:
                metrics["throughput"] = (
                    metrics["generated_tokens"] / metrics["wall_time"]
                )

            # 复制 edge-cloud 的带宽、top-k 和起草长度历史数据
            metrics["edge_cloud_bandwidth_history"] = (
                comm_simulator.edge_cloud_bandwidth_history.copy()
            )
            metrics["edge_cloud_topk_history"] = (
                comm_simulator.edge_cloud_topk_history.copy()
            )
            metrics["edge_cloud_draft_len_history"] = (
                comm_simulator.edge_cloud_draft_len_history.copy()
            )

            # 在性能分析结束后，可以在这里自动打印占比数据：
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

        return prefix, metrics
