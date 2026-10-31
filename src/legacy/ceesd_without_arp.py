"""Legacy `ceesd_without_arp` 实现（旧验证/回滚血缘，仅历史复现）。

新代码不要调用；无 ARP 消融请用 ``cee_sd_no_arp``
（``adaptive_tridecoding(_ablation="no_arp")``）。
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple, cast

import torch

from src.communication import CommunicationSimulator
from src.metrics import DecodingMetrics


def run(
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
    from src.baselines import (
        _add_per_model_wall_time,
        _finalize_comm_metrics,
        _graph_mode_cache_kwargs,
        _make_comm_simulator,
        _reward_comm_s,
        _send_downlink_token,
        apply_transfer_top_k_cap,
        collect_verification_payload,
        get_empty_metrics,
        proposal_top_k,
        reject_tail_scalar_bytes,
        resolve_stage_verification,
        sample_accept_token,
        stage_prob_history,
        stage_topk_proposal_history,
        state_entropy,
    )

    batch_delay = self.comm_env.batch_delay
    queuing_time = 0.0
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

    # B19：little/draft/target 都做多 token 验证前向（调度开销主体）⇒ 三个
    # 都开图并跨样本复用；此前只给 little/draft 单步图、无验证档位、且每
    # 样本重捕获（~534ms/样本把收益吃光）。target 此前完全没接线。
    graph_kw = _graph_mode_cache_kwargs(
        self.args,
        cap=int(getattr(self.args, "gamma1", 1))
        + int(getattr(self.args, "gamma2", 1))
        + 4,
    )
    caches = self._acquire_three_layer_caches(
        "_ceesd_without_arp_caches", graph_kw, draft_top_k
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
    wall_time = 0
    little_comp_time = 0.0
    draft_comp_time = 0.0
    target_comp_time = 0.0
    arp_overhead_time = 0.0
    dra_overhead_time = 0.0

    idx = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    current_tokens = prefix.clone()  # 用于计算生成token数

    start_event.record(stream=torch.cuda.current_stream())

    comm_simulator.transfer(prefix, None, "edge_end")  # 将 prompt 传输到 edge

    _tri_prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点
    # D4/B20：RL 选出的草稿长度走方法局部变量——此前直接改写
    # self.args.gamma1/gamma2，动作残留在全局 Namespace 上污染后续方法
    gamma2 = int(self.args.gamma2)
    gamma1 = int(self.args.gamma1)
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
        little_stage_probs: Optional[torch.Tensor] = None
        draft_stage_probs: Optional[torch.Tensor] = None

        # 第一层 speculative
        step_start_time = time.time()

        x, little_rebuilt_probs, little_rebuilt_meta, q = self._generate_with_optional_rebuilt_proposal(
            little_model_cache,
            prefix.to(little_device),
            gamma2,
            current_proposal_top_k,
            need_topk_metadata=True,
        )

        if self.little_rl_adapter is not None:
            bandwidth = comm_simulator.bandwidth_edge_end_mbps
            latency = comm_simulator.ntt_edge_end_ms
            acc_probs = []  # No ARP head
            # 注意：ceesd_without_arp 不带 ARP adapter 且可能没开 top-k 压缩，
            # 此时 `_generate_with_optional_rebuilt_proposal` 会合法地返回 q=None
            # （它只在需要重建 top-k 概率时才返回 q）。熵因此必须走缓存里按原始
            # logits 算好的 last_entropy——原先的 assert 会让这个消融直接崩溃。
            entropy = state_entropy(q, little_model_cache)
            task_name = getattr(self, "task", "unknown")
            next_k, _ = self.little_rl_adapter.select_config(
                bandwidth,
                latency,
                acc_probs,
                entropy,
                task_name,
                training=not getattr(self.args, "disable_rl_update", False),
            )
            gamma2 = int(next_k)

        actual_gamma2 = x.shape[1] - prefix_len

        _ = draft_model_cache.generate(x.to(draft_device), 1)

        little_model_forward_times += actual_gamma2
        draft_model_forward_times += 1
        total_little_model_generated_tokens += actual_gamma2

        n1: int = prefix_len + actual_gamma2 - 1

        little_accepted_this_iter = 0
        if actual_gamma2 > 0:
            little_stage_probs = stage_prob_history(
                little_model_cache,
                prefix_len,
                little_rebuilt_probs,
            )
            draft_tokens, draft_probs = collect_verification_payload(
                little_stage_probs,
                x,
                prefix_len,
                actual_gamma2,
            )
            comm_simulator.transfer(draft_tokens, draft_probs, "edge_end")
            little_stage_kwargs = {
                "proposer_cache": little_model_cache,
                "verifier_cache": draft_model_cache,
                "x": x,
                "prefix_len": prefix_len,
                "gamma": actual_gamma2,
                "output_device": little_device,
                "draft_probs_override": cast(torch.Tensor, little_stage_probs),
            }
            little_topk_history = stage_topk_proposal_history(
                little_rebuilt_meta,
                actual_gamma2,
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
            t = sample_accept_token(
                draft_model_cache.prob_history[:, -1, : self.vocab_size],
                output_device=little_device,
            )
            little_all_accepted = True

        total_little_model_accepted_tokens += little_accepted_this_iter

        if self.little_rl_adapter is not None:
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_comm_time = _reward_comm_s(
                self.args, comm_simulator, "edge_end"
            )

            # 去掉分子 +1
            tps_part = little_accepted_this_iter / (
                step_time + step_comm_time + 1e-9
            )
            reward = math.exp(min(tps_part, 100) / 20.0)

            # 平滑的幂次惩罚
            if actual_gamma2 > 1:
                acc_rate = little_accepted_this_iter / actual_gamma2
                reward *= acc_rate**2

            if not getattr(self.args, "disable_rl_update", False):
                self.little_rl_adapter.step(reward)

        assert n1 >= prefix_len - 1, f"n1 {n1}, prefix_len {prefix_len}"
        prefix = x[:, : n1 + 1]

        if not little_all_accepted:
            # reject someone, sample from the pos n1
            little_stage_probs = stage_prob_history(
                little_model_cache,
                prefix_len,
                little_rebuilt_probs,
            )
            comm_simulator.transfer(
                None,
                little_stage_probs[:, n1, : self.vocab_size],
                "edge_end",
                transfer_top_k is not None and transfer_top_k > 0,
                transfer_top_k,
            )
            if charge_residual:
                # 统一口径（§3.4）：压缩行已含 k×(4+元素)，补尾部标量。
                comm_simulator.simulate_transfer(
                    reject_tail_scalar_bytes(
                        little_stage_probs[:, n1, : self.vocab_size],
                        transfer_top_k,
                    ),
                    "edge_end",
                )

        # 传输索引
        _send_downlink_token(comm_simulator, t, "edge_end")

        prefix = torch.cat((prefix, t), dim=1)
        new_generated_token = prefix[:, prefix_len:]

        # 第二层 speculative
        step_start_time = time.time()

        if idx == 1:
            comm_simulator.transfer(prefix, None, "edge_cloud")
        else:
            comm_simulator.transfer(new_generated_token, None, "edge_cloud")

        x, draft_rebuilt_probs, _, q = self._generate_with_optional_rebuilt_proposal(
            draft_model_cache,
            prefix.to(draft_device),
            gamma1,
            current_proposal_top_k,
        )

        if self.rl_adapter is not None:
            bandwidth = comm_simulator.bandwidth_edge_cloud_mbps
            latency = comm_simulator.ntt_edge_cloud_ms
            acc_probs = []
            # 同上：q 可能为 None（无 ARP adapter / 未开 top-k），熵来自缓存
            entropy = state_entropy(q, draft_model_cache)
            task_name = getattr(self, "task", "unknown")
            next_k, _ = self.rl_adapter.select_config(
                bandwidth,
                latency,
                acc_probs,
                entropy,
                task_name,
                training=not getattr(self.args, "disable_rl_update", False),
            )
            gamma1 = int(next_k)

        actual_gamma1 = x.shape[1] - prefix.shape[1]

        queuing_time += batch_delay
        _ = target_model_cache.generate(x.to(target_device), 1)

        draft_model_forward_times += actual_gamma1
        target_model_forward_times += 1
        total_draft_model_generated_tokens += (
            new_generated_token.shape[1] + actual_gamma1
        )

        n2: int = prefix_len + new_generated_token.shape[1] + actual_gamma1 - 1
        draft_accepted_this_iter = 0
        total_gamma_layer2 = new_generated_token.shape[1] + actual_gamma1
        if total_gamma_layer2 > 0:
            draft_stage_probs = stage_prob_history(
                draft_model_cache,
                prefix_len + new_generated_token.shape[1],
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
            (
                draft_accepted_this_iter,
                n2,
                t,
                draft_all_accepted,
            ) = resolve_stage_verification(
                proposer_cache=draft_model_cache,
                verifier_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=total_gamma_layer2,
                output_device=draft_device,
                draft_probs_override=draft_stage_probs,
            )
            if not draft_all_accepted:
                comm_simulator.send_reject_message("edge_cloud")
        else:
            t = sample_accept_token(
                target_model_cache.prob_history[:, -1, : self.vocab_size],
                output_device=draft_device,
            )
            draft_all_accepted = True
        total_draft_model_accepted_tokens += draft_accepted_this_iter

        if self.rl_adapter is not None:
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_comm_time = _reward_comm_s(
                self.args, comm_simulator, "edge_cloud"
            )

            # 去掉分子 +1
            tps_part = draft_accepted_this_iter / (
                step_time + step_comm_time + 1e-9
            )
            reward = math.exp(min(tps_part, 100) / 20.0)

            # 平滑的幂次惩罚
            if actual_gamma1 > 1:
                acc_rate = draft_accepted_this_iter / actual_gamma1
                reward *= acc_rate**2

            if not getattr(self.args, "disable_rl_update", False):
                self.rl_adapter.step(reward)

        assert n2 >= prefix_len - 1, (
            f"n {n2} should be greater or equal than prefix_len {prefix_len}"
        )
        prefix = x[:, : n2 + 1]
        if n2 <= little_model_cache.current_length:
            little_model_cache.rollback(n2 + 1)
        if not draft_all_accepted:
            draft_stage_probs = stage_prob_history(
                draft_model_cache,
                prefix_len + new_generated_token.shape[1],
                draft_rebuilt_probs,
            )
            comm_simulator.transfer(
                None,
                draft_stage_probs[:, n2, : self.vocab_size],
                "edge_cloud",
                transfer_top_k is not None and transfer_top_k > 0,
                transfer_top_k,
            )
            if charge_residual:
                # 统一口径（§3.4）：同阶段一，补尾部标量。
                comm_simulator.simulate_transfer(
                    reject_tail_scalar_bytes(
                        draft_stage_probs[:, n2, : self.vocab_size],
                        transfer_top_k,
                    ),
                    "edge_cloud",
                )
            new_generated_token = prefix[:, prefix_len:]
        else:
            new_generated_token = prefix[:, prefix_len:]

        prefix = torch.cat((prefix, t), dim=1)
        # 传输索引
        _send_downlink_token(comm_simulator, t, "edge_cloud")
        _send_downlink_token(comm_simulator, t, "edge_end")

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
    # 带宽/top-k/草稿历史 + 逐消息账 + 口径标签（与其余方法同一 helper）
    _finalize_comm_metrics(metrics, comm_simulator, self.args)

    metrics["arp_overhead_time"] = arp_overhead_time
    metrics["dra_overhead_time"] = dra_overhead_time

    return prefix, metrics
