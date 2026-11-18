"""置信度校准数据采集（实验脚本，非标准评测入口）。

``adaptive_tridecoding`` 生产实现只返回 metrics，不暴露逐 token 的 acceptance
概率，因此本脚本保留一份**与旧版逐行等价**的采集循环，直接从 fresh decoder
的 acceptance head adapter 上读取 ``last_acc_prob``。算法本身不变；变化的只是
资源获取方式——不再继承旧的 ``EvalMTBench``，而是显式创建一个
``ModelStore`` decoder，并复用共享的 ``eval.data.load_task_data`` 取 MT-Bench
样本。

输出：当前目录下 ``confidence_data.json``，结构仍是
``{"little_draft": [{"prob", "accepted"}], "draft_target": [...]}``。
"""

import json
import os
import sys
from typing import Dict, List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from eval.data import load_task_data
from eval.decoder import ModelStore
from src.communication import CommunicationSimulator, PreciseCommunicationSimulator
from src.metrics import INT_SIZE, DecodingMetrics, get_empty_metrics
from src.model_gpu import KVCacheModel
from src.utils import max_fn, parse_arguments, sample

# 旧脚本固定只采前 100 条 MT-Bench 样本。
MAX_SAMPLES = 100
OUTPUT_FILE = "confidence_data.json"


@torch.no_grad()
def collect_adaptive_tridecoding(
    decoder,
    prefix,
    confidence_data: Dict[str, List[dict]],
    transfer_top_k=300,
    use_precise_comm_sim=False,
    ntt_ms_edge_cloud=10,
    ntt_ms_edge_end=1,
    **kwargs,
) -> Tuple[torch.Tensor, DecodingMetrics]:
    """采集一次 adaptive_tridecoding rollout 的逐 token 置信度。

    与原实现相比只把 ``self.*`` 换成显式 ``decoder.*``；接受/拒绝判定、
    通信计费和 metrics 口径保持原样。
    """
    args = decoder.args
    vocab_size = decoder.vocab_size
    max_tokens = prefix.shape[1] + args.max_tokens
    little_device = decoder.get_model_input_device(decoder.little_model)
    draft_device = decoder.get_model_input_device(decoder.draft_model)
    target_device = decoder.get_model_input_device(decoder.target_model)
    little_model_cache = KVCacheModel(
        decoder.little_model, args.temp, args.top_k, args.top_p
    )
    little_model_cache.vocab_size = vocab_size
    draft_model_cache = KVCacheModel(
        decoder.draft_model, args.temp, args.top_k, args.top_p
    )
    draft_model_cache.vocab_size = vocab_size
    target_model_cache = KVCacheModel(
        decoder.target_model, args.temp, args.top_k, args.top_p
    )
    target_model_cache.vocab_size = vocab_size

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
            bandwidth_edge_cloud=args.edge_cloud_bandwidth,
            bandwidth_edge_end=args.edge_end_bandwidth,
            bandwidth_cloud_end=args.cloud_end_bandwidth,
            transfer_top_k=transfer_top_k,
            dimension="Mbps",
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
        )

    # Metrics tracking
    little_model_forward_times = 0
    draft_model_forward_times = 0
    target_forward_times = 0
    total_little_model_generated_tokens = 0
    total_draft_model_generated_tokens = 0
    total_little_model_accepted_tokens = 0
    total_draft_model_accepted_tokens = 0
    wall_time = 0

    idx = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    current_tokens = prefix.clone()  # 用于计算生成 token 数

    start_event.record(stream=torch.cuda.current_stream())

    comm_simulator.transfer(prefix, None, "edge_end")  # 将 prompt 传输到 edge

    while prefix.shape[1] < max_tokens:
        idx += 1

        prefix_len = prefix.shape[1]

        # 第一层 speculative

        x = prefix.clone().to(little_device)

        little_probs = []  # Store probabilities for this iteration

        for _ in range(args.gamma2):
            adapter = decoder.small_draft_adapter
            q = little_model_cache._forward_with_kvcache(x)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
            hidden_states = little_model_cache.hidden_states
            assert hidden_states is not None
            stop = adapter.predict(hidden_states)

            # Collect probability
            if adapter.last_acc_prob is not None:
                little_probs.append(adapter.last_acc_prob)
            else:
                little_probs.append(0.0)  # Should not happen if adapter is correct

            if stop:
                break

        actual_gamma2 = x.shape[1] - prefix_len

        _ = draft_model_cache.generate(x.to(draft_device), 1)

        little_model_forward_times += actual_gamma2
        draft_model_forward_times += 1
        total_little_model_generated_tokens += actual_gamma2

        n1: int = prefix_len + actual_gamma2 - 1

        little_accepted_this_iter = 0
        for i in range(actual_gamma2):
            r = torch.rand(1, device=little_device)
            j = x[:, prefix_len + i]

            # 传输 token id 和 prob 用于 reject sampling
            comm_simulator.transfer(
                j,
                little_model_cache.prob_history[:, prefix_len + i - 1, j],
                "edge_end",
            )

            if (
                r
                > (
                    draft_model_cache.prob_history.to(little_device)[
                        :, prefix_len + i - 1, j
                    ]
                )
                / (little_model_cache.prob_history[:, prefix_len + i - 1, j])
            ):
                comm_simulator.send_reject_message("edge_end")
                n1 = prefix_len + i - 1

                # Record data (Rejected)
                if i < len(little_probs):
                    confidence_data["little_draft"].append(
                        {"prob": little_probs[i], "accepted": False}
                    )

                break

            else:
                little_accepted_this_iter += 1
                # Record data (Accepted)
                if i < len(little_probs):
                    confidence_data["little_draft"].append(
                        {"prob": little_probs[i], "accepted": True}
                    )

        total_little_model_accepted_tokens += little_accepted_this_iter

        assert n1 >= prefix_len - 1, f"n {n1}, prefix_len {prefix_len}"
        prefix = x[:, : n1 + 1]

        little_model_cache.rollback(n1 + 1)

        if n1 < prefix_len + actual_gamma2 - 1:
            # reject someone, sample from the pos n1
            rebuild_probs = comm_simulator.rebuild_full_probs(
                little_model_cache.prob_history[:, n1, :vocab_size]
            )
            little_model_cache.prob_history[:, n1, :vocab_size] = rebuild_probs

            comm_simulator.transfer(
                None,
                little_model_cache.prob_history[:, n1, :vocab_size],
                "edge_end",
                transfer_top_k is not None and transfer_top_k > 0,
                transfer_top_k,
            )

            t = sample(
                max_fn(
                    draft_model_cache.prob_history[:, n1, :vocab_size].to(little_device)
                    - little_model_cache.prob_history[:, n1, :vocab_size]
                )
            )

            draft_model_cache.rollback(n1 + 1)

        else:
            t = sample(draft_model_cache.prob_history[:, -1, :vocab_size]).to(
                little_device
            )

            draft_model_cache.rollback(n1 + 2)

        # 传输索引
        comm_simulator.simulate_transfer(INT_SIZE, "edge_end")
        comm_simulator.transfer(t, None, "edge_end")

        prefix = torch.cat((prefix, t), dim=1)
        new_generated_token = prefix[:, prefix_len:]

        # 第二层 speculative

        if idx == 1:
            comm_simulator.transfer(prefix, None, "edge_cloud")
        else:
            comm_simulator.transfer(new_generated_token, None, "edge_cloud")

        x = prefix.clone().to(draft_device)

        draft_probs = []

        for _ in range(args.gamma1):
            adapter = decoder.draft_target_adapter
            q = draft_model_cache._forward_with_kvcache(x)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
            hidden_states = draft_model_cache.hidden_states
            assert hidden_states is not None
            stop = adapter.predict(hidden_states)

            if adapter.last_acc_prob is not None:
                draft_probs.append(adapter.last_acc_prob)
            else:
                draft_probs.append(0.0)

            if stop:
                break

        actual_gamma1 = x.shape[1] - prefix.shape[1]

        _ = target_model_cache.generate(x.to(target_device), 1)

        draft_model_forward_times += actual_gamma1
        target_forward_times += 1
        total_draft_model_generated_tokens += actual_gamma1

        n2: int = prefix_len + new_generated_token.shape[1] + actual_gamma1 - 1
        draft_accepted_this_iter = 0
        for i in range(
            new_generated_token.shape[1] + actual_gamma1,
        ):
            r = torch.rand(1, device=draft_device)
            j = x[:, prefix_len + i]

            # 传输 token id 和 prob 用于 reject sampling
            comm_simulator.transfer(
                j,
                draft_model_cache.prob_history[:, prefix_len + i - 1, j],
                "edge_cloud",
            )

            if (
                r
                > (
                    target_model_cache.prob_history.to(draft_device)[
                        :, prefix_len + i - 1, j
                    ]
                )
                / (draft_model_cache.prob_history[:, prefix_len + i - 1, j])
            ):
                n2 = prefix_len + i - 1
                comm_simulator.send_reject_message("edge_cloud")

                # Record data (Rejected)
                # Note: i starts from new_generated_token.shape[1] for draft
                # tokens; the first new_generated_token.shape[1] tokens are
                # from the previous stage (already accepted by draft model).
                # We only care about tokens generated by draft model here.
                draft_token_idx = i - new_generated_token.shape[1]
                if draft_token_idx >= 0 and draft_token_idx < len(draft_probs):
                    confidence_data["draft_target"].append(
                        {"prob": draft_probs[draft_token_idx], "accepted": False}
                    )

                break
            else:
                draft_accepted_this_iter += 1

                # Record data (Accepted)
                draft_token_idx = i - new_generated_token.shape[1]
                if draft_token_idx >= 0 and draft_token_idx < len(draft_probs):
                    confidence_data["draft_target"].append(
                        {"prob": draft_probs[draft_token_idx], "accepted": True}
                    )

        total_draft_model_accepted_tokens += draft_accepted_this_iter

        assert n2 >= prefix_len - 1, (
            f"n {n2} should be greater or equal than prefix_len {prefix_len}"
        )
        prefix = x[:, : n2 + 1]
        draft_model_cache.rollback(n2 + 1)
        if n2 <= little_model_cache.current_length:
            little_model_cache.rollback(n2 + 1)
        if n2 < prefix_len + new_generated_token.shape[1] + actual_gamma1 - 1:
            rebuild_probs = comm_simulator.rebuild_full_probs(
                draft_model_cache.prob_history[:, n2, :vocab_size]
            )
            draft_model_cache.prob_history[:, n2, :vocab_size] = rebuild_probs

            comm_simulator.transfer(
                None,
                draft_model_cache.prob_history[:, n2, :vocab_size],
                "edge_cloud",
                transfer_top_k is not None and transfer_top_k > 0,
                transfer_top_k,
            )
            t = sample(
                max_fn(
                    target_model_cache.prob_history[:, n2, :vocab_size].to(draft_device)
                    - draft_model_cache.prob_history[:, n2, :vocab_size]
                )
            )
            new_generated_token = prefix[:, prefix_len:]

            target_model_cache.rollback(n2 + 1)

        else:
            t = sample(target_model_cache.prob_history[:, -1, :vocab_size]).to(
                draft_device
            )
            new_generated_token = prefix[:, prefix_len:]

            target_model_cache.rollback(n2 + 2)

        prefix = torch.cat((prefix, t), dim=1)
        # 传输索引
        comm_simulator.simulate_transfer(INT_SIZE, "edge_cloud")
        comm_simulator.transfer(t, None, "edge_cloud")
        comm_simulator.simulate_transfer(INT_SIZE, "edge_end")
        comm_simulator.transfer(t, None, "edge_end")
        # 同步

    end_event.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000.0

    wall_time += elapsed_time
    generated_tokens = prefix.shape[1] - current_tokens.shape[1]
    wall_time += comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time

    metrics = get_empty_metrics()
    metrics["little_forward_times"] = little_model_forward_times
    metrics["draft_forward_times"] = draft_model_forward_times
    metrics["target_forward_times"] = target_forward_times
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
    return prefix, metrics


def main():
    args = parse_arguments()

    # 采集循环使用 adaptive_tridecoding（两级 acceptance head 都会装配）。
    args.eval_mode = "adaptive_tridecoding"

    store = ModelStore()
    try:
        decoder = store.new_decoder(args, "mt_bench")
        data = load_task_data("mt_bench", args)
        confidence_data = {"little_draft": [], "draft_target": []}

        print("Starting data collection...")
        for i, datum in enumerate(data):
            if i >= MAX_SAMPLES:
                break

            print(f"Processing sample {i + 1}...")
            prompt = datum["turns"][0]
            input_ids = decoder.tokenizer(prompt, return_tensors="pt").input_ids.to(
                decoder.get_model_input_device(decoder.target_model)
            )

            collect_adaptive_tridecoding(
                decoder,
                input_ids,
                confidence_data,
                transfer_top_k=args.transfer_top_k,
                ntt_ms_edge_cloud=args.ntt_ms_edge_cloud,
                ntt_ms_edge_end=args.ntt_ms_edge_end,
            )

        with open(OUTPUT_FILE, "w") as f:
            json.dump(confidence_data, f, indent=2)

        print(f"Data saved to {OUTPUT_FILE}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
