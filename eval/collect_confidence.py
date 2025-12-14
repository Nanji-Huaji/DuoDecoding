import os
import sys
import json
import torch
import argparse
from typing import Tuple, List, Dict

sys.path.append(os.path.join(sys.path[0], "../"))
from src.utils import seed_everything, parse_arguments, max_fn, sample
from src.engine import DecodingMetrics, get_empty_metrics, INT_SIZE
from src.model_gpu import KVCacheModel
from src.communication import CommunicationSimulator, PreciseCommunicationSimulator
from eval_mt_bench import EvalMTBench

class CollectConfidence(EvalMTBench):
    def __init__(self, args):
        super().__init__(args)
        self.confidence_data = {
            "little_draft": [],
            "draft_target": []
        }

    @torch.no_grad()
    def adaptive_tridecoding(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        ntt_ms_edge_cloud=10,
        ntt_ms_edge_end=1,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        max_tokens = prefix.shape[1] + self.args.max_tokens
        little_device = self.little_model.device
        draft_device = self.draft_model.device
        target_device = self.target_model.device
        little_model_cache = KVCacheModel(
            self.little_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        little_model_cache.vocab_size = self.vocab_size
        draft_model_cache = KVCacheModel(
            self.draft_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        draft_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(
            self.target_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        target_model_cache.vocab_size = self.vocab_size

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
            )

        # Metrics tracking
        little_model_forward_times = 0
        draft_model_forward_times = 0
        target_model_forward_times = 0
        total_little_model_generated_tokens = 0
        total_draft_model_generated_tokens = 0
        total_little_model_accepted_tokens = 0
        total_draft_model_accepted_tokens = 0
        wall_time = 0

        idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()  # 用于计算生成token数

        start_event.record(stream=torch.cuda.current_stream())

        comm_simulator.transfer(
            prefix, None, "edge_end"
        )  # 将 prompt 传输到 edge

        while prefix.shape[1] < max_tokens:

            idx += 1

            prefix_len = prefix.shape[1]

            # 第一层 speculative

            x = prefix.clone().to(little_device)
            
            little_probs = [] # Store probabilities for this iteration

            for _ in range(self.args.gamma2):
                adapter = self.small_draft_adapter
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
                    little_probs.append(0.0) # Should not happen if adapter is correct

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
                    little_model_cache._prob_history[:, prefix_len + i - 1, j],
                    "edge_end",
                )
                
                is_accepted = True
                if r > (
                    draft_model_cache._prob_history.to(little_device)[
                        :, prefix_len + i - 1, j
                    ]
                ) / (
                    little_model_cache._prob_history[:, prefix_len + i - 1, j]
                ):
                    comm_simulator.send_reject_message("edge_end")
                    n1 = prefix_len + i - 1
                    is_accepted = False
                    
                    # Record data (Rejected)
                    if i < len(little_probs):
                        self.confidence_data["little_draft"].append({
                            "prob": little_probs[i],
                            "accepted": False
                        })
                    
                    break

                else:
                    little_accepted_this_iter += 1
                    # Record data (Accepted)
                    if i < len(little_probs):
                        self.confidence_data["little_draft"].append({
                            "prob": little_probs[i],
                            "accepted": True
                        })

            total_little_model_accepted_tokens += little_accepted_this_iter

            assert n1 >= prefix_len - 1, f"n {n1}, prefix_len {prefix_len}"
            prefix = x[:, : n1 + 1]

            little_model_cache.rollback(n1 + 1)

            if n1 < prefix_len + actual_gamma2 - 1:
                # reject someone, sample from the pos n1
                rebuild_probs = comm_simulator.rebuild_full_probs(
                    little_model_cache._prob_history[:, n1, : self.vocab_size]
                )
                little_model_cache._prob_history[:, n1, : self.vocab_size] = (
                    rebuild_probs
                )

                comm_simulator.transfer(
                    None,
                    little_model_cache._prob_history[:, n1, : self.vocab_size],
                    "edge_end",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )

                t = sample(
                    max_fn(
                        draft_model_cache._prob_history[
                            :, n1, : self.vocab_size
                        ].to(little_device)
                        - little_model_cache._prob_history[
                            :, n1, : self.vocab_size
                        ]
                    )
                )

                draft_model_cache.rollback(n1 + 1)

            else:
                t = sample(
                    draft_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(little_device)

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

            for _ in range(self.args.gamma1):
                adapter = self.draft_target_adapter
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
            target_model_forward_times += 1
            total_draft_model_generated_tokens += actual_gamma1

            n2: int = (
                prefix_len + new_generated_token.shape[1] + actual_gamma1 - 1
            )
            draft_accepted_this_iter = 0
            for i in range(
                new_generated_token.shape[1] + actual_gamma1,
            ):
                r = torch.rand(1, device=draft_device)
                j = x[:, prefix_len + i]

                # 传输 token id 和 prob 用于 reject sampling
                comm_simulator.transfer(
                    j,
                    draft_model_cache._prob_history[:, prefix_len + i - 1, j],
                    "edge_cloud",
                )

                if r > (
                    target_model_cache._prob_history.to(draft_device)[
                        :, prefix_len + i - 1, j
                    ]
                ) / (draft_model_cache._prob_history[:, prefix_len + i - 1, j]):
                    n2 = prefix_len + i - 1
                    comm_simulator.send_reject_message("edge_cloud")
                    
                    # Record data (Rejected)
                    # Note: i starts from new_generated_token.shape[1] for draft tokens
                    # The first new_generated_token.shape[1] tokens are from previous stage (already accepted by draft model)
                    # We only care about tokens generated by draft model in this stage
                    draft_token_idx = i - new_generated_token.shape[1]
                    if draft_token_idx >= 0 and draft_token_idx < len(draft_probs):
                         self.confidence_data["draft_target"].append({
                            "prob": draft_probs[draft_token_idx],
                            "accepted": False
                        })
                    
                    break
                else:
                    draft_accepted_this_iter += 1
                    
                    # Record data (Accepted)
                    draft_token_idx = i - new_generated_token.shape[1]
                    if draft_token_idx >= 0 and draft_token_idx < len(draft_probs):
                         self.confidence_data["draft_target"].append({
                            "prob": draft_probs[draft_token_idx],
                            "accepted": True
                        })
                        
            total_draft_model_accepted_tokens += draft_accepted_this_iter

            assert (
                n2 >= prefix_len - 1
            ), f"n {n2} should be greater or equal than prefix_len {prefix_len}"
            prefix = x[:, : n2 + 1]
            draft_model_cache.rollback(n2 + 1)
            if n2 <= little_model_cache.current_length:
                little_model_cache.rollback(n2 + 1)
            if n2 < prefix_len + new_generated_token.shape[1] + actual_gamma1 - 1:

                rebuild_probs = comm_simulator.rebuild_full_probs(
                    draft_model_cache._prob_history[:, n2, : self.vocab_size]
                )
                draft_model_cache._prob_history[:, n2, : self.vocab_size] = (
                    rebuild_probs
                )

                comm_simulator.transfer(
                    None,
                    draft_model_cache._prob_history[:, n2, : self.vocab_size],
                    "edge_cloud",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )
                t = sample(
                    max_fn(
                        target_model_cache._prob_history[
                            :, n2, : self.vocab_size
                        ].to(draft_device)
                        - draft_model_cache._prob_history[
                            :, n2, : self.vocab_size
                        ]
                    )
                )
                new_generated_token = prefix[:, prefix_len:]

                target_model_cache.rollback(n2 + 1)

            else:
                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
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
        wall_time += (
            comm_simulator.edge_cloud_comm_time
            + comm_simulator.edge_end_comm_time
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
            comm_simulator.edge_cloud_comm_time
            + comm_simulator.edge_end_comm_time
        )
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times
        return prefix, metrics

if __name__ == "__main__":
    args = parse_arguments()
    
    # Force adaptive_tridecoding
    args.eval_mode = "adaptive_tridecoding"
    
    evaluator = CollectConfidence(args)
    
    # Run evaluation on a few samples
    # We can just run the standard eval loop but break early if needed, or run full
    # For simplicity, we run the standard eval loop which iterates over data
    
    # We need to call eval() but EvalMTBench doesn't implement eval() directly in the snippet I saw?
    # Ah, EvalMTBench inherits from Baselines. Baselines has eval()?
    # Let's check baselines.py again. It has abstract method eval().
    # EvalMTBench must implement it.
    # Wait, I didn't see eval() implementation in EvalMTBench in the snippet.
    # Let me check eval/eval_mt_bench.py content again.
    
    # Assuming EvalMTBench has an eval method or we can just iterate over data and call adaptive_tridecoding
    
    # Let's implement a simple loop here to be safe
    
    print("Starting data collection...")
    for i, datum in enumerate(evaluator.data):
        if i >= 100: # Limit to 100 samples for more data
            break
        
        print(f"Processing sample {i+1}...")
        prompt = datum["turns"][0] # Just take the first turn
        
        input_ids = evaluator.tokenizer(prompt, return_tensors="pt").input_ids.to(evaluator.target_model.device)
        
        evaluator.adaptive_tridecoding(
            input_ids,
            transfer_top_k=args.transfer_top_k,
            ntt_ms_edge_cloud=args.ntt_ms_edge_cloud,
            ntt_ms_edge_end=args.ntt_ms_edge_end
        )
        
    # Save data
    output_file = "confidence_data.json"
    with open(output_file, "w") as f:
        json.dump(evaluator.confidence_data, f, indent=2)
        
    print(f"Data saved to {output_file}")
