import os
import sys

sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import random
import shortuuid
from src.utils import seed_everything, parse_arguments
from src.engine import Decoding
from fastchat.model import get_conversation_template
from typing import List, Tuple, Dict, Any

from src.baselines import get_empty_metrics, DecodingMetrics
from src.baselines import Baselines
from eval_mt_bench import get_class_methods

from functools import partial
from few_shot_examples import get_few_shot_prompt
import inspect
from rouge_score import rouge_scorer
import numpy as np

decoding_metrics = get_empty_metrics()


class EvalXSum(Baselines):
    def __init__(self, args):
        super().__init__(args)
        self.load_tokenizer()
        self.load_model()
        self.load_data()

        self.task = "xsum"
        
        if "Llama-2" in str(self.args.draft_model) and "Llama-2" in str(
            self.args.target_model
        ):
            self.model_id = "llama-2-chat"
        elif "Llama-2" in str(self.args.target_model):
            self.model_id = "vicuna"
        elif "vicuna" in str(self.args.draft_model) and "vicuna" in str(
            self.args.target_model
        ):
            self.model_id = "vicuna"
        elif "Llama-3.1" in str(self.args.draft_model) and "Llama-3.1" in str(
            self.args.target_model
        ):
            self.model_id = "llama-3.1"
        elif "llama" in str(self.args.draft_model):
            self.model_id = "vicuna"
        elif "Qwen" in str(self.args.target_model) or "qwen" in str(self.args.target_model):
            self.model_id = "qwen"
        else:
            self.model_id = "vicuna"

    def load_data(self):
        self.color_print(f"Loading XSum data...", 3)
        try:
            import datasets
            dataset = datasets.load_dataset("xsum", split="test")
            self.data = [dict(item) for item in dataset]
            
            # Filter data if needed (e.g. for testing)
            if hasattr(self.args, 'eval_data_num') and self.args.eval_data_num is not None:
                self.data = self.data[:self.args.eval_data_num]

            self.color_print(f"Loaded {len(self.data)} samples.", 3)
        except Exception as e:
            self.color_print(f"Error loading XSum data: {e}", 1)
            self.data = []

    def preprocess(self, input_text):
        few_shot_prompt = get_few_shot_prompt("xsum", self.args.num_shots)
        full_input = few_shot_prompt + "Article: " + input_text

        qs = f"Summarize the following article:\n\n{full_input}"
        if self.model_id == "llama-3.1" or self.model_id == "qwen":
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": qs}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            conv = get_conversation_template(self.model_id)
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " "
        return prompt

    @torch.no_grad()
    def eval(self, total: int | None = 80):
        global decoding_metrics
        decoding = self.get_decoding_method()

        decoding = partial(
            decoding,
            transfer_top_k=self.args.transfer_top_k,
            use_precise_comm_sim=self.args.use_precise,
            use_stochastic_comm=self.args.use_stochastic_comm,
            ntt_ms_edge_cloud=self.args.ntt_ms_edge_cloud,
            ntt_ms_edge_end=self.args.ntt_ms_edge_end,
            use_early_stopping=self.args.use_early_stopping,
        )

        out_path = os.path.join(
            self.args.exp_name, f"{self.args.eval_mode}_xsum.jsonl"
        )
        # Ensure directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_f = open(out_path, "a")
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

        # Warmup
        print(f"Start warm up...")
        n = 5
        for item in tqdm.tqdm(
            self.data,
            total=len(self.data),
            disable=not self.accelerator.is_main_process,
            ncols=50,
        ):
            n -= 1
            if n == 0:
                break
            
            article = str(item["document"])
            # qs = f"Summarize the following article:\n\n{article[:1000]}"
            prompt = self.preprocess(article[:1000])
            if self.model_id == "llama-3.1" or self.model_id == "qwen":
                input_ids = torch.tensor(self.tokenizer([prompt], add_special_tokens=False).input_ids)
            else:
                input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
            
            input_ids = input_ids.to(self.accelerator.device)
            decoding(input_ids)


        total_wall_time = []
        total_num_tokens = []

        loop_index = 0

        print(f"Start evaluation...")
        for item in tqdm.tqdm(
            self.data,
            total=min(len(self.data), total) if total is not None else len(self.data),
            disable=not self.accelerator.is_main_process,
            ncols=50,
        ):
            if total is not None and loop_index >= total:
                break
            
            article = str(item["document"])
            reference = str(item["summary"])
            # qs = f"Summarize the following article:\n\n{article[:4000]}" # Truncate

            loop_index += 1
            if loop_index > min(len(self.data), total) and total is not None:
                break

            # set random seed
            for i in range(self.args.num_samples_per_task):
                while self.seed in self.seed_set:
                    self.seed = random.randint(0, 1000000)
                seed_everything(self.seed)

                prompt = self.preprocess(article[:4000])

                if self.model_id == "llama-3.1" or self.model_id == "qwen":
                    input_ids = torch.tensor(self.tokenizer([prompt], add_special_tokens=False).input_ids)
                else:
                    input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)

                input_ids = input_ids.to(self.accelerator.device)

                torch.cuda.synchronize()
                start_time = time.time()
                try:
                    output_ids = decoding(input_ids)
                    if isinstance(output_ids, tuple):
                        output_ids, metrics = output_ids[:2] # Handle 2 or 3 return values
                        for key in decoding_metrics.keys():
                            if key in metrics and key not in ["little_acceptance_rate", "draft_acceptance_rate"] and hasattr(metrics[key], "__add__"):
                                decoding_metrics[key] += metrics[key]
                            elif isinstance(metrics.get(key), dict):
                                if isinstance(decoding_metrics[key], dict):
                                    for sub_key in metrics[key]:
                                        decoding_metrics[key][sub_key] = decoding_metrics[key].get(sub_key, 0) + metrics[key][sub_key]
                except Exception as e:
                    print(f"Error during decoding: {e}")
                    output_ids = torch.tensor([[self.tokenizer.eos_token_id]]).to(self.accelerator.device)

                torch.cuda.synchronize()
                end_time = time.time()

                # Slice generated part
                if output_ids.shape[1] > input_ids.shape[1]:
                     generated_ids = output_ids[0][input_ids.shape[1]:]
                     output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                else:
                     output_text = ""
                
                # Calculate ROUGE
                scores = scorer.score(reference, output_text)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
                
                wall_time = end_time - start_time
                num_tokens = output_ids.shape[1] - input_ids.shape[1]
                total_wall_time.append(wall_time)
                total_num_tokens.append(num_tokens)

                ans_json = {
                    "id": item.get("id", str(shortuuid.uuid())),
                    "reference": reference,
                    "prediction": output_text,
                    "rouge1": scores['rouge1'].fmeasure,
                    "rouge2": scores['rouge2'].fmeasure,
                    "rougeL": scores['rougeL'].fmeasure,
                    "wall_time": wall_time,
                    "num_tokens": num_tokens
                }
                
                if self.accelerator.is_main_process:
                    out_f.write(json.dumps(ans_json, ensure_ascii=False) + "\n")
                    out_f.flush()

        out_f.close()

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            avg_speed = sum(total_num_tokens) / sum(total_wall_time) if sum(total_wall_time) > 0 else 0
            avg_r1 = sum(rouge1_scores) / len(rouge1_scores) if len(rouge1_scores) > 0 else 0
            avg_r2 = sum(rouge2_scores) / len(rouge2_scores) if len(rouge2_scores) > 0 else 0
            avg_rl = sum(rougeL_scores) / len(rougeL_scores) if len(rougeL_scores) > 0 else 0

            self.color_print(f"Eval Mode: {self.args.eval_mode}", 0)
            self.color_print(f"Average Generation Speed: {avg_speed:.2f} tokens/s", 2)
            self.color_print(f"Average ROUGE-1: {avg_r1:.4f}", 2)
            self.color_print(f"Average ROUGE-2: {avg_r2:.4f}", 2)
            self.color_print(f"Average ROUGE-L: {avg_rl:.4f}", 2)
            
            self.color_print("-------Decoding Metrics-------")
            self.color_print(f"{decoding_metrics}")
            self.color_print("-------Decoding Metrics-------")

            # Save summaries
            eval_result = dict(decoding_metrics)
            eval_result["little_model"] = self.args.little_model
            eval_result["draft_model"] = self.args.draft_model
            eval_result["target_model"] = self.args.target_model
            eval_result["eval_mode"] = self.args.eval_mode
            eval_result["gamma"] = self.args.gamma
            eval_result["gamma1"] = self.args.gamma1
            eval_result["gamma2"] = self.args.gamma2

            decoding_metrics["accuracy"] = {
                "rouge1": avg_r1,
                "rouge2": avg_r2,
                "rougeL": avg_rl
            }
            eval_result["accuracy"] = decoding_metrics["accuracy"]

            if decoding_metrics["wall_time"] != 0:
                decoding_metrics["throughput"] = (
                    decoding_metrics["generated_tokens"]
                    / decoding_metrics["wall_time"]
                )
                eval_result["throughput"] = decoding_metrics["throughput"]

            decoding_metrics_path = os.path.join(
                self.args.exp_name, f"{self.args.eval_mode}_xsum_metrics.json"
            )
            os.makedirs(os.path.dirname(decoding_metrics_path), exist_ok=True)
            with open(decoding_metrics_path, "w") as f:
                json.dump(eval_result, f, indent=4)
            self.color_print(f"Decoding metrics saved to {decoding_metrics_path}", 2)

    def preprocess(self, input_text):
        pass

    def postprocess(self, input_text, output_text):
        pass


if __name__ == "__main__":
    args = parse_arguments()
    alg = EvalXSum(args)
    alg.eval()
