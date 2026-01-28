import os
import sys

sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import re
import numpy as np

from src.utils import seed_everything, parse_arguments
from src.baselines import Baselines, get_empty_metrics
from fastchat.model import get_conversation_template
from datasets import load_dataset
from eval_mt_bench import get_class_methods

from functools import partial
from eval.few_shot_examples import get_few_shot_prompt

decoding_metrics = get_empty_metrics()

INVALID_ANS = "[invalid]"

def extract_answer_from_gold(completion):
    if completion.find("####") >= 0:
        ans = completion.split("####")[1].strip()
        return ans.replace(",", "")
    else:
        return INVALID_ANS

def extract_answer_from_output(completion):
    text = completion.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', text)]
    if not pred:
        return INVALID_ANS
    return pred[-1]

class EvalGSM8K(Baselines):
    def __init__(self, args):
        super().__init__(args)
        self.load_tokenizer()
        self.load_model()
        self.load_data()

        self.task = "gsm8k"
        
        # Determine model_id for chat template
        if "Llama-2" in str(self.args.draft_model) and "Llama-2" in str(self.args.target_model):
            self.model_id = "llama-2-chat"
        elif "Llama-2" in str(self.args.target_model):
            self.model_id = "vicuna"
        elif "vicuna" in str(self.args.draft_model) and "vicuna" in str(self.args.target_model):
            self.model_id = "vicuna"
        elif "Llama-3.1" in str(self.args.draft_model) and "Llama-3.1" in str(self.args.target_model):
            self.model_id = "llama-3.1"
        elif "llama" in str(self.args.draft_model):
            self.model_id = "vicuna"
        elif "Qwen" in str(self.args.target_model) or "qwen" in str(self.args.target_model):
            self.model_id = "qwen"
        else:
            self.model_id = "vicuna"

        self.acc_list = []

    def load_data(self):
        self.color_print(f"Loading GSM8K data...", 3)
        try:
            dataset = load_dataset("gsm8k", "main", split="test")
            self.data = [dict(item) for item in dataset]
            self.color_print(f"Loaded {len(self.data)} samples from GSM8K.", 2)
        except Exception as e:
            self.color_print(f"Error loading GSM8K data: {e}", 1)
            self.data = []

    def preprocess(self, input_text):
        few_shot_prompt = get_few_shot_prompt("gsm8k", self.args.num_shots)
        full_input = few_shot_prompt + "Question: " + input_text

        if self.model_id == "llama-3.1" or self.model_id == "qwen":
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step and end your answer with 'The answer is <number>'."},
                {"role": "user", "content": full_input}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            conv = get_conversation_template(self.model_id)
            conv.append_message(conv.roles[0], f"{full_input}\nAnswer:")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        return prompt

    def is_correct(self, model_completion, gt_completion):
        gt_ans = extract_answer_from_gold(gt_completion)
        if gt_ans == INVALID_ANS:
            return False
            
        # Try to find the ground truth number in the model's output specifically at the end
        # But GSM8K eval often just looks for the last number
        pred_ans = extract_answer_from_output(model_completion)
        
        return pred_ans == gt_ans

    def postprocess(self, output_text):
        return output_text.strip()

    @torch.no_grad()
    def eval(self, total: int | None = 80):
        global decoding_metrics
        # Select decoding method
        decoding = self.get_decoding_method()

        decoding = partial(
            decoding,
            transfer_top_k=self.args.transfer_top_k,
            use_precise_comm_sim=self.args.use_precise,
            use_stochastic_comm=self.args.use_stochastic_comm,
            ntt_ms_edge_cloud=self.args.ntt_ms_edge_cloud,
            ntt_ms_edge_end=self.args.ntt_ms_edge_end,
        )

        out_path = os.path.join(
            self.args.exp_name, f"{self.args.eval_mode}_gsm8k.jsonl"
        )
        out_f = open(out_path, "a")
        wall_times = {"time": [], "num_tokens": []}

        # Warmup
        print(f"Start warm up...")
        # (Simplified warmup to save tokens/time if needed, or keep standard)
        
        eval_data = self.data
        if total is not None:
            eval_data = self.data[:total]

        for datum in tqdm.tqdm(
            eval_data,
            disable=not self.accelerator.is_main_process,
            ncols=50,
        ):
            question = datum["question"]
            answer_gt = datum["answer"]
            
            prompt = self.preprocess(question)
            
            if self.model_id == "llama-3.1" or self.model_id == "qwen":
                input_ids = torch.tensor(
                    self.tokenizer([prompt], add_special_tokens=False).input_ids
                )
            else:
                input_ids = torch.tensor(
                    self.tokenizer.encode(prompt, add_special_tokens=True)
                ).unsqueeze(0)
            
            # Run Decoding
            torch.cuda.synchronize()
            start_time = time.time()
            
            output_ids = decoding(input_ids)
            if isinstance(output_ids, tuple):
                output_ids, metrics = output_ids
                # Merge metrics
                for key in decoding_metrics.keys():
                    if key in metrics and hasattr(metrics[key], "__add__"):
                        decoding_metrics[key] += metrics[key]
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Decode output
            output_text = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Check correctness
            clean_output = output_text.strip()
            is_correct = self.is_correct(clean_output, answer_gt)
            self.acc_list.append(1 if is_correct else 0)

            # Record stats
            if self.accelerator.is_main_process:
                new_tokens = output_ids.shape[1] - input_ids.shape[1]
                wall_times["time"].append(end_time - start_time)
                wall_times["num_tokens"].append(new_tokens)
                
                result_json = {
                    "question": question,
                    "generated_answer": clean_output,
                    "ground_truth": answer_gt,
                    "correct": is_correct,
                    "time": end_time - start_time,
                    "new_tokens": new_tokens
                }
                out_f.write(json.dumps(result_json, ensure_ascii=False) + "\n")
                out_f.flush()

        out_f.close()
        
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            # Calculate Speed
            if len(wall_times["time"]) > 0 and sum(wall_times["time"]) > 0:
                speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"])
                speed_std = np.std(np.array(wall_times["num_tokens"]) / np.array(wall_times["time"]))
                self.color_print(
                    f"generate speed (tokens / second):  {speed:.2f} with std {speed_std:.2f}",
                    2,
                )
            else:
                 self.color_print("No generation time recorded.", 3)

            # Calculate Accuracy
            if len(self.acc_list) > 0:
                acc = sum(self.acc_list) / len(self.acc_list)
                decoding_metrics["accuracy"] = acc
                self.color_print(f"GSM8K Accuracy: {acc:.4f}", 2)
            
            if decoding_metrics["wall_time"] != 0:
                decoding_metrics["throughput"] = (
                    decoding_metrics["generated_tokens"]
                    / decoding_metrics["wall_time"]
                )
            else:
                decoding_metrics["throughput"] = 0.0

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

            eval_result["throughput"] = decoding_metrics["throughput"]

            decoding_metrics_path = os.path.join(
                self.args.exp_name, f"{self.args.eval_mode}_gsm8k_metrics.json"
            )
            os.makedirs(os.path.dirname(decoding_metrics_path), exist_ok=True)
            with open(decoding_metrics_path, "w") as f:
                json.dump(eval_result, f, indent=4)
            self.color_print(f"Decoding metrics saved to {decoding_metrics_path}", 2)

if __name__ == "__main__":
    args = parse_arguments()
    alg = EvalGSM8K(args)
    alg.eval()
