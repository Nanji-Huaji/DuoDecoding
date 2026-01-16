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
        if self.model_id == "llama-3.1":
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step and end your answer with 'The answer is <number>'."},
                {"role": "user", "content": input_text}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            conv = get_conversation_template(self.model_id)
            conv.append_message(conv.roles[0], f"{input_text}\nAnswer:")
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
        if self.args.eval_mode == "small" or self.args.eval_mode == "large":
            decoding = self.autoregressive_sampling
        elif self.args.eval_mode == "sd":
            decoding = self.speculative_decoding
        elif self.args.eval_mode == "para_sd":
            decoding = self.parallel_speculative_decoding
        elif self.args.eval_mode == "duodec":
            decoding = self.duodecoding
        elif self.args.eval_mode == "lade":
            decoding = self.lookahead_forward
        elif self.args.eval_mode == "rest":
            decoding = self.rest_forward
        elif self.args.eval_mode == "tridecoding":
            decoding = self.tridecoding
        elif self.args.eval_mode == "tridecoding_with_bandwidth":
            decoding = self.tridecoding_with_bandwidth
        elif self.args.eval_mode == "uncertainty_decoding":
            decoding = self.uncertainty_decoding
        elif self.args.eval_mode == "speculative_decoding_with_bandwidth":
            decoding = self.speculative_decoding_with_bandwidth
        elif self.args.eval_mode == "speculative_decoding_with_bandwidth_full_prob":
            decoding = self.speculative_decoding_with_bandwidth_full_prob
        elif self.args.eval_mode in get_class_methods(Baselines):
            decoding = getattr(self, self.args.eval_mode)
        else:
            try:
                methods = getattr(self, self.args.eval_mode)
                if callable(methods):
                    decoding = methods
                else:
                    self.color_print("Unknown eval mode", 3)
                    raise NotImplementedError
            except AttributeError:
                self.color_print("Unknown eval mode", 3)
                raise NotImplementedError

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
            
            if self.model_id == "llama-3.1":
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
            
            self.color_print("-------Decoding Metrics-------")
            self.color_print(f"{decoding_metrics}")
            self.color_print("-------Decoding Metrics-------")

if __name__ == "__main__":
    args = parse_arguments()
    alg = EvalGSM8K(args)
    alg.eval()
