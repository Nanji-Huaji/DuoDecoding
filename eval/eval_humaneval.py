import os
import sys

sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import ipdb
import random
import multiprocessing
from datasets import load_dataset
from src.utils import seed_everything, parse_arguments
from src.engine import Decoding
from collections import Counter
from fastchat.model import get_conversation_template

from src.engine import DecodingMetrics, get_empty_metrics

from src.baselines import Baselines

from eval_mt_bench import get_class_methods

from functools import partial
from few_shot_examples import get_few_shot_prompt

decoding_metrics = get_empty_metrics()

def check_correctness_worker(check_program, result_queue):
    try:
        exec(check_program, {})
        result_queue.put("passed")
    except Exception as e:
        result_queue.put(f"failed")

def check_correctness(completion, test_code, entry_point, timeout=3.0):
    check_program = completion + "\n" + test_code + "\n" + f"check({entry_point})"
    
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=check_correctness_worker, args=(check_program, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return 0 # Timeout
    
    if not queue.empty():
        result = queue.get()
        return 1 if result == "passed" else 0
    return 0 # Failed/Empty

class EvalHumaneval(Baselines):
    def __init__(self, args):
        super().__init__(args)

        # load relative resources
        self.load_tokenizer()
        self.load_model()

        self.task = "humaneval"

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
        elif "gemma" in str(self.args.target_model) or "gemma" in str(self.args.draft_model):
            self.model_id = "gemma"
        else:
            self.model_id = "vicuna"
            
        self.load_data()

        self.draft_time = []
        self.target_time = []
        self.acc_num = []

    def load_data(self):
        # * load evaluation data
        self.color_print(f"Loading HumanEval data from HuggingFace...", 3)
        data = []
        try:
            hf_data = load_dataset("openai_humaneval", split="test")
        except Exception as e:
            self.color_print(f"Failed to load dataset from Hugging Face: {e}", 1)
            raise e

        for datum_item in hf_data:
            datum = dict(datum_item)
            datum["input_text"] = self.preprocess(datum["prompt"])
            encode_special_token_flag = not (
                "Llama-3.1" in self.args.draft_model
                and "Llama-3.1" in self.args.target_model
            )
            input_ids = self.tokenizer.encode(
                datum["input_text"], add_special_tokens=encode_special_token_flag
            )
            datum["input_ids"] = torch.tensor(input_ids).unsqueeze(0)
            data.append(datum)
            
        self.data = data
        self.color_print(f"Loaded {len(self.data)} items from Hugging Face openai_humaneval", 2)

    def preprocess(self, input_text):
        few_shot_prompt = get_few_shot_prompt("humaneval", self.args.num_shots)
        full_input = few_shot_prompt + input_text

        if self.model_id == "llama-3.1" or self.model_id == "qwen" or self.model_id == "gemma":
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Please complete the following python code."},
                {"role": "user", "content": full_input}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        elif "vicuna" in self.model_id:
            text = full_input.strip()
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            conv.stop_str = "</s>"
            prompt = conv.get_prompt()
            return prompt
        else:
            return full_input.strip()

    def postprocess(self, input_text, output_text):
        bos_token = self.tokenizer.bos_token
        if bos_token and output_text.startswith(bos_token):
            generation = output_text[
                len(input_text) + len(bos_token) + 1 :
            ]  # tokenizer will add a '<s> ' at the beginning of the text.
        else:
            generation = output_text[len(input_text) :]
        stop_words = [
            "\nclass",
            "\ndef",
            "\n#",
            "\n@",
            "\nprint",
            "\nif",
            "\n```",
        ]
        if self.tokenizer.eos_token:
            stop_words.append(self.tokenizer.eos_token)

        for stop_word in stop_words:
            if stop_word in generation:
                next_line = generation.index(stop_word)
                generation = generation[:next_line].strip()
        output_text = input_text + "\n    " + generation
        output_text = output_text.replace("\t", "    ")

        return output_text

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
            stop_sequences=["\nclass", "\ndef", "\nif", "\nprint"],
        )

        out_path = os.path.join(
            self.args.exp_name, f"{self.args.eval_mode}_humaneval.jsonl"
        )
        out_f = open(out_path, "a")
        wall_times = {"time": [], "num_tokens": [], "ttft": []}
        for _ in range(self.args.num_samples_per_task):
            # set random seed. Ensure each experiment runs with a unique random seed.
            while self.seed in self.seed_set:
                self.seed = random.randint(0, 1000000)
            seed_everything(self.seed)
            self.seed_set.add(self.seed)

            # warm up
            n = 10
            all_time = 0
            print(f"Start warm up...")
            for datum in tqdm.tqdm(
                self.data,
                total=len(self.data),
                disable=not self.accelerator.is_main_process,
                ncols=50,
            ):
                input_ids = datum["input_ids"]
                if self.args.eval_mode == "para_sd_hybrid":
                    generate_ids = decoding(input_ids)
                else:
                    generate_ids = decoding(input_ids)
                if isinstance(generate_ids, tuple):
                    generate_ids, _ = generate_ids
                t = 0
                all_time += max(0, t)
                n = n - 1
                if n == 0:
                    break

            self.draft_forward_times = 0
            self.target_forward_times = 0
            self.num_acc_tokens = []
            self.prob_with_flag = []  # draft每个token的prob与他是否被接收

            eval_data = self.data
            if total is not None:
                eval_data = self.data[:total]

            for datum in tqdm.tqdm(
                eval_data,
                disable=not self.accelerator.is_main_process,
                ncols=50,
            ):
                input_ids = datum["input_ids"]
                torch.cuda.synchronize()
                start_time = time.time()
                generate_ids = decoding(input_ids)
                if isinstance(generate_ids, tuple):
                    generate_ids, metrics = generate_ids
                    for key in decoding_metrics.keys():
                        if key not in [""] and hasattr(decoding_metrics[key], "__add__"):
                            decoding_metrics[key] += metrics[key]
                t = 0
                torch.cuda.synchronize()
                end_time = time.time()
                if self.accelerator.is_main_process:
                    wall_times["time"].append(end_time - start_time)
                    wall_times["num_tokens"].append(
                        generate_ids.shape[1] - input_ids.shape[1]
                    )
                    output = self.postprocess(
                        datum["input_text"], self.tokenizer.decode(generate_ids[0, :])
                    )
                    passed = check_correctness(output, datum["test"], datum["entry_point"])
                    self.acc_num.append(passed)
                    out_f.write(
                        json.dumps(
                            {
                                "task_id": datum["task_id"],
                                "time": end_time - start_time,
                                "new_tokens": generate_ids.shape[1]
                                - input_ids.shape[1],
                                "completion": output,
                                "passed": passed,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                out_f.flush()
            # print(all_time)
        out_f.close()

        self.color_print(f"current eval mode: {self.args.eval_mode}", 0)
        self.color_print(f"draft model forward times: {self.draft_forward_times}", 2)

        self.accelerator.wait_for_everyone()

        if (
            self.accelerator.num_processes == 1 and self.accelerator.is_main_process
        ) or (
            self.accelerator.num_processes == 2 and not self.accelerator.is_main_process
        ):
            print(
                f"\033[92mtarget model forward times: {self.target_forward_times}\033[0m"
            )

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            if len(wall_times["time"]) > 0 and sum(wall_times["time"]) > 0:
                speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"])
                speed_std = (
                    (
                        torch.tensor(wall_times["num_tokens"])
                        / torch.tensor(wall_times["time"])
                    )
                    .std()
                    .item()
                )
                self.color_print(
                    f"generate speed (tokens / second):  {speed:.2f} with std {speed_std}",
                    2,
                )
            else:
                 self.color_print("No generation time recorded.", 3)

            if len(self.acc_num) > 0:
                decoding_metrics["accuracy"] = sum(self.acc_num) / len(self.acc_num)

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
                self.args.exp_name, f"{self.args.eval_mode}_humaneval_metrics.json"
            )
            os.makedirs(os.path.dirname(decoding_metrics_path), exist_ok=True)
            with open(decoding_metrics_path, "w") as f:
                json.dump(eval_result, f, indent=4)
            self.color_print(f"Decoding metrics saved to {decoding_metrics_path}", 2)

        # if self.accelerator.is_main_process:
        #     try:
        #         self.color_print(
        #             f"Mean accepted tokens: {sum(self.num_acc_tokens) / len(self.num_acc_tokens)}"
        #         )
        #     except:
        #         pass


if __name__ == "__main__":
    args = parse_arguments()
    alg = EvalHumaneval(args)
    alg.eval()
