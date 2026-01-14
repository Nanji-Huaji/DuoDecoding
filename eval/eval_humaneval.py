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
        self.load_data()
        self.load_model()

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
        if "vicuna" in self.args.target_model:
            text = input_text.strip()
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            conv.stop_str = "</s>"
            prompt = conv.get_prompt()
        else:
            return input_text.strip()
        return prompt

    def postprocess(self, input_text, output_text):
        if output_text.startswith(self.tokenizer.bos_token):
            generation = output_text[
                len(input_text) + len(self.tokenizer.bos_token) + 1 :
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
            self.tokenizer.eos_token,
        ]
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
        elif (
            self.args.eval_mode
            == "speculative_decoding_with_bandwidth_full_prob"
        ):
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
            print(f"Unknown eval mode: {self.args.eval_mode}")
            raise NotImplementedError

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

            self.color_print("-------Decoding Metrics-------")
            self.color_print(f"{decoding_metrics}")
            self.color_print("-------Decoding Metrics-------")

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
