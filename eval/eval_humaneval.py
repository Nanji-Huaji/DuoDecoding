"""HumanEval：代码补全评测，子进程隔离执行测试用例判分。

任务只负责数据、prompt 构造与逐样本判分（含超时熔断的 correctness
helper，安全语义原样保留）；warmup、decoder 组装与结果落盘由
``eval.runner`` 负责。
"""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing

import torch
import tqdm
from fastchat.model import get_conversation_template

from eval.data import load_task_data  # noqa: E402
from eval.few_shot_examples import get_few_shot_prompt  # noqa: E402
from eval.model_ids import determine_model_id  # noqa: E402
from eval.runner import Evaluation  # noqa: E402
from src.utils import parse_arguments  # noqa: E402


def postprocess_completion(tokenizer, input_text, output_text):
    """从整段解码文本截出补全（HumanEval 与 SpecBench 的同一口径）。"""
    bos_token = tokenizer.bos_token
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
    if tokenizer.eos_token:
        stop_words.append(tokenizer.eos_token)

    for stop_word in stop_words:
        if stop_word in generation:
            next_line = generation.index(stop_word)
            generation = generation[:next_line].strip()
    output_text = input_text + "\n    " + generation
    output_text = output_text.replace("\t", "    ")
    return output_text


def check_correctness_worker(check_program, result_queue):
    try:
        exec(check_program, {})
        result_queue.put("passed")
    except Exception:
        result_queue.put("failed")


def check_correctness(completion, test_code, entry_point, timeout=3.0):
    check_program = completion + "\n" + test_code + "\n" + f"check({entry_point})"

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=check_correctness_worker, args=(check_program, queue)
    )
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return 0  # Timeout

    if not queue.empty():
        result = queue.get()
        return 1 if result == "passed" else 0
    return 0  # Failed/Empty


class HumanevalTask:
    name = "humaneval"

    # B34：preprocess 里会走 chat template（模板自带 BOS）的 model_id 集合。
    # 该集合必须与下面 preprocess 的模板分支保持一致。
    CHAT_TEMPLATE_MODEL_IDS = ("llama-3.1", "llama-3.2", "llama-3", "qwen", "gemma")

    # 旧入口的解码停串：warmup 与评测同样绑定。
    STOP_SEQUENCES = ["\nclass", "\ndef", "\nif", "\nprint"]

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        # D1：model_id 判定单点化（原内联链规则原样入表，task=humaneval）
        self.model_id = determine_model_id(
            "humaneval", args.draft_model, args.target_model
        )
        self.acc_num = []

    def _uses_chat_template(self) -> bool:
        return self.model_id in self.CHAT_TEMPLATE_MODEL_IDS

    def load_data(self):
        data = []
        for datum_item in load_task_data("humaneval", self.args):
            datum = dict(datum_item)
            datum["input_text"] = self.preprocess(datum["prompt"])
            # B34：走 chat template 的模型族，模板自带 BOS，encode 时必须关
            # add_special_tokens，否则 Llama-3/3.2/gemma 会双 BOS。
            input_ids = self.tokenizer.encode(
                datum["input_text"], add_special_tokens=not self._uses_chat_template()
            )
            datum["input_ids"] = torch.tensor(input_ids).unsqueeze(0)
            data.append(datum)
        return data

    def preprocess(self, input_text):
        few_shot_prompt = get_few_shot_prompt("humaneval", self.args.num_shots)
        full_input = few_shot_prompt + input_text

        if self.model_id in self.CHAT_TEMPLATE_MODEL_IDS and self.model_id != "gemma":
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. "
                    "Please complete the following python code.",
                },
                {"role": "user", "content": full_input},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        elif self.model_id == "gemma":
            messages = [
                {
                    "role": "user",
                    "content": "Please complete the following python code.\n"
                    + full_input,
                }
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
            conv.append_message(conv.roles[1], None)  # type: ignore
            conv.stop_str = "</s>"
            prompt = conv.get_prompt()
            return prompt
        else:
            return full_input.strip()

    def _warmup(self, run, data):
        count = min(10, len(data))
        if count <= 0:
            return
        print("Start warm up...")
        for datum in tqdm.tqdm(
            data[:count], total=count, disable=not run.is_main_process, ncols=50
        ):
            run.generate(
                datum["input_ids"], record=False, stop_sequences=self.STOP_SEQUENCES
            )

    def run(self, run, data):
        wall_times = {"time": [], "num_tokens": []}
        num_samples = int(getattr(run.args, "num_samples_per_task", 1) or 1)
        for _ in range(num_samples):
            # next_seed 同时负责 seed_everything 与跨样本种子去重
            # （旧实现的 seed_set 循环）。
            run.next_seed()

            self._warmup(run, data)

            # 每轮重复重置运行计数器，只统计末轮——旧口径。
            decoder = run.decoder
            decoder.draft_forward_times = 0
            decoder.target_forward_times = 0
            decoder.num_acc_tokens = []

            for datum in tqdm.tqdm(data, disable=not run.is_main_process, ncols=50):
                output_ids, _, elapsed = run.generate(
                    datum["input_ids"], stop_sequences=self.STOP_SEQUENCES
                )
                if run.is_main_process:
                    new_tokens = int(output_ids.shape[1] - datum["input_ids"].shape[1])
                    wall_times["time"].append(elapsed)
                    wall_times["num_tokens"].append(new_tokens)
                    output = postprocess_completion(
                        self.tokenizer,
                        datum["input_text"],
                        self.tokenizer.decode(output_ids[0, :]),
                    )
                    # Timing jobs need the decoded text, not execution of an
                    # untrusted completion on the machine holding the weights.
                    passed = None
                    if not getattr(run.args, "humaneval_generate_only", False):
                        passed = check_correctness(
                            output, datum["test"], datum["entry_point"]
                        )
                        self.acc_num.append(passed)
                    run.write(
                        {
                            "task_id": datum["task_id"],
                            "time": elapsed,
                            "new_tokens": new_tokens,
                            "completion": output,
                            "passed": passed,
                        }
                    )

        print(f"current eval mode: {run.args.eval_mode}")
        print(f"draft model forward times: {run.decoder.draft_forward_times}")

        accelerator = run.decoder.accelerator
        num_processes = getattr(accelerator, "num_processes", 1)
        if (num_processes == 1 and run.is_main_process) or (
            num_processes == 2 and not run.is_main_process
        ):
            print(
                f"\033[92mtarget model forward times: "
                f"{run.decoder.target_forward_times}\033[0m"
            )

        if run.is_main_process:
            if wall_times["time"] and sum(wall_times["time"]) > 0:
                speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"])
                speed_std = (
                    (
                        torch.tensor(wall_times["num_tokens"])
                        / torch.tensor(wall_times["time"])
                    )
                    .std()
                    .item()
                )
                print(
                    f"generate speed (tokens / second):  {speed:.2f} "
                    f"with std {speed_std}"
                )
            else:
                print("No generation time recorded.")

            if self.acc_num:
                run.metrics["accuracy"] = sum(self.acc_num) / len(self.acc_num)
        return {}


class EvalHumaneval(Evaluation):
    task_type = HumanevalTask


if __name__ == "__main__":
    args = parse_arguments()
    EvalHumaneval(args).eval(args.eval_data_num)
