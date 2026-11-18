"""SpecBench：按 sub_domain 读本地 JSONL 的多域评测。

任务只负责数据、prompt 构造与逐样本生成；warmup、decoder 组装与结果
落盘由 ``eval.runner`` 负责。JSONL 文件名带 sub_domain（record_name），
metrics 汇总文件仍用任务名。
"""

import os
import sys
from collections import Counter
from statistics import stdev

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tqdm
from fastchat.model import get_conversation_template

from eval.data import load_task_data  # noqa: E402
from eval.eval_humaneval import postprocess_completion  # noqa: E402
from eval.few_shot_examples import get_few_shot_prompt  # noqa: E402
from eval.model_ids import determine_model_id  # noqa: E402
from eval.runner import Evaluation  # noqa: E402
from src.utils import parse_arguments  # noqa: E402


class SpecbenchTask:
    name = "specbench"

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        # D1：model_id 判定单点化（原内联链规则原样入表，task=specbench）
        self.model_id = determine_model_id(
            "specbench", args.draft_model, args.target_model
        )
        # JSONL 记录名带 sub_domain；metrics 汇总文件仍用任务名。
        self.record_name = f"specbench_{args.sub_domain}"

    def load_data(self):
        data = []
        for datum in load_task_data("specbench", self.args):
            datum["input_text"] = self.preprocess(datum["turns"][0])
            # 与 humaneval 的 B34 同因（chat template 自带 BOS），但旧入口
            # 只豁免 draft/target 同为 Llama-3.1 的组合——原样保留，不与
            # _uses_chat_template 的集合统一。
            encode_special_token_flag = not (
                "Llama-3.1" in self.args.draft_model
                and "Llama-3.1" in self.args.target_model
            )
            input_ids = self.tokenizer.encode(
                datum["input_text"], add_special_tokens=encode_special_token_flag
            )
            datum["input_ids"] = torch.tensor(input_ids).unsqueeze(0)
            data.append(datum)
        return data

    def preprocess(self, input_text):
        task_map = {
            "math_reasoning": "gsm8k",
            "summarization": "summarization",
            "translation": "translation",
            "qa": "gsm8k",
        }
        task = task_map.get(self.args.sub_domain, "gsm8k")
        few_shot_prompt = get_few_shot_prompt(task, self.args.num_shots)

        if self.model_id == "llama-3.1" or self.model_id == "qwen":
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": few_shot_prompt + input_text},
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
                    "content": "You are a helpful assistant.\n"
                    + few_shot_prompt
                    + input_text,
                }
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        elif "vicuna" in self.model_id:
            text = (few_shot_prompt + input_text).strip()
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            conv.stop_str = "</s>"
            prompt = conv.get_prompt()
            return prompt
        else:
            if self.args.sub_domain == "summarization":
                text = input_text.strip()
                prompt = few_shot_prompt + text + " TL;DR: "
            elif self.args.sub_domain == "translation":
                prompt = (
                    few_shot_prompt
                    + "Translate German to English. German: "
                    + input_text[len("Translate German to English: ") :]
                    + " English: "
                )
            elif self.args.sub_domain == "math_reasoning":
                prompt = (
                    f"{few_shot_prompt}{input_text} Let's think step by step.\nStep 1:"
                )
            elif self.args.sub_domain == "rag":
                prompt = few_shot_prompt + input_text
            else:
                # 其他子域使用简单格式，避免对话标记导致的格式冲突
                prompt = (few_shot_prompt + input_text).strip()
        return prompt

    def _warmup(self, run, data):
        count = min(10, len(data))
        if count <= 0:
            return
        print("Start warm up...")
        for datum in tqdm.tqdm(
            data[:count], total=count, disable=not run.is_main_process, ncols=50
        ):
            run.generate(datum["input_ids"], record=False, use_early_stopping=False)

    def run(self, run, data):
        wall_times = {"time": [], "num_tokens": []}
        # Only retain the scalars needed for mean_*; token histories are already
        # aggregated by the runner and can be much larger than the samples.
        sums, counts = Counter(), Counter()
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

            for datum in tqdm.tqdm(
                data, total=len(data), disable=not run.is_main_process, ncols=50
            ):
                # Legacy SpecBench measures the full token budget even when the
                # general CLI early-stop flag is present.
                output_ids, sample_metrics, elapsed = run.generate(
                    datum["input_ids"], use_early_stopping=False
                )
                if run.is_main_process:
                    for key, value in sample_metrics.items():
                        if isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        ):
                            sums[key] += value
                            counts[key] += 1
                    new_tokens = int(output_ids.shape[1] - datum["input_ids"].shape[1])
                    wall_times["time"].append(elapsed)
                    wall_times["num_tokens"].append(new_tokens)
                    output = postprocess_completion(
                        self.tokenizer,
                        datum["input_text"],
                        self.tokenizer.decode(output_ids[0, :]),
                    )
                    run.write(
                        {
                            "question_id": datum["question_id"],
                            "time": elapsed,
                            "new_tokens": new_tokens,
                            "completion": output,
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

        if not run.is_main_process:
            return {}
        speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"])
        speeds = [
            n / t
            for n, t in zip(wall_times["num_tokens"], wall_times["time"], strict=True)
        ]
        speed_std = stdev(speeds) if len(speeds) > 1 else 0.0
        print(f"generate speed (tokens / second):  {speed:.2f} with std {speed_std}")

        # 逐样本 metrics 汇总（mean_*），随 runner 一并写入
        # {eval_mode}_specbench_metrics.json。
        summary = {
            "speed_tokens_per_second": speed,
            "speed_std": speed_std,
            "num_samples": len(wall_times["time"]),
        }
        summary.update(
            {f"mean_{key}": total / counts[key] for key, total in sums.items()}
        )

        try:
            print(
                "Mean accepted tokens: "
                f"{sum(run.decoder.num_acc_tokens) / len(run.decoder.num_acc_tokens)}"
            )
        except Exception:
            pass
        return summary


class EvalSpecbench(Evaluation):
    task_type = SpecbenchTask


if __name__ == "__main__":
    args = parse_arguments()
    EvalSpecbench(args).eval()
