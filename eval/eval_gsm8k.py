"""GSM8K：few-shot 数学题评测，按 ``####`` 抽取最终答案算 accuracy。

任务只负责数据、prompt 构造与逐样本判分；warmup（旧入口本就没有）、
decoder 组装与结果落盘由 ``eval.runner`` 负责。
"""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re

import numpy as np
import torch
import tqdm

from eval.data import load_task_data  # noqa: E402
from eval.few_shot_examples import get_few_shot_prompt  # noqa: E402
from eval.model_ids import determine_model_id  # noqa: E402
from eval.runner import Evaluation  # noqa: E402
from src.utils import parse_arguments  # noqa: E402

INVALID_ANS = "[invalid]"


def extract_answer_from_gold(completion):
    if completion.find("####") >= 0:
        ans = completion.split("####")[1].strip()
        return ans.replace(",", "")
    else:
        return INVALID_ANS


def extract_answer_from_output(completion):
    """改进的答案提取逻辑，按优先级尝试多种方法"""
    # 方法 1: 寻找 #### 标记（官方 GSM8K 格式）
    if "####" in completion:
        try:
            answer = completion.split("####")[1].strip()
            answer = answer.split("\n")[0].strip()  # 取第一行
            answer = answer.replace(",", "").replace("$", "")
            # 提取数字
            numbers = re.findall(r"-?\d+\.?\d*", answer)
            if numbers:
                return numbers[0]
        except Exception:
            pass

    # 方法 2: 寻找 "The answer is" 格式
    answer_patterns = [
        r"[Tt]he answer is:?\s*([\-\$]?[\d,\.]+)",
        r"[Aa]nswer:?\s*([\-\$]?[\d,\.]+)",
        r"^####\s*([\-\$]?[\d,\.]+)",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, completion)
        if match:
            answer = match.group(1).replace(",", "").replace("$", "")
            return answer

    # 方法 3: 提取最后一个数字（作为后备）
    text = completion.replace(",", "").replace("$", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return INVALID_ANS


class Gsm8kTask:
    name = "gsm8k"

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.model_id = determine_model_id("gsm8k", args.draft_model, args.target_model)
        self.acc_list = []

    def load_data(self):
        return load_task_data("gsm8k", self.args)

    def preprocess(self, input_text):
        few_shot_prompt = get_few_shot_prompt("gsm8k", self.args.num_shots)
        full_input = few_shot_prompt + "Question: " + input_text

        # 使用与官方 GSM8K CoT 一致的提示格式
        if self.model_id in ["llama-3.1", "llama-3.2", "llama-3", "qwen"]:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": full_input
                    + "\n\nPlease solve this step by step and put your "
                    "final answer after #### at the end.",
                },
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif self.model_id == "gemma":
            messages = [
                {
                    "role": "user",
                    "content": full_input
                    + "\n\nPlease solve this step by step and put your "
                    "final answer after #### at the end.",
                }
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Llama-2 和其他模型：使用简单的文本续写格式，避免对话标记冲突
            # 直接续写Few-shot示例，不添加额外的对话结构
            prompt = full_input + "\nAnswer:"
        return prompt

    def is_correct(self, model_completion, gt_completion):
        gt_ans = extract_answer_from_gold(gt_completion)
        model_ans = extract_answer_from_output(model_completion)
        if model_ans == INVALID_ANS:
            return False
        try:
            if abs(float(model_ans) - float(gt_ans)) < 1e-6:
                return True
        except Exception:
            pass
        return model_ans.strip() == gt_ans.strip()

    def _encode(self, prompt):
        # 走 chat template 的模型族：模板自带 BOS，encode 关 add_special_tokens。
        if self.model_id in ("llama-3.1", "llama-3.2", "llama-3", "qwen"):
            return torch.tensor(
                self.tokenizer([prompt], add_special_tokens=False).input_ids
            )
        return torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=True)
        ).unsqueeze(0)

    def run(self, run, data):
        wall_times = {"time": [], "num_tokens": []}
        # `--num_samples_per_task K`：同一 prompt 重复采样 K 次（分布检验）。
        # 旧口径用 seed + rep*7919 派生种子（可复现），不走 next_seed 的
        # 随机去重方案，保持分布检验脚本的确定性。
        n_rep = max(1, int(getattr(run.args, "num_samples_per_task", 1) or 1))
        for datum in tqdm.tqdm(data, disable=not run.is_main_process, ncols=50):
            question = datum["question"]
            answer_gt = datum["answer"]
            prompt = self.preprocess(question)
            input_ids = self._encode(prompt)

            for rep in range(n_rep):
                if n_rep > 1:
                    torch.manual_seed(int(run.args.seed) + rep * 7919)

                output_ids, _, elapsed = run.generate(input_ids)
                output_text = self.tokenizer.decode(
                    output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
                )

                clean_output = output_text.strip()
                correct = self.is_correct(clean_output, answer_gt)
                self.acc_list.append(1 if correct else 0)

                if run.is_main_process:
                    new_tokens = int(output_ids.shape[1] - input_ids.shape[1])
                    wall_times["time"].append(elapsed)
                    wall_times["num_tokens"].append(new_tokens)
                    # 逐 token 保存生成 id：仅存文本无法做逐 token 恒等检验。
                    run.write(
                        {
                            "question": question,
                            "gen_ids": output_ids[0][input_ids.shape[1] :].tolist(),
                            "prompt_len": int(input_ids.shape[1]),
                            "sample_idx": rep,
                            "generated_answer": clean_output,
                            "ground_truth": answer_gt,
                            "correct": correct,
                            "time": elapsed,
                            "new_tokens": new_tokens,
                        }
                    )

        if run.is_main_process:
            if wall_times["time"] and sum(wall_times["time"]) > 0:
                speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"])
                speed_std = np.std(
                    np.array(wall_times["num_tokens"]) / np.array(wall_times["time"])
                )
                print(
                    f"generate speed (tokens / second):  {speed:.2f} "
                    f"with std {speed_std:.2f}"
                )
            else:
                print("No generation time recorded.")

        if self.acc_list:
            acc = sum(self.acc_list) / len(self.acc_list)
            run.metrics["accuracy"] = acc
            if run.is_main_process:
                print(f"GSM8K Accuracy: {acc:.4f}")
        return {}


class EvalGSM8K(Evaluation):
    task_type = Gsm8kTask


if __name__ == "__main__":
    args = parse_arguments()
    EvalGSM8K(args).eval(args.eval_data_num)
