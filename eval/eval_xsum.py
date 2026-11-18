"""XSum（BBC 单句摘要）评测。

运行循环（warmup、多样本种子、异常剔除、ROUGE、落盘）与 CNN/DM 共用
``eval.summarization``；本入口只保留 XSum 特有的 prompt 与输出后处理
（单句摘要、无停串、无摘要 token 预算）。
"""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re

from fastchat.model import get_conversation_template

from eval.few_shot_examples import get_few_shot_prompt  # noqa: E402
from eval.runner import Evaluation  # noqa: E402
from eval.summarization import SummarizationTask  # noqa: E402
from src.utils import parse_arguments  # noqa: E402


class XsumTask(SummarizationTask):
    name = "xsum"
    # XSum 走 determine_model_id 的专用规则链。
    model_id_task = "xsum"
    doc_field = "document"
    ref_field = "summary"

    def preprocess(self, input_text):
        few_shot_prompt = get_few_shot_prompt("xsum", self.args.num_shots)
        full_input = few_shot_prompt + "Article: " + input_text

        qs = (
            "Summarize the following BBC news article in one sentence. "
            "Write only the summary sentence.\n\n"
            f"{full_input}\nSummary:"
        )
        if self.model_id in ("llama-3.1", "llama-3.2", "llama-3", "qwen"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": qs},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif self.model_id == "gemma":
            messages = [
                {"role": "user", "content": "You are a helpful assistant.\n" + qs}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif "vicuna" in self.model_id or "llama-2-chat" in self.model_id:
            # Chat 模型使用 conversation template
            conv = get_conversation_template(self.model_id)
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)  # type: ignore
            prompt = conv.get_prompt() + " "
        else:
            # Base 模型（如 llama-2）使用简单格式，避免对话标记导致的格式冲突
            prompt = qs
        return prompt

    def postprocess(self, input_text, output_text):
        del input_text
        text = output_text.strip()
        text = re.sub(r"^(Summary|TL;DR)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r"^Here(?: is|'s) (?:a )?summary\s*:\s*", "", text, flags=re.IGNORECASE
        )
        text = text.split("\n")[0].strip()
        sentence_match = re.match(r"(.+?[.!?](?:[\"')\]]+)?)($|\s)", text)
        if sentence_match is not None:
            text = sentence_match.group(1).strip()
        return text


class EvalXSum(Evaluation):
    task_type = XsumTask


if __name__ == "__main__":
    args = parse_arguments()
    EvalXSum(args).eval(args.eval_data_num)
