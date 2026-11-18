"""CNN/DailyMail 摘要评测。

运行循环（warmup、多样本种子、异常剔除、ROUGE、落盘）与 XSum 共用
``eval.summarization``；本入口只保留 CNN/DM 特有的 prompt、输出后处理、
摘要 token 预算与解码停串。
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


class CnndmTask(SummarizationTask):
    name = "cnndm"
    model_id_task = "cnndm"
    doc_field = "article"
    ref_field = "highlights"
    # 旧入口的类常量原样保留：摘要预算 160、停串剥离开启。
    summary_max_tokens = 160
    stop_sequences = ["\nArticle:", "\nSummary:"]

    def preprocess(self, input_text):
        few_shot_prompt = get_few_shot_prompt("cnndm", self.args.num_shots)
        full_input = few_shot_prompt + "Article: " + input_text

        qs = (
            "Write a concise summary of the following CNN/DailyMail news article "
            "in a few sentences. "
            "Write only the summary.\n\n"
            f"{full_input}\nSummary:"
        )
        if self.model_id in ("llama-3.1", "llama-3.2", "llama-3", "qwen"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": qs},
            ]
            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if "Qwen3" in str(self.args.target_model) or "Qwen3" in str(
                self.args.draft_model
            ):
                chat_template_kwargs["enable_thinking"] = False
            prompt = self.tokenizer.apply_chat_template(
                messages, **chat_template_kwargs
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
        text = re.sub(r"^(Summary|Highlights?)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r"^Here(?: is|'s) (?:a )?summary\s*:\s*", "", text, flags=re.IGNORECASE
        )
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"^<think>.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"^</think>\s*", "", text, flags=re.IGNORECASE)
        for stop_sequence in self.stop_sequences:
            if stop_sequence in text:
                text = text.split(stop_sequence, 1)[0]
        return text.strip()


class EvalCNNDM(Evaluation):
    task_type = CnndmTask


if __name__ == "__main__":
    args = parse_arguments()
    EvalCNNDM(args).eval(args.eval_data_num)
