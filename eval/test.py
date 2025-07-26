import os
import sys

sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import ipdb
import random
from src.utils import seed_everything, parse_arguments
from src.engine import Decoding
from collections import Counter
from fastchat.model import get_conversation_template


class EvalDebug(Decoding):
    def __init__(self, args):
        super().__init__(args)

        # 1. 加载模型和分词器
        self.load_tokenizer()
        self.load_gpt_fast_model()

        # 2. 设置模型对话模板 (为保持逻辑完整性，此处保留)
        self.model_id = "vicuna"
        print(f"Using conversation template for: {self.model_id}")

        # 3. 加载测试数据
        self.load_data()

    def load_data(self):
        """
        使用两个简单的 prompt 进行测试，而不是从文件加载。
        """
        self.color_print("Loading simple hardcoded prompts for debugging...", 3)
        self.data = [
            {
                "question_id": 1,
                "category": "test-simple",
                "turns": ["Hello, what is your name?"],
            },
            {
                "question_id": 2,
                "category": "test-multi-turn",
                "turns": ["What is the capital of France?", "And what is its population?"],
            },
        ]

    @torch.no_grad()
    def eval(self):
        # 模式固定为自回归 (Autoregressive)
        decoding_function = self.autoregressive_sampling

        # 遍历所有测试用例
        for question in self.data:
            print(f"\n===== Running Test Case: question_id={question['question_id']} =====")

            # 初始化对话模板
            conv = get_conversation_template(self.model_id)

            # 处理多轮对话
            for turn_idx, user_query in enumerate(question["turns"]):
                print(f"\n--- Turn {turn_idx + 1} ---")

                # --- START: 分词逻辑 (Tokenization) ---
                conv.append_message(conv.roles[0], user_query)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt() + " "  # 添加空格以适配某些模型

                # 将 prompt 转换为 token IDs
                input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
                # --- END: 分词逻辑 ---

                print(f"Input Prompt: \n{prompt}")
                print(f"Input Token IDs shape: {input_ids.shape}")

                # 同步以获得准确的计时
                torch.cuda.synchronize()
                start_time = time.time()

                # --- START: 生成逻辑 (Generation) ---
                # 调用自回归生成函数
                output_ids = decoding_function(input_ids)
                # --- END: 生成逻辑 ---

                torch.cuda.synchronize()
                end_time = time.time()

                # --- START: 反分词逻辑 (Detokenization) ---
                # 从输出中解码出文本，去除输入部分
                generated_token_ids = output_ids[0][input_ids.shape[1] :]
                output_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                output_text = output_text.strip()
                # --- END: 反分词逻辑 ---

                # 更新对话历史，为下一轮做准备
                conv.messages[-1][-1] = output_text

                # 打印本次生成的结果
                print(f"Generated Text: \n{output_text}")
                print(f"Time taken: {end_time - start_time:.2f} seconds")
                print(f"Number of generated tokens: {len(generated_token_ids)}")

        print("\n===== Debugging Finished =====")

    def preprocess(self, text):
        pass

    def postprocess(self, text):
        pass


if __name__ == "__main__":
    # 请确保您的 parse_arguments() 能够提供必要的模型路径等参数
    args = parse_arguments()

    # 将模式固定为 'small' 或 'large'，因为它们都使用自回归
    args.eval_mode = "large"

    debugger = EvalDebug(args)
    debugger.eval()
