import torch
import json
import torch.distributed as dist
import numpy as np
import os
import transformers
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

target_models = ["llama/Llama-2-13b-hf", "vicuna/tiny-vicuna-1b", "vicuna/vicuna-68m", "vicuna/vicuna-13b-v1.5"]

if __name__ == "__main__":
    # 遍历所有目标模型
    for target_model in target_models:
        print(f"\n{'='*50}")
        print(f"正在处理模型: {target_model}")
        print(f"{'='*50}")

        try:
            # 加载tokenizer
            print(f"正在加载模型 {target_model} 的tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(target_model)

            # 输出tokenizer的长度（词汇表大小）
            vocab_size = len(tokenizer)
            print(f"Tokenizer词汇表大小: {vocab_size}")

            # 输出一些额外的tokenizer信息
            print(f"模型最大长度: {tokenizer.model_max_length}")
            print(f"Padding token: {tokenizer.pad_token}")
            print(f"EOS token: {tokenizer.eos_token}")
            print(f"BOS token: {tokenizer.bos_token}")

        except Exception as e:
            print(f"加载模型 {target_model} 时出错: {str(e)}")
            continue
