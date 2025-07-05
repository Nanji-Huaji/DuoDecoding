import torch
import json
import torch.distributed as dist
import numpy as np
import os
import transformers
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer


draft_model = AutoModelForCausalLM.from_pretrained(
    "llama/llama-68m",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    cache_dir="llama/.cache/huggingface",
    local_files_only=True,
)
draft_tokenizer = AutoTokenizer.from_pretrained(
    "llama/llama-68m",
    trust_remote_code=True,
    cache_dir="llama/.cache/huggingface",
    local_files_only=True,
)
