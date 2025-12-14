import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import torch

def get_model(model_path) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer

