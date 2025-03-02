import os
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np


def seed_everything(seed: int):
    "set all random seed for reproducible results."
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def model_zoo(args):
    vocab_size = {
        "codellama-7b": 32000,
        "codellama-34b": 32000,
        "codellama-70b": 32000,
        "llama-2-7b": 32000,
        "llama-2-70b": 32000,
        "deepseek-1.3b": 32256,
        "deepseek-6.7b": 32256,
        "deepseek-33b": 32256,
        "llama-68m-q5-gguf": 32000,
        "llama-68m-q8-gguf": 32000,
        "llama-68m": 32000,
        "llama-68m-fp16": 32000,
        "llama-160m-q5-gguf": 32000,
        "llama-160m": 32000,
        "vicuna-68m-q5-gguf": 32000,
        "vicuna-68m": 32000,
        "vicuna-7b-v1.5": 32000,
        "vicuna-7b-v1.3": 32000,
        "llama-290m-q5-gguf": 32000,
        "llama-290m": 32000,
        "llama-543m": 32000,
        "llama-543m-q5-gguf": 32000,
        "llama-2-7b-chat": 32000,
        "llama-68m-chat-q5-gguf": 32000,
    }

    zoo = {
        "llama-2-7b": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/Llama-2-7b-hf",
        "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-base",
        "deepseek-6.7b": "deepseek-ai/deepseek-coder-6.7b-base",
        "llama-68m-q5-gguf": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/llama-68m.Q5_K_M.gguf",
        "llama-68m-q8-gguf": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/llama-68m.Q8_0.gguf",
        "llama-68m-fp16": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/llama-68m.fp16.bin",
        "llama-68m": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/llama-68m",
        "llama-160m-q5-gguf": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/llama-160m.q5_k_m.gguf",
        "llama-160m": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/llama-160m",
        "vicuna-68m-q5-gguf": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/vicuna-68m.Q5_K_M.gguf",
        "vicuna-68m": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/vicuna-68m",
        "vicuna-7b-v1.5": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/vicuna-7b-v1.5",
        "vicuna-7b-v1.3": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/vicuna-7b-v1.3",
        "llama-2-7b-chat": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/Llama-2-7b-chat-hf",
        "llama-68m-chat-q5-gguf": "/cpfs02/llm/shared/public/lvkai/workspace/sd/data/Llama-68M-Chat-v1-Q5_K_M.gguf",
    }

    args.vocab_size = vocab_size[args.draft_model]
    args.draft_model = zoo[args.draft_model]
    args.target_model = zoo[args.target_model]


def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description="args for this file")

    parser.add_argument(
        "--data_path",
        type=str,
        default="/cpfs02/llm/shared/public/lvkai/workspace/sd/DuoDecoding/data",
    )

    parser.add_argument("--draft_model", type=str, default="codellama-7b")
    parser.add_argument("--target_model", type=str, default="codellama-70b")

    parser.add_argument(
        "--exp_name",
        "-e",
        type=str,
        default="test",
        help="folder name for storing results.",
    )
    parser.add_argument("--eval_mode", type=str, default="small", help="eval mode.")
    parser.add_argument(
        "--num_samples_per_task",
        "-n",
        type=int,
        default=1,
        help="num_samples for a task (prompt) in humaneval dataset.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=1234,
        help="set a random seed, which can makes the result reproducible",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="max token number generated."
    )
    parser.add_argument(
        "--temp", type=float, default=0.2, help="temperature for generating new tokens."
    )
    parser.add_argument(
        "--top_k", type=int, default=0, help="top_k for ungreedy sampling strategy."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="top_p for ungreedy sampling strategy.",
    )
    parser.add_argument("--gamma", type=int, default=4, help="guess time.")
    parser.add_argument(
        "--sub_domain",
        type=str,
        default="math_reasoning",
        help="sub domain in specbench.",
        choices=[
            "math_reasoning",
            "mt-bench",
            "qa",
            "rag",
            "summarization",
            "translation",
        ],
    )

    # for lookahead decoding
    parser.add_argument(
        "--level",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--guess",
        type=int,
        default=10,
    )
    # end for lookahead decoding

    # for rest
    parser.add_argument(
        "--max-token-span",
        type=int,
        default=16,
        help="The maximum length of suffix for retrieval.",
    )
    parser.add_argument(
        "--datastore-path",
        type=str,
        default="/cpfs02/llm/shared/public/lvkai/workspace/sd/DuoDecoding/src/model/rest/datastore/datastore_chat_large.idx",
        help="The path of the datastore for retrival.",
    )
    parser.add_argument(
        "--num-draft",
        type=int,
        default=64,
        help="The maximum number of draft tokens.",
    )
    # end for rest

    args = parser.parse_args()
    args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    model_zoo(args)
    return args


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float("-inf")
    return logits


def norm_logits(
    logits: torch.Tensor, temperature: float, top_k: float, top_p: float
) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    if temperature == 0:
        idx = logits.argmax(dim=1)
        new_logits = torch.zeros_like(logits, device=logits.device)
        new_logits[:, idx] = 1
        return new_logits.float()
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def norm_numpy_logits(
    logits: np.ndarray, temperature: float, top_k: float, top_p: float
) -> np.ndarray:
    assert logits.ndim == 2
    if temperature == 0:
        idx = logits.argmax(axis=1)
        new_logits = np.zeros_like(logits, dtype=np.float32)
        new_logits[np.arange(new_logits.shape[0]), idx] = 1
        return new_logits
    logits = logits / temperature
    # logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    return probs


def sample(probs: torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    return idx_next


def max_fn(x):
    """
    norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum
