import torch

from typing import Optional, Union, List, Dict, Any
import os, sys
from pathlib import Path

from .gpt_fast.model import Transformer, KVCache, find_multiple
from .gpt_fast.generate import decode_one_token, decode_n_tokens, causal_mask, create_block_mask, prefill, _prefill
from .utils import norm_logits
from .gpt_fast.generate import sample

from torch.nn.attention.flex_attention import create_block_mask

from torch.nn.attention.flex_attention import BlockMask


def causal_mask(b, h, q, kv):
    return q >= kv


class GPTFastWarpper:
    """
    GPTFastWarpper 是 GPTFast 模型的封装器，提供在 duodecoding 上调用 gpt-fast 模型的方法。
    """

    def __init__(
        self,
        model: Transformer,
        temperature: float = 1.0,
        top_k: int = 200,
        max_seq_length: int = 2048,
        device: Union[str, torch.device] = "cuda:0",
        arg_max_token: int = 128,  # 由 duodecoding 传入的参数，表示每次生成的最大 token 数量，用于确定 KVCache大小和 empty tensor 的大小。
        compile: bool = True,
        compile_prefill: bool = False,
    ) -> None:
        self.model = model.to(device=device)
        self.temperature = temperature
        self.top_k = top_k
        self.max_seq_length = max_seq_length
        self.kvcache_is_init = False

        self.cached_prompt: Optional[torch.Tensor] = None

        self.batch_size: int = 1
        self.device: torch.device = torch.device(device)

        self.arg_max_token = arg_max_token

        # 生成中间变量
        self.seq: Optional[torch.Tensor] = None
        self.next_token: Optional[torch.Tensor] = None
        self.input_pos: Optional[torch.Tensor] = None
        # 将 sampling_kwargs 存储起来以便传递
        self.sampling_kwargs = {"temperature": self.temperature, "top_k": self.top_k}

        self.current_seq_capacity: int = 0  # 当前 KVCache 所对应的序列容量

        if compile:
            global decode_one_token, decode_n_tokens, prefill, _prefill
            torch._inductor.config.triton.cudagraph_trees = False
            decode_one_token = torch.compile(decode_one_token)

            if compile_prefill:
                prefill = torch.compile(prefill, fullgraph=True)
                _prefill = torch.compile(_prefill, fullgraph=True)

        # print(f"GPTFastWarpper initialized with device: {self.device}, max_seq_length: {self.max_seq_length}")

    def _is_prefix(self, x: torch.Tensor) -> bool:
        """
        判断 x 是否以 self.cached_prompt 为前缀。
        如果 self.cached_prompt 比 x 长，返回 False。
        如果 self.cached_prompt 在其长度内与 x 相同，返回 True。
        如果 self.cached_prompt 为空，返回 False。
        """
        if self.cached_prompt is None:
            return False
        if self.cached_prompt.size(-1) > x.size(-1):
            return False
        if torch.equal(self.cached_prompt, x[..., : self.cached_prompt.size(-1)]):
            return True
        return torch.equal(self.cached_prompt, x[..., : self.cached_prompt.size(-1)])

    def _init_kvcache(self, max_seq_length: int):
        with torch.device(self.model.device):
            self.model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
        self.kvcache_is_init = True
        self.current_seq_capacity = max(max_seq_length, self.current_seq_capacity)

    def _reset_kvcache(self):
        """Reset the KV cache to the initial state."""
        if self.input_pos is not None:
            self.input_pos = None
        self.model.reset_caches()
        self.cached_prompt = None

    def _init_generation(
        self, x: torch.Tensor, max_new_tokens: int = 128, batch_size: int = 1, **sampling_kwargs
    ) -> None:
        """初始化生成过程所需的中间变量，一次性创建max_new_tokens个新 token 的 KV cache 和 empty tensor"""
        self.T = x.size(-1)
        self.T_new = self.T + max_new_tokens
        max_seq_length = min(self.T_new, self.model.config.block_size)
        self._init_kvcache(max_seq_length)
        self._reset_kvcache()  # 尝试调换顺序
        aligned_max_seq_length = find_multiple(max_seq_length, 8)
        aligned_capacity_seq_length = find_multiple(self.current_seq_capacity, 8)

        self.empty = torch.empty(batch_size, self.T_new, dtype=x.dtype, device=self.device)
        x = x.to(torch.long)  # 确保 token id 是整数类型
        x = x.view(1, -1).repeat(batch_size, 1)
        self.empty[:, : self.T] = x
        self.seq = self.empty
        input_pos = torch.arange(0, self.T, device=self.device).to(self.device)
        x = x.to(self.device)
        self.next_token = _prefill(
            self.model, x.view(batch_size, -1), input_pos, max_seq_length=aligned_capacity_seq_length, **sampling_kwargs
        ).clone()
        self.seq[:, self.T] = self.next_token.squeeze()
        self.input_pos = torch.tensor([self.T], device=self.device, dtype=torch.int)
        self.cached_prompt = x.clone()

    def _generate_with_cache(self, x: torch.Tensor, gamma: int) -> torch.Tensor:
        """
        根据输入的 prompt x 生成 gamma 个新 token。
        returns: 包含新生成的 token 在内的完整序列。
        """
        assert (
            self.kvcache_is_init and self.input_pos and (self.seq is not None) and (self.next_token is not None)
        ), "KV cache is not initialized. Call _init_generation first."
        x = x.to(torch.long)  # 确保 token id 是整数类型
        generated_tokens, _ = decode_n_tokens(
            self.model,
            self.next_token.view(self.batch_size, -1),
            self.input_pos,
            gamma,
        )
        start_pos = self.input_pos.item() + 1
        end_pos = start_pos + gamma
        # 确保 KVCache 的大小足够
        if end_pos > self.seq.size(-1):
            raise ValueError(
                f"Cannot generate {gamma} tokens, as it exceeds the maximum sequence length {self.seq.size(-1)}."
            )
        # 更新 seq 和 next_token
        if not generated_tokens:
            print("[DEBUG] `decode_n_tokens` returned no tokens. Returning current sequence.")
            return self.seq[:, :start_pos]

        self.seq[:, start_pos:end_pos] = torch.cat(generated_tokens, dim=-1)
        # self.input_pos += gamma
        self.next_token = self.seq[:, self.input_pos.item()].view(self.batch_size, -1).to(torch.long)
        # 更新 cached_prompt
        new_full_sequence = self.seq[:, :end_pos]
        self.cached_prompt = new_full_sequence.clone()
        res = self.seq[:, :end_pos]
        if torch.any(res > 32000):
            print(f"Warning: Generated tokens contain values greater than 32000, which may indicate an issue.")
            mask = res > 32000
            res[mask] = 0
            return res
        return self.seq[:, :end_pos]

    def generate(self, x: torch.Tensor, gamma: int, max_new_token: int = 128) -> torch.Tensor:
        x = x.to(torch.long)  # 确保 token id 是整数类型
        if not self.kvcache_is_init or not self._is_prefix(x):
            self._init_generation(x, max_new_token, batch_size=self.batch_size)

        elif self.cached_prompt is not None and x.size(-1) > self.cached_prompt.size(-1):
            # 如果 x 是 cached_prompt 的前缀
            prev_len = self.cached_prompt.size(-1)
            new_part = x[:, prev_len:]

            input_pos = torch.arange(prev_len, x.size(-1), device=self.device)
            next_token = prefill(self.model, new_part.view(1, -1), input_pos, **self.sampling_kwargs).clone()
            if self.seq is None:
                raise ValueError("KV cache is not initialized. Call _init_generation first.")

            self.seq[:, prev_len : x.size(-1)] = new_part
            self.seq[:, x.size(-1)] = next_token.squeeze()
            self.T = x.size(-1)
            self.input_pos = torch.tensor([self.T], device=self.device, dtype=torch.int)
            self.next_token = next_token
            self.cached_prompt = x.clone()
        res = self._generate_with_cache(x, gamma)
        return self._generate_with_cache(x, gamma)

    @torch.no_grad()
    def rollback(self, end_pos: int):
        """Rollback the KV cache to a specific position"""
        raise NotImplementedError("Rollback is not implemented in GPTFastWarpper.")
