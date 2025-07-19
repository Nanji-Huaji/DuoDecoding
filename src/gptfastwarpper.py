import torch
from . import model_gpu
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

        print(f"GPTFastWarpper initialized with device: {self.device}, max_seq_length: {self.max_seq_length}")

    # def _is_prefix(self, new_prompt: torch.Tensor) -> bool:
    #     """检查 self.cached_prompt 是否是 new_prompt 的前缀。"""
    #     print("\n--- [DEBUG] Inside _is_prefix ---")

    #     if self.cached_prompt is None or new_prompt.size(-1) < self.cached_prompt.size(-1):
    #         print("Result: False (Reason: No cached_prompt available)")
    #         print("--- End _is_prefix ---\n")
    #         return False
    #     # 检查两个 prompt 的形状是否兼容（除了序列长度）
    #     if self.cached_prompt.ndim != new_prompt.ndim or self.cached_prompt.shape[:-1] != new_prompt.shape[:-1]:
    #         print("Result: False (Reason: Shape mismatch)")
    #         print("--- End _is_prefix ---\n")
    #         return False

    #     prefix_len = self.cached_prompt.size(-1)
    #     # 使用你之前的修复，这是正确的
    #     new_prompt_prefix = new_prompt[..., :prefix_len]
    #     print(f"Cached prompt (shape {self.cached_prompt.shape}):")
    #     print(self.cached_prompt)
    #     print(f"New prompt (shape {new_prompt.shape}):")
    #     print(new_prompt)
    #     print(f"Sliced new prompt prefix to compare (shape {new_prompt_prefix.shape}):")
    #     print(new_prompt_prefix)
    #     are_equal = torch.equal(self.cached_prompt, new_prompt_prefix)

    #     print(f"Result of torch.equal: {are_equal}")
    #     print("--- End _is_prefix ---\n")

    #     prefix_len = self.cached_prompt.size(-1)
    #     # 正确的切片方式：对最后一个维度进行切片
    #     return torch.equal(self.cached_prompt, new_prompt[..., :prefix_len])

    def _is_prefix(self, x: torch.Tensor) -> bool:
        print("\n--- [DEBUG] Inside _is_prefix ---")
        if self.cached_prompt is None or self.cached_prompt.size(-1) == 0:
            print("Result: False (Reason: No cached_prompt available)")
            print("--- End _is_prefix ---")
            return False
        if self.cached_prompt.size(-1) > x.size(-1):
            print("Result: False (Reason: New prompt is shorter than cache)")
            print("--- End _is_prefix ---")
            return False

        # 关键修复：在比较前，将 self.cached_prompt 转换回 torch.long
        cached_prompt_long = self.cached_prompt.to(torch.long)

        # 取出 x 中与缓存长度相同的前缀部分
        prefix_to_compare = x[:, : self.cached_prompt.size(-1)]

        # 打印用于调试
        print(f"Cached prompt (as long, shape {cached_prompt_long.shape})")
        print(f"New prompt's prefix (shape {prefix_to_compare.shape})")
        # 进行严格的逐元素比较
        result = torch.equal(cached_prompt_long, prefix_to_compare)
        print(f"Result of torch.equal: {result}")
        print("--- End _is_prefix ---")
        return result

    def _init_kvcache(self, max_seq_length: int):
        with torch.device(self.model.device):
            self.model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
        self.kvcache_is_init = True

    def _reset_kvcache(self):
        """Reset the KV cache to the initial state."""
        if self.input_pos is not None:
            self.input_pos.zero_()
        self.cached_prompt = None

    def _init_generation(
        self, x: torch.Tensor, max_new_tokens: int = 128, batch_size: int = 1, **sampling_kwargs
    ) -> None:
        """初始化生成过程所需的中间变量，一次性创建128个新 token 的 KV cache 和 empty tensor"""
        self.T = x.size(-1)
        self.T_new = self.T + max_new_tokens
        max_seq_length = min(self.T_new, self.model.config.block_size)
        self._reset_kvcache()
        self._init_kvcache(max_seq_length)
        aligned_max_seq_length = find_multiple(max_seq_length, 8)

        self.empty = torch.empty(batch_size, self.T_new, dtype=self.model.dtype, device=self.device)
        x = x.to(torch.long)  # 确保 token id 是整数类型
        x = x.view(1, -1).repeat(batch_size, 1)
        self.empty[:, : self.T] = x
        self.seq = self.empty
        input_pos = torch.arange(0, self.T, device=self.device).to(self.device)
        x = x.to(self.device)
        self.next_token = _prefill(
            self.model, x.view(batch_size, -1), input_pos, max_seq_length=aligned_max_seq_length, **sampling_kwargs
        ).clone()
        self.seq[:, self.T] = self.next_token.squeeze()
        self.input_pos = torch.tensor([self.T], device=self.device, dtype=torch.int)
        self.cached_prompt = x

    def _generate_with_cache(self, x: torch.Tensor, gamma: int) -> torch.Tensor:
        """
        根据输入的 prompt x 生成 gamma 个新 token。
        returns: 包含新生成的 token 在内的完整序列。
        """
        assert (
            self.kvcache_is_init and self.input_pos and (self.seq is not None) and (self.next_token is not None)
        ), "KV cache is not initialized. Call _init_generation first."
        if self.cached_prompt is not None:
            print(f"  - `self.cached_prompt` shape before generation: {self.cached_prompt.shape}")
        else:
            print(f"  - `self.cached_prompt` before generation: None")
        # ------------------------------------
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

        return self.seq[:, :end_pos]

    def generate(self, x: torch.Tensor, gamma: int, max_new_token: int = 128) -> torch.Tensor:
        x = x.to(torch.long)  # 确保 token id 是整数类型
        # --- 调试信息：进入 generate 方法 ---
        print("\n" + "#" * 20 + " [Enter `generate` method] " + "#" * 20)
        print(f"  - Input prompt `x` shape: {x.shape}")
        if self.cached_prompt is not None:
            print(f"  - Current `cached_prompt` shape: {self.cached_prompt.shape}")
        else:
            print(f"  - Current `cached_prompt` is None.")
        # ------------------------------------
        if not self.kvcache_is_init or not self._is_prefix(x):
            print(f"DEBUG: x 不是 cached_prompt 的前缀，重新初始化 KV cache")
            print("\n  [Decision] Path 1: Initializing/Re-initializing cache.")
            print(f"    - Reason: kvcache_is_init={self.kvcache_is_init}, is_prefix={self._is_prefix(x)}")
            self._init_generation(x, max_new_token, batch_size=self.batch_size)  # gamma 好像应该 -1，但先不管

        elif self.cached_prompt is not None and x.size(-1) > self.cached_prompt.size(-1):
            # 如果 x 是 cached_prompt 的前缀
            print("\n  [Decision] Path 2: Extending existing cache.")
            prev_len = self.cached_prompt.size(-1)
            new_part = x[:, prev_len:]
            print(f"    - Extending from length {prev_len} to {x.size(-1)}. New part length: {new_part.size(-1)}")
            print(f"    - New part tokens: {new_part.view(-1).tolist()}")

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
            print("    - Extension prefill complete.")
            print(f"    - `self.input_pos` is now: {self.input_pos.item()}")
            print(f"    - `self.next_token` for generation is: {self.next_token.view(-1).tolist()}")
            print(f"    - `self.cached_prompt` updated to new input `x`.")
        else:
            # 这种情况可能是 x 和 cached_prompt 完全相同，或者 x 比 cached_prompt 短 (逻辑错误)
            print("\n  [Decision] Path 3: No re-init, no extension. Proceeding directly.")
            if self.cached_prompt is not None:
                print(f"    - `x` length ({x.size(-1)}) vs `cached_prompt` length ({self.cached_prompt.size(-1)})")
                print("#" * 20 + " [Calling _generate_with_cache from `generate`] " + "#" * 20)
        return self._generate_with_cache(x, gamma)

    @torch.no_grad()
    def rollback(self, end_pos: int):
        """Rollback the KV cache to a specific position"""
        raise NotImplementedError("Rollback is not implemented in GPTFastWarpper.")
