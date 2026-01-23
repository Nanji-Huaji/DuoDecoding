import torch
from .utils import norm_logits, sample
from typing import Tuple
from typing import Literal




class KVCacheModel:
    def __init__(
        self,
        model: torch.nn.Module,
        temperature: float = 1,
        top_k: int = 0,
        top_p: float = 0,
        return_hidden_states: bool = False,
    ) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history_buffer: torch.Tensor | None = None
        self._logits_history_buffer: torch.Tensor | None = None
        self._input_ids_buffer: torch.Tensor | None = None
        self._current_len = 0
        self._capacity = 0

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

        self.hidden_states: torch.Tensor | None = None
        self.vocab_size = model.config.vocab_size

    @property
    def _prob_history(self) -> torch.Tensor | None:
        if self._prob_history_buffer is None:
            return None
        return self._prob_history_buffer[:, :self._current_len, :]

    @property
    def logits_history(self) -> torch.Tensor | None:
        if self._logits_history_buffer is None:
            return None
        return self._logits_history_buffer[:, :self._current_len, :]

    @property
    def input_ids(self) -> torch.Tensor | None:
        if self._input_ids_buffer is None:
            return None
        return self._input_ids_buffer[:, :self._current_len]

    def _grow_buffers(self, min_capacity: int):
        new_capacity = max(min_capacity, self._capacity * 2 if self._capacity > 0 else 1024)
        
        if self._logits_history_buffer is not None:
            new_logits = torch.zeros(
                (self._logits_history_buffer.shape[0], new_capacity, self._logits_history_buffer.shape[2]),
                device=self._logits_history_buffer.device, dtype=self._logits_history_buffer.dtype
            )
            new_logits[:, :self._current_len, :] = self._logits_history_buffer[:, :self._current_len, :]
            self._logits_history_buffer = new_logits
            
        if self._prob_history_buffer is not None:
            new_probs = torch.zeros(
                (self._prob_history_buffer.shape[0], new_capacity, self._prob_history_buffer.shape[2]),
                device=self._prob_history_buffer.device, dtype=self._prob_history_buffer.dtype
            )
            new_probs[:, :self._current_len, :] = self._prob_history_buffer[:, :self._current_len, :]
            self._prob_history_buffer = new_probs

        if self._input_ids_buffer is not None:
            new_ids = torch.zeros(
                (self._input_ids_buffer.shape[0], new_capacity),
                device=self._input_ids_buffer.device, dtype=self._input_ids_buffer.dtype
            )
            new_ids[:, :self._current_len] = self._input_ids_buffer[:, :self._current_len]
            self._input_ids_buffer = new_ids
            
        self._capacity = new_capacity

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        既支持传入全量 input_ids (Prefill)，也支持在增量阶段仅传入新 Token 或全量序列。
        """
        if self._past_key_values is None:
            # Prefill 阶段
            outputs = self._model(input_ids)
            
            init_logits = outputs.logits
            init_probs = init_logits[:, :, : self.vocab_size]
            
            self._current_len = init_logits.shape[1]
            self._capacity = max(self._current_len + 512, 1024)
            
            self._logits_history_buffer = torch.zeros(
                (init_logits.shape[0], self._capacity, init_logits.shape[2]),
                device=init_logits.device, dtype=init_logits.dtype
            )
            self._prob_history_buffer = torch.zeros(
                (init_probs.shape[0], self._capacity, init_probs.shape[2]),
                device=init_probs.device, dtype=init_probs.dtype
            )
            self._input_ids_buffer = torch.zeros(
                (input_ids.shape[0], self._capacity),
                device=input_ids.device, dtype=input_ids.dtype
            )
            
            self._logits_history_buffer[:, :self._current_len, :] = init_logits
            self._prob_history_buffer[:, :self._current_len, :] = init_probs
            self._input_ids_buffer[:, :self._current_len] = input_ids
            
            # 归一化概率
            active_probs = self._prob_history_buffer[:, :self._current_len, :]
            for i in range(active_probs.shape[1]):
                active_probs[:, i, :] = norm_logits(
                    active_probs[:, i, :],
                    self._temperature,
                    self._top_k,
                    self._top_p,
                )
            
            self._past_key_values = outputs.past_key_values
            last_q = active_probs[:, -1, :]
            self.hidden_states = outputs.hidden_states
        else:
            # 增量阶段
            cached_len = self._current_len
            
            # 兼容逻辑
            if input_ids.shape[1] > cached_len:
                # 传入了比缓存长的序列，说明包含新 token (如 speculative decoding 的批量验证)
                new_input_ids = input_ids[:, cached_len:]
            elif input_ids.shape[1] == cached_len:
                # 调用者可能传了全量已缓存序列，不需要再次进行模型推理
                return self._prob_history_buffer[:, cached_len - 1, :]
            else:
                # 传入的是单独的新 token
                new_input_ids = input_ids
                
            outputs = self._model(new_input_ids, past_key_values=self._past_key_values, use_cache=True)
            
            # 强制 clone 以打破与 torch.compile (reduce-overhead) 内部静态缓冲区的关联
            logits = outputs.logits.clone()
            
            new_tokens_len = logits.shape[1]
            new_total_len = self._current_len + new_tokens_len
            
            if new_total_len > self._capacity:
                self._grow_buffers(new_total_len)
            
            # 更新缓冲区
            self._logits_history_buffer[:, self._current_len:new_total_len, :] = logits
            self._input_ids_buffer[:, self._current_len:new_total_len] = new_input_ids

            not_cached_q = logits[:, :, : self.vocab_size]
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)

            for i in range(not_cached_q.shape[1]):
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)

            self._prob_history_buffer[:, self._current_len:new_total_len, :] = not_cached_q
            
            self._current_len = new_total_len
            last_q = self._prob_history_buffer[:, self._current_len - 1, :]
            self._past_key_values = outputs.past_key_values
            self.hidden_states = outputs.hidden_states

        return last_q

    @property
    def last_hidden_state(self) -> torch.Tensor:
        if self.hidden_states is None:
            raise ValueError("hidden_states is None")
        return self.hidden_states[-1]

    def _generate_with_kvcache(self, prefix: torch.Tensor, gamma: int) -> torch.Tensor:
        """forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        # Prefill or ensure prefix is in cache
        q = self._forward_with_kvcache(prefix)
        
        for _ in range(gamma):
            next_tok = sample(q)
            q = self._forward_with_kvcache(next_tok)

        return self.input_ids

    @torch.inference_mode()
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        return self._generate_with_kvcache(input, gamma)

    @torch.inference_mode()
    def rollback(self, end_pos: int):
        if self._past_key_values is None:
            return

        if hasattr(self._past_key_values, 'crop'):
            self._past_key_values.crop(end_pos)
        else:
            past_key_values_trimmed = []
            assert self._past_key_values
            for kv in self._past_key_values:
                k, v = kv
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            self._past_key_values = past_key_values_trimmed

        self._current_len = end_pos

    @property
    def current_length(self) -> int:
        # 当前KVCache的长度
        return self._current_len

    def __len__(self) -> int:
        if self._past_key_values is None:
            return 0
        return self.current_length
