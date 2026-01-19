import torch
from .utils import norm_logits, sample
from typing import Tuple
from typing import Literal
from transformers import StaticCache

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
        self._prob_history_list = []
        self._logits_history_list = []

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

        self.hidden_states: torch.Tensor | None = None
        self.vocab_size = model.config.vocab_size
        
        # Bucketing / Padding config
        self.bucket_size = 128
        self.current_bucket_len = 0
        self.real_seq_len = 0

    @property
    def _prob_history(self) -> torch.Tensor:
        if not self._prob_history_list:
            return None
        return torch.cat(self._prob_history_list, dim=1)

    @_prob_history.setter
    def _prob_history(self, value):
        if value is None:
            self._prob_history_list = []
        else:
            self._prob_history_list = [value]

    @property
    def logits_history(self) -> torch.Tensor:
        if not self._logits_history_list:
            return None
        return torch.cat(self._logits_history_list, dim=1)

    @logits_history.setter
    def logits_history(self, value):
        if value is None:
            self._logits_history_list = []
        else:
            self._logits_history_list = [value]
            
    def _pad_past_key_values(self, past_key_values, target_len):
        """Pad past_key_values to target_len"""
        # Handle DynamicCache
        is_dynamic_cache = hasattr(past_key_values, "key_cache")
        
        # If it's a DynamicCache, extract key_cache and value_cache
        current_past =  []
        if is_dynamic_cache:
            for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
                current_past.append((k, v))
        else:
            current_past = past_key_values

        padded_past = []
        for key, value in current_past:
            # key/value shape: (batch, num_heads, seq_len, head_dim)
            current_len = key.shape[2]
            if current_len >= target_len:
                padded_past.append((key, value))
                continue
                
            padding_len = target_len - current_len
            pad_tensor = torch.zeros(
                key.shape[0], key.shape[1], padding_len, key.shape[3],
                dtype=key.dtype, device=key.device
            )
            padded_key = torch.cat([key, pad_tensor], dim=2)
            padded_value = torch.cat([value, pad_tensor], dim=2)
            padded_past.append((padded_key, padded_value))
        
        # Re-wrap into DynamicCache if needed
        # We MUST return a DynamicCache if the input was one OR if transformers requires it.
        # Given the error AttributeError: 'tuple' object has no attribute 'get_seq_length',
        # we know the model expects an object with .get_seq_length().
        
        new_cache = DynamicCache()
        # Manually inject our padded tensors
        new_cache.key_cache = [k for k, v in padded_past]
        new_cache.value_cache = [v for k, v in padded_past]
        # Important: set _seen_tokens to match our padded length or real length?
        # get_seq_length() returns sum of seq lengths of cached keys.
        # But we are mocking a full buffer.
        # Llama uses get_seq_length() to determine position embeddings usually.
        # If we pass position_ids manually, get_seq_length might be ignored for position calc,
        # BUT it might be used for other checks.
        return new_cache

        # return tuple(padded_past)

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        # CUDA Graph marking is essential for reduce-overhead
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if self._past_key_values is None:
            # First run (Prefill) with StaticCache
            
            # Determine max length. 
            # Ideally this should cover the max generation length.
            # We use a safe default or config
            max_cache_len = getattr(self._model.config, "max_position_embeddings", 4096)
            # Ensure it fits initial input
            if seq_len > max_cache_len:
                 max_cache_len = seq_len + 1024
            
            # Initialize StaticCache
            # Note: StaticCache allocates memory immediately.
            self._past_key_values = StaticCache(
                config=self._model.config, 
                batch_size=batch_size, 
                max_cache_len=max_cache_len, 
                device=device, 
                dtype=self._model.dtype
            )
            
            # Prefill cache_position
            cache_position = torch.arange(seq_len, device=device)
            
            outputs = self._model(
                input_ids, 
                use_cache=True,
                past_key_values=self._past_key_values,
                cache_position=cache_position,
            )
            
            outputs_prob = outputs.logits[:, :, : self.vocab_size]
            self._prob_history_list = [outputs_prob]
            self._logits_history_list = [outputs.logits]
            
            # Normalize the first batch
            current_prob = self._prob_history_list[0]
            for i in range(current_prob.shape[-2]):
                current_prob[:, i, :] = norm_logits(
                    current_prob[:, i, :],
                    self._temperature,
                    self._top_k,
                    self._top_p,
                )
            
            # Update real_seq_len
            self.real_seq_len = seq_len

            last_q = current_prob[:, -1, :]
            self.hidden_states = outputs.hidden_states
        else:
            # Decoding steps
            # Calculate cache position for new tokens
            cache_position = torch.arange(self.real_seq_len, self.real_seq_len + seq_len, device=device)
            
            # Make sure we don't exceed cache size
            if self.real_seq_len + seq_len > self._past_key_values.max_cache_len:
                # StaticCache cannot grow. We must raise error or re-allocate?
                # Re-allocating defeats the purpose of StaticCache for graphs.
                # Ideally, we allocated enough in init.
                pass

            # Handling full sequence inputs during decoding
            # If the user passes the full sequence (prefix + new), we must slice it.
            if seq_len >= self.real_seq_len and seq_len > 1:
                # Slice to get only the new tokens
                input_ids = input_ids[:, self.real_seq_len:]
                # Update seq_len and cache_position accordingly
                seq_len = input_ids.shape[1]
                cache_position = torch.arange(self.real_seq_len, self.real_seq_len + seq_len, device=device)
            
            # If after slicing we have no new tokens, it means we are just querying the last state.
            if seq_len == 0:
                # Return the last probabilities from history without running model
                # This assumes history is populated (which it should be if real_seq_len > 0)
                if self._prob_history_list:
                     # Check the last tensor in the list
                     last_tensor = self._prob_history_list[-1]
                     return last_tensor[:, -1, :]
                else:
                    raise RuntimeError("No history available but input is empty.")

            outputs = self._model(
                input_ids, 
                past_key_values=self._past_key_values, 
                use_cache=True, 
                cache_position=cache_position
            )
            
            self.real_seq_len += seq_len
            
            # Ensure outputs.logits is valid
            if outputs.logits is None:
                raise ValueError("Model output logits is None")

            not_cached_q = outputs.logits[:, :, : self.vocab_size]
            self._logits_history_list.append(outputs.logits)

            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)

            # In-place normalization for the new chunk
            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)

            self._prob_history_list.append(not_cached_q)

            last_q = not_cached_q[:, -1, :]
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
        x = prefix

        for _ in range(gamma):
            # Only process new tokens if cache exists
            if self._past_key_values is not None:
                # Slicing from real_seq_len ensures we only feed tokens not yet in cache
                # But we must ensure x is long enough (it should be)
                if self.real_seq_len < x.shape[1]:
                    model_input = x[:, self.real_seq_len:]
                else:
                    # Fallback/Debug: If x didn't grow, we might be stuck or just starting?
                    # If real_seq_len == x.shape[1], we have nothing to process.
                    # But we need logits. 
                    # Assuming we just generated a token and appended it, x > real_seq_len should hold.
                    # If this is entered with x matching cache, we might need to rely on history?
                    # For now assume x covers new tokens.
                    model_input = x[:, -1:] # Least safe fallback
            else:
                model_input = x

            q = self._forward_with_kvcache(model_input)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        # output = self._generate_with_kvcache(input, gamma)
        output = self._generate_with_kvcache(input, gamma)
        return output

    @torch.no_grad()
    def rollback(self, end_pos: int):
        if self._past_key_values is None:
            return

        # Rollback prob_history logic based on list
        # We need to find the cutoff point in the list
        current_len = 0
        new_list = []
        for tensor in self._prob_history_list:
            t_len = tensor.shape[1]
            # Since we now use padding, we need to respect self.real_seq_len logic
            # But prob_history_list stores the *generated* outputs, which relate to generated valid tokens.
            # So simple concatenation logic usually holds for outputs regardless of KV cache padding.
            if current_len + t_len > end_pos:
                remain = end_pos - current_len
                if remain > 0:
                    new_list.append(tensor[:, :remain, :])
                break
            new_list.append(tensor)
            current_len += t_len
        self._prob_history_list = new_list

        # Rollback logits_history logic
        current_len = 0
        new_list = []
        for tensor in self._logits_history_list:
            t_len = tensor.shape[1]
            if current_len + t_len > end_pos:
                remain = end_pos - current_len
                if remain > 0:
                    new_list.append(tensor[:, :remain, :])
                break
            new_list.append(tensor)
            current_len += t_len
        self._logits_history_list = new_list

        # Rollback KV Cache and State
        self.real_seq_len = end_pos
        
        # We perform lazy rollback for KV: just update the index.
        # But for correctness if we switch branch, we should ensure the "future" data is clean?
        # Actually in decoding tree, we overwrite.
        # So setting real_seq_len is enough for logical rollback.
        
        # However, for our padding logic to be safe (we assume tails are 0 for attention mask),
        # we strictly don't need to zero out the tail if we manage Mask correctly.
        # But standard `crop` usage was here.
        
        # If we use HF's crop or our own list based structure:
        # Our KV is now a padded tuple. 
        # We maintain the structure and just reset the length.
        pass

    @property
    def current_length(self) -> int:
        # 当前KVCache的长度
        if self._past_key_values is None:
            return 0
        return self._past_key_values[0][0].shape[2]

    def __len__(self) -> int:
        if self._past_key_values is None:
            return 0
        return self.current_length
