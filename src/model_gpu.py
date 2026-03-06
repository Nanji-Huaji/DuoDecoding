import torch

from .utils import norm_logits, sample


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
        self._prob_history: torch.Tensor | None = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

        self.hidden_states: torch.Tensor | None = None

        self.logits_history: torch.Tensor | None = (
            None  # 不确定性方法需要存储logits历史
        )

        if hasattr(model.config, "vocab_size"):
            self.vocab_size = model.config.vocab_size
        elif hasattr(model.config, "text_config") and hasattr(
            model.config.text_config, "vocab_size"
        ):
            self.vocab_size = model.config.text_config.vocab_size
        else:
            raise AttributeError("Vocab size not found in model config")

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 确保 input_ids 是 long 类型
        if input_ids.dtype != torch.long:
            print(
                f"🔍 DEBUG [_forward_with_kvcache]: input_ids dtype={input_ids.dtype}, converting to long"
            )
            input_ids = input_ids.long()

        # 使用实际模型embedding层的vocab_size进行验证
        actual_vocab_size = self._model.get_input_embeddings().weight.shape[0]
        original_max = input_ids.max().item()
        original_min = input_ids.min().item()
        if original_max >= actual_vocab_size or original_min < 0:
            print(
                f"⚠️  [Token Clamp Warning] Detected out-of-range token IDs: min={original_min}, max={original_max}, actual_vocab_size={actual_vocab_size}"
            )
            input_ids = torch.clamp(input_ids, 0, actual_vocab_size - 1)
            print(
                f"   Clamped to valid range: min={input_ids.min().item()}, max={input_ids.max().item()}"
            )

        if self._past_key_values is None:
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits[:, :, : self.vocab_size]
            self.logits_history = outputs.logits

            # 检查logits是否包含inf/nan
            if torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any():
                print("⚠️  [NaN/Inf Warning] Model forward produced invalid logits!")
                print(f"   NaN count: {torch.isnan(outputs.logits).sum().item()}")
                print(f"   Inf count: {torch.isinf(outputs.logits).sum().item()}")
                print(f"   Logits shape: {outputs.logits.shape}")
                print(
                    f"   Logits range: [{outputs.logits.min().item():.2f}, {outputs.logits.max().item():.2f}]"
                )

            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(
                    self._prob_history[:, i, :],
                    self._temperature,
                    self._top_k,
                    self._top_p,
                )
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
            self.hidden_states = outputs.hidden_states
        else:
            # return the last token's logits
            cached_len = self._past_key_values[0][0].shape[2]

            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)

            # 确保是 long 类型
            if last_input_id.dtype != torch.long:
                print(
                    f"🔍 DEBUG [_forward_with_kvcache cached]: last_input_id dtype={last_input_id.dtype}, converting to long"
                )
                last_input_id = last_input_id.long()

            # 钳位input_ids，防止越界访问
            original_max = last_input_id.max().item()
            original_min = last_input_id.min().item()
            if original_max >= self.vocab_size or original_min < 0:
                print(
                    f"⚠️  [Token Clamp Warning] Detected out-of-range token IDs in cached forward: min={original_min}, max={original_max}, vocab_size={self.vocab_size}"
                )
                last_input_id = torch.clamp(last_input_id, 0, self.vocab_size - 1)
                print(
                    f"   Clamped to valid range: min={last_input_id.min().item()}, max={last_input_id.max().item()}"
                )

            outputs = self._model(
                last_input_id, past_key_values=self._past_key_values, use_cache=True
            )

            not_cached_q = outputs.logits[:, :, : self.vocab_size]
            self.logits_history = torch.cat(
                [self.logits_history, outputs.logits], dim=1
            )

            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)

            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(
                    not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p
                )

            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)

            last_q = not_cached_q[:, -1, :]
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
        x = prefix

        # 确保输入是 long 类型
        if x.dtype != torch.long:
            print(
                f"🔍 DEBUG [_generate_with_kvcache]: prefix dtype={x.dtype}, converting to long"
            )
            x = x.long()

        for _ in range(gamma):
            q = self._forward_with_kvcache(x)
            next_tok = sample(q)

            # 确保 next_tok 是 long 类型
            if next_tok.dtype != torch.long:
                print(
                    f"🔍 DEBUG [_generate_with_kvcache]: next_tok dtype={next_tok.dtype}, converting to long"
                )
                next_tok = next_tok.long()

            x = torch.cat((x, next_tok), dim=1)

            # 确保拼接后仍是 long 类型
            if x.dtype != torch.long:
                print(
                    f"🔍 DEBUG [_generate_with_kvcache]: x after cat dtype={x.dtype}, converting to long"
                )
                x = x.long()

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

        if hasattr(self._past_key_values, "crop"):
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

        self._prob_history = self._prob_history[:, :end_pos, :]
        self.logits_history = self.logits_history[:, :end_pos, :]

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device

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
