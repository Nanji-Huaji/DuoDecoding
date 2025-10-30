import torch
from .utils import norm_logits, sample
from typing import Tuple
from typing import Literal


class PastKeyValues:
    # TODO
    # 思考中
    # 目的是让KVCache和prob_history可以在一开始的时候就被初始化，避免gpu内存占用高峰导致OOM
    def __init__(self, *args, **kwargs):
        self.past_key_values: torch.Tensor 
        self.shape: torch.Size
        self.used_ptr: Tuple[int, ...]
        self.max_len: int
        raise NotImplementedError("PastKeyValues class is not implemented yet.")
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.past_key_values[index]
    
    def __setitem__(self, index: int, value: torch.Tensor) -> None:
        self.past_key_values[index] = value


class ProbHistory:
    # TODO
    # 目的同上
    def __init__(self, *args, **kwargs):
        self.prob_history: torch.Tensor 
        self.shape: torch.Size
        self.used_ptr: int
        self.max_len: int
        raise NotImplementedError("ProbHistory class is not implemented yet.")
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.prob_history[index]
    
    def __setitem__(self, index: int, value: torch.Tensor) -> None:
        self.prob_history[index] = value

    def cat(self, other: 'ProbHistory' | torch.Tensor, dim: int) -> ProbHistory:
        raise NotImplementedError("ProbHistory.cat method is not implemented yet.")

    def as_type(self, type: Literal['torch.Tensor', 'numpy.ndarray']) -> 'ProbHistory' | torch.Tensor | 'np.ndarray':
        raise NotImplementedError("ProbHistory.as_type method is not implemented yet.")

    def to(self, target: torch.device | torch.dtype) -> 'ProbHistory':
        pass



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

        self.logits_history: torch.Tensor | None = None # 不确定性方法需要存储logits历史

        self.vocab_size = model.config.vocab_size

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits[:, :, : self.vocab_size]
            self.logits_history = outputs.logits
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

            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)

            not_cached_q = outputs.logits[:, :, : self.vocab_size]
            self.logits_history = torch.cat([self.logits_history, outputs.logits], dim=1)

            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)

            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)

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

        for _ in range(gamma):
            q = self._forward_with_kvcache(x)
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
        past_key_values_trimmed = []
        if self._past_key_values is None:
            return
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
    def current_length(self) -> int:
        # 当前KVCache的长度
        if self._past_key_values is None:
            return 0
        return self._past_key_values[0][0].shape[2]

    def __len__(self) -> int:
        if self._past_key_values is None:
            return 0
        return self.current_length
