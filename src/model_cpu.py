import math
import time
import json
import torch
import numpy as np
import llama_cpp, ctypes
from .utils import norm_logits, sample, norm_numpy_logits


class KVCacheCppModel:
    def __init__(
        self, model, temperature: float = 1, top_k: int = 0, top_p: float = 0
    ) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._unnorm_prob_history = None
        self.stop_signal = torch.zeros(1)

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self.kv_cache = [[(ctypes.c_uint8 * 25600000)(), _, _, _] for _ in range(8)]

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits[:, :, : self.vocab_size]
            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(
                    self._prob_history[:, i, :],
                    self._temperature,
                    self._top_k,
                    self._top_p,
                )
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            # return the last token's logits
            cached_len = self._past_key_values[0][0].shape[2]

            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)

            outputs = self._model(
                last_input_id, past_key_values=self._past_key_values, use_cache=True
            )

            not_cached_q = outputs.logits[:, :, : self.vocab_size]

            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)

            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(
                    not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p
                )

            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)

            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values

        return last_q

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

    def _generate_llama_cpp_kseq(self, prefix: list[int], gamma: int) -> list[int]:
        x = prefix
        window_size = gamma
        for token in self._model.generate(
            prefix, top_k=self._top_k, top_p=self._top_p, temp=self._temperature
        ):
            x.append(token)
            gamma -= 1
            if gamma == 0:
                break
        return x

    def _generate_llama_cpp(self, prefix: list[int], gamma: int) -> list[int]:
        x = prefix
        window_size = gamma
        for token in self._model.generate(
            prefix, top_k=self._top_k, top_p=self._top_p, temp=self._temperature
        ):
            x.append(token)
            gamma -= 1
            if gamma == 0:
                break

        # get prob history
        if self._past_key_values is None:
            self._prob_history = self._model._scores  # (seq_len, vocab)
            self._prob_history = norm_numpy_logits(
                self._model._scores,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
            )
            self._prob_history = np.expand_dims(self._prob_history, axis=0)
            self._past_key_values = 1
        else:
            last_logit = self._model._scores[-window_size:, :]
            last_prob = norm_numpy_logits(
                last_logit,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
            )
            last_prob = np.expand_dims(last_prob, axis=0)

            self._prob_history = np.concatenate([self._prob_history, last_prob], axis=1)
        return x

    @torch.no_grad()
    def generate(self, input: list[int], gamma: int) -> list[int]:
        output = self._generate_llama_cpp(input, gamma)
        return output

    @torch.no_grad()
    def generate_k_seq(self, input_ids: list[int], gamma: int) -> list[int]:
        prefix_len = len(input_ids)

        max_k = 8

        first_token = self._generate_llama_cpp_kseq(input_ids, 2)

        second_token = first_token[-1]
        first_token = first_token[-2]

        first_logits = self._model._scores[-2, :]
        first_logprob = self._model.logits_to_logprobs(first_logits)

        second_logits = self._model._scores[-1, :]
        second_logprob = self._model.logits_to_logprobs(second_logits)
        second_max_logprob = second_logprob[second_token]

        top8_logprob = np.partition(first_logprob, -max_k)[-max_k:].tolist()

        sorted_top8_logprob = sorted(top8_logprob, reverse=True)

        prob_12 = sorted_top8_logprob[0] + second_max_logprob

        k = 1
        if gamma == 16:  # 2 4 8
            for i, prob in enumerate(sorted_top8_logprob):
                if i == 0:
                    continue
                if prob_12 < prob:
                    k += 1
                else:
                    break
            if k == 3:
                k = 2
            if k == 5 or k == 6:
                k = 4
            if k == 7:
                k = 4
        elif gamma == 24:  # 2 3 4 6 8
            for i, prob in enumerate(sorted_top8_logprob):
                if i == 0:
                    continue
                if prob_12 < prob:
                    k += 1
                else:
                    break
            if k == 5:
                k = 4
            if k == 7:
                k = 6
        elif gamma == 12:  # 2 3 4 6
            for i, prob in enumerate(sorted_top8_logprob):
                if i == 0:
                    continue
                if i == 7:
                    break
                if prob_12 < prob:
                    k += 1
                else:
                    break
            if k == 5:
                k = 4
            if k == 7:
                k = 6

        assert gamma % k == 0, f"{gamma=}, {k=}, {gamma % k=}"

        input_ids = input_ids[:prefix_len]

        start_tokens = [first_token]
        gen_token = int(gamma / k) - 1
        if k > 1:
            top_k_token = np.argpartition(first_logprob, -k)[
                -k:
            ]  # 注意这里topk是无序的
            sorted_top_k_token = top_k_token[
                np.argsort(first_logprob[top_k_token])[::-1]
            ]
            start_tokens = sorted_top_k_token

        flatten_draft_k_seq_ids = []
        draft_k_seq_prob = []
        num = 0

        if gen_token == 0:
            flatten_draft_k_seq_ids = start_tokens

        else:
            for i in range(k):

                if i == 0:
                    input = input_ids + [start_tokens[i], second_token]
                    if gen_token - 1 > 0:

                        output = self._generate_llama_cpp_kseq(input, gen_token - 1)
                    else:
                        output = input
                else:
                    input = input_ids + [start_tokens[i]]
                    output = self._generate_llama_cpp_kseq(input, gen_token)

                flatten_draft_k_seq_ids += output[prefix_len:]

                if k != 1:
                    # kv cache related
                    kv_cache_size = llama_cpp.llama_get_state_size(self._model.ctx)

                    llama_cpp.llama_state_get_data(
                        self._model.ctx, self.kv_cache[num][0], kv_cache_size
                    )

                    self.kv_cache[num][1] = np.array(output[:-1])
                    self.kv_cache[num][2] = self._model.n_tokens
                    self.kv_cache[num][3] = kv_cache_size
                num += 1

        return flatten_draft_k_seq_ids, draft_k_seq_prob, k

    @torch.no_grad()
    def rollback(self, end_pos: int):
        # past_key_values_trimmed = []
        # assert self._past_key_values
        # for kv in self._past_key_values:
        #     k, v = kv
        #     k = k[:, :, :end_pos, :]
        #     v = v[:, :, :end_pos, :]
        #     kv_trimmed = (k, v)
        #     past_key_values_trimmed.append(kv_trimmed)

        # self._past_key_values = past_key_values_trimmed

        self._prob_history = self._prob_history[:, :end_pos, :]
