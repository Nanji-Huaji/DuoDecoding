import json
import os

import torch
from torch import nn
from collections.abc import Sequence

from .model_gpu import KVCacheModel

# 可选：把 acc-head 的逐 token 接受概率写出来（RL_ACC_PROB_TRACE=/path.jsonl）。
# 动机：ARP 的停止判据依赖这些概率，但它们的分布从未被观测过——而"阈值维度是否
# 可能可分辨"完全取决于它（诊断文档 F18）。
ACC_PROB_TRACE_PATH = os.environ.get("RL_ACC_PROB_TRACE")


class DecodingAdapter:
    def __init__(
        self,
        acc_head: nn.Module,
        threshold: float | None,
        model: KVCacheModel | None = None,
        stop_mode: str = "cumulative",
    ):
        self.acc_head = acc_head
        self.model: KVCacheModel | None = model
        self.threshold = threshold
        # "cumulative" (default, historical): stop when 1 - prod_i p_i > threshold.
        # The product saturates after ~2 drafted tokens whatever the threshold, which
        # makes the whole threshold dimension inert (measured: 0.05..0.95 give
        # byte-identical rollouts -- diagnosis doc F18).
        # "per_token": stop when the *latest* token's rejection probability exceeds
        # the threshold, i.e. 1 - p_last > threshold.  Monotone in the threshold, so
        # the action space dimension becomes discriminable again.
        assert stop_mode in {"cumulative", "per_token"}, stop_mode
        self.stop_mode = stop_mode
        self.last_acc_prob = 0.5
        self.step_acc_probs = []

    def reset_step(self):
        self.step_acc_probs = []

    @torch.inference_mode()
    def predict(self, hidden_states: Sequence[torch.Tensor]) -> bool:
        """
        Predict whether to stop generation based on the hidden states.
        Input:
            hidden_states: Sequence of layer hidden states, where
                hidden_states[-1] has shape (1, seq_len, hidden_size).
        Output:
            stop_prediction: bool, whether to stop generation
        """
        cum_acc_prob = 1.0
        stop_prediction = False
        # If we have history, we might want to use it, but for now we follow original logic
        # but store each token's prob in step_acc_probs.

        logits = self.acc_head((hidden_states[-1]).to(self.device).to(self.dtype))[
            0, -1
        ].float()
        if self.threshold is None:
            predicted = logits.argmax(dim=-1)
            stop_prediction = predicted == 0
            acc_prob = 1.0 if predicted == 1 else 0.0  # Approximate
        else:
            acc_prob = logits.softmax(dim=-1)[1].item()

        self.last_acc_prob = acc_prob
        self.step_acc_probs.append(acc_prob)
        if ACC_PROB_TRACE_PATH:
            try:
                with open(ACC_PROB_TRACE_PATH, "a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "step_pos": len(self.step_acc_probs),
                                "acc_prob": float(acc_prob),
                                "threshold": None if self.threshold is None else float(self.threshold),
                                "stop_mode": self.stop_mode,
                            }
                        )
                        + "\n"
                    )
            except OSError:
                pass

        if self.threshold is not None:
            # Re-calculate cumulative rejection prob for stopping decision
            # Note: This is a simplification. Usually we'd track cumulative.
            # But the original code was:
            # rej_prob = 1 - cum_acc_prob
            # stop_prediction = (rej_prob > self.threshold)
            # However, cum_acc_prob should be product of all probs in this step.

            p_prod = 1.0
            for p in self.step_acc_probs:
                p_prod *= p
            rej_prob = 1.0 - p_prod
            stop_prediction = rej_prob > self.threshold

        return stop_prediction

    @property
    def device(self):
        return next(self.acc_head.parameters()).device

    def to(self, device):
        self.acc_head.to(device)
        return self

    @property
    def dtype(self):
        return next(self.acc_head.parameters()).dtype
