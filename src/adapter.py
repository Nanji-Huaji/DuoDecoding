import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import os
from torch import nn


from .model_gpu import KVCacheModel

from typing import TypedDict

import logger


class DecodingAdapter:
    def __init__(self, acc_head: nn.Module, threshold: float | None, model: KVCacheModel | None = None):
        self.acc_head = acc_head
        self.model: KVCacheModel | None = model
        self.threshold = threshold
        self.device = next(acc_head.parameters()).device
        self.dtype = next(acc_head.parameters()).dtype

    @torch.no_grad()
    def predict(self, hidden_states: torch.Tensor) -> bool:
        """
        Predict whether to stop generation based on the hidden states.
        Input:
            hidden_states: Tensor of shape (1, seq_len, hidden_size)
        Output:
            stop_prediction: bool, whether to stop generation
        """
        cum_acc_prob = 1.0
        logits = self.acc_head((hidden_states[-1]).to(self.device).to(self.dtype))[0, -1].float()
        if self.threshold is None:
                    predicted = logits.argmax(dim = -1)
                    stop_prediction = (predicted == 0)
        else:
                    acc_prob = logits.softmax(dim = -1)[1].item()
                    cum_acc_prob *= acc_prob
                    rej_prob = 1 - cum_acc_prob
                    
                    stop_prediction = (rej_prob > self.threshold)
        return stop_prediction
    
    @torch.no_grad()
    def __call__(self, hidden_states: torch.Tensor) -> bool:
        stop = self.predict(hidden_states)
        if not stop:
            return False
        else:
            if self.model is None:
                return True
            else:
                # rollback kvcache
                self.model.rollback(1)
                return True
            # reject token to stop generation
