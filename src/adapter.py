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
        self.last_acc_prob = 0.5
        self.step_acc_probs = []

    def reset_step(self):
        self.step_acc_probs = []

    @torch.inference_mode()
    def predict(self, hidden_states: torch.Tensor) -> bool:
        """
        Predict whether to stop generation based on the hidden states. 
        Input:
            hidden_states: Tensor of shape (1, seq_len, hidden_size). Whole hidden states of the current generation.
        Output:
            stop_prediction: bool, whether to stop generation
        """
        cum_acc_prob = 1.0
        # If we have history, we might want to use it, but for now we follow original logic
        # but store each token's prob in step_acc_probs.
        
        logits = self.acc_head((hidden_states[-1]).to(self.device).to(self.dtype))[0, -1].float()
        if self.threshold is None:
            predicted = logits.argmax(dim = -1)
            stop_prediction = (predicted == 0)
            acc_prob = 1.0 if predicted == 1 else 0.0 # Approximate
        else:
            acc_prob = logits.softmax(dim = -1)[1].item()
            
        self.last_acc_prob = acc_prob
        self.step_acc_probs.append(acc_prob)
        
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
            stop_prediction = (rej_prob > self.threshold)
        
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
