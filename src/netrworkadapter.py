from typing import List, Tuple, Dict, Any
from src.communication import CommunicationSimulator
import torch

class NetworkAdapter:
    def __init__(self, args):
        self.args = args
        self.bandwidth_history: List[float] = []

    def predict_next_bandwidth(self) -> float:
        # harmonic mean of past bandwidths
        if not self.bandwidth_history:
            return self.args.edge_cloud_bandwidth
        return len(self.bandwidth_history) / sum(1.0 / bw for bw in self.bandwidth_history)
    
    def compress_k(self, vocab: torch.Tensor, draft_acc_prob: List) -> int:
        raise NotImplementedError