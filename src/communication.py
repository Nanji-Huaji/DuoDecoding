import numpy as np
from typing import TypedDict, List, Tuple, Literal, Union, Optional
import torch

class TransferUnit(TypedDict):
    data_size_bytes: int
    transfer_time: float

class Statistics(TypedDict):
    edge_cloud: List[TransferUnit]
    edge_end: List[TransferUnit]
    cloud_end: List[TransferUnit]


LinkType = Literal["edge_cloud", "edge_end", "cloud_end"]
datatype = Union[torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.int16, torch.int8]


class CommunicationSimulator:
    def __init__(self, bandwidth_edge_cloud, bandwidth_edge_end, bandwidth_cloud_end):
        self.bandwidth_edge_cloud = bandwidth_edge_cloud
        self.bandwidth_edge_end = bandwidth_edge_end
        self.bandwidth_cloud_end = bandwidth_cloud_end
        self.stats = Statistics(
            edge_cloud=[],
            edge_end=[],
            cloud_end=[],
        )

    @property
    def edge_cloud_comm_time(self):
        return sum(self.stats["edge_cloud"][i]["transfer_time"] for i in range(len(self.stats["edge_cloud"])))

    @property
    def edge_end_comm_time(self):
        return sum(self.stats["edge_end"][i]["transfer_time"] for i in range(len(self.stats["edge_end"])))
    
    @property
    def cloud_end_comm_time(self):
        return sum(self.stats["cloud_end"][i]["transfer_time"] for i in range(len(self.stats["cloud_end"])))

    def simulate_transfer(self, data_size_bytes, link_type, add_to_stats=True):
        if link_type == "edge_cloud":
            bandwidth = self.bandwidth_edge_cloud
        elif link_type == "edge_end":
            bandwidth = self.bandwidth_edge_end
        elif link_type == "cloud_end":
            bandwidth = self.bandwidth_cloud_end
        else:
            raise ValueError(f"Unknown link type: {link_type}")

        transfer_time = data_size_bytes / bandwidth
        
        if add_to_stats:
            transfer_unit = TransferUnit(
                data_size_bytes=data_size_bytes,
                transfer_time=transfer_time
            )
            self.stats[link_type].append(transfer_unit)
            
        return transfer_time
    
    def transfer(self, tokens: torch.Tensor, prob_history: torch.Tensor, link_type: LinkType) -> float:
        token_bytes = 0
        prob_bytes = 0

        # Token data size (int32 or int64)
        if tokens is not None and tokens.numel() > 0:
            token_bytes = tokens.element_size() * tokens.numel()
        
        # Probability history data size (float32 or float16)
        if prob_history is not None and prob_history.numel() > 0:
            prob_bytes = prob_history.element_size() * prob_history.numel()
        
        total_bytes = token_bytes + prob_bytes
        
        return self.simulate_transfer(total_bytes, link_type)
    

    def __call__(self, tokens: torch.Tensor, prob_history: Optional[torch.Tensor] =None, link_type: LinkType="edge_cloud", prob_history_dtype=torch.float16, description="") -> Tuple[float, str]:
        if link_type not in ["edge_cloud", "edge_end", "cloud_end"]:
            raise ValueError(f"Unknown link type: {link_type}")
        
        token_bytes = 0
        prob_bytes = 0
        total_bytes = 0

        if tokens is not None and tokens.numel() > 0:
            token_bytes = tokens.element_size() * tokens.numel()
            total_bytes += token_bytes

        if prob_history is not None and prob_history.numel() > 0:
            if prob_history_dtype is not None:
                prob_history = prob_history.to(prob_history_dtype)
            prob_bytes = prob_history.element_size() * prob_history.numel()
            total_bytes += prob_bytes

        transfer_time = self.simulate_transfer(total_bytes, link_type)

        return transfer_time, link_type


