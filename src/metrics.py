from typing import Any, List, Optional, TypedDict
from dataclasses import dataclass

INT_SIZE = 4

@dataclass
class ModelTime:
    comp_time: float
    comm_time: float
    queuing_time: float

    def __add__(self, other: 'ModelTime') -> 'ModelTime':
        return ModelTime(
            comp_time=self.comp_time + other.comp_time,
            comm_time=self.comm_time + other.comm_time,
            queuing_time=self.queuing_time + other.queuing_time
        )

    def __radd__(self, other: 'ModelTime') -> 'ModelTime':
        if other == 0:
            return self
        else:
            return self.__add__(other)


@dataclass
class ModelTimeComposition:
    target_model_time: ModelTime
    draft_model_time: ModelTime
    little_model_time: ModelTime

    def __add__(self, other: 'ModelTimeComposition') -> 'ModelTimeComposition':
        return ModelTimeComposition(
            target_model_time=self.target_model_time + other.target_model_time,
            draft_model_time=self.draft_model_time + other.draft_model_time,
            little_model_time=self.little_model_time + other.little_model_time
        
        )
    
    def __radd__(self, other: 'ModelTimeComposition') -> 'ModelTimeComposition':
        if other == 0:
            return self
        else:
            return self.__add__(other)


class DecodingMetrics(TypedDict):
    """
    TypedDict class that defines metrics for tracking decoding performance and resource usage.

    This class serves as a type annotation for dictionaries containing comprehensive
    metrics about the decoding process, including forward pass counts, token statistics,
    timing information, communication overhead, and energy consumption.
    """

    little_forward_times: int
    draft_forward_times: int
    target_forward_times: int
    little_computation_time: float
    draft_computation_time: float
    target_computation_time: float
    generated_tokens: int
    little_generated_tokens: int
    draft_generated_tokens: int
    little_accepted_tokens: int
    draft_accepted_tokens: int
    wall_time: float
    throughput: float
    communication_time: float
    computation_time: float
    edge_end_comm_time: float
    edge_cloud_data_bytes: int | float
    edge_end_data_bytes: int | float
    cloud_end_data_bytes: int | float
    loop_times: int
    each_loop_draft_tokens: float
    comm_energy: float
    connect_times: dict
    accuracy: Optional[Any]
    queuing_time: int | float
    arp_overhead_time: float
    dra_overhead_time: float
    avg_top_k: float
    avg_draft_len: float
    edge_cloud_bandwidth_history: List[float]
    edge_cloud_topk_history: List[int]
    edge_cloud_draft_len_history: List[int]
    little_entropy_history: List[float]
    draft_entropy_history: List[float]
    little_accept_rate_history: List[float]
    draft_accept_rate_history: List[float]
    little_accepted_vocab_rank_history: List[int]
    draft_accepted_vocab_rank_history: List[int]
    little_accepted_in_transfer_topk_history: List[bool]
    draft_accepted_in_transfer_topk_history: List[bool]
    little_accepted_transfer_topk_rank_history: List[int]
    draft_accepted_transfer_topk_rank_history: List[int]


def get_empty_metrics() -> DecodingMetrics:
    """
    Create and return an empty DecodingMetrics object with all fields initialized to zero.
    """
    return DecodingMetrics(
        little_forward_times=0,
        draft_forward_times=0,
        target_forward_times=0,
        little_computation_time=0.0,
        draft_computation_time=0.0,
        target_computation_time=0.0,
        generated_tokens=0,
        little_generated_tokens=0,
        draft_generated_tokens=0,
        little_accepted_tokens=0,
        draft_accepted_tokens=0,
        wall_time=0.0,
        throughput=0.0,
        communication_time=0.0,
        computation_time=0.0,
        edge_end_comm_time=0.0,
        edge_cloud_data_bytes=0,
        edge_end_data_bytes=0,
        cloud_end_data_bytes=0,
        loop_times=0,
        each_loop_draft_tokens=0.0,
        comm_energy=0.0,
        connect_times={},
        accuracy=None,
        queuing_time=0.0,
        arp_overhead_time=0.0,
        dra_overhead_time=0.0,
        avg_top_k=0.0,
        avg_draft_len=0.0,
        edge_cloud_bandwidth_history=[],
        edge_cloud_topk_history=[],
        edge_cloud_draft_len_history=[],
        little_entropy_history=[],
        draft_entropy_history=[],
        little_accept_rate_history=[],
        draft_accept_rate_history=[],
        little_accepted_vocab_rank_history=[],
        draft_accepted_vocab_rank_history=[],
        little_accepted_in_transfer_topk_history=[],
        draft_accepted_in_transfer_topk_history=[],
        little_accepted_transfer_topk_rank_history=[],
        draft_accepted_transfer_topk_rank_history=[],
    )
