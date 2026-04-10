from typing import Any, List, Optional, TypedDict

INT_SIZE = 4


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
