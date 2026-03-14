import os
import warnings
import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist
import transformers

transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, TypedDict, Literal

from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .communication import (
    CUHLM,
    CommunicationSimulator,
    PreciseCommunicationSimulator,
    PreciseCUHLM,
)
from .model_gpu import KVCacheModel
from .register import Register
from .utils import (
    log_prob_tensor_if_invalid,
    log_ratio_if_invalid,
    max_fn,
    sample,
    seed_everything,
)

try:
    import flash_attn
except ImportError:
    pass

from functools import partial

# from .register import Register

flash_attn_available = "flash_attn" in globals()
logger = logging.getLogger(__name__)

attn_impl = "sdpa" if not flash_attn_available else "flash_attention_2"

INT_SIZE = 4


@dataclass
class VerificationInputs:
    draft_probs_batch: torch.Tensor
    target_probs_batch: torch.Tensor
    draft_tokens: torch.Tensor
    draft_token_indices: torch.Tensor
    prefix_len: int
    gamma: int
    actual_gamma: int
    max_idx: int


@dataclass
class AcceptanceResult:
    accepted_count: int
    n: int
    selected_draft_p: torch.Tensor
    selected_target_p: torch.Tensor
    accept_mask: torch.Tensor


@dataclass
class RollbackPlan:
    draft_end_pos: int
    target_end_pos_reject: int
    target_end_pos_accept: int
    all_accepted: bool


def _sd_alignment_debug_enabled() -> bool:
    return os.environ.get("DUODEC_DEBUG_SD_ALIGNMENT", "0") == "1"


def _format_cache_state(name: str, cache: KVCacheModel) -> str:
    state = cache.debug_state()
    return (
        f"{name}(current={state['current_length']}, "
        f"tracked={state['tracked_seq_len']}, "
        f"prob={state['prob_history_len']}, "
        f"logits={state['logits_history_len']}, "
        f"max={state['max_length']})"
    )


def _log_sd_alignment_snapshot(
    stage: str,
    prefix_len: int,
    approx_model_cache: KVCacheModel,
    target_model_cache: KVCacheModel,
    *,
    x_len: Optional[int] = None,
    gamma: Optional[int] = None,
    note: str = "",
) -> None:
    if not _sd_alignment_debug_enabled():
        return

    message = (
        f"[SD-ALIGN] stage={stage}, prefix_len={prefix_len}, "
        f"x_len={x_len}, gamma={gamma}, "
        f"{_format_cache_state('approx', approx_model_cache)}, "
        f"{_format_cache_state('target', target_model_cache)}"
    )
    if note:
        message += f", note={note}"
    logger.warning(message)


def _log_invalid_batch_details(
    *,
    prefix_len: int,
    gamma: int,
    max_idx: int,
    actual_gamma: int,
    x: torch.Tensor,
    draft_model_cache: KVCacheModel,
    target_model_cache: KVCacheModel,
    draft_probs_batch: torch.Tensor,
    target_probs_batch: torch.Tensor,
    selected_draft_p: torch.Tensor,
    selected_target_p: torch.Tensor,
) -> None:
    draft_row_sums = draft_probs_batch[0].detach().float().sum(dim=-1).cpu().tolist()
    target_row_sums = target_probs_batch[0].detach().float().sum(dim=-1).cpu().tolist()
    draft_tokens = x[:, prefix_len : prefix_len + actual_gamma].detach().cpu().tolist()

    target_window_start = max(0, prefix_len - 2)
    target_window_end = max_idx + 2
    approx_window_start = max(0, prefix_len - 2)
    approx_window_end = max_idx + 2

    logger.warning(
        "[SD-ALIGN][invalid-batch] prefix_len=%s gamma=%s max_idx=%s actual_gamma=%s "
        "draft_tokens=%s selected_draft_p=%s selected_target_p=%s "
        "draft_row_sums=%s target_row_sums=%s "
        "target_window_row_sums=%s approx_window_row_sums=%s "
        "%s %s",
        prefix_len,
        gamma,
        max_idx,
        actual_gamma,
        draft_tokens,
        selected_draft_p.detach().float().cpu().tolist(),
        selected_target_p.detach().float().cpu().tolist(),
        draft_row_sums,
        target_row_sums,
        target_model_cache.debug_row_sums(target_window_start, target_window_end),
        draft_model_cache.debug_row_sums(approx_window_start, approx_window_end),
        _format_cache_state("draft", draft_model_cache),
        _format_cache_state("target", target_model_cache),
    )


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
    )


class Decoding(Register, ABC):
    def __init__(self, args):
        Register.__init__(self, args)
        self.args = args
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            size = int(os.environ["WORLD_SIZE"])
            if "duodec" in args.eval_mode:
                dist.init_process_group(
                    "gloo", init_method="env://", rank=rank, world_size=size
                )
            else:
                dist.init_process_group(
                    "nccl", init_method="env://", rank=rank, world_size=size
                )
        self.accelerator = Accelerator()

        seed_everything(self.args.seed)
        self.seed = self.args.seed
        self.seed_set = set()

        # record metrics for report
        self.draft_forward_times = 0
        self.little_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []
        self.prob_with_flag = []

        self.vocab_size = -1
        self.stop_tokens_matrix = None

    def _prepare_stop_tokens(self, stop_sequences: List[str]):
        """
        预处理停止词序列，将其转换为 GPU 上的张量矩阵，以便在生成过程中进行高效的广播检查。
        """
        if not stop_sequences or not getattr(self, "tokenizer", None):
            raise ValueError("Stop sequences provided but tokenizer is not available.")

        # 1. 获取完整的 ID 序列
        stop_ids_list = [
            self.tokenizer.encode(s, add_special_tokens=False) for s in stop_sequences
        ]

        if not stop_ids_list:
            self.stop_tokens_matrix = None
            return

        # 2. 找出最长的停止词长度
        max_len = max(len(ids) for ids in stop_ids_list)

        # 3. 填充并转为 Tensor (用 -1 填充左侧，方便右对齐比对)
        # 形状: [停止词个数, 最长长度]
        # 确保 device 正确，这里假设 self.target_model 已经加载
        device = (
            self.target_model.device
            if hasattr(self, "target_model") and self.target_model is not None
            else "cpu"
        )

        matrix = torch.full(
            (len(stop_ids_list), max_len),
            -1,
            dtype=torch.long,
            device=device,
        )
        for i, ids in enumerate(stop_ids_list):
            matrix[i, -len(ids) :] = torch.tensor(ids, device=device)

        self.stop_tokens_matrix = matrix

    def _should_stop(
        self,
        prefix: torch.Tensor,
        max_tokens: int,
        use_early_stopping: bool = False,
    ) -> bool:
        """
        Unified stopping criteria check.
        Prioritizes checks on GPU to minimize synchronization overhead.
        """
        # 1. Length check
        if prefix.shape[1] >= max_tokens:
            return True

        if not use_early_stopping:
            return False

        # 2. EOS check
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            # prefix: [batch, seq_len]
            if prefix[0, -1] == self.tokenizer.eos_token_id:
                return True

        # 3. Stop tokens matrix check
        if self.stop_tokens_matrix is not None:
            max_stop_len = self.stop_tokens_matrix.size(1)
            # Only check a reasonable window at the end
            check_window = max(64, max_stop_len + 10)

            # Get the trailing sequence
            seq = prefix[0, -check_window:]

            # If sequence is shorter than any stop token, skip
            if seq.size(0) < max_stop_len:
                return False

            # Use unfold to create sliding windows: [num_windows, max_stop_len]
            windows = seq.unfold(0, max_stop_len, 1)

            # Broadcasting comparison:
            # windows: [1, W, L]
            # matrix:  [S, 1, L]
            targets = windows.unsqueeze(0)
            stops = self.stop_tokens_matrix.unsqueeze(1)

            # Check matches, treating -1 in stops as always matching (padding)
            matches = (targets == stops) | (stops == -1)

            # Check if any full sequence matches in any window
            # dimensions: [S, W, L] -> all(dim=-1) -> [S, W] -> any() -> bool
            if matches.all(dim=-1).any():
                return True

        return False

    def _check_stopping_criteria(
        self, input_ids: torch.Tensor, stop_sequences: Optional[List[str]] = None
    ) -> bool:
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            return False

        # Check for EOS at the last position only
        if (
            input_ids.shape[1] > 0
            and input_ids[0, -1].item() == self.tokenizer.eos_token_id
        ):
            return True

        # Check for stop sequences
        if stop_sequences:
            decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            for stop_seq in stop_sequences:
                if decoded_text.endswith(stop_seq):
                    return True
        return False

    @staticmethod
    def _prepare_verification_inputs(
        draft_model_cache: KVCacheModel,
        target_model_cache: KVCacheModel,
        x: torch.Tensor,
        prefix_len: int,
        gamma: int,
        draft_probs_override: Optional[torch.Tensor] = None,
    ) -> VerificationInputs:
        draft_device = draft_model_cache.device
        draft_probs = (
            draft_probs_override
            if draft_probs_override is not None
            else draft_model_cache._prob_history
        )
        if draft_probs is None or target_model_cache._prob_history is None:
            raise ValueError("Probability history is not initialized for verification")

        max_idx = min(
            prefix_len + gamma - 1,
            draft_probs.shape[1],
            target_model_cache._prob_history.shape[1],
        )
        actual_gamma = max_idx - (prefix_len - 1)
        if actual_gamma <= 0:
            empty_tokens = x[:, 0:0]
            empty_indices = empty_tokens.unsqueeze(-1)
            empty_probs = draft_probs[:, 0:0, :]
            return VerificationInputs(
                draft_probs_batch=empty_probs,
                target_probs_batch=target_model_cache._prob_history[:, 0:0, :].to(
                    draft_device
                ),
                draft_tokens=empty_tokens,
                draft_token_indices=empty_indices,
                prefix_len=prefix_len,
                gamma=gamma,
                actual_gamma=0,
                max_idx=max_idx,
            )

        draft_probs_batch = draft_probs[:, prefix_len - 1 : max_idx, :]
        target_probs_batch = target_model_cache._prob_history[
            :, prefix_len - 1 : max_idx, :
        ].to(draft_device)
        draft_tokens = x[:, prefix_len : prefix_len + actual_gamma]
        draft_token_indices = draft_tokens.unsqueeze(-1)

        return VerificationInputs(
            draft_probs_batch=draft_probs_batch,
            target_probs_batch=target_probs_batch,
            draft_tokens=draft_tokens,
            draft_token_indices=draft_token_indices,
            prefix_len=prefix_len,
            gamma=gamma,
            actual_gamma=actual_gamma,
            max_idx=max_idx,
        )

    @staticmethod
    def _compute_acceptance_result(
        verification_inputs: VerificationInputs,
        *,
        r: Optional[torch.Tensor] = None,
    ) -> AcceptanceResult:
        if verification_inputs.actual_gamma <= 0:
            return AcceptanceResult(
                accepted_count=0,
                n=verification_inputs.prefix_len - 1,
                selected_draft_p=verification_inputs.draft_probs_batch[:, 0:0, 0],
                selected_target_p=verification_inputs.target_probs_batch[:, 0:0, 0],
                accept_mask=torch.zeros(
                    (verification_inputs.draft_probs_batch.shape[0], 0),
                    dtype=torch.bool,
                    device=verification_inputs.draft_probs_batch.device,
                ),
            )

        selected_draft_p = torch.gather(
            verification_inputs.draft_probs_batch,
            2,
            verification_inputs.draft_token_indices,
        ).squeeze(-1)
        selected_target_p = torch.gather(
            verification_inputs.target_probs_batch,
            2,
            verification_inputs.draft_token_indices,
        ).squeeze(-1)

        if r is None:
            r = torch.rand(
                (selected_draft_p.shape[0], verification_inputs.actual_gamma),
                device=selected_draft_p.device,
            )

        accept_mask = r <= (selected_target_p / selected_draft_p)
        continuous_accept, _ = accept_mask.to(torch.int8).cummin(dim=1)
        accepted_count = int(continuous_accept[0].sum().item())
        n = verification_inputs.prefix_len + accepted_count - 1

        if accepted_count == verification_inputs.actual_gamma:
            n = verification_inputs.prefix_len + verification_inputs.actual_gamma - 1

        return AcceptanceResult(
            accepted_count=accepted_count,
            n=int(n),
            selected_draft_p=selected_draft_p,
            selected_target_p=selected_target_p,
            accept_mask=accept_mask,
        )

    @staticmethod
    def _compute_residual_distribution(
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
    ) -> torch.Tensor:
        return max_fn(target_probs - draft_probs)

    @staticmethod
    def _build_rollback_plan(prefix_len: int, gamma: int, n: int) -> RollbackPlan:
        all_accepted = n >= prefix_len + gamma - 1
        return RollbackPlan(
            draft_end_pos=n + 1,
            target_end_pos_reject=n + 1,
            target_end_pos_accept=n + 2,
            all_accepted=all_accepted,
        )

    @staticmethod
    def _apply_rollback(
        draft_model_cache: KVCacheModel,
        target_model_cache: KVCacheModel,
        rollback_plan: RollbackPlan,
    ) -> None:
        draft_model_cache.rollback(rollback_plan.draft_end_pos)
        if rollback_plan.all_accepted:
            target_model_cache.rollback(rollback_plan.target_end_pos_accept)
        else:
            target_model_cache.rollback(rollback_plan.target_end_pos_reject)

    @staticmethod
    def _sample_reject_token(
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
        output_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if target_probs.device != draft_probs.device:
            target_probs = target_probs.to(draft_probs.device)

        residual_probs = Decoding._compute_residual_distribution(
            target_probs,
            draft_probs,
        )
        log_prob_tensor_if_invalid(
            residual_probs,
            "Decoding._sample_reject_token.residual_probs",
        )
        token = sample(residual_probs)
        if output_device is not None and token.device != output_device:
            token = token.to(output_device)
        return token

    @staticmethod
    def _sample_accept_token(
        target_next_probs: torch.Tensor,
        output_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        log_prob_tensor_if_invalid(
            target_next_probs,
            "Decoding._sample_accept_token.target_next_probs",
        )
        token = sample(target_next_probs)
        if output_device is not None and token.device != output_device:
            token = token.to(output_device)
        return token

    @staticmethod
    def _verify_draft_sequence(
        draft_model_cache: KVCacheModel,
        target_model_cache: KVCacheModel,
        x: torch.Tensor,
        prefix_len: int,
        gamma: int,
        comm_simulator: Optional[CommunicationSimulator] = None,
        comm_link: Literal["edge_cloud", "edge_end", "cloud_end"] = "edge_cloud",
        transfer_mode: Literal["none", "serial", "batch_before"] = "serial",
        send_reject_message: bool = True,
        draft_probs_override: Optional[torch.Tensor] = None,
        decoding_metrics: Optional[DecodingMetrics] = None,
    ) -> Tuple[int, int]:
        """
        Verify draft sequence using rejection sampling with target model probabilities.

        This function supports multiple verification scenarios used across different decoding methods:
        - Standard speculative decoding (dist_spec)
        - Tridecoding (two-layer verification)
        - Adaptive decoding (with serial transfer)
        - Uncertainty-based methods

        Args:
            draft_model_cache: KVCache of the draft/approximation model
            target_model_cache: KVCache of the target/verification model
            x: Generated token sequence [batch_size, seq_len]
            prefix_len: Length of the prefix before draft tokens
            gamma: Number of draft tokens to verify
            comm_simulator: Communication simulator (optional)
            comm_link: Communication link type ("edge_cloud", "edge_end", "cloud_end")
            transfer_mode: How to handle token/prob transfer:
                - "none": No transfer (already transferred externally)
                - "serial": Transfer token and prob one by one during verification
                - "batch_before": Batch transfer before verification (caller handles this)
            send_reject_message: Whether to send reject message on rejection
            draft_probs_override: Override draft probs (if None, use draft_model_cache._prob_history)
            decoding_metrics: Optional metrics dict to update

        Returns:
            Tuple[int, int]: (accepted_count, final_position)
                - accepted_count: Number of tokens accepted
                - final_position: Final position index (n)
        """
        draft_device = draft_model_cache.device
        _log_sd_alignment_snapshot(
            "verify_enter",
            prefix_len,
            draft_model_cache,
            target_model_cache,
            x_len=x.shape[1],
            gamma=gamma,
        )

        verification_inputs = Decoding._prepare_verification_inputs(
            draft_model_cache=draft_model_cache,
            target_model_cache=target_model_cache,
            x=x,
            prefix_len=prefix_len,
            gamma=gamma,
            draft_probs_override=draft_probs_override,
        )
        if verification_inputs.actual_gamma <= 0:
            return 0, prefix_len - 1

        draft_probs_batch = verification_inputs.draft_probs_batch
        target_probs_batch = verification_inputs.target_probs_batch
        invalid_draft_probs = log_prob_tensor_if_invalid(
            draft_probs_batch,
            "Decoding._verify_draft_sequence.draft_probs_batch",
        )
        invalid_target_probs = log_prob_tensor_if_invalid(
            target_probs_batch,
            "Decoding._verify_draft_sequence.target_probs_batch",
        )

        # Batch transfer mode: transfer all tokens and probs before verification
        if transfer_mode == "batch_before" and comm_simulator is not None:
            batch_probs = torch.gather(
                draft_probs_batch,
                2,
                verification_inputs.draft_token_indices,
            ).squeeze(-1)
            comm_simulator.transfer(
                verification_inputs.draft_tokens,
                batch_probs,
                comm_link,
            )

        # Rejection sampling masking
        r = torch.rand(
            (x.shape[0], verification_inputs.actual_gamma),
            device=draft_device,
        )
        acceptance_result = Decoding._compute_acceptance_result(
            verification_inputs,
            r=r,
        )
        invalid_acceptance_ratio = log_ratio_if_invalid(
            acceptance_result.selected_target_p,
            acceptance_result.selected_draft_p,
            "Decoding._verify_draft_sequence.acceptance_ratio",
        )
        if invalid_draft_probs or invalid_target_probs or invalid_acceptance_ratio:
            _log_invalid_batch_details(
                prefix_len=prefix_len,
                gamma=gamma,
                max_idx=verification_inputs.max_idx,
                actual_gamma=verification_inputs.actual_gamma,
                x=x,
                draft_model_cache=draft_model_cache,
                target_model_cache=target_model_cache,
                draft_probs_batch=draft_probs_batch,
                target_probs_batch=target_probs_batch,
                selected_draft_p=acceptance_result.selected_draft_p,
                selected_target_p=acceptance_result.selected_target_p,
            )
        accepted_counts = acceptance_result.accepted_count
        n = acceptance_result.n
        if accepted_counts < verification_inputs.actual_gamma:
            if send_reject_message and comm_simulator:
                comm_simulator.send_reject_message(comm_link)

        # Simulate serial transfer correctly even when computing is vectorized
        if transfer_mode == "serial" and comm_simulator is not None:
            # Just transfer up to the rejected token
            for i in range(
                accepted_counts
                + (1 if accepted_counts < verification_inputs.actual_gamma else 0)
            ):
                comm_simulator.transfer(
                    verification_inputs.draft_token_indices[0, i, 0],
                    draft_probs_batch[:, i, :].squeeze(0),
                    comm_link,
                )

        # Update metrics if provided
        if decoding_metrics is not None:
            decoding_metrics["draft_generated_tokens"] += gamma
            decoding_metrics["draft_accepted_tokens"] += int(n - prefix_len + 1)

        return acceptance_result.accepted_count, int(n)

    @staticmethod
    def _finalize_verification(
        approx_model_cache: KVCacheModel,
        target_model_cache: KVCacheModel,
        x: torch.Tensor,
        prefix_len: int,
        gamma: int,
        n: int,
    ) -> torch.Tensor:
        """
        Finalize the verification phase by rolling back KV caches and sampling the next token.
        """
        # Truncate x to the accepted prefix length
        prefix = x[:, : n + 1]
        rollback_plan = Decoding._build_rollback_plan(prefix_len, gamma, n)

        # Finalize-time cache invariants:
        # - during verification: target.current_length == draft.current_length + 1
        # - after finalize: draft.current_length == target.current_length
        # - after finalize: cache.current_length == len(prefix) - 1, because the
        #   new sampled token is appended to `prefix` before it is forwarded
        approx_model_cache.rollback(rollback_plan.draft_end_pos)

        if not rollback_plan.all_accepted:
            # Rejection: Sample from residual distribution max(0, target_prob - approx_prob)
            target_prob_slice = target_model_cache._prob_history[
                :, n, : target_model_cache.vocab_size
            ]
            approx_prob_slice = approx_model_cache._prob_history[
                :, n, : approx_model_cache.vocab_size
            ]

            t = Decoding._sample_reject_token(
                target_prob_slice,
                approx_prob_slice,
                output_device=prefix.device,
            )
            target_model_cache.rollback(rollback_plan.target_end_pos_reject)
        else:
            # All accepted: sample the already-computed next-token distribution.
            next_target_probs = target_model_cache._prob_history[
                :, -1, : target_model_cache.vocab_size
            ]
            t = Decoding._sample_accept_token(
                next_target_probs,
                output_device=prefix.device,
            )
            target_model_cache.rollback(rollback_plan.target_end_pos_accept)

        # Append the new sampled token
        return torch.cat((prefix, t), dim=1)

    def _get_device_map_strategy(self, model_name: str, num_gpus_available: int) -> str:
        """
        Determine the appropriate device_map strategy based on model size and available GPUs.

        新策略：避免单模型跨卡，大模型使用Q4量化放在单卡

        Args:
            model_name: Name of the model to load
            num_gpus_available: Number of GPUs available for model loading

        Returns:
            device_map strategy string
        """
        # Extract model size
        pattern = r"(\d+(?:\.\d+)?(?:[xX]\d+)?)[bB]"
        match = re.search(pattern, model_name)
        params = float(match.group(1)) if match else 0

        # 对于所有模型，优先使用单卡加载，避免跨卡通信
        # 大模型会在load_model中使用Q4量化
        return "auto"

    def _get_available_gpu_count(self) -> int:
        """
        Get the number of available GPUs, considering CUDA_VISIBLE_DEVICES.

        Returns:
            Number of available GPUs
        """
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            if visible_devices:
                # Count the number of devices in CUDA_VISIBLE_DEVICES
                return len([d for d in visible_devices.split(",") if d.strip()])

        # Fallback to torch.cuda.device_count()
        return torch.cuda.device_count()

    def load_model(self):
        # * load models according to different evaluation methods.
        self.color_print(
            f"Loading models:\n{self.args.draft_model}\n{self.args.target_model}",
            3,
        )

        # Get available GPU count
        num_gpus = self._get_available_gpu_count()
        self.color_print(f"Available GPUs: {num_gpus}", 3)

        # Helper function to extract model size
        def get_model_size(model_name):
            pattern = r"(\d+(?:\.\d+)?(?:[xX]\d+)?)[bB]"
            match = re.search(pattern, model_name)
            return float(match.group(1)) if match else 0

        # Helper function to determine if model needs quantization
        # A6000 有 48GB 显存，约可容纳 ~24B 全精度或 ~16B bf16模型
        # 使用保守阈值：>20B 使用 Q4 量化
        def should_quantize(model_name):
            size = get_model_size(model_name)
            is_awq = "awq" in model_name.lower()
            return size > 20 and not is_awq

        # Helper function to get quantization config
        def get_quant_config(model_name):
            if should_quantize(model_name):
                print(
                    f"📦 Model {model_name} ({get_model_size(model_name)}B) will use 4-bit quantization"
                )
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            return None

        loader = partial(
            AutoModelForCausalLM.from_pretrained,
            local_files_only=False,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if self.args.eval_mode == "small":
            device_map = "cuda:0"
            self.color_print(f"Loading draft model on {device_map}", 3)
            draft_quant = get_quant_config(self.args.draft_model)
            if draft_quant:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=device_map,
                    quantization_config=draft_quant,
                ).eval()
            else:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=device_map,
                ).eval()

        elif self.args.eval_mode == "large":
            device_map = "cuda:0"
            self.color_print(f"Loading target model on {device_map}", 3)
            target_quant = get_quant_config(self.args.target_model)
            if target_quant:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=device_map,
                    quantization_config=target_quant,
                ).eval()
            else:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=device_map,
                ).eval()

        elif self.args.eval_mode in [
            "sd",
            "dsd",
            "dssd",
            "dist_spec",
            "dist_split_spec",
            "uncertainty_decoding",
            "cuhlm",
            "speculative_decoding_with_bandwidth",
            "speculative_decoding_with_bandwidth_full_prob",
        ]:
            # 双模型场景：大模型在一张卡，小模型在另一张卡
            draft_size = get_model_size(self.args.draft_model)
            target_size = get_model_size(self.args.target_model)

            # 将大模型放在GPU 0，小模型放在GPU 1（如果有多个GPU）
            if target_size > draft_size:
                draft_device = "cuda:1" if num_gpus > 1 else "cuda:0"
                target_device = "cuda:0"
                print(
                    f"🎯 Target ({target_size}B) -> GPU 0, Draft ({draft_size}B) -> GPU {1 if num_gpus > 1 else 0}"
                )
            else:
                draft_device = "cuda:0"
                target_device = "cuda:1" if num_gpus > 1 else "cuda:0"
                print(
                    f"🎯 Draft ({draft_size}B) -> GPU 0, Target ({target_size}B) -> GPU {1 if num_gpus > 1 else 0}"
                )

            self.color_print(f"Loading draft model on {draft_device}", 3)
            draft_quant = get_quant_config(self.args.draft_model)
            if draft_quant:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=draft_device,
                    quantization_config=draft_quant,
                ).eval()
            else:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=draft_device,
                ).eval()

            self.color_print(f"Loading target model on {target_device}", 3)
            target_quant = get_quant_config(self.args.target_model)
            if target_quant:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=target_device,
                    quantization_config=target_quant,
                ).eval()
            else:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=target_device,
                ).eval()

        elif self.args.eval_mode == "adaptive_decoding":
            # adaptive_decoding: 将大模型放GPU 0，小模型放GPU 1
            draft_size = get_model_size(self.args.draft_model)
            target_size = get_model_size(self.args.target_model)

            if target_size > draft_size:
                draft_device = "cuda:1" if num_gpus > 1 else "cuda:0"
                target_device = "cuda:0"
            else:
                draft_device = "cuda:0"
                target_device = "cuda:1" if num_gpus > 1 else "cuda:0"

            self.color_print(f"Loading draft model on {draft_device}", 3)
            draft_quant = get_quant_config(self.args.draft_model)
            if draft_quant:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=draft_device,
                    output_hidden_states=True,
                    quantization_config=draft_quant,
                ).eval()
            else:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=draft_device,
                    output_hidden_states=True,
                ).eval()

            self.color_print(f"Loading target model on {target_device}", 3)
            target_quant = get_quant_config(self.args.target_model)
            if target_quant:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=target_device,
                    quantization_config=target_quant,
                ).eval()
            else:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=target_device,
                ).eval()

        elif self.args.eval_mode in [
            "tridecoding",
            "adaptive_tridecoding",
            "cee_sd",
            "ceesd_without_arp",
            "ceesd_w/o_arp",
            "cee_cuhlm",
            "cee_dsd",
            "cee_dssd",
        ]:
            output_hidden_states = self.args.eval_mode in [
                "adaptive_tridecoding",
                "cee_sd",
                "cee_cuhlm",
            ]

            # 分配策略：
            # 1. 计算三个模型的大小
            # 2. 将最大的模型放在一张卡上（使用Q4量化如果需要）
            # 3. 将其余两个模型放在另一张卡上
            # 4. 避免单模型跨卡

            little_size = get_model_size(self.args.little_model)
            draft_size = get_model_size(self.args.draft_model)
            target_size = get_model_size(self.args.target_model)

            model_sizes = [
                (little_size, "little", self.args.little_model),
                (draft_size, "draft", self.args.draft_model),
                (target_size, "target", self.args.target_model),
            ]
            model_sizes.sort(reverse=True, key=lambda x: x[0])  # 按大小降序

            largest_model = model_sizes[0]
            print(
                f"🎯 Largest model: {largest_model[1]} ({largest_model[0]}B) -> GPU 0"
            )
            print(
                f"📍 Other models: {model_sizes[1][1]} ({model_sizes[1][0]}B), {model_sizes[2][1]} ({model_sizes[2][0]}B) -> GPU 1"
            )

            # 分配设备
            little_device = "cuda:1" if largest_model[1] != "little" else "cuda:0"
            draft_device = "cuda:1" if largest_model[1] != "draft" else "cuda:0"
            target_device = "cuda:1" if largest_model[1] != "target" else "cuda:0"

            # 如果只有1个GPU，全部放在cuda:0
            if num_gpus == 1:
                little_device = draft_device = target_device = "cuda:0"
                print("⚠️  Only 1 GPU available, all models will be loaded on cuda:0")

            self.color_print(f"Loading little model on {little_device}", 3)
            little_quant = get_quant_config(self.args.little_model)
            if little_quant:
                self.little_model = loader(
                    self.args.little_model,
                    device_map=little_device,
                    output_hidden_states=output_hidden_states,
                    quantization_config=little_quant,
                ).eval()
            else:
                self.little_model = loader(
                    self.args.little_model,
                    device_map=little_device,
                    output_hidden_states=output_hidden_states,
                ).eval()

            self.color_print(f"Loading draft model on {draft_device}", 3)
            draft_quant = get_quant_config(self.args.draft_model)
            if draft_quant:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=draft_device,
                    output_hidden_states=output_hidden_states,
                    quantization_config=draft_quant,
                ).eval()
            else:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=draft_device,
                    output_hidden_states=output_hidden_states,
                ).eval()

            self.color_print(f"Loading target model on {target_device}", 3)
            target_quant = get_quant_config(self.args.target_model)
            if target_quant:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=target_device,
                    quantization_config=target_quant,
                ).eval()
            else:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=target_device,
                ).eval()

        # 从实际模型embedding层获取vocab_size
        if hasattr(self, "target_model") and self.target_model is not None:
            actual_vocab_size = self.target_model.get_input_embeddings().weight.shape[0]
            self.vocab_size = actual_vocab_size
            print(f"✅ Using vocab_size from target model embedding: {self.vocab_size}")
        elif hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.vocab_size = self.tokenizer.vocab_size
            print(f"⚠️  Using vocab_size from tokenizer: {self.vocab_size}")
        else:
            self.vocab_size = int(self.args.vocab_size)
            print(f"⚠️  Using vocab_size from args: {self.vocab_size}")

        # Print device allocation for loaded models
        self._print_model_device_info()

    def _print_model_device_info(self):
        """Print device allocation information for all loaded models."""
        self.color_print("=" * 60, 3)
        self.color_print("Model Device Allocation:", 3)
        self.color_print("=" * 60, 3)

        if hasattr(self, "little_model") and self.little_model is not None:
            self._print_single_model_device_info("Little Model", self.little_model)

        if hasattr(self, "draft_model") and self.draft_model is not None:
            self._print_single_model_device_info("Draft Model", self.draft_model)

        if hasattr(self, "target_model") and self.target_model is not None:
            self._print_single_model_device_info("Target Model", self.target_model)

        self.color_print("=" * 60, 3)

    def _print_single_model_device_info(self, model_name: str, model):
        """Print device allocation for a single model."""
        self.color_print(f"\n{model_name}:", 3)

        if hasattr(model, "hf_device_map"):
            device_map = model.hf_device_map
            device_summary = {}

            for layer_name, device in device_map.items():
                device_str = str(device)
                if device_str not in device_summary:
                    device_summary[device_str] = []
                device_summary[device_str].append(layer_name)

            for device, layers in sorted(device_summary.items()):
                self.color_print(f"  {device}: {len(layers)} layers", 3)

        elif hasattr(model, "device"):
            self.color_print(f"  Device: {model.device}", 3)
        else:
            # Try to get device from first parameter
            try:
                first_param = next(model.parameters())
                self.color_print(f"  Device: {first_param.device}", 3)
            except StopIteration:
                self.color_print("  Device: Unknown (no parameters)", 3)

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.target_model}...", 3)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.target_model,
            trust_remote_code=True,
            local_files_only=False,
        )
        self.tokenizer.padding_side = "right"

        if self.tokenizer.pad_token_id is None:
            model_name = str(self.args.target_model).lower()
            if "llama" in model_name:
                self.tokenizer.pad_token_id = 2
            elif self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess(self, input_text):
        pass

    @abstractmethod
    def postprocess(self, input_text, output_text):
        pass

    @Register.register_decoding("large")
    @Register.register_decoding("small")
    @torch.inference_mode()
    def autoregressive_sampling(
        self,
        prefix,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        if self.args.eval_mode == "small":
            model = self.draft_model
        elif self.args.eval_mode == "large":
            model = self.target_model
        else:
            raise RuntimeError(
                "Auto-Regressive Decoding can be used only in small / large eval mode!"
            )
        prefix = prefix.to(model.device)
        model = KVCacheModel(model, self.args.temp, self.args.top_k, self.args.top_p)
        model.vocab_size = self.args.vocab_size

        prefix_len = prefix.shape[1]
        max_tokens = prefix_len + self.args.max_tokens

        x = prefix

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        target_forward_times = 0

        start_event.record(stream=torch.cuda.current_stream())
        queuing_time = 0
        batch_delay = getattr(self.args, "batch_delay", 0)
        while x.shape[1] < max_tokens:
            queuing_time += batch_delay
            x = model.generate(x, 1)
            target_forward_times += 1

            if use_early_stopping and self._check_stopping_criteria(x, stop_sequences):
                break

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = (
            start_event.elapsed_time(end_event) / 1000.0
        )  # Convert to seconds
        generated_tokens = x.shape[1] - prefix_len

        metrics = get_empty_metrics()
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = elapsed_time + queuing_time
        metrics["throughput"] = (
            generated_tokens / metrics["wall_time"] if metrics["wall_time"] > 0 else 0
        )

        return x, metrics

    @Register.register_decoding("sd")
    @torch.inference_mode()
    def speculative_decoding(
        self,
        prefix,
        transfer_top_k: int | None = 300,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.draft_model.device
        target_device = self.target_model.device

        approx_model_cache = KVCacheModel(
            self.draft_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(
            self.target_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        target_model_cache.vocab_size = self.vocab_size

        draft_forward_times = 0
        target_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        loop_idx = 0

        start_event.record(stream=torch.cuda.current_stream())

        while prefix.shape[1] < max_tokens:
            loop_idx += 1

            prefix_len = prefix.shape[1]

            draft_kvcache_length = approx_model_cache.current_length
            target_kvcache_length = target_model_cache.current_length

            # 确保不会生成超过max_tokens的token
            remaining_tokens = max_tokens - prefix_len
            if remaining_tokens <= 0:
                break

            # 调整gamma以不超过剩余的token数量
            current_gamma = min(
                self.args.gamma, remaining_tokens - 1
            )  # 减1是为了留给最后的采样token
            if current_gamma <= 0:
                # 如果只剩1个token，直接用target model生成
                _ = target_model_cache.generate(prefix.to(target_device), 1)
                target_forward_times += 1
                if self.accelerator.is_main_process:
                    self.target_forward_times += 1

                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                prefix = torch.cat((prefix, t), dim=1)
                self.num_acc_tokens.append(1)
                break

            x = approx_model_cache.generate(prefix.to(draft_device), current_gamma)
            draft_forward_times += current_gamma
            total_drafted_tokens += current_gamma

            # Verification-time cache invariant: approx has forwarded states up to
            # x[:, :-1], while target forwards the full x and ends up one step ahead.
            _ = target_model_cache.generate(x.to(target_device), 1)
            target_forward_times += 1

            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            this_step_accepted_tokens, n = self._verify_draft_sequence(
                draft_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=current_gamma,
                transfer_mode="none",
                send_reject_message=False,
            )
            _log_sd_alignment_snapshot(
                "verify_exit",
                prefix_len,
                approx_model_cache,
                target_model_cache,
                x_len=x.shape[1],
                gamma=current_gamma,
                note=f"accepted={this_step_accepted_tokens}, n={n}",
            )

            total_accepted_tokens += this_step_accepted_tokens

            self.num_acc_tokens.append(this_step_accepted_tokens)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"

            # 检查是否还有空间添加一个token（在接受序列过长时提前截断）
            if n + 1 >= max_tokens:
                prefix = x[:, :max_tokens]
                break

            prefix = self._finalize_verification(
                approx_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=current_gamma,
                n=n,
            )
            _log_sd_alignment_snapshot(
                "finalize_exit",
                prefix.shape[1],
                approx_model_cache,
                target_model_cache,
                x_len=prefix.shape[1],
                gamma=current_gamma,
            )

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        generated_tokens = prefix.shape[1] - current_tokens.shape[1]

        batch_delay = getattr(self.args, "batch_delay", 0)
        queuing_time = target_forward_times * batch_delay
        wall_time = elapsed_time + queuing_time

        throughput = generated_tokens / wall_time if wall_time > 0 else 0

        metrics = get_empty_metrics()
        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = total_drafted_tokens
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["wall_time"] = wall_time
        metrics["throughput"] = throughput
        metrics["loop_times"] = loop_idx
        metrics["queuing_time"] = queuing_time
        metrics["each_loop_draft_tokens"] = (
            total_drafted_tokens / loop_idx if loop_idx > 0 else 0
        )

        return prefix, metrics

    @torch.inference_mode()
    def speculative_decoding_with_bandwidth(
        self,
        prefix,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim: bool = False,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        if use_precise_comm_sim:
            comm_simulator = PreciseCommunicationSimulator(
                bandwidth_hz=1e6,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
            )
        else:
            comm_simulator = CommunicationSimulator(
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=float("inf"),
                bandwidth_cloud_end=float("inf"),
                dimension="Mbps",
            )
        self.color_print(f"Using transfer_top_k: {transfer_top_k}", 2)

        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.draft_model.device
        target_device = self.target_model.device

        approx_model_cache = KVCacheModel(
            self.draft_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(
            self.target_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        target_model_cache.vocab_size = self.vocab_size

        draft_forward_times = 0
        target_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        start_event.record(stream=torch.cuda.current_stream())

        idx: int = 0

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]

            idx += 1

            # 确保不会生成超过max_tokens的token
            remaining_tokens = max_tokens - prefix_len
            if remaining_tokens <= 0:
                break

            # 调整gamma以不超过剩余的token数量
            current_gamma = min(
                self.args.gamma, remaining_tokens - 1
            )  # 减1是为了留给最后的采样token
            if current_gamma <= 0:
                # 如果只剩1个token，直接用target model生成
                _ = target_model_cache.generate(prefix.to(target_device), 1)
                target_forward_times += 1
                if self.accelerator.is_main_process:
                    self.target_forward_times += 1

                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                prefix = torch.cat((prefix, t), dim=1)
                self.num_acc_tokens.append(1)
                break

            x = approx_model_cache.generate(prefix.to(draft_device), current_gamma)
            draft_forward_times += current_gamma
            total_drafted_tokens += current_gamma

            _ = target_model_cache.generate(x.to(target_device), 1)

            target_forward_times += 1

            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            this_step_accepted_tokens, n = self._verify_draft_sequence(
                draft_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=current_gamma,
                comm_simulator=comm_simulator,
                comm_link="edge_cloud",
                transfer_mode="serial",
                send_reject_message=True,
            )

            total_accepted_tokens += this_step_accepted_tokens

            self.num_acc_tokens.append(this_step_accepted_tokens)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"

            # 检查是否还有空间添加一个token
            if n + 1 >= max_tokens:
                prefix = x[:, :max_tokens]
                break

            # 针对使用了通信压缩的传输逻辑
            if n < prefix_len + current_gamma - 1:
                if transfer_top_k is not None and transfer_top_k > 0:
                    rebuild_probs = comm_simulator._apply_top_k_compression(
                        approx_model_cache._prob_history[:, n, : self.vocab_size],
                        transfer_top_k,
                    )
                    rebuild_probs = comm_simulator.rebuild_full_probs(rebuild_probs)
                    approx_model_cache._prob_history[:, n, : self.vocab_size] = (
                        rebuild_probs
                    )

                comm_simulator.transfer(
                    None,
                    approx_model_cache._prob_history[:, n, : self.vocab_size],
                    "edge_cloud",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )

            # finalize_verification自动回流状态与重新采样（或简单采样下一个字符）
            prefix = self._finalize_verification(
                approx_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=current_gamma,
                n=n,
            )

            # 传输新生成的 token id
            comm_simulator.simulate_transfer(INT_SIZE, "edge_cloud")

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        generated_tokens = prefix.shape[1] - current_tokens.shape[1]

        batch_delay = getattr(self.args, "batch_delay", 0)
        queuing_time = target_forward_times * batch_delay
        wall_time = elapsed_time + comm_simulator.edge_cloud_comm_time + queuing_time

        throughput = generated_tokens / wall_time if wall_time > 0 else 0

        metrics = get_empty_metrics()
        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = total_drafted_tokens
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["wall_time"] = wall_time
        metrics["throughput"] = throughput
        metrics["queuing_time"] = queuing_time
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy

        return prefix, metrics

    @torch.no_grad()
    def uncertainty_decoding(
        self,
        prefix,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim=False,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """
        Implement of the method raised in "Communication-Efficient Hybrid Language Model via Uncertainty-Aware Opportunistic and Compressed Transmission"
        """
        if use_precise_comm_sim:
            comm_simulator = PreciseCUHLM(
                bandwidth_hz=1e6,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
            )
        else:
            comm_simulator = CUHLM(
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                uncertainty_threshold=0.8,
                dimension="Mbps",
            )

        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.draft_model.device
        target_device = self.target_model.device

        approx_model_cache = KVCacheModel(
            self.draft_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(
            self.target_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        target_model_cache.vocab_size = self.vocab_size

        # Metrics Tracking
        target_forward_times = 0
        draft_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0
        queuing_time = 0
        batch_delay = getattr(self.args, "batch_delay", 0)

        loop_idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record(stream=torch.cuda.current_stream())

        input_len = prefix.shape[1]

        is_accepted_last_step = False

        while prefix.shape[1] < max_tokens:
            loop_idx += 1
            prefix_len = prefix.shape[1]

            # 传输 prompt
            if loop_idx == 1:
                comm_simulator.transfer(prefix, None, link_type="edge_cloud")

            # Sync
            x = approx_model_cache.generate(prefix.to(draft_device), 1)
            queuing_time += batch_delay
            _ = target_model_cache.generate(x.to(target_device), 1)

            # 无论接受与否，都要传输起草的 token
            comm_simulator.transfer(x, None, link_type="edge_cloud")
            current_logit = approx_model_cache.logits_history[:, -1, : self.vocab_size]
            assert current_logit is not None, "Logits history should not be None"
            uncertainty = comm_simulator.calculate_uncertainty(
                current_logit, M=20, theta_max=2.0, draft_token=x[0, -1].item()
            )
            should_transfer, vocab_size = comm_simulator.determine_transfer_strategy(
                uncertainty, current_logit
            )

            draft_forward_times += 1
            if not is_accepted_last_step:
                target_forward_times += 1
            else:
                # 如果上一个token被接受了，等下一次没有被接受，这么做是为了实现简单
                target_forward_times += 0

            total_drafted_tokens += 1

            n = prefix_len + 1 - 1

            if not should_transfer:
                is_accepted_last_step = True

                # 接受draft token - 仿照接受所有token的情况
                accepted_token = x[:, -1:]  # draft token
                prefix = torch.cat((prefix, accepted_token), dim=1)

                comm_simulator.send_accept_message(
                    linktype="edge_cloud"
                )  # 发送消息告知应该接受

                # KVCache管理：仿照接受所有token的情况
                # 由于我们接受了draft token，需要从target model采样一个新token
                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)

                # rollback target_model_cache，因为我们已经消费了它的输出
                # 这里n相当于prefix_len（接受了1个token）
                n = prefix_len  # 接受了位置为prefix_len的token
                target_model_cache.rollback(n + 2)  # 等同于rollback(prefix_len + 2)

                # 将新采样的token添加到序列中
                if prefix.shape[1] < max_tokens:
                    prefix = torch.cat((prefix, t), dim=1)

                comm_simulator.transfer(
                    t, None, link_type="edge_cloud"
                )  # 传输接受的token和新采样的token

                continue

            is_accepted_last_step = False

            # 拒绝采样

            # 压缩
            current_probs = comm_simulator._get_current_probs(
                approx_model_cache._prob_history
            )
            compressed_prob = comm_simulator._apply_top_k_compression(
                current_probs, vocab_size
            )

            rebuild_probs = comm_simulator.rebuild_full_probs(compressed_prob)
            approx_model_cache._prob_history[:, -1, : self.vocab_size] = (
                rebuild_probs  # 完成概率的重建
            )

            r = torch.rand(1, device=draft_device)
            j = x[:, prefix_len]

            self.color_print(
                f"Uncertainty: {uncertainty:.4f}, Vocab size: {vocab_size}", 3
            )

            if r > (
                target_model_cache._prob_history.to(draft_device)[:, prefix_len - 1, j]
            ) / (approx_model_cache._prob_history[:, prefix_len - 1, j]):
                n = prefix_len - 1
                comm_simulator.send_reject_message(
                    linktype="edge_cloud"
                )  # 发送消息告知应该拒绝、
                comm_simulator.transfer(
                    None,  # 一开始已经传输过
                    approx_model_cache._prob_history[:, -1, : self.vocab_size],
                    link_type="edge_cloud",
                    is_compressed=True,
                    compressed_k=vocab_size,
                )

            total_accepted_tokens += n - prefix_len + 1

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, : n + 1]

            approx_model_cache.rollback(n + 1)

            if n < prefix_len:
                # reject someone, sample from the pos n
                t = sample(
                    max_fn(
                        target_model_cache._prob_history[:, n, : self.vocab_size].to(
                            draft_device
                        )
                        - approx_model_cache._prob_history[:, n, : self.vocab_size]
                    )
                )
                target_model_cache.rollback(n + 1)
            else:
                # all approx model decoding accepted
                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                target_model_cache.rollback(n + 2)

            comm_simulator.transfer(
                t, None, link_type="edge_cloud"
            )  # 传输新采样的token
            prefix = torch.cat((prefix, t), dim=1)

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        metrics = get_empty_metrics()

        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = prefix.shape[1] - input_len
        metrics["draft_generated_tokens"] = draft_forward_times
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = (
            elapsed_time + queuing_time + comm_simulator.edge_cloud_comm_time
        )
        metrics["throughput"] = (
            (prefix.shape[1] - input_len) / metrics["wall_time"]
            if metrics["wall_time"] > 0
            else 0
        )
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy

        return prefix, metrics

    @torch.inference_mode()
    def lookahead_forward(self, prefix, **kwargs):
        input_ids = prefix.cuda()

        output_ids, idx, accept_length_list = self.target_model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            temperature=0.0,
            max_new_tokens=self.args.max_tokens,
            num_assistant_tokens_schedule="constant",
        )
        new_token = len(output_ids[0][len(input_ids[0]) :])
        return output_ids

    @torch.inference_mode()
    def verify_first_token_for_k_seq(
        self, draft_tokens_k_seq, draft_prob_k_seq, target_prob, **kwargs
    ):
        flag = False  # if any accepted
        resampled_token_id = 0
        chosen_draft_tokens_seq_idx = 0

        first_token_k_seq = draft_tokens_k_seq[:, 0]

        r = torch.rand(1, device=target_prob.device)

        if r > target_prob[:, 0, first_token_k_seq[0]]:
            t = torch.where(target_prob[0, 0, :] == 1)[0].unsqueeze(0)
            resampled_token_id = t
            idx = 0
            for idx, first_token in enumerate(first_token_k_seq[1:], 1):
                if t == first_token:
                    flag = True
                    chosen_draft_tokens_seq_idx = idx
                    break
        else:
            flag = True
            chosen_draft_tokens_seq_idx = 0

        return flag, resampled_token_id, chosen_draft_tokens_seq_idx

    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int = 4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")
