import os
import torch
import logging
from typing import Optional
from .metrics import DecodingMetrics
from .model_gpu import KVCacheModel

logger = logging.getLogger(__name__)

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
