"""Small shared proposal/stop/payload operations.

These are the primitives genuinely used by more than one algorithm family. They
do not sequence a round, own caches or policies, or dispatch between algorithms.
"""

from typing import Optional

import torch

from ..adapter import DecodingAdapter
from ..baselines_helpers import _quantize_probs_logspace
from ..communication import PROB_QUANT_PASS_THROUGH_BITS
from ..model_gpu import KVCacheModel
from ..proposal_utils import (
    build_topk_proposal_history_step,
    concat_topk_proposal_history,
)
from ..utils import rebuild_topk_uniform_probs, sample


def _uplink_prob_payload(
    probs: Optional[torch.Tensor], bits: Optional[int]
) -> tuple[Optional[torch.Tensor], Optional[int]]:
    """B15/B16：上行概率载荷的**原子决策**——返回的 (张量, 计费位宽)
    必须成对使用：张量继续流入下游验证路径，位宽传给 transfer/simulate
    计费。调用点因此不可能"声明 bits 却不量化"或反过来。

    - bits < 16：对数域真实量化，返回 (q̂, bits)——验证判据与计费
      看到同一个 q̂
    - bits >= 16 或 probs 为 None：原样返回 (probs, None)——按
      element_size 全宽计费（历史口径）
    """
    if probs is None:
        return None, None
    bits = int(bits or 16)
    if bits >= PROB_QUANT_PASS_THROUGH_BITS:
        return probs, None
    return _quantize_probs_logspace(probs, bits), bits


class SharedOperations:
    """Proposal, stop and payload primitives shared by algorithm families."""

    def _generate_with_optional_rebuilt_proposal(
        self,
        cache: KVCacheModel,
        prefix: torch.Tensor,
        gamma: int,
        proposal_top_k: Optional[int],
        adapter: Optional[DecodingAdapter] = None,
        *,
        need_topk_metadata: bool = False,
    ) -> tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[object], Optional[torch.Tensor]
    ]:
        if adapter is None and proposal_top_k is not None:
            if need_topk_metadata and hasattr(
                cache, "generate_with_rebuilt_topk_metadata"
            ):
                x, rebuilt_draft_probs, rebuilt_draft_meta = (
                    cache.generate_with_rebuilt_topk_metadata(
                        prefix,
                        gamma,
                        proposal_top_k,
                    )
                )
            else:
                x, rebuilt_draft_probs = cache.generate_with_rebuilt_topk(
                    prefix,
                    gamma,
                    proposal_top_k,
                )
                rebuilt_draft_meta = None
            return x, rebuilt_draft_probs, rebuilt_draft_meta, None
        if adapter is None and proposal_top_k is None:
            return cache.generate(prefix, gamma), None, None, None

        x = prefix.clone()
        rebuilt_rows: list[torch.Tensor] = []
        proposal_steps = []
        q: Optional[torch.Tensor] = None
        for _ in range(gamma):
            q = cache.forward_context(x)
            sample_probs = rebuild_topk_uniform_probs(q, proposal_top_k)
            if proposal_top_k is not None:
                rebuilt_rows.append(sample_probs.unsqueeze(1))
            # F43：记录每一步的 (top-k 索引, top-k 概率, 均匀尾)。验证方在**拒绝**时
            # 必须从 norm(max(0, p−q)) 采样，这需要提案分布本身；有了这个精确表示，
            # 跨链路只需 k×(4+p)+p 字节/位置，而不是整行 V×p（32k 词表下差 ~70×）。
            if need_topk_metadata:
                _step_meta = build_topk_proposal_history_step(q, proposal_top_k)
                if _step_meta is not None:
                    proposal_steps.append(_step_meta)
            next_tok = sample(sample_probs)
            x = torch.cat((x, next_tok), dim=1)

            if adapter is None:
                continue
            hidden_states = cache.hidden_states
            assert hidden_states is not None
            if adapter.predict(hidden_states):
                break

        rebuilt_draft_probs = None
        if rebuilt_rows:
            rebuilt_draft_probs = torch.cat(rebuilt_rows, dim=1)
        rebuilt_meta = (
            concat_topk_proposal_history(proposal_steps)
            if need_topk_metadata
            else None
        )
        return x, rebuilt_draft_probs, rebuilt_meta, q

    # ------------------------------------------------------------------
    # EOS 停止
    # ------------------------------------------------------------------
    # 项目里本来就有 `_check_stopping_criteria`（engine.py，检查**最后一个位置**
    # 的 EOS），并且已接进各解码循环，但 `eval/eval_mixed.py` 从不传
    # `use_early_stopping`，所以默认 False、在混合评测里从未生效。后果是结构性
    # 的：GSM8K 的 "#### <答案>" 不出现、HumanEval 函数体被截断，任务质量无法
    # 评测（实测 20/20 样本全部触顶 max_tokens）。
    #
    # 这里补两件事：
    #  1) 让混合评测显式启用 `use_early_stopping`（见 eval/eval_mixed.py）；
    #  2) `_check_stopping_criteria` 只看最后一个位置，而 gamma>1 时一个 chunk 里
    #     可能有多个 token、EOS 出现在中间就会被漏掉；`_stop_at_eos` 在生成段里
    #     找**第一个** EOS 并截断到它，避免多生成无意义的后缀。
    # 需要旧行为（不截断）时传 --disable_eos_stop。
    def _residual_payload_bytes(
        self,
        stage_probs: Optional[torch.Tensor],
        top_k: Optional[int],
    ) -> float:
        """拒绝位置残差所需的**增量**载荷（字节）——F43 修正版。

        既有代码已经在两处 `simulate_transfer` 里为拒绝位置计费了
        **概率部分**：``top_k × element_size``（``transfer_top_k`` 有效时）或整行
        ``V × element_size``（`src/baselines.py` 阶段一/阶段二各一处）。本函数只补上
        被漏掉的部分：

          · top-k 表示还需要 **k 个索引**（每个 4 B：int32）与 **1 个尾部标量**；
          · 走整行时既有的 ``V × element_size`` 已经完整，无需再加。

        因此返回值仅为 ``k×4 + element``（top-k 有效时），否则为 0。
        """
        if stage_probs is None or stage_probs.numel() == 0:
            return 0.0
        vocab = int(stage_probs.shape[-1])
        element = int(stage_probs.element_size())
        if top_k is not None and 0 < int(top_k) < vocab:
            return float(int(top_k)) * 4 + element
        return 0.0

    def _eos_token_id(self) -> Optional[int]:
        if getattr(self.args, "disable_eos_stop", False):
            return None
        tokenizer = getattr(self, "tokenizer", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        return int(eos_id) if eos_id is not None else None

    def _stop_at_eos(
        self, prefix: torch.Tensor, prompt_len: int
    ) -> tuple[torch.Tensor, bool]:
        """生成段里出现 EOS 时截断到该 token（含），并报告应当停止。"""
        eos_id = self._eos_token_id()
        if eos_id is None or prefix.shape[1] <= prompt_len:
            return prefix, False
        hits = (prefix[:, prompt_len:] == eos_id).nonzero(as_tuple=False)
        if hits.numel() == 0:
            return prefix, False
        end = prompt_len + int(hits[0, 1].item()) + 1
        return prefix[:, :end], True
