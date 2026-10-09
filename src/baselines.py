import json
import os
import math
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple, cast

import torch
import transformers

from .SpecDec_pp.specdec_pp.wrap_model import AcceptancePredictionHead

transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")

from .adapter import DecodingAdapter
from .communication import (
    CUHLM,
    CommunicationSimulator,
    PreciseCommunicationSimulator,
    PreciseCUHLM,
    PROB_QUANT_PASS_THROUGH_BITS,
    cuhlm_uplink_payload_bytes,
    tk_slt_uplink_payload_bytes,
)
from .decoding_ops import (
    apply_rollback,
    build_rollback_plan,
    collect_verification_payload,
    compute_acceptance_result,
    compute_residual_distribution,
    materialize_acceptance,
    prepare_verification_inputs,
    reject_residual_payload_bytes,
    reject_tail_scalar_bytes,
    resolve_stage_verification,
    sample_accept_token,
    sample_reject_token,
    sample_reject_token_from_topk_proposal,
    verify_draft_sequence_result,
)
from .acc_head_registry import anchor_repo_path, is_usable_acc_head_dir
from .engine import Decoding
from .graph_decode import (  # B19：CUDA Graph 接线的单一入口
    acquire_graph_caches,
    graph_mode_cache_kwargs as _graph_mode_cache_kwargs,
)
from .metrics import INT_SIZE, DecodingMetrics, get_empty_metrics
from .model_gpu import KVCacheModel
from .mode_features import MODE_FEATURES
from .proposal_utils import (
    apply_transfer_top_k_cap,
    build_stage_prefix_topk_history,
    build_draft_probs_override,
    build_topk_proposal_history_step,
    concat_topk_proposal_history,
    merge_stage_topk_histories,
    proposal_top_k,
    stage_topk_proposal_history,
    stage_prob_history,
)
from .register import Register
from .rl_agent_registry import (
    ROLE_LITTLE,
    ROLE_MAIN,
    get_rl_agent_spec,
    resolve_legacy_rl_agent_load_path,
)
from .rl_adapter import RLNetworkAdapter
from .utils import (
    max_fn,
    norm_logits,
    rebuild_topk_uniform_probs,
    sample,
    skip_token_validation,
    state_entropy,
)



def _add_comm_accounting_metrics(metrics: dict, args, comm_simulator) -> None:
    """L1：把 edge-cloud 动态 NTT 观测历史挂进 per-sample metrics。

    列表字段会被 eval 层用 += 拼接成全运行历史；口径标签等运行级常量由
    eval/utils.py 的 get_save_dict 从 args 写入最终 json（单一事实源）。
    """
    metrics["edge_cloud_ntt_history"] = list(
        getattr(comm_simulator, "ntt_edge_cloud_history", [])
    )


def _send_downlink_token(comm_simulator, token: torch.Tensor, link_type: str) -> None:
    """下行回传一个采样 token 及其位置索引，**合并为一次传输**。

    B17：旧写法把它拆成两次调用
    （``transfer(t, None, link)`` 与 ``simulate_transfer(INT_SIZE, link)``），而
    ``CommunicationSimulator._charge_transfer`` 每被调用一次就累加一次 NTT 与
    ``connect_times`` —— 同一次 WAN 往返被计费两次。

    ``transfer`` 本身只是"tensor → 字节"的编码转发（communication.py 里它只
    调用 ``simulate_transfer`` 一次），且全仓没有任何调用点给
    ``protocol_overhead_bytes`` 传过值（默认 0），所以合并前后**数据字节数逐位
    相同**，只有往返次数（以及由它派生的 comm_time/connect_times）会变。
    """
    comm_simulator.simulate_transfer(
        INT_SIZE + token.element_size() * token.numel(), link_type
    )


def _send_downlink_index_only(comm_simulator, link_type: str) -> None:
    """CUHLM 论文口径的下行：响应 token 索引 **negligible**（§II-B）。

    论文的成本分析只计上行词表分布（式 (5)），token 索引一律不计字节——
    所以这里按 **0 字节报文**计。报文物理上仍发生（设备必须拿到响应
    token），repo 链路模型的 per-message NTT 照付；论文的 Shannon 时延
    模型（式 (6)）没有 per-message 固定成本，这一项是**链路模型差异**
    而非字节口径差异——docs/protocol.md §3 的 CUHLM 小节有完整说明。
    """
    comm_simulator.simulate_transfer(0, link_type)


def load_acceptance_prediction_head(model_path: str) -> AcceptancePredictionHead:
    # 先把仓库相对路径锚定到仓库根（与 CWD 无关），并在调用 HF 之前确认目录里
    # 真的有一个 head。否则路径无效时 huggingface_hub 会把它当成 repo id 校验，
    # 抛出与真实原因无关的 HFValidationError（acc head 没下载被误读成模型名写错）。
    if not is_usable_acc_head_dir(model_path):
        raise FileNotFoundError(
            f"acc head 不可用：{model_path}"
            f"（锚定仓库根后为 {anchor_repo_path(model_path)}）"
            "——目录不存在或缺少 config.json。"
            "权重不在 git 里，请先跑 `python scripts/setup/download_assets.py` 下载；"
            "或确认该模型的 local_path"
            "（src/SpecDec_pp/checkpoints/acc_head_registry.json）"
            "与 src/SpecDec_pp/checkpoints/ 下的实际布局是否一致。"
        )
    path = anchor_repo_path(model_path)
    try:
        return AcceptancePredictionHead.from_pretrained(str(path))
    except FileNotFoundError:
        config_path = path / "config.json"
        bin_path = path / "pytorch_model.bin"
        safetensors_path = path / "model.safetensors"
        if (
            not config_path.is_file()
            or not bin_path.is_file()
            or safetensors_path.is_file()
        ):
            raise

        with config_path.open() as f:
            config = json.load(f)

        head = AcceptancePredictionHead(config)
        state_dict = torch.load(
            bin_path, map_location="cpu", weights_only=True
        )  # 纯 state_dict；weights_only 防篡改文件执行代码（R5）
        head.load_state_dict(state_dict, strict=True)
        return head


def _build_cache(
    model,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    vocab_size: int,
    **cache_kwargs,
) -> KVCacheModel:
    cache = KVCacheModel(model, temperature, top_k, top_p, **cache_kwargs)
    cache.vocab_size = vocab_size
    return cache


def _move_token_tensor(tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tokens.device == device:
        return tokens
    if tokens.dtype != torch.long:
        tokens = tokens.to(torch.long)
    return tokens.to("cpu", non_blocking=False).to(device, non_blocking=True)


def _quantize_probs_logspace(probs: torch.Tensor, bits: int) -> torch.Tensor:
    """概率载荷量化的**真实**实现（不只是记账）：对数域均匀量化 → 还原 → 重归一化。

    · 对数域量化使各项的相对误差均匀 ✓（接受判据 min(1,p/q) 关心的是相对误差）
    · 量化映射单调 ⇒ **保持序关系** ✓（top-k 选择不受影响）
    · 重归一化保证和为 1 ✓（否则接受判据的比值会失真）
    · bits >= 16 时原样返回 ⇒ 默认路径与历史数字完全一致 ✓
    """
    if (
        bits is None
        or bits >= PROB_QUANT_PASS_THROUGH_BITS
        or probs is None
        or probs.numel() == 0
    ):
        return probs
    levels = float((1 << int(bits)) - 1)
    p32 = probs.to(torch.float32)
    logp = torch.log(p32.clamp_min(1e-12))
    lo = logp.amin(dim=-1, keepdim=True)
    hi = logp.amax(dim=-1, keepdim=True)
    span = (hi - lo).clamp_min(1e-9)
    q = torch.round((logp - lo) / span * levels)
    out = torch.exp(lo + q / levels * span)
    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return out.to(probs.dtype)


def _uplink_prob_payload(
    probs: Optional[torch.Tensor], args
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
    bits = int(getattr(args, "prob_payload_bits", 16) or 16)
    if bits >= PROB_QUANT_PASS_THROUGH_BITS:
        return probs, None
    return _quantize_probs_logspace(probs, bits), bits


def _simulate_topk_prob_transfer(
    comm_simulator: CommunicationSimulator,
    *,
    link_type: str,
    draft_len: int,
    transfer_top_k: Optional[int],
    prob_dtype: torch.dtype,
) -> None:
    if draft_len <= 0:
        return

    prob_bytes = torch.tensor([], dtype=prob_dtype).element_size()
    effective_topk = transfer_top_k if transfer_top_k is not None and transfer_top_k > 0 else 0
    if hasattr(comm_simulator, "_compressed_topk_payload_bytes"):
        total_bytes = comm_simulator._compressed_topk_payload_bytes(
            compressed_k=effective_topk,
            seq_length=draft_len,
            prob_element_size=prob_bytes,
        )
    else:
        index_bytes = 4
        total_bytes = draft_len * effective_topk * (prob_bytes + index_bytes)
    comm_simulator.simulate_transfer(
        total_bytes,
        cast(str, link_type),
        topk=effective_topk,
        draft_len=draft_len,
    )


#: TK-SLT ODLD（--tk_slt_odld）的 γ* 上限：估计噪声下防病态值；
#: 论文 Table I 的最大 γ* 是 14（α=0.8、L=0.01），64 只在极端估计下触及。
_TK_SLT_ODLD_GAMMA_CAP = 64


def _last_edge_cloud_tx_seconds(comm_simulator) -> Optional[float]:
    """最近一次 edge-cloud 传输的纯发射时长（秒，不含 NTT）——ODLD 的 b̂ 用。

    链路模型的 TransferUnit.tx_time 只计发射时长，与论文 T_V = D_V/R_up
    的口径一致（式 (2)/(3) 无 per-message 固定成本）。读不到（如测试里的
    FakeCommSimulator 没有 stats）时返回 None，该轮不更新估计。
    """
    try:
        units = comm_simulator.stats["edge_cloud"]
        if units:
            return float(units[-1]["tx_time"])
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return None
    return None


def _quantize_probs_fp16(probs: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """TK-SLT 上行载荷的**真实** FP16 量化（论文 §VI-B：half precision 传输）。

    论文的通信模式把传输的概率值量化到 FP16；这里让验证判据与计费看到
    **同一个** q̂（B15/B16 的"计费与数据一致"原则）：

    · fp16 roundtrip 后按行重归一化（softmax over top-K 本来和为 1，
      量化误差 ~1e-3，重归一化后仍是合法分布，接受判据 p/q 不失真）✓
    · 0 值（top-K 之外）量化后仍是 0，稀疏支撑不变 ✓
    · 值为 0/1 的行（temp=0 的 one-hot）逐位不变 ⇒ 协议 temp=0 下
      与未量化路径完全一致 ✓
    """
    if probs is None or probs.numel() == 0:
        return probs
    out = probs.to(torch.float16).to(probs.dtype)
    row_sum = out.sum(dim=-1, keepdim=True)
    tiny = torch.finfo(out.dtype).tiny
    return out / row_sum.clamp_min(tiny)


def _lambert_w_minus_one(z: float) -> float:
    """Lambert W 的 −1 分支：解 w·e^w = z，z ∈ (−1/e, 0)，返回 w ≤ −1。

    论文 Theorem 2 式 (21) 需要 W_{−1}。g(w)=w·e^w 在 (−∞,−1] 上从 0⁻
    单调降到 −1/e，故对给定 z ∈ (−1/e, 0) 解唯一，二分法无依赖、确定
    可测（100 次二分到 ~1e-28 绝对精度，对 γ 的影响 < 1e-26）。
    """
    if not (-1.0 / math.e < z < 0.0):
        raise ValueError(f"W_{{-1}} 的定义域是 (−1/e, 0)，收到 z={z!r}")
    lo, hi = -200.0, -1.0  # g(lo)≈0⁻ > z > g(hi)=−1/e
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if mid * math.exp(mid) > z:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def tk_slt_speedup_ratio(gamma: int, alpha: float, L: float) -> float:
    """TK-SLT 论文式 (11)：S_inf(γ) = (1−α^{γ+1}) / ((1−α)(1+γL))。

    α=1 时取极限 (γ+1)/(1+γL)（式 (12) 的期望 token 数除以式 (7) 的
    归一化时延）；α=0 时为 1/(1+γL)。
    """
    gamma = int(gamma)
    if alpha >= 1.0:
        return (gamma + 1) / (1.0 + gamma * L)
    if alpha <= 0.0:
        return 1.0 / (1.0 + gamma * L)
    return (1.0 - alpha ** (gamma + 1)) / ((1.0 - alpha) * (1.0 + gamma * L))


def tk_slt_optimal_draft_length(
    alpha: float,
    b: float,
    c: float,
    gamma_cap: int = 64,
) -> Tuple[int, float]:
    """论文 Algorithm 1（ODLD）：闭式最优草稿长度 γ*（Theorem 2 式 (21)）。

        γ0 = (1/ln α)·(W_{−1}(−(1/e)·α^{1/L−1}) + 1) − 1/L，L = b + c

    γ0 < 1 ⇒ γ* = 1；否则在 {⌊γ0⌋, ⌈γ0⌉} 里取 S_inf 更大者（平局取
    ⌈γ0⌉，与论文 Algorithm 1 的 ≤ 分支一致）。返回 (γ*, S_inf(γ*))。

    论文约束外的退化输入（估计噪声 early rounds 常见）按保守方向处理：
    · α ≥ 1 / L ≤ 0：接受率饱和或相对开销为零 ⇒ 加大 γ 只赚不亏，
      取上限 gamma_cap（防噪声下的病态值）；
    · α ≤ 0：全拒 ⇒ γ* = 1（AS² 随后会选择 standalone LLM）；
    · L ≥ 1：论文式 (19) 要求 L < 1，此时 z 越界无解；取 γ* = 1，
      S_inf(1) = (1+α)/(1+L) < 1 恒成立 ⇒ AS² 会退回 standalone LLM。
    """
    L = float(b) + float(c)
    gamma_cap = max(1, int(gamma_cap))
    if L <= 0.0 or alpha >= 1.0:
        return gamma_cap, tk_slt_speedup_ratio(gamma_cap, alpha, L)
    if alpha <= 0.0 or L >= 1.0:
        return 1, tk_slt_speedup_ratio(1, alpha, L)

    z = -(1.0 / math.e) * alpha ** (1.0 / L - 1.0)
    w = _lambert_w_minus_one(z)
    gamma0 = (w + 1.0) / math.log(alpha) - 1.0 / L
    if gamma0 < 1.0:
        return 1, tk_slt_speedup_ratio(1, alpha, L)

    floor_g, ceil_g = math.floor(gamma0), math.ceil(gamma0)
    s_floor = tk_slt_speedup_ratio(floor_g, alpha, L)
    s_ceil = tk_slt_speedup_ratio(ceil_g, alpha, L)
    if s_floor <= s_ceil:
        gamma_star = ceil_g
    else:
        gamma_star = floor_g
    gamma_star = max(1, min(gamma_star, gamma_cap))
    return gamma_star, tk_slt_speedup_ratio(gamma_star, alpha, L)


def tk_slt_select_speculative(
    alpha: float,
    b: float,
    c: float,
    gamma_cap: int = 64,
) -> Tuple[bool, int, float]:
    """论文 Algorithm 2（AS²）：S_inf(γ*) > 1 才用 DSD，否则 standalone LLM。

    返回 (use_dsd, γ*, S*)。与 ODLD 一样吃 (α, b, c)——在线使用时由
    运行估计喂入（接受率、T_V/T_LLM、T_SLM/T_LLM）。
    """
    gamma_star, s_star = tk_slt_optimal_draft_length(alpha, b, c, gamma_cap)
    return s_star > 1.0, gamma_star, s_star


def _validate_token_range(
    tokens: torch.Tensor,
    *,
    vocab_size: int,
    label: str,
) -> None:
    if skip_token_validation():
        return
    if tokens.numel() == 0:
        return
    if tokens.dtype != torch.long:
        tokens = tokens.to(torch.long)
    min_id = int(tokens.min().item())
    max_id = int(tokens.max().item())
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"Invalid token ids at {label}: min={min_id}, max={max_id}, vocab_size={vocab_size}"
        )


def _ensure_token_shape(tokens: torch.Tensor, *, label: str) -> torch.Tensor:
    if tokens.dtype != torch.long:
        tokens = tokens.to(torch.long)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(-1)
    if tokens.dim() != 2:
        raise ValueError(f"Unexpected token tensor shape at {label}: {tuple(tokens.shape)}")
    return tokens


def _sample_token_from_probs(
    probs: torch.Tensor,
    *,
    output_device: torch.device,
    vocab_size: int,
    label: str,
) -> torch.Tensor:
    token = sample(probs).to(torch.long)
    token = _ensure_token_shape(token, label=label)
    _validate_token_range(token, vocab_size=vocab_size, label=label)
    return _move_token_tensor(token, output_device)


def _compute_token_vocab_rank(probs: torch.Tensor, token_id: int) -> int:
    token_prob = probs[..., token_id]
    return int((probs > token_prob).sum().item()) + 1


def _compute_transfer_topk_rank(
    probs: torch.Tensor,
    token_id: int,
    transfer_top_k: Optional[int],
    vocab_rank: int,
) -> tuple[bool, int]:
    vocab_size = probs.shape[-1]
    if transfer_top_k is None or transfer_top_k <= 0 or transfer_top_k >= vocab_size:
        return True, vocab_rank

    topk_count = min(transfer_top_k, vocab_size)
    topk_indices = torch.topk(probs, topk_count, sorted=True).indices
    matches = (topk_indices == token_id).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return False, 0
    return True, int(matches[0].item()) + 1


def _record_accepted_token_ranks(
    *,
    stage_probs: Optional[torch.Tensor],
    x: torch.Tensor,
    prefix_len: int,
    accepted_count: int,
    transfer_top_k: Optional[int],
    vocab_rank_history: List[int],
    in_transfer_topk_history: List[bool],
    transfer_topk_rank_history: List[int],
) -> None:
    if stage_probs is None or accepted_count <= 0:
        return

    for i in range(accepted_count):
        logit_idx = prefix_len + i - 1
        token_id = int(x[:, prefix_len + i].item())
        probs = stage_probs[0, logit_idx, :]
        vocab_rank = _compute_token_vocab_rank(probs, token_id)
        in_transfer_topk, transfer_topk_rank = _compute_transfer_topk_rank(
            probs, token_id, transfer_top_k, vocab_rank
        )
        vocab_rank_history.append(vocab_rank)
        in_transfer_topk_history.append(in_transfer_topk)
        transfer_topk_rank_history.append(transfer_topk_rank)


def _finalize_cuhlm_verification(
    *,
    proposer_cache: KVCacheModel,
    verifier_cache: KVCacheModel,
    verification_inputs,
    x: torch.Tensor,
    prefix_len: int,
    accepted_count: int,
    reject_offset: Optional[int],
    output_device: torch.device,
) -> tuple[int, torch.Tensor, bool]:
    actual_gamma = verification_inputs.actual_gamma
    all_accepted = reject_offset is None
    if all_accepted:
        n = prefix_len + actual_gamma - 1
    else:
        n = prefix_len + cast(int, reject_offset) - 1

    rollback_plan = build_rollback_plan(prefix_len, actual_gamma, n)
    if rollback_plan.all_accepted:
        t = sample_accept_token(
            verifier_cache.prob_history[:, -1, : verifier_cache.vocab_size],
            output_device=output_device,
        )
    else:
        assert reject_offset is not None
        vocab_limit = min(proposer_cache.vocab_size, verifier_cache.vocab_size)
        t = sample_reject_token(
            verification_inputs.target_probs_batch[:, reject_offset, :vocab_limit],
            verification_inputs.draft_probs_batch[:, reject_offset, :vocab_limit],
            output_device=output_device,
        )

    apply_rollback(
        proposer_cache,
        verifier_cache,
        rollback_plan,
    )
    return n, t, rollback_plan.all_accepted


def _add_per_model_wall_time(
    metrics: DecodingMetrics,
    *,
    elapsed_time: float,
    comm_time: float,
    queuing_time: float,
    little_comp_time: float = 0.0,
    draft_comp_time: float = 0.0,
    target_comp_time: float = 0.0,
) -> None:
    """Split wall_time into per-model contributions.

    wall_time = little_wall + draft_wall + target_wall + comm + queuing

    When per-model CPU timing is available, it is used to proportionally split
    the GPU computation time.  Communication and queuing remain separate.
    """
    total_comp = little_comp_time + draft_comp_time + target_comp_time
    if total_comp > 0:
        metrics["little_wall_time"] = (little_comp_time / total_comp) * elapsed_time
        metrics["draft_wall_time"] = (draft_comp_time / total_comp) * elapsed_time
        metrics["target_wall_time"] = (target_comp_time / total_comp) * elapsed_time
    else:
        # Fallback: split by forward times if available
        lf = metrics.get("little_forward_times", 0)
        df = metrics.get("draft_forward_times", 0)
        tf = metrics.get("target_forward_times", 0)
        total_fwd = lf + df + tf
        if total_fwd > 0:
            metrics["little_wall_time"] = (lf / total_fwd) * elapsed_time
            metrics["draft_wall_time"] = (df / total_fwd) * elapsed_time
            metrics["target_wall_time"] = (tf / total_fwd) * elapsed_time
        else:
            metrics["little_wall_time"] = 0.0
            metrics["draft_wall_time"] = 0.0
            metrics["target_wall_time"] = 0.0

    metrics["little_computation_time"] = little_comp_time
    metrics["draft_computation_time"] = draft_comp_time
    metrics["target_computation_time"] = target_comp_time


def get_decoding_fn(instance: "Baselines", name: str) -> Callable:
    if hasattr(instance, name):
        method = getattr(instance, name)
        if callable(method):
            return method
        else:
            raise ValueError(
                f"Attribute '{name}' in {instance.__class__.__name__} is not callable"
            )
    else:
        raise ValueError(
            f"Decoding method '{name}' not found in class {instance.__class__.__name__}"
        )


def compute_stage_reward(
    tps_part: float,
    generated_tokens: int,
    accepted_tokens: int,
    opportunistic: bool,
) -> float:
    if opportunistic:
        return tps_part

    reward = math.exp(min(tps_part, 100) / 20.0)
    if generated_tokens > 1:
        reward *= (accepted_tokens / generated_tokens) ** 2
    return reward


class Baselines(Decoding):
    """
    用于实验的方法。
    包含：
    - dssd
    - Uncertainty Decoding
    - dsd
    - Tridecoding
    """

    def __init__(self, args):
        super().__init__(args)
        # 真实 RTT trace 回放：模块级一次性配置（所有模拟器实例共享，含基线）
        ntt_trace_file = getattr(args, "ntt_trace_file", "")
        if ntt_trace_file:
            from src.communication import configure_ntt_trace
            from src.utils import read_trace_file

            trace_vals: list = []
            for run_id in (1, 2, 3):
                try:
                    trace_vals = read_trace_file(ntt_trace_file, run_id)
                    if trace_vals:
                        break
                except Exception:
                    continue
            if not trace_vals:
                raise FileNotFoundError(f"NTT trace 无法加载: {ntt_trace_file}")
            configure_ntt_trace(
                trace_vals,
                scale=getattr(args, "ntt_trace_scale", 1.0),
                src=os.path.basename(ntt_trace_file),
            )
        # self.load_acc_head() # Moved to load_model
        eval_mode = getattr(args, "eval_mode", "")
        # D2：模式能力查询单点化（原两处硬编码集合移入 mode_features）
        _spec = MODE_FEATURES.get(eval_mode)
        uses_main_rl = bool(_spec and _spec.uses_main_rl)
        uses_little_rl = bool(_spec and _spec.uses_little_rl)
        if getattr(args, "use_rl_adapter", False):
            checkpoint_root = getattr(
                args, "rl_checkpoint_root", "checkpoints/rl_agents"
            )
            init_seed = getattr(args, "rl_init_seed", None)
            init_strategy = getattr(args, "rl_init_strategy", "resume")
            legacy_load_paths = eval_mode != "cee_sd_opportunistic"
            opportunistic = eval_mode == "cee_sd_opportunistic"
            little_prefers_latest = opportunistic and not getattr(
                args, "disable_rl_update", False
            )
            epsilon_decay = getattr(args, "rl_epsilon_decay", None)
            reward_scale = getattr(args, "rl_reward_scale", None)
            batch_size = getattr(args, "rl_batch_size", None)
            if epsilon_decay is None:
                epsilon_decay = 0.95 if opportunistic else 0.9995
            if reward_scale is None:
                reward_scale = 1.0 if opportunistic else 0.01
            if batch_size is None:
                batch_size = 16 if opportunistic else 32
            if uses_main_rl:
                main_spec = get_rl_agent_spec(
                    ROLE_MAIN,
                    little_model=getattr(args, "little_model", None),
                    draft_model=args.draft_model,
                    target_model=args.target_model,
                    checkpoint_root=checkpoint_root,
                )
                self.rl_adapter = RLNetworkAdapter(
                    args,
                    model_path=getattr(args, "main_rl_path", None)
                    or main_spec.latest_path,
                    best_model_path=getattr(args, "main_rl_best_path", None)
                    or main_spec.best_path,
                    agent_name=main_spec.agent_name,
                    legacy_load_paths=[
                        path
                        for path in [
                            resolve_legacy_rl_agent_load_path(
                                ROLE_MAIN,
                                getattr(args, "little_model", None),
                                args.draft_model,
                                args.target_model,
                            )
                        ]
                        if path is not None
                    ]
                    if legacy_load_paths
                    else [],
                    threshold_candidates=main_spec.threshold_candidates,
                    init_seed=init_seed,
                    init_strategy=init_strategy,
                    frozen=opportunistic,
                    prefer_latest=False,
                    epsilon_decay=epsilon_decay,
                    reward_scale=reward_scale,
                    batch_size=batch_size,
                    force_threshold_override=getattr(
                        args, "rl_force_threshold", None
                    ),
                    force_topk_override=getattr(args, "rl_force_topk", None),
                )
            else:
                self.rl_adapter = None

            if uses_little_rl:
                little_spec = get_rl_agent_spec(
                    ROLE_LITTLE,
                    little_model=args.little_model,
                    draft_model=args.draft_model,
                    target_model=args.target_model,
                    checkpoint_root=checkpoint_root,
                )
                self.little_rl_adapter = RLNetworkAdapter(
                    args,
                    model_path=getattr(args, "little_rl_path", None)
                    or little_spec.latest_path,
                    best_model_path=getattr(args, "little_rl_best_path", None)
                    or little_spec.best_path,
                    agent_name=little_spec.agent_name,
                    legacy_load_paths=[
                        path
                        for path in [
                            resolve_legacy_rl_agent_load_path(
                                ROLE_LITTLE,
                                args.little_model,
                                args.draft_model,
                                args.target_model,
                            )
                        ]
                        if path is not None
                    ]
                    if legacy_load_paths
                    else [],
                    threshold_candidates=little_spec.threshold_candidates,
                    k_candidates=[1]
                    if eval_mode == "cee_sd_opportunistic"
                    else little_spec.topk_candidates,
                    init_seed=None if init_seed is None else init_seed + 1,
                    init_strategy=init_strategy,
                    frozen=False,
                    prefer_latest=little_prefers_latest,
                    epsilon_decay=epsilon_decay,
                    reward_scale=reward_scale,
                    batch_size=batch_size,
                    # 默认 None：little 级不继承 --rl_force_threshold（那一级是
                    # opportunistic，可能接受未验证 token，混淆主阈值的消融结论）。
                    # 用 --rl_force_threshold_little 单独指定。
                    force_threshold_override=(
                        getattr(args, "rl_force_threshold_little", None)
                        if getattr(args, "rl_force_threshold_little", None)
                        is not None
                        else -1.0  # 哨兵：显式"不强制"，避免继承主级 flag
                    ),
                )
            else:
                self.little_rl_adapter = None
        else:
            self.rl_adapter = None
            self.little_rl_adapter = None

        self.task = "unknown"  # This attribute should be set in the subclass

    def load_model(self):
        super().load_model()
        self.load_acc_head()

    def _save_adaptive_rl_checkpoints(self, throughput: float) -> None:
        if getattr(self.args, "disable_rl_update", False):
            return
        if self.rl_adapter is not None:
            self.rl_adapter.save(throughput)
        if self.little_rl_adapter is not None:
            self.little_rl_adapter.save(throughput)

    def _acquire_three_layer_caches(
        self,
        attr: str,
        graph_kw: dict,
        draft_top_k: int,
    ) -> dict[str, KVCacheModel]:
        """little/draft/target 三层缓存的统一入口（B19）。

        三个模型每轮都做多 token 验证前向（little 验证 draft、draft 验证 little
        的接受段、target 验证 draft —— 这是调度开销的主体），所以**三个都开图**；
        图开启时经 acquire_graph_caches 跨样本复用（StaticCache 原地 reset ⇒ 图
        不重捕获）。little/draft 用 draft_top_k 压缩、target 不压缩，与各方法
        历史取值一致。
        """
        return acquire_graph_caches(
            self,
            attr,
            graph_kw,
            {
                "little": lambda: _build_cache(
                    self.little_model,
                    temperature=self.args.temp,
                    top_k=draft_top_k,
                    top_p=self.args.top_p,
                    vocab_size=self.vocab_size,
                    **graph_kw,
                ),
                "draft": lambda: _build_cache(
                    self.draft_model,
                    temperature=self.args.temp,
                    top_k=draft_top_k,
                    top_p=self.args.top_p,
                    vocab_size=self.vocab_size,
                    **graph_kw,
                ),
                "target": lambda: _build_cache(
                    self.target_model,
                    temperature=self.args.temp,
                    top_k=0,
                    top_p=0,
                    vocab_size=self.vocab_size,
                    **graph_kw,
                ),
            },
        )

    def _acquire_draft_target_caches(
        self,
        attr: str,
        graph_kw: dict,
        draft_top_k: int,
        target_top_k: int,
        target_top_p: float,
    ) -> Tuple[KVCacheModel, KVCacheModel]:
        """草稿 + 目标两条缓存（草稿走多 token 前向 = 图收益点，目标按需 eager）。

        图开启时经 acquire_graph_caches 跨样本复用（草稿缓存不重捕获）。target 的
        top-k/top-p 由调用方给：多数方法目标不压缩（0/0），dsd 传采样 top-k。
        """
        caches = acquire_graph_caches(
            self,
            attr,
            graph_kw,
            {
                "draft": lambda: _build_cache(
                    self.draft_model,
                    temperature=self.args.temp,
                    top_k=draft_top_k,
                    top_p=self.args.top_p,
                    vocab_size=self.vocab_size,
                    **graph_kw,
                ),
                "target": lambda: _build_cache(
                    self.target_model,
                    temperature=self.args.temp,
                    top_k=target_top_k,
                    top_p=target_top_p,
                    vocab_size=self.vocab_size,
                ),
            },
        )
        return caches["draft"], caches["target"]

    def build_adaptive_tridecoding_caches(
        self,
        transfer_top_k: Optional[int],
    ) -> dict[str, KVCacheModel]:
        draft_top_k = (
            transfer_top_k
            if (transfer_top_k is not None and transfer_top_k > 0)
            else self.args.top_k
        )
        # B19：此前这里直接 _build_cache —— 既没透传 use_cuda_graph（命令行开图
        # 对 tridecoding 完全无效），也没有跨样本复用（每样本重捕获）。
        graph_kw = _graph_mode_cache_kwargs(
            self.args,
            cap=int(getattr(self.args, "gamma1", 1))
            + int(getattr(self.args, "gamma2", 1))
            + 4,
        )
        return self._acquire_three_layer_caches(
            "_tridecoding_caches", graph_kw, draft_top_k
        )

    def load_acc_head(self):
        # Load acc head if adaptive method is used
        args = self.args
        _spec = MODE_FEATURES.get(self.args.eval_mode)
        _acc = _spec.acc_head if _spec else "none"
        if _acc == "draft_target":
            draft_target_threshold: float | int = self.args.draft_target_threshold
            self.acc_head_path = args.acc_head_path
            self.acc_head = load_acceptance_prediction_head(
                self.acc_head_path,
            )
            self.acc_head.eval()
            if hasattr(self, "draft_model"):
                self.acc_head.to(self.draft_model.device)
            self.adapter = DecodingAdapter(
                self.acc_head,
                draft_target_threshold,
                stop_mode=getattr(self.args, "arp_stop_mode", "cumulative"),
            )
        elif _acc == "both":
            small_draft_threshold: float | int = self.args.small_draft_threshold
            draft_target_threshold: float | int = self.args.draft_target_threshold
            self.small_draft_acc_head_path = args.small_draft_acc_head_path
            self.small_draft_acc_head = load_acceptance_prediction_head(
                self.small_draft_acc_head_path,
            )
            self.small_draft_acc_head.eval()
            if hasattr(self, "little_model"):
                self.small_draft_acc_head.to(self.little_model.device)
            self.draft_target_acc_head_path = args.draft_target_acc_head_path
            self.draft_target_acc_head = load_acceptance_prediction_head(
                self.draft_target_acc_head_path,
            )
            self.draft_target_acc_head.eval()
            if hasattr(self, "draft_model"):
                self.draft_target_acc_head.to(self.draft_model.device)
            self.small_draft_adapter = DecodingAdapter(
                self.small_draft_acc_head,
                small_draft_threshold,
                stop_mode=getattr(self.args, "arp_stop_mode", "cumulative"),
            )
            self.draft_target_adapter = DecodingAdapter(
                self.draft_target_acc_head,
                draft_target_threshold,
                stop_mode=getattr(self.args, "arp_stop_mode", "cumulative"),
            )

    @staticmethod
    def _collect_dssd_uplink_payload(
        prob_history: torch.Tensor,
        x: torch.Tensor,
        prefix_len: int,
        gamma: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return collect_verification_payload(prob_history, x, prefix_len, gamma)

    def _generate_with_optional_rebuilt_proposal(
        self,
        cache: KVCacheModel,
        prefix: torch.Tensor,
        gamma: int,
        proposal_top_k: Optional[int],
        adapter: Optional[DecodingAdapter] = None,
        *,
        need_topk_metadata: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[object], Optional[torch.Tensor]]:
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
            q = cache._forward_with_kvcache(x)
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

    def _select_cuhlm_stage_config(
        self,
        *,
        stage: str,
        transfer_top_k: Optional[int],
        uncertainty_threshold: float,
    ) -> tuple[Optional[int], float]:
        """
        Central hook for future dynamic policies.
        `cee_cuhlm` currently uses static CUHLM settings and leaves RL disabled.
        """
        default_threshold = getattr(
            self.args,
            "uncertainty_threshold",
            uncertainty_threshold,
        )
        stage_threshold = default_threshold
        if stage == "little_to_draft":
            stage_threshold = getattr(
                self.args,
                "small_draft_threshold",
                default_threshold,
            )
        elif stage == "draft_to_target":
            stage_threshold = getattr(
                self.args,
                "draft_target_threshold",
                default_threshold,
            )

        return transfer_top_k, float(stage_threshold)

    @Register.register_decoding("dist_split_spec")
    @Register.register_decoding("dssd")
    @torch.no_grad()
    def dist_split_spec(
        self,
        prefix: torch.Tensor,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim: bool = False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 200,
        ntt_ms_edge_end: float = 20,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """
        Distributed Split Speculative Decoding (DSSD) under a symmetric-link
        assumption.

        Protocol-level behavior:
        - Uplink: send draft token ids and scalar q_j(x_j) values only.
        - Edge: verify accept/reject with target probabilities.
        - Reject path: downlink sends the full target distribution P_j(x) for the
          rejected position, and the device resamples locally from norm(max(P-Q, 0)).
        - All-accepted path: downlink sends only the next sampled token.
        """
        if use_precise_comm_sim:
            comm_simulator: CommunicationSimulator = PreciseCommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=float("inf"),
                bandwidth_cloud_end=float("inf"),
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=bool(getattr(self.args, "stochastic_ntt", False)),
            )
        # 统一计费口径（docs/protocol.md §3）：与 adaptive_tridecoding 同一套
        # 开关。CUHLM 系（uncertainty_decoding / cee_cuhlm）按决策暂不接。
        # 开关名必须出现在本方法代码里（protocols.py 运行时内省的依据）。
        comm_simulator.coalesce_rounds = (
            str(getattr(self.args, "comm_round_trip_mode", "per_transfer"))
            == "per_round"
        )
        charge_residual = bool(getattr(self.args, "charge_residual_payload", False))
        _topk_cap = int(getattr(self.args, "transfer_top_k_cap", 0) or 0)
        transfer_top_k = apply_transfer_top_k_cap(transfer_top_k, _topk_cap)
        comm_simulator.transfer_top_k = transfer_top_k
        self.color_print(f"Using transfer_top_k: {transfer_top_k}", 2)

        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        # 使用 transfer_top_k 作为草稿模型的 top-k 压缩参数
        draft_top_k = (
            transfer_top_k
            if (transfer_top_k is not None and transfer_top_k > 0)
            else self.args.top_k
        )

        # CUDA Graph：草稿缓存走 γ 步循环是图收益点；target 整段前向不图化。
        # 图开启时挂 self 跨样本复用（StaticCache 原地 reset，不重捕获）。
        _graph_kw = _graph_mode_cache_kwargs(
            self.args, cap=int(getattr(self.args, "gamma", 5)) + 4
        )
        _reused = getattr(self, "_dssd_caches", None) if _graph_kw else None
        if _reused is not None:
            approx_model_cache, target_model_cache = _reused
            approx_model_cache.reset_for_new_sample()
            target_model_cache.reset_for_new_sample()
        else:
            approx_model_cache = KVCacheModel(
                self.draft_model, self.args.temp, draft_top_k, self.args.top_p,
                **_graph_kw,
            )
            target_model_cache = KVCacheModel(
                self.target_model,
                self.args.temp,
                0,
                0,  # 目标模型不压缩
            )
            if _graph_kw:
                setattr(self, "_dssd_caches", (approx_model_cache, target_model_cache))
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache.vocab_size = self.vocab_size

        draft_forward_times = 0
        target_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0
        queuing_time = 0
        batch_delay = getattr(self.args, "batch_delay", 0)

        # 追踪 top-k 和 draft length
        total_draft_steps = 0
        sum_draft_len = 0.0
        sum_top_k = 0.0

        # 原始 prompt 长度：_stop_at_eos 需要（与 adaptive_tridecoding 一致）
        _tri_prompt_len = prefix.shape[1]

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        start_event.record(stream=torch.cuda.current_stream())

        idx: int = 0

        draft_comp_time = 0.0
        target_comp_time = 0.0

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            prefix = _ensure_token_shape(prefix, label="dssd.prefix")
            _validate_token_range(
                prefix, vocab_size=self.vocab_size, label="dssd.prefix"
            )

            idx += 1
            comm_simulator.set_round(idx)

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
                queuing_time += batch_delay
                t0 = time.time()
                _ = target_model_cache.generate(
                    _move_token_tensor(prefix, target_device), 1
                )
                target_comp_time += time.time() - t0
                target_forward_times += 1
                if self.accelerator.is_main_process:
                    self.target_forward_times += 1

                t = sample(
                    target_model_cache.prob_history[:, -1, : self.vocab_size]
                ).to(prefix.device)
                prefix = torch.cat((prefix, t), dim=1)
                self.num_acc_tokens.append(1)
                break

            current_proposal_top_k = proposal_top_k(transfer_top_k)
            rebuilt_draft_probs = None
            rebuilt_draft_meta = None
            t0 = time.time()
            if current_proposal_top_k is not None:
                x, rebuilt_draft_probs, rebuilt_draft_meta = (
                    approx_model_cache.generate_with_rebuilt_topk_metadata(
                        _move_token_tensor(prefix, draft_device),
                        current_gamma,
                        current_proposal_top_k,
                    )
                )
            else:
                x = approx_model_cache.generate(
                    _move_token_tensor(prefix, draft_device), current_gamma
                )
            draft_comp_time += time.time() - t0
            x = _ensure_token_shape(x, label="dssd.generated_x")
            _validate_token_range(
                x, vocab_size=self.vocab_size, label="dssd.generated_x"
            )
            draft_forward_times += current_gamma
            total_drafted_tokens += current_gamma

            # 累积追踪指标
            total_draft_steps += 1
            sum_draft_len += current_gamma
            sum_top_k += (
                current_proposal_top_k if current_proposal_top_k is not None else 0
            )

            draft_tokens, draft_token_probs = self._collect_dssd_uplink_payload(
                approx_model_cache.prob_history, x, prefix_len, current_gamma
            )

            # DSSD uplink only carries token ids and the scalar q_j(x_j) values.
            comm_simulator.transfer(draft_tokens, draft_token_probs, "edge_cloud")

            queuing_time += batch_delay
            t0 = time.time()
            _ = target_model_cache.generate(_move_token_tensor(x, target_device), 1)
            target_comp_time += time.time() - t0

            target_forward_times += 1
            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            verification_inputs = prepare_verification_inputs(
                draft_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=current_gamma,
                draft_probs_override=build_draft_probs_override(
                    approx_model_cache,
                    prefix_len,
                    rebuilt_draft_probs,
                ),
                draft_topk_history=stage_topk_proposal_history(
                    rebuilt_draft_meta,
                    current_gamma,
                ),
            )
            acceptance_result = compute_acceptance_result(verification_inputs)
            (
                this_step_accepted_tokens,
                n,
                _,
            ) = materialize_acceptance(verification_inputs, acceptance_result)
            total_accepted_tokens += this_step_accepted_tokens

            self.num_acc_tokens.append(this_step_accepted_tokens)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, : n + 1]
            rollback_plan = build_rollback_plan(
                prefix_len,
                verification_inputs.actual_gamma,
                n,
            )

            # 检查是否还有空间添加一个token
            if prefix.shape[1] >= max_tokens:
                apply_rollback(
                    approx_model_cache,
                    target_model_cache,
                    rollback_plan,
                )
                break

            if not rollback_plan.all_accepted:
                # Reject path: edge sends the rejected position and the full target
                # distribution P_j(x); the device resamples locally using its own
                # cached Q_j(x).
                rejection_offset = n - (prefix_len - 1)
                target_prob_row = verification_inputs.target_probs_batch[
                    :, rejection_offset, :
                ]
                if charge_residual:
                    # 统一口径（§3.4）：残差按 top-k 表示计 k*(4+元素)+元素，
                    # 不再按整行 V×元素（legacy 路径保持原样以复现历史数字）。
                    comm_simulator.simulate_transfer(
                        reject_residual_payload_bytes(target_prob_row, transfer_top_k),
                        "edge_cloud",
                    )
                else:
                    comm_simulator.simulate_transfer(INT_SIZE, "edge_cloud")
                    comm_simulator.transfer(None, target_prob_row, "edge_cloud")

                residual_probs = compute_residual_distribution(
                    target_prob_row,
                    verification_inputs.draft_probs_batch[
                        :, rejection_offset, : self.vocab_size
                    ],
                )
                t = _sample_token_from_probs(
                    residual_probs,
                    output_device=prefix.device,
                    vocab_size=self.vocab_size,
                    label="dssd.reject_sampled_t",
                )
            else:
                # All-accepted path: edge sends only the next token.
                t = _sample_token_from_probs(
                    target_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=prefix.device,
                    vocab_size=self.vocab_size,
                    label="dssd.accept_sampled_t",
                )
            t = _ensure_token_shape(t, label="dssd.sampled_t")
            _validate_token_range(t, vocab_size=self.vocab_size, label="dssd.sampled_t")

            apply_rollback(
                approx_model_cache,
                target_model_cache,
                rollback_plan,
            )

            # 最后检查添加token后是否会超出限制
            if prefix.shape[1] < max_tokens:
                prefix = torch.cat((prefix, t), dim=1)
                prefix = _ensure_token_shape(prefix, label="dssd.prefix_after_concat")
                _validate_token_range(
                    prefix,
                    vocab_size=self.vocab_size,
                    label="dssd.prefix_after_concat",
                )

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

            # Downlink returns the final continuation token and its position index.
            _send_downlink_token(comm_simulator, t, "edge_cloud")

        comm_simulator.flush_round()  # 结算最后一轮（按轮合并模式下必须）
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        # EOS 截断必须在 metrics 结算前（B9）：被截掉的 token 不应计入
        # generated_tokens/吞吐。此前截断放在函数末尾，EOS 后继续解码的
        # token 全部混进了 TPS。计算耗时是真实花费，wall_time 不回调。
        prefix, _ = self._stop_at_eos(prefix, _tri_prompt_len)

        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        throughput = (
            generated_tokens / (elapsed_time + comm_simulator.edge_cloud_comm_time)
            if (elapsed_time + comm_simulator.edge_cloud_comm_time) > 0
            else 0
        )

        metrics = get_empty_metrics()
        metrics["avg_top_k"] = (
            sum_top_k / total_draft_steps if total_draft_steps > 0 else 0
        )
        metrics["avg_draft_len"] = (
            sum_draft_len / total_draft_steps if total_draft_steps > 0 else 0
        )
        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["draft_computation_time"] = draft_comp_time
        metrics["target_computation_time"] = target_comp_time
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = total_drafted_tokens
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["wall_time"] = elapsed_time + comm_simulator.edge_cloud_comm_time
        metrics["throughput"] = throughput
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times

        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = (
            elapsed_time + queuing_time + comm_simulator.edge_cloud_comm_time
        )
        if metrics["wall_time"] > 0:
            metrics["throughput"] = metrics["generated_tokens"] / metrics["wall_time"]

        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time,
            queuing_time=queuing_time,
        )

        # 复制 edge-cloud 的带宽、top-k 和起草长度历史数据
        metrics["edge_cloud_bandwidth_history"] = (
            comm_simulator.edge_cloud_bandwidth_history.copy()
        )
        # 逐消息记账记录（[字节, 轮号]）——离线重放（scripts/rebill.py）的
        # 输入：换带宽/NTT/地板/口径的敏感性分析不用重跑 GPU。
        # getattr 兜底：测试替身可能只实现聚合口径。
        metrics["comm_trace_edge_cloud"] = getattr(
            comm_simulator, "edge_cloud_trace", []
        )
        _add_comm_accounting_metrics(metrics, self.args, comm_simulator)
        metrics["edge_cloud_topk_history"] = (
            comm_simulator.edge_cloud_topk_history.copy()
        )
        metrics["edge_cloud_draft_len_history"] = (
            comm_simulator.edge_cloud_draft_len_history.copy()
        )

        return prefix, metrics

    @Register.register_decoding("dist_spec")
    @Register.register_decoding("dsd")
    @torch.no_grad()
    def dist_spec(
        self,
        prefix,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim: bool = False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 200,
        ntt_ms_edge_end: float = 20,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        if use_precise_comm_sim:
            comm_simulator: CommunicationSimulator = PreciseCommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=float("inf"),
                bandwidth_cloud_end=float("inf"),
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=bool(getattr(self.args, "stochastic_ntt", False)),
            )
        # 统一计费口径（docs/protocol.md §3）：与 adaptive_tridecoding 同一套
        # 开关。CUHLM 系（uncertainty_decoding / cee_cuhlm）按决策暂不接。
        # 开关名必须出现在本方法代码里（protocols.py 运行时内省的依据）。
        comm_simulator.coalesce_rounds = (
            str(getattr(self.args, "comm_round_trip_mode", "per_transfer"))
            == "per_round"
        )
        charge_residual = bool(getattr(self.args, "charge_residual_payload", False))
        _topk_cap = int(getattr(self.args, "transfer_top_k_cap", 0) or 0)
        transfer_top_k = apply_transfer_top_k_cap(transfer_top_k, _topk_cap)
        comm_simulator.transfer_top_k = transfer_top_k
        self.color_print(f"Using transfer_top_k: {transfer_top_k}", 2)

        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        # CUDA Graph：草稿缓存走 γ 步循环是图收益点；target 整段前向不图化。
        # 图开启时挂 self 跨样本复用（StaticCache 原地 reset，不重捕获）。
        _graph_kw = _graph_mode_cache_kwargs(
            self.args, cap=int(getattr(self.args, "gamma", 5)) + 4
        )
        _reused = getattr(self, "_dsd_caches", None) if _graph_kw else None
        if _reused is not None:
            approx_model_cache, target_model_cache = _reused
            approx_model_cache.reset_for_new_sample()
            target_model_cache.reset_for_new_sample()
        else:
            approx_model_cache = KVCacheModel(
                self.draft_model, self.args.temp, self.args.top_k, self.args.top_p,
                **_graph_kw,
            )
            target_model_cache = KVCacheModel(
                self.target_model, self.args.temp, self.args.top_k, self.args.top_p
            )
            if _graph_kw:
                setattr(self, "_dsd_caches", (approx_model_cache, target_model_cache))
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache.vocab_size = self.vocab_size

        draft_forward_times = 0
        target_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0
        queuing_time = 0
        batch_delay = getattr(self.args, "batch_delay", 0)

        # 追踪 top-k 和 draft length
        total_draft_steps = 0
        sum_draft_len = 0.0
        sum_top_k = 0.0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        start_event.record(stream=torch.cuda.current_stream())

        idx: int = 0

        draft_comp_time = 0.0
        target_comp_time = 0.0

        _tri_prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点

        while prefix.shape[1] < max_tokens:

            # B10：循环内 EOS 早停（与 dssd/adaptive_* 已接线方法语义对齐——
            # 计算与通信在 EOS 后立即停，时延口径跨方法可比）
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            idx += 1
            comm_simulator.set_round(idx)

            prefix_len = prefix.shape[1]

            # 确保不会生成超过max_tokens的token
            remaining_tokens = max_tokens - prefix_len
            if remaining_tokens <= 0:
                break

            if idx == 1:
                comm_simulator.transfer(prefix, None, "edge_cloud")  # 初始上下文传输

            # 调整gamma以不超过剩余的token数量
            current_gamma = min(
                self.args.gamma, remaining_tokens - 1
            )  # 减1是为了留给最后的采样token
            if current_gamma <= 0:
                # 如果只剩1个token，直接用target model生成
                queuing_time += batch_delay
                t0 = time.time()
                _ = target_model_cache.generate(prefix.to(target_device), 1)
                target_comp_time += time.time() - t0
                target_forward_times += 1
                if self.accelerator.is_main_process:
                    self.target_forward_times += 1

                t = sample(
                    target_model_cache.prob_history[:, -1, : self.vocab_size]
                ).to(prefix.device)
                prefix = torch.cat((prefix, t), dim=1)
                self.num_acc_tokens.append(1)
                break

            current_proposal_top_k = proposal_top_k(transfer_top_k)
            rebuilt_draft_probs = None
            rebuilt_draft_meta = None
            t0 = time.time()
            if current_proposal_top_k is not None:
                x, rebuilt_draft_probs, rebuilt_draft_meta = (
                    approx_model_cache.generate_with_rebuilt_topk_metadata(
                        prefix.to(draft_device),
                        current_gamma,
                        current_proposal_top_k,
                    )
                )
            else:
                x = approx_model_cache.generate(prefix.to(draft_device), current_gamma)
            draft_comp_time += time.time() - t0
            draft_forward_times += current_gamma
            total_drafted_tokens += current_gamma

            # 累积追踪指标
            total_draft_steps += 1
            sum_draft_len += current_gamma
            # B18：avg_top_k 只统计"传输压缩 top-k"，未压缩记 0（与 dssd/
            # tridecoding/adaptive_decoding 同名列一致）。此前回退到
            # self.args.top_k（采样 top-k），与传输/DRA 的选择无关，导致同一
            # 指标列在 dsd 上是另一个量。
            sum_top_k += (
                transfer_top_k
                if transfer_top_k is not None and transfer_top_k > 0
                else 0
            )

            # 上行只计本轮新草稿（x = prefix + 本轮γ个新token）。KV/前缀留存云端，
            # 已确认前缀不重发；拒绝回滚后 x 会变短，故按"本轮输入前缀之后"切片，
            # 不做跨轮长度记账。旧实现按全长 x 计费 ⇒ O(L²) 字节膨胀（曾实测
            # t5a_dsd 3068B/生成tok，名义 ~8B）。
            comm_simulator.transfer(x[:, prefix.shape[1] :], None, "edge_cloud")
            draft_prob_window = (
                rebuilt_draft_probs
                if rebuilt_draft_probs is not None
                else approx_model_cache.prob_history[:, -(1 + current_gamma) : -1, :]
            )
            queuing_time += batch_delay
            t0 = time.time()
            _ = target_model_cache.generate(x.to(target_device), 1)
            target_comp_time += time.time() - t0

            target_forward_times += 1
            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            comm_simulator.transfer(
                None,
                draft_prob_window,
                "edge_cloud",
                transfer_top_k is not None and transfer_top_k > 0,
                transfer_top_k,
            )

            verification_inputs = prepare_verification_inputs(
                draft_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=current_gamma,
                draft_probs_override=build_draft_probs_override(
                    approx_model_cache,
                    prefix_len,
                    rebuilt_draft_probs,
                ),
                draft_topk_history=stage_topk_proposal_history(
                    rebuilt_draft_meta,
                    current_gamma,
                ),
            )
            acceptance_result = compute_acceptance_result(verification_inputs)
            accepted_count, n, _ = materialize_acceptance(
                verification_inputs, acceptance_result
            )
            should_send_reject_signal = (
                verification_inputs.actual_gamma < current_gamma
                or accepted_count < verification_inputs.actual_gamma
            )

            if should_send_reject_signal:
                comm_simulator.send_reject_message("edge_cloud")

            this_step_accepted_tokens = accepted_count
            total_accepted_tokens += this_step_accepted_tokens

            self.num_acc_tokens.append(this_step_accepted_tokens)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, : n + 1]

            rollback_plan = build_rollback_plan(
                prefix_len,
                verification_inputs.actual_gamma,
                n,
            )

            # 检查是否还有空间添加一个token
            if prefix.shape[1] >= max_tokens:
                apply_rollback(
                    approx_model_cache,
                    target_model_cache,
                    rollback_plan,
                )
                break

            if not rollback_plan.all_accepted:
                rejection_offset = n - (prefix_len - 1)
                target_prob_row = verification_inputs.target_probs_batch[
                    :, rejection_offset, :
                ]
                if charge_residual:
                    # 统一口径（§3.4）：legacy 的 dsd 拒绝位置零计费（残差被
                    # 无声地省掉）；honest 补 k*(4+元素)+元素。
                    comm_simulator.simulate_transfer(
                        reject_residual_payload_bytes(target_prob_row, transfer_top_k),
                        "edge_cloud",
                    )

                t = sample_reject_token(
                    target_prob_row,
                    verification_inputs.draft_probs_batch[
                        :, rejection_offset, : self.vocab_size
                    ],
                    output_device=prefix.device,
                )
            else:
                t = sample_accept_token(
                    target_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=prefix.device,
                )

            apply_rollback(
                approx_model_cache,
                target_model_cache,
                rollback_plan,
            )

            # 最后检查添加token后是否会超出限制
            if prefix.shape[1] < max_tokens:
                prefix = torch.cat((prefix, t), dim=1)

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

            # dist_spec 下行：采样的 token 与其位置索引合并成一次往返
            #（B17 统一口径；此前只付了 INT_SIZE，token 本体没计）。
            # 草稿序列、概率窗口与拒绝信号仍走上面的协议专属传输。
            _send_downlink_token(comm_simulator, t, "edge_cloud")

        comm_simulator.flush_round()  # 结算最后一轮（按轮合并模式下必须）
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        throughput = (
            generated_tokens / (elapsed_time + comm_simulator.edge_cloud_comm_time)
            if (elapsed_time + comm_simulator.edge_cloud_comm_time) > 0
            else 0
        )

        metrics = get_empty_metrics()
        metrics["avg_top_k"] = (
            sum_top_k / total_draft_steps if total_draft_steps > 0 else 0
        )
        metrics["avg_draft_len"] = (
            sum_draft_len / total_draft_steps if total_draft_steps > 0 else 0
        )
        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["draft_computation_time"] = draft_comp_time
        metrics["target_computation_time"] = target_comp_time
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = total_drafted_tokens
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["wall_time"] = elapsed_time + comm_simulator.edge_cloud_comm_time
        metrics["throughput"] = throughput
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times

        batch_delay = getattr(self.args, "batch_delay", 0)
        queuing_time = target_forward_times * batch_delay
        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] += queuing_time
        if metrics["wall_time"] > 0:
            metrics["throughput"] = metrics["generated_tokens"] / metrics["wall_time"]

        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time,
            queuing_time=queuing_time,
            draft_comp_time=draft_comp_time,
            target_comp_time=target_comp_time,
        )

        # 复制 edge-cloud 的带宽、top-k 和起草长度历史数据
        metrics["edge_cloud_bandwidth_history"] = (
            comm_simulator.edge_cloud_bandwidth_history.copy()
        )
        # 逐消息记账记录（[字节, 轮号]）——离线重放（scripts/rebill.py）的
        # 输入：换带宽/NTT/地板/口径的敏感性分析不用重跑 GPU。
        # getattr 兜底：测试替身可能只实现聚合口径。
        metrics["comm_trace_edge_cloud"] = getattr(
            comm_simulator, "edge_cloud_trace", []
        )
        _add_comm_accounting_metrics(metrics, self.args, comm_simulator)
        metrics["edge_cloud_topk_history"] = (
            comm_simulator.edge_cloud_topk_history.copy()
        )
        metrics["edge_cloud_draft_len_history"] = (
            comm_simulator.edge_cloud_draft_len_history.copy()
        )

        # 遵守 max_tokens：投机解码按整块追加，最后一轮可能多出若干 token。
        # 参照 dist_spec（它靠 max(0, remaining-1) 截断 γ 来保证不越界），这里显式
        # 截断，保证与 target_only / 普通 SD 的长度契约一致——否则会多拿 token，
        # 使配对质量比较与时延/吞吐统计都不公平。
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, :max_tokens]
        return prefix, metrics

    @Register.register_decoding("uncertainty_decoding")
    @Register.register_decoding("cuhlm")
    @torch.no_grad()
    def uncertainty_decoding(
        self,
        prefix,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim=False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 200,
        ntt_ms_edge_end: float = 20,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """
        Implement of the method raised in "Communication-Efficient Hybrid Language Model via Uncertainty-Aware Opportunistic and Compressed Transmission"
        """
        if use_precise_comm_sim:
            comm_simulator: CUHLM = PreciseCUHLM(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                uncertainty_threshold=getattr(
                    self.args, "uncertainty_threshold", 0.8
                ),
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            threshold = getattr(self.args, "uncertainty_threshold", 0.8)
            comm_simulator = CUHLM(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                uncertainty_threshold=threshold,
                dimension="Mbps",
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=bool(getattr(self.args, "stochastic_ntt", False)),
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )

        # F-CUHLM 口径自述：随 metrics 落盘（eval/utils.py get_save_dict
        # 读取 args.protocol_deviations），保证新口径跑出的工件可辨识、
        # 不与 2026-10-05 前的旧口径结果混排。
        # 2026-10-09 统一往返（docs/protocol.md §3.4）：per_round 下每次
        # 触发的上行+下行合并成一次云请求往返（1×NTT）。载荷字节仍是
        # 论文式 (5) 口径（cuhlm_uplink_payload_bytes）。
        comm_simulator.coalesce_rounds = (
            str(getattr(self.args, "comm_round_trip_mode", "per_transfer"))
            == "per_round"
        )
        _dev = (
            "cuhlm_fair_accounting: skip=0RTT(buffered resync), "
            "trigger=1 uplink(resync+draft+topk)+1 downlink, "
            "queue=per-trigger, reject-resample=x_hat; "
            "round_trip=unified(1xNTT/trigger since 2026-10-09)"
        )
        _devs = list(getattr(self.args, "protocol_deviations", ()) or ())
        if _dev not in _devs:
            self.args.protocol_deviations = tuple(_devs + [_dev])

        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        # 使用 transfer_top_k 作为草稿模型的 top-k 压缩参数
        draft_top_k = (
            transfer_top_k
            if (transfer_top_k is not None and transfer_top_k > 0)
            else self.args.top_k
        )

        # CUDA Graph：只有草稿缓存跑 γ 步单 token 循环，是图的收益点；
        # target 是整段前向，开图无收益反而多占显存（与 engine.py 的取舍一致）。
        # 图开启时缓存挂 self 跨样本复用（StaticCache 原地 reset，不重捕获）。
        _graph_kw = _graph_mode_cache_kwargs(
            self.args, cap=int(getattr(self.args, "gamma", 5)) + 4
        )
        _reused = (
            getattr(self, "_uncertainty_decoding_caches", None) if _graph_kw else None
        )
        if _reused is not None:
            approx_model_cache, target_model_cache = _reused
            approx_model_cache.reset_for_new_sample()
            target_model_cache.reset_for_new_sample()
        else:
            approx_model_cache = KVCacheModel(
                self.draft_model, self.args.temp, draft_top_k, self.args.top_p,
                **_graph_kw,
            )
            target_model_cache = KVCacheModel(
                self.target_model,
                self.args.temp,
                0,
                0,  # 目标模型不压缩
            )
            if _graph_kw:
                setattr(
                    self,
                    "_uncertainty_decoding_caches",
                    (approx_model_cache, target_model_cache),
                )
        approx_model_cache.vocab_size = self.vocab_size
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

        draft_comp_time = 0.0
        target_comp_time = 0.0

        _tri_prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点

        # F-CUHLM 口径（2026-10-05 对齐论文 Algorithm 1 的通信模式；
        # 2026-04 起字节按论文式 (5) 计，见 docs/protocol.md §3 CUHLM 小节）：
        # - 跳过分支：零通信、零排队。被跳过的 token 缓存在端侧
        #   （_pending_resync 计数），下次触发时随上行消息捎带——论文 III-B
        #   Step 5 "lightweight index-based resynchronization...negligible and
        #   therefore omitted from the cost analysis"：索引不计字节、不单独
        #   付出 RTT。
        # - 触发分支：一次上行（一次 NTT），载荷 = 论文式 (5) 的压缩形式
        #   k(t)·(b_prob+b_index) bits（b_prob=8、b_index=⌈log₂V⌉，即
        #   cuhlm_uplink_payload_bytes）；draft token 索引 negligible 不计
        #   字节。旧口径：token 按 element_size、概率按 fp32+int32（k×8B），
        #   比论文口径高约 2.8×。
        #   + 一次 0 字节下行（一次 NTT，_send_downlink_index_only）。
        #   batch_delay 只在触发时计（与 DSD/DSSD/CEE-SD 的排队口径一致，
        #   它们均为 target_forward_times × batch_delay）。
        # - reject 重采样改用压缩重构分布 x̂（论文式 17）；旧实现用全量分布，
        #   压缩包只计费不上场（精度口径偏向该基线）。reject 在服务器端
        #   基于已持有的分布重采样（论文式 17），不产生额外传输——这与
        #   统一口径的 reject 残差计费不同，是论文自身的协议设计。
        _pending_resync = 0

        while prefix.shape[1] < max_tokens:

            # B10：循环内 EOS 早停（与 dssd/adaptive_* 已接线方法语义对齐——
            # 计算与通信在 EOS 后立即停，时延口径跨方法可比）
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            loop_idx += 1
            prefix_len = prefix.shape[1]
            # 统一往返（§3.4）：标记新一轮，上一轮累积的字节在此结算
            # （per_round = 每次云端交互 1×NTT；per_transfer 下是 no-op）
            comm_simulator.set_round(loop_idx)

            if loop_idx == 1:
                comm_simulator.transfer(prefix, None, link_type="edge_cloud")

            # Draft generates 1 token
            t0 = time.time()
            x = approx_model_cache.generate(prefix.to(draft_device), 1)
            draft_comp_time += time.time() - t0
            if approx_model_cache.logits_history is not None:
                current_logit = approx_model_cache.logits_history[
                    :, -1, : self.vocab_size
                ]
            else:
                raise ValueError("Approx model logits history is None")

            # Calculate uncertainty BEFORE target call (as in the paper)
            uncertainty = comm_simulator.calculate_uncertainty(
                current_logit, M=20, theta_max=2.0, draft_token=int(x[0, -1].item())
            )
            # 压缩词表公式（论文式 (16)/(23)/(26)）定义在概率分布 x(t) 上：
            # 不确定度用原始 logits 算（温度扰动采样，设计如此），
            # 但 k* 的求解必须吃 softmax 后的概率分布。
            current_probs = comm_simulator._get_current_probs(
                approx_model_cache.prob_history
            )
            should_transfer, vocab_size = comm_simulator.determine_transfer_strategy(
                uncertainty, current_probs
            )

            draft_forward_times += 1
            total_drafted_tokens += 1

            if not should_transfer:
                # Low uncertainty: truly skip target model (as in the paper)
                # target_forward_times NOT incremented (target genuinely not called)
                # target_comp_time NOT incremented

                accepted_token = x[:, -1:]
                prefix = torch.cat(
                    (prefix.to(accepted_token.device), accepted_token), dim=1
                )

                # F-CUHLM：跳过即零通信——不发 accept 消息、不即时上行；
                # token 计入待同步缓冲，下次触发时随上行捎带（见循环上方口径说明）。
                _pending_resync += 1

                # No bonus token from target (target was not called)
                # No KVCache rollback needed (target cache was not advanced)

                if use_early_stopping and self._check_stopping_criteria(
                    prefix, stop_sequences
                ):
                    break

                continue

            # High uncertainty: run target model for verification
            # F-CUHLM：排队只在真正发起云端调用时计（对齐 DSD/DSSD/CEE-SD）。
            queuing_time += batch_delay

            # F-CUHLM：触发 = 一次合并上行（一次 NTT）。载荷按论文式 (5) 的
            # 压缩形式计：k(t)·(b_prob+b_index) bits（b_prob=8、b_index=
            # ⌈log₂V⌉）。捎带的重同步 token 与当前 draft token 的索引按论文
            # §II-B/III-B Step 5 的 negligible 假设**不计字节**（旧口径按
            # element_size 逐 token 计）；topk/draft_len 仍进历史记录，
            # avg_top_k 反映的是自适应 k(t)。
            comm_simulator.simulate_transfer(
                cuhlm_uplink_payload_bytes(int(vocab_size), self.vocab_size),
                "edge_cloud",
                topk=int(vocab_size),
                draft_len=_pending_resync + 1,
            )
            _pending_resync = 0

            t0 = time.time()
            _ = target_model_cache.generate(x.to(target_device), 1)
            target_comp_time += time.time() - t0
            target_forward_times += 1

            # Rejection sampling with compressed probability distribution
            n = prefix_len + 1 - 1

            verification_inputs = prepare_verification_inputs(
                draft_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=1,
            )
            acceptance_result = compute_acceptance_result(verification_inputs)
            accepted_count, n, _ = materialize_acceptance(
                verification_inputs, acceptance_result
            )

            self.color_print(
                f"Uncertainty: {uncertainty:.4f}, Vocab size: {vocab_size}", 3
            )

            # F-CUHLM：压缩分布已在触发时的合并上行中计费（每次触发必传，
            # 与论文一致）；reject 不再有独立的 reject 消息与二次压缩包传输，
            # 下行只回传最终 token（见下方 _send_downlink_token）。
            total_accepted_tokens += accepted_count

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, : n + 1]

            rollback_plan = build_rollback_plan(
                prefix_len,
                verification_inputs.actual_gamma,
                n,
            )

            if not rollback_plan.all_accepted:
                target_prob_row = verification_inputs.target_probs_batch[:, 0, :]
                # F-CUHLM：重采样基于服务器实际持有的压缩重构分布 x̂
                # （论文式 (17)，top-k 保留 + 残差均匀）。旧实现用端侧全量
                # 分布，压缩失真对输出零影响——精度口径曾偏向该基线。
                # compress_rebuild_probs 严格要求 (batch, seq, vocab) 三维，
                # 取单行后切回 (1, vocab) 喂 sample_reject_token。
                _draft_row_hat = comm_simulator.compress_rebuild_probs(
                    approx_model_cache.prob_history[:, n : n + 1, : self.vocab_size],
                    int(vocab_size),
                )[:, 0, :]
                t = sample_reject_token(
                    target_prob_row,
                    _draft_row_hat,
                    output_device=prefix.device,
                )
            else:
                t = sample_accept_token(
                    target_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=prefix.device,
                )

            apply_rollback(
                approx_model_cache,
                target_model_cache,
                rollback_plan,
            )

            # F-CUHLM（论文口径）：响应 token 索引 negligible（§II-B）——
            # 0 字节下行报文，保留一次 NTT（见 _send_downlink_index_only）。
            # 旧口径（B17）：INT_SIZE + token 字节。
            _send_downlink_index_only(comm_simulator, "edge_cloud")
            prefix = torch.cat((prefix, t), dim=1)

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        # 最后一轮（或中途 break 时）的字节可能仍在合并桶里，结算必须在
        # wall_time/communication_time 读数之前。
        comm_simulator.flush_round()
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        metrics = get_empty_metrics()

        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["draft_computation_time"] = draft_comp_time
        metrics["target_computation_time"] = target_comp_time
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
        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time,
            queuing_time=queuing_time,
            draft_comp_time=draft_comp_time,
            target_comp_time=target_comp_time,
        )
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times

        # 复制 edge-cloud 的带宽、top-k 和起草长度历史数据
        metrics["edge_cloud_bandwidth_history"] = (
            comm_simulator.edge_cloud_bandwidth_history.copy()
        )
        # 逐消息记账记录（[字节, 轮号]）——离线重放（scripts/rebill.py）的
        # 输入：换带宽/NTT/地板/口径的敏感性分析不用重跑 GPU。
        # getattr 兜底：测试替身可能只实现聚合口径。
        metrics["comm_trace_edge_cloud"] = getattr(
            comm_simulator, "edge_cloud_trace", []
        )
        _add_comm_accounting_metrics(metrics, self.args, comm_simulator)
        metrics["edge_cloud_topk_history"] = (
            comm_simulator.edge_cloud_topk_history.copy()
        )
        metrics["edge_cloud_draft_len_history"] = (
            comm_simulator.edge_cloud_draft_len_history.copy()
        )

        return prefix, metrics

    @Register.register_decoding("tk_slt")
    @Register.register_decoding("tkslt")
    @torch.no_grad()
    def tk_slt(
        self,
        prefix,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim: bool = False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 200,
        ntt_ms_edge_end: float = 20,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """Top-K Sparse Logits Transmission（TK-SLT）基线。

        论文 "Communication-Efficient Collaborative LLM Inference via
        Distributed Speculative Decoding"（WCSP'25, Zheng & Yang）：DSD 框架
        下，端侧 SLM 只对 top-K logits 做 softmax——草稿分布 Y_i 的支撑集
        就是 top-K（§III Solution 1），采样自该稀疏分布；上行每个草稿位置
        只传 K 个概率值 + 词表索引，BS 用重建的稀疏 Q 做标准投机验证，
        拒绝时从 norm(max(0, P−Q)) 重采样（稀疏 Q ⇒ 非 top-K 位置拿到
        完整的 P，正是论文的残差定义）。

        通信口径 = 论文自身（与 CUHLM 系同一处理方式，见 docs/protocol.md
        §3.1/§3.3），不接仓库统一计费开关：
        - 上行：γ·K·b_prob bits，b_prob=16（FP16，§VI-B）。概率值**真实**
          量化到 fp16 再按行重归一化——验证判据与计费看到同一个 q̂
          （B15/B16 的"计费与数据一致"原则）；
        - 草稿 token 索引、下行响应（结果 token + 位置 j）按 §II-B 判为
          negligible：0 字节报文，链路模型仍按报文计一次 NTT；
        - K<=0/None ⇒ 整词表载荷 + 全量 softmax 提议（vanilla DSD，
          该论文自己的不压缩基线）。

        可选 --tk_slt_odld：按论文 §V Algorithm 1/2（ODLD/AS²）在线估计
        (α, b, c)（接受率、T_V/T_LLM、T_SLM/T_LLM 的运行均值）并逐轮选
        γ*；S*<=1 的轮次退回 standalone LLM（target 直出、无上行分布）。
        默认关闭（固定 γ，与 dsd/dssd/cuhlm 同口径可比）；首轮无估计，
        用 --gamma 兜底。
        """
        if use_precise_comm_sim:
            comm_simulator: CommunicationSimulator = PreciseCommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=float("inf"),
                bandwidth_cloud_end=float("inf"),
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=bool(getattr(self.args, "stochastic_ntt", False)),
            )
        comm_simulator.transfer_top_k = transfer_top_k
        self.color_print(f"TK-SLT using top-K: {transfer_top_k}", 2)
        # 统一往返口径（2026-10-09，docs/protocol.md §3.4）：per_round 下
        # 一轮的上行载荷与 0B 下行合并成一次云请求往返（1×NTT）。
        # 此前逐报文各付 50ms（每轮 2×NTT），比 dsd/dssd/ours 每轮多付
        # 50ms 纯记账差异。载荷字节仍是论文口径（tk_slt_uplink_payload_bytes）。
        comm_simulator.coalesce_rounds = (
            str(getattr(self.args, "comm_round_trip_mode", "per_transfer"))
            == "per_round"
        )

        # 口径自述随 metrics 落盘（F-CUHLM 同款），保证论文口径跑出的工件
        # 可辨识、不与仓库统一口径的结果混排。
        _dev = (
            "tk_slt_paper_accounting: uplink=gamma*K*b_prob(FP16 16b, "
            "real-quantized), indices negligible, "
            "reject-resample=server-side sparse fp16 Q; "
            "round_trip=unified(1xNTT/round since 2026-10-09)"
        )
        _devs = list(getattr(self.args, "protocol_deviations", ()) or ())
        if _dev not in _devs:
            self.args.protocol_deviations = tuple(_devs + [_dev])

        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        # K 同时定义采样稀疏性（softmax 只作用于 top-K logits）与传输载荷。
        # K<=0/None ⇒ 不压缩：全量 softmax 提议 + 整词表载荷（vanilla DSD）。
        topk_enabled = transfer_top_k is not None and int(transfer_top_k) > 0
        draft_sampling_top_k = int(transfer_top_k) if topk_enabled else 0
        topk_history_val = int(transfer_top_k) if topk_enabled else 0

        odld_enabled = bool(getattr(self.args, "tk_slt_odld", False))

        # CUDA Graph：草稿缓存走 γ 步循环是图收益点；target 整段前向不图化
        # （与 dist_spec 同一取舍）。ODLD 开启时 γ* 可超过 --gamma，cap 按
        # ODLD 上限取，保证验证图档位覆盖。
        default_gamma_cap = int(getattr(self.args, "gamma", 5))
        _odld_cap = _TK_SLT_ODLD_GAMMA_CAP if odld_enabled else default_gamma_cap
        _graph_kw = _graph_mode_cache_kwargs(self.args, cap=_odld_cap + 4)
        _reused = getattr(self, "_tk_slt_caches", None) if _graph_kw else None
        if _reused is not None:
            approx_model_cache, target_model_cache = _reused
            approx_model_cache.reset_for_new_sample()
            target_model_cache.reset_for_new_sample()
        else:
            approx_model_cache = KVCacheModel(
                self.draft_model,
                self.args.temp,
                draft_sampling_top_k,
                self.args.top_p,
                **_graph_kw,
            )
            target_model_cache = KVCacheModel(
                self.target_model,
                self.args.temp,
                0,
                0,  # 目标模型不压缩
            )
            if _graph_kw:
                self._tk_slt_caches = (approx_model_cache, target_model_cache)
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache.vocab_size = self.vocab_size

        draft_forward_times = 0
        target_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0
        queuing_time = 0
        batch_delay = getattr(self.args, "batch_delay", 0)

        # 追踪 top-k 和 draft length（B18：avg_top_k 只统计传输压缩 top-k，
        # 未压缩记 0，与 dsd/dssd/tridecoding 同名列口径一致）
        total_draft_steps = 0
        sum_draft_len = 0.0
        sum_top_k = 0.0

        # ODLD/AS² 的在线估计（论文式 (5)：b=T_V/T_LLM、c=T_SLM/T_LLM；
        # α=期望接受率）。运行均值跨轮更新；standalone 轮不产生新估计
        # （无草稿/验证），只有 DSD 轮喂入。
        alpha_num = 0.0  # Σ 接受 token 数
        alpha_den = 0.0  # Σ 草稿 token 数
        t_slm_seconds = 0.0  # Σ 草稿前向耗时
        t_slm_tokens = 0
        t_llm_seconds = 0.0  # Σ 验证前向耗时
        t_llm_runs = 0
        t_v_seconds = 0.0  # Σ 上行纯发射耗时（不含 NTT，同论文 T_V 口径）
        t_v_dists = 0  # Σ 已传分布数
        standalone_rounds = 0
        last_gamma_star = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        start_event.record(stream=torch.cuda.current_stream())

        idx: int = 0

        draft_comp_time = 0.0
        target_comp_time = 0.0

        _tri_prompt_len = prefix.shape[1]  # EOS 早停的生成段起点（B10）

        while prefix.shape[1] < max_tokens:
            # 循环内 EOS 早停（B10）：计算与通信在 EOS 后立即停，
            # 时延口径与 dsd/dssd/adaptive_* 对齐
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            idx += 1
            prefix_len = prefix.shape[1]
            # 轮号标注（离线重放的分组依据；per_round 下轮首 bucket 已空，
            # set_round 的防御性 flush 是 no-op，不影响轮末结算语义）
            comm_simulator.set_round(idx)
            prefix = _ensure_token_shape(prefix, label="tk_slt.prefix")
            _validate_token_range(
                prefix, vocab_size=self.vocab_size, label="tk_slt.prefix"
            )

            remaining_tokens = max_tokens - prefix_len
            if remaining_tokens <= 0:
                break

            if idx == 1:
                # 初始上下文上传（全仓一次性约定：论文不建模 prompt）
                comm_simulator.transfer(prefix, None, "edge_cloud")

            # AS²（论文 Algorithm 2）：有估计后才启用；S*<=1 的轮次退回
            # standalone LLM。首轮（无估计）用 --gamma。
            base_gamma = int(self.args.gamma)
            if odld_enabled and t_llm_runs >= 1 and alpha_den > 0:
                alpha_hat = alpha_num / alpha_den
                llm_per_run = t_llm_seconds / t_llm_runs
                b_hat = (
                    (t_v_seconds / t_v_dists) / llm_per_run if t_v_dists > 0 else 0.0
                )
                c_hat = (
                    (t_slm_seconds / t_slm_tokens) / llm_per_run
                    if t_slm_tokens > 0
                    else 0.0
                )
                use_dsd, gamma_star, _s_star = tk_slt_select_speculative(
                    alpha_hat, b_hat, c_hat, _TK_SLT_ODLD_GAMMA_CAP
                )
                last_gamma_star = gamma_star
                self.color_print(
                    f"ODLD: alpha={alpha_hat:.3f} b={b_hat:.3f} c={c_hat:.3f} "
                    f"-> gamma*={gamma_star} S*={_s_star:.3f}",
                    3,
                )
                if not use_dsd:
                    # standalone LLM 轮（AS² 分支）：target 直出 1 token。
                    # 无草稿、无上行分布；下行响应 negligible（0 字节报文）。
                    # 1-token 前向 ≠ 验证前向，不进 T_LLM 估计。
                    standalone_rounds += 1
                    queuing_time += batch_delay
                    t0 = time.time()
                    _ = target_model_cache.generate(
                        _move_token_tensor(prefix, target_device), 1
                    )
                    target_comp_time += time.time() - t0
                    target_forward_times += 1
                    if self.accelerator.is_main_process:
                        self.target_forward_times += 1

                    t = _sample_token_from_probs(
                        target_model_cache.prob_history[:, -1, : self.vocab_size],
                        output_device=prefix.device,
                        vocab_size=self.vocab_size,
                        label="tk_slt.standalone_t",
                    )
                    prefix = torch.cat((prefix, t), dim=1)
                    self.num_acc_tokens.append(1)

                    if use_early_stopping and self._check_stopping_criteria(
                        prefix, stop_sequences
                    ):
                        break
                    _send_downlink_index_only(comm_simulator, "edge_cloud")
                    continue
                base_gamma = gamma_star

            # 调整 gamma 以不超过剩余的 token 数（减 1 留给最后的采样 token）
            current_gamma = min(base_gamma, remaining_tokens - 1)
            if current_gamma <= 0:
                # 只剩 1 个 token：target 直出（与 dist_spec 的边界分支同口径，
                # 不计通信）
                queuing_time += batch_delay
                t0 = time.time()
                _ = target_model_cache.generate(
                    _move_token_tensor(prefix, target_device), 1
                )
                target_comp_time += time.time() - t0
                target_forward_times += 1
                if self.accelerator.is_main_process:
                    self.target_forward_times += 1

                t = _sample_token_from_probs(
                    target_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=prefix.device,
                    vocab_size=self.vocab_size,
                    label="tk_slt.fallback_t",
                )
                prefix = torch.cat((prefix, t), dim=1)
                self.num_acc_tokens.append(1)
                break

            # ---- Draft：top-K 稀疏采样（softmax 只作用于 top-K logits）----
            t0 = time.time()
            x = approx_model_cache.generate(
                _move_token_tensor(prefix, draft_device), current_gamma
            )
            draft_comp_time += time.time() - t0
            x = _ensure_token_shape(x, label="tk_slt.generated_x")
            _validate_token_range(
                x, vocab_size=self.vocab_size, label="tk_slt.generated_x"
            )
            draft_forward_times += current_gamma
            total_drafted_tokens += current_gamma
            t_slm_seconds += time.time() - t0
            t_slm_tokens += current_gamma

            total_draft_steps += 1
            sum_draft_len += current_gamma
            sum_top_k += topk_history_val

            # ---- 上行：K 个概率/草稿位置，FP16 量化后传输 ----
            # prob_history 行 = 稀疏分布（temp>0 时非零项 ≤ K；temp=0 时
            # one-hot）。窗口按"本轮输入前缀之后"切片，与计费的 γ 一致。
            window_end = min(
                prefix_len + current_gamma - 1, approx_model_cache.prob_history.shape[1]
            )
            draft_prob_window = approx_model_cache.prob_history[
                :, prefix_len - 1 : window_end, :
            ]
            window_rows = int(draft_prob_window.shape[1])
            quantized_window = _quantize_probs_fp16(draft_prob_window)

            comm_simulator.simulate_transfer(
                tk_slt_uplink_payload_bytes(
                    transfer_top_k, window_rows, self.vocab_size
                ),
                "edge_cloud",
                topk=topk_history_val,
                draft_len=current_gamma,
            )
            # ODLD 的 b̂ 估计读"本轮上行"的纯发射时长（TransferUnit.tx_time）。
            # per_transfer（直充）模式下上行报文此刻已在 stats 里；per_round
            # 合并模式下字节推迟到轮末 flush 才落账，改在 flush 后读。
            if not comm_simulator.coalesce_rounds:
                _tx_seconds = _last_edge_cloud_tx_seconds(comm_simulator)
                if _tx_seconds is not None:
                    t_v_seconds += _tx_seconds
                    t_v_dists += window_rows

            # ---- Verify：BS 用收到的稀疏 FP16 分布做标准投机验证 ----
            queuing_time += batch_delay
            t0 = time.time()
            _ = target_model_cache.generate(_move_token_tensor(x, target_device), 1)
            target_comp_time += time.time() - t0
            t_llm_seconds += time.time() - t0
            t_llm_runs += 1

            target_forward_times += 1
            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            verification_inputs = prepare_verification_inputs(
                draft_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=current_gamma,
                draft_probs_override=build_draft_probs_override(
                    approx_model_cache,
                    prefix_len,
                    quantized_window,
                ),
            )
            acceptance_result = compute_acceptance_result(verification_inputs)
            (
                this_step_accepted_tokens,
                n,
                _,
            ) = materialize_acceptance(verification_inputs, acceptance_result)
            total_accepted_tokens += this_step_accepted_tokens
            alpha_num += this_step_accepted_tokens
            alpha_den += verification_inputs.actual_gamma

            self.num_acc_tokens.append(this_step_accepted_tokens)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, : n + 1]
            rollback_plan = build_rollback_plan(
                prefix_len,
                verification_inputs.actual_gamma,
                n,
            )

            # 检查是否还有空间添加一个 token
            if prefix.shape[1] >= max_tokens:
                apply_rollback(
                    approx_model_cache,
                    target_model_cache,
                    rollback_plan,
                )
                break

            if not rollback_plan.all_accepted:
                # 拒绝路径：BS 基于已持有的稀疏 FP16 分布 Q̂ 从
                # norm(max(0, P−Q̂)) 重采样（论文 §II-A 3b；稀疏 Q ⇒
                # 非 top-K 位置拿到完整的 P）。无额外传输。
                rejection_offset = n - (prefix_len - 1)
                t = sample_reject_token(
                    verification_inputs.target_probs_batch[
                        :, rejection_offset, : self.vocab_size
                    ],
                    verification_inputs.draft_probs_batch[
                        :, rejection_offset, : self.vocab_size
                    ],
                    output_device=prefix.device,
                )
            else:
                # 全接受路径：从 P_{γ+1} 采样 bonus token
                t = sample_accept_token(
                    target_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=prefix.device,
                )
            t = _ensure_token_shape(t, label="tk_slt.sampled_t")
            _validate_token_range(
                t, vocab_size=self.vocab_size, label="tk_slt.sampled_t"
            )

            apply_rollback(
                approx_model_cache,
                target_model_cache,
                rollback_plan,
            )

            if prefix.shape[1] < max_tokens:
                prefix = torch.cat((prefix, t), dim=1)
                prefix = _ensure_token_shape(prefix, label="tk_slt.prefix_after_concat")
                _validate_token_range(
                    prefix,
                    vocab_size=self.vocab_size,
                    label="tk_slt.prefix_after_concat",
                )

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

            # ---- 下行：结果 token + 位置 j（§II-B negligible ⇒ 0 字节报文）----
            _send_downlink_index_only(comm_simulator, "edge_cloud")

            # ---- 轮末结算：per_round 下本轮字节在此落账（1×NTT/轮）----
            # per_transfer 模式下 flush_round 是 no-op（直充已落账）。
            comm_simulator.flush_round()
            if comm_simulator.coalesce_rounds:
                _tx_seconds = _last_edge_cloud_tx_seconds(comm_simulator)
                if _tx_seconds is not None:
                    t_v_seconds += _tx_seconds
                    t_v_dists += window_rows

        # 中途 break（max_tokens/EOS/fallback）时最后一轮字节可能仍在合并桶里，
        # 结算必须在 wall_time/communication_time 读数之前。
        comm_simulator.flush_round()
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        # 遵守 max_tokens（整块追加可能越界）+ EOS 截断必须在 metrics 结算前
        # （B9：被截掉的 token 不计入吞吐；最后一轮越过 EOS 的情况由这里兜底）
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, :max_tokens]
        prefix, _ = self._stop_at_eos(prefix, _tri_prompt_len)

        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        throughput = (
            generated_tokens / (elapsed_time + comm_simulator.edge_cloud_comm_time)
            if (elapsed_time + comm_simulator.edge_cloud_comm_time) > 0
            else 0
        )

        metrics = get_empty_metrics()
        metrics["avg_top_k"] = (
            sum_top_k / total_draft_steps if total_draft_steps > 0 else 0
        )
        metrics["avg_draft_len"] = (
            sum_draft_len / total_draft_steps if total_draft_steps > 0 else 0
        )
        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["draft_computation_time"] = draft_comp_time
        metrics["target_computation_time"] = target_comp_time
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = total_drafted_tokens
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["wall_time"] = elapsed_time + comm_simulator.edge_cloud_comm_time
        metrics["throughput"] = throughput
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times

        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = (
            elapsed_time + queuing_time + comm_simulator.edge_cloud_comm_time
        )
        if metrics["wall_time"] > 0:
            metrics["throughput"] = metrics["generated_tokens"] / metrics["wall_time"]

        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time,
            queuing_time=queuing_time,
            draft_comp_time=draft_comp_time,
            target_comp_time=target_comp_time,
        )

        # ODLD 观测：落盘最终估计与 standalone 轮数（分析用）
        metrics["tk_slt_odld"] = odld_enabled
        metrics["tk_slt_standalone_rounds"] = standalone_rounds
        if odld_enabled:
            metrics["tk_slt_alpha_hat"] = (
                alpha_num / alpha_den if alpha_den > 0 else 0.0
            )
            llm_per_run = t_llm_seconds / t_llm_runs if t_llm_runs > 0 else 0.0
            metrics["tk_slt_b_hat"] = (
                (t_v_seconds / t_v_dists) / llm_per_run
                if t_v_dists > 0 and llm_per_run > 0
                else 0.0
            )
            metrics["tk_slt_c_hat"] = (
                (t_slm_seconds / t_slm_tokens) / llm_per_run
                if t_slm_tokens > 0 and llm_per_run > 0
                else 0.0
            )
            metrics["tk_slt_gamma_star_final"] = last_gamma_star

        # 复制 edge-cloud 的带宽、top-k 和起草长度历史数据
        metrics["edge_cloud_bandwidth_history"] = (
            comm_simulator.edge_cloud_bandwidth_history.copy()
        )
        # 逐消息记账记录（[字节, 轮号]）——离线重放（scripts/rebill.py）的
        # 输入：换带宽/NTT/地板/口径的敏感性分析不用重跑 GPU。
        # getattr 兜底：测试替身可能只实现聚合口径。
        metrics["comm_trace_edge_cloud"] = getattr(
            comm_simulator, "edge_cloud_trace", []
        )
        _add_comm_accounting_metrics(metrics, self.args, comm_simulator)
        metrics["edge_cloud_topk_history"] = (
            comm_simulator.edge_cloud_topk_history.copy()
        )
        metrics["edge_cloud_draft_len_history"] = (
            comm_simulator.edge_cloud_draft_len_history.copy()
        )

        return prefix, metrics

    @Register.register_decoding("tridecoding")
    @torch.no_grad()
    def tridecoding(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 10,
        ntt_ms_edge_end: float = 1,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ):
        max_tokens = prefix.shape[1] + self.args.max_tokens
        little_device = self.get_model_input_device(self.little_model)
        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        caches = self.build_adaptive_tridecoding_caches(transfer_top_k)
        little_model_cache = caches["little"]
        draft_model_cache = caches["draft"]
        target_model_cache = caches["target"]

        if use_precise_comm_sim:
            comm_simulator: CommunicationSimulator = PreciseCommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=self.args.edge_end_bandwidth,
                bandwidth_cloud_end=self.args.cloud_end_bandwidth,
                transfer_top_k=transfer_top_k,
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=bool(getattr(self.args, "stochastic_ntt", False)),
            )

        # 统一计费口径（docs/protocol.md §3）：与 adaptive_tridecoding 同一套
        # 开关。CUHLM 系（uncertainty_decoding / cee_cuhlm）按决策暂不接。
        # 开关名必须出现在本方法代码里（protocols.py 运行时内省的依据）。
        comm_simulator.coalesce_rounds = (
            str(getattr(self.args, "comm_round_trip_mode", "per_transfer"))
            == "per_round"
        )
        charge_residual = bool(getattr(self.args, "charge_residual_payload", False))
        _topk_cap = int(getattr(self.args, "transfer_top_k_cap", 0) or 0)
        transfer_top_k = apply_transfer_top_k_cap(transfer_top_k, _topk_cap)
        comm_simulator.transfer_top_k = transfer_top_k

        # Metrics tracking
        little_model_forward_times = 0
        draft_model_forward_times = 0
        target_model_forward_times = 0
        total_little_model_generated_tokens = 0
        total_draft_model_generated_tokens = 0
        total_little_model_accepted_tokens = 0
        total_draft_model_accepted_tokens = 0
        queuing_time = 0
        batch_delay = getattr(self.args, "batch_delay", 0)
        wall_time = 0
        total_draft_steps = 0
        sum_draft_len = 0
        sum_top_k = 0

        idx = 0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()  # 用于计算生成token数

        start_event.record(stream=torch.cuda.current_stream())

        comm_simulator.transfer(prefix, None, "edge_end")  # 将 prompt 传输到 edge

        _tri_prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点

        while prefix.shape[1] < max_tokens:

            # B10：循环内 EOS 早停（与 dssd/adaptive_* 已接线方法语义对齐——
            # 计算与通信在 EOS 后立即停，时延口径跨方法可比）
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            idx += 1
            comm_simulator.set_round(idx)

            prefix_len = prefix.shape[1]

            # 第一层 speculative

            current_proposal_top_k = proposal_top_k(transfer_top_k)
            little_rebuilt_probs = None
            little_rebuilt_meta = None
            if current_proposal_top_k is not None:
                x, little_rebuilt_probs, little_rebuilt_meta = (
                    little_model_cache.generate_with_rebuilt_topk_metadata(
                        prefix.to(little_device),
                        self.args.gamma2,
                        current_proposal_top_k,
                    )
                )
            else:
                x = little_model_cache.generate(
                    prefix.to(little_device), self.args.gamma2
                )
            _ = draft_model_cache.generate(x.to(draft_device), 1)

            little_model_forward_times += self.args.gamma2
            draft_model_forward_times += 1
            total_little_model_generated_tokens += self.args.gamma2

            # 累积追踪指标 - 记录第一层 draft
            total_draft_steps += 1
            sum_draft_len += self.args.gamma2
            sum_top_k += (
                current_proposal_top_k if current_proposal_top_k is not None else 0
            )

            n1: int = prefix_len + self.args.gamma2 - 1

            little_accepted_this_iter = 0
            # 批量传输 draft tokens 和对应的 probabilities 以节省 RTT
            if self.args.gamma2 > 0:
                little_stage_probs = build_draft_probs_override(
                    little_model_cache,
                    prefix_len,
                    little_rebuilt_probs,
                )
                if little_stage_probs is None:
                    little_stage_probs = little_model_cache.prob_history
                # B15：量化与计费原子绑定——第一级上行此前静默忽略位宽
                little_stage_probs, _little_bits = _uplink_prob_payload(
                    little_stage_probs, self.args
                )
                draft_tokens, draft_probs = collect_verification_payload(
                    little_stage_probs,
                    x,
                    prefix_len,
                    self.args.gamma2,
                )
                comm_simulator.transfer(
                    draft_tokens,
                    draft_probs,
                    "edge_end",
                    prob_bits=_little_bits,
                )

            first_stage_inputs, first_stage_acceptance = verify_draft_sequence_result(
                draft_model_cache=little_model_cache,
                target_model_cache=draft_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=self.args.gamma2,
                # B15：验证看到的 q̂ 与上行载荷同源（此前重建未量化副本，
                # 位宽生效时验证与计费看到的是两个分布）
                draft_probs_override=(
                    cast(torch.Tensor, little_stage_probs)
                    if self.args.gamma2 > 0
                    else build_draft_probs_override(
                        little_model_cache,
                        prefix_len,
                        little_rebuilt_probs,
                    )
                ),
                draft_topk_history=stage_topk_proposal_history(
                    little_rebuilt_meta,
                    self.args.gamma2,
                ),
            )
            # 注意：AcceptanceResult 只有 accepted_count（接受计数），绝对位置 n
            # 必须经 materialize_acceptance 求（n = prefix_len + accepted_count - 1，
            # 全部接受时取 prefix_len + actual_gamma - 1）。这里原先写的 `.n` 是
            # 重构前的旧字段，会让 tridecoding/ceesd_without_arp 直接崩溃。
            little_accepted_this_iter, n1, _ = materialize_acceptance(
                first_stage_inputs, first_stage_acceptance
            )

            total_little_model_accepted_tokens += little_accepted_this_iter

            assert n1 >= prefix_len - 1, f"n {n1}, prefix_len {prefix_len}"
            prefix = x[:, : n1 + 1]

            first_stage_rollback_plan = build_rollback_plan(
                prefix_len,
                first_stage_inputs.actual_gamma,
                n1,
            )
            little_model_cache.rollback(first_stage_rollback_plan.draft_end_pos)

            if not first_stage_rollback_plan.all_accepted:
                # reject someone, sample from the pos n1
                # rebuild_probs = comm_simulator.rebuild_full_probs(
                #     little_model_cache.prob_history[:, n1, : self.vocab_size]
                # )
                # little_model_cache.prob_history[:, n1, : self.vocab_size] = (
                #     rebuild_probs
                # )

                _rej_probs, _rej_bits = _uplink_prob_payload(
                    first_stage_inputs.draft_probs_batch[
                        :, n1 - (prefix_len - 1), : self.vocab_size
                    ],
                    self.args,
                )
                comm_simulator.transfer(
                    None,
                    _rej_probs,
                    "edge_end",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                    prob_bits=_rej_bits,
                )
                if charge_residual:
                    # 统一口径（§3.4）：压缩行已含 k×(4+元素)（索引在内），
                    # 与统一式 k*(4+元素)+元素 相比只差尾部标量。
                    comm_simulator.simulate_transfer(
                        reject_tail_scalar_bytes(_rej_probs, transfer_top_k),
                        "edge_end",
                    )

                rejection_offset = n1 - (prefix_len - 1)
                if little_rebuilt_meta is not None:
                    t = sample_reject_token_from_topk_proposal(
                        draft_model_cache.prob_history[:, n1, : self.vocab_size],
                        stage_topk_proposal_history(
                            little_rebuilt_meta,
                            self.args.gamma2,
                        ),
                        rejection_offset,
                        output_device=little_device,
                    )
                else:
                    t = sample_reject_token(
                        draft_model_cache.prob_history[:, n1, : self.vocab_size],
                        first_stage_inputs.draft_probs_batch[
                            :, rejection_offset, : self.vocab_size
                        ],
                        output_device=little_device,
                    )

                draft_model_cache.rollback(
                    first_stage_rollback_plan.target_end_pos_reject
                )

            else:
                t = sample_accept_token(
                    draft_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=little_device,
                )

                draft_model_cache.rollback(
                    first_stage_rollback_plan.target_end_pos_accept
                )

            # 传输索引
            _send_downlink_token(comm_simulator, t, "edge_end")

            prefix = torch.cat((prefix, t), dim=1)
            new_generated_token = prefix[:, prefix_len:]

            # 第二层 speculative

            if idx == 1:
                comm_simulator.transfer(prefix, None, "edge_cloud")
            else:
                comm_simulator.transfer(new_generated_token, None, "edge_cloud")

            x, draft_rebuilt_probs, draft_rebuilt_meta, _ = self._generate_with_optional_rebuilt_proposal(
                draft_model_cache,
                prefix.to(draft_device),
                self.args.gamma1,
                current_proposal_top_k,
            )

            queuing_time += batch_delay
            _ = target_model_cache.generate(x.to(target_device), 1)

            draft_model_forward_times += self.args.gamma1
            target_model_forward_times += 1
            total_draft_model_generated_tokens += (
                new_generated_token.shape[1] + self.args.gamma1
            )

            total_gamma = new_generated_token.shape[1] + self.args.gamma1
            n2: int = prefix_len + total_gamma - 1

            # 批量传输 draft tokens 和对应的 probabilities 以节省 RTT
            if total_gamma > 0:
                draft_stage_probs = build_draft_probs_override(
                    draft_model_cache,
                    prefix_len,
                    draft_rebuilt_probs,
                )
                if draft_stage_probs is None:
                    draft_stage_probs = draft_model_cache.prob_history
                # B15：原子化——原实现量化只作用于计费载荷，verify 的
                # override 却重建未量化副本（注释承诺的"验证看到同一个
                # q̂"从未兑现）。现在两者同源；默认 16 时不量化、数字不变
                draft_stage_probs, _draft_bits = _uplink_prob_payload(
                    draft_stage_probs, self.args
                )
                draft_tokens_second, draft_probs_second = collect_verification_payload(
                    draft_stage_probs,
                    x,
                    prefix_len,
                    total_gamma,
                )
                comm_simulator.transfer(
                    draft_tokens_second,
                    draft_probs_second,
                    "edge_cloud",
                    prob_bits=_draft_bits,
                )

            second_stage_inputs, second_stage_acceptance = verify_draft_sequence_result(
                draft_model_cache=draft_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=total_gamma,
                draft_probs_override=(
                    cast(torch.Tensor, draft_stage_probs)
                    if total_gamma > 0
                    else build_draft_probs_override(
                        draft_model_cache,
                        prefix_len,
                        draft_rebuilt_probs,
                    )
                ),
                draft_topk_history=stage_topk_proposal_history(
                    draft_rebuilt_meta,
                    total_gamma,
                ),
            )
            # 同上：用 materialize_acceptance 求绝对位置，而不是旧字段 `.n`
            draft_accepted_this_iter, n2, _ = materialize_acceptance(
                second_stage_inputs, second_stage_acceptance
            )
            total_draft_model_accepted_tokens += draft_accepted_this_iter

            assert n2 >= prefix_len - 1, (
                f"n {n2} should be greater or equal than prefix_len {prefix_len}"
            )
            prefix = x[:, : n2 + 1]
            second_stage_rollback_plan = build_rollback_plan(
                prefix_len,
                second_stage_inputs.actual_gamma,
                n2,
            )
            draft_model_cache.rollback(second_stage_rollback_plan.draft_end_pos)
            if n2 <= little_model_cache.current_length:
                little_model_cache.rollback(second_stage_rollback_plan.draft_end_pos)
            if not second_stage_rollback_plan.all_accepted:
                # rebuild_probs = comm_simulator.rebuild_full_probs(
                #     draft_model_cache.prob_history[:, n2, : self.vocab_size]
                # )
                # draft_model_cache.prob_history[:, n2, : self.vocab_size] = (
                #     rebuild_probs
                # )

                _rej_probs2, _rej_bits2 = _uplink_prob_payload(
                    second_stage_inputs.draft_probs_batch[
                        :, n2 - (prefix_len - 1), : self.vocab_size
                    ],
                    self.args,
                )
                comm_simulator.transfer(
                    None,
                    _rej_probs2,
                    "edge_cloud",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                    prob_bits=_rej_bits2,
                )
                if charge_residual:
                    # 统一口径（§3.4）：同阶段一，补尾部标量。
                    comm_simulator.simulate_transfer(
                        reject_tail_scalar_bytes(_rej_probs2, transfer_top_k),
                        "edge_cloud",
                    )
                rejection_offset = n2 - (prefix_len - 1)
                if draft_rebuilt_meta is not None:
                    t = sample_reject_token_from_topk_proposal(
                        target_model_cache.prob_history[:, n2, : self.vocab_size],
                        stage_topk_proposal_history(
                            draft_rebuilt_meta,
                            total_gamma,
                        ),
                        rejection_offset,
                        output_device=draft_device,
                    )
                else:
                    t = sample_reject_token(
                        target_model_cache.prob_history[:, n2, : self.vocab_size],
                        second_stage_inputs.draft_probs_batch[
                            :, rejection_offset, : self.vocab_size
                        ],
                        output_device=draft_device,
                    )
                new_generated_token = prefix[:, prefix_len:]

                target_model_cache.rollback(
                    second_stage_rollback_plan.target_end_pos_reject
                )

            else:
                t = sample_accept_token(
                    target_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=draft_device,
                )
                new_generated_token = prefix[:, prefix_len:]

                target_model_cache.rollback(
                    second_stage_rollback_plan.target_end_pos_accept
                )

            prefix = torch.cat((prefix, t), dim=1)
            # 传输索引
            _send_downlink_token(comm_simulator, t, "edge_cloud")
            _send_downlink_token(comm_simulator, t, "edge_end")
            # 同步

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        comm_simulator.flush_round()  # 结算最后一轮（按轮合并模式下必须）
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        wall_time += elapsed_time
        # 遵守 max_tokens：投机按整块追加，最后一轮会多出若干 token。对照基线
        # dist_spec 严格 128，而本方法此前 14/20 样本超预算（最多 +9），会让配对
        # 质量比较与时延/吞吐统计都不公平（F33）。
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, :max_tokens]
        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        wall_time += (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )

        metrics = get_empty_metrics()
        metrics["avg_top_k"] = (
            sum_top_k / total_draft_steps if total_draft_steps > 0 else 0
        )
        metrics["avg_draft_len"] = (
            sum_draft_len / total_draft_steps if total_draft_steps > 0 else 0
        )
        metrics["little_forward_times"] = little_model_forward_times
        metrics["draft_forward_times"] = draft_model_forward_times
        metrics["target_forward_times"] = target_model_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["little_generated_tokens"] = total_little_model_generated_tokens
        metrics["draft_generated_tokens"] = total_draft_model_generated_tokens
        metrics["little_accepted_tokens"] = total_little_model_accepted_tokens
        metrics["draft_accepted_tokens"] = total_draft_model_accepted_tokens
        metrics["wall_time"] = wall_time
        metrics["throughput"] = (
            metrics["generated_tokens"] / wall_time if wall_time > 0 else 0
        )
        metrics["communication_time"] = (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times

        batch_delay = getattr(self.args, "batch_delay", 0)
        queuing_time = target_model_forward_times * batch_delay
        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] += queuing_time
        if metrics["wall_time"] > 0:
            metrics["throughput"] = metrics["generated_tokens"] / metrics["wall_time"]

        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time,
            queuing_time=queuing_time,
        )

        # 复制 edge-cloud 的带宽、top-k 和起草长度历史数据
        metrics["edge_cloud_bandwidth_history"] = (
            comm_simulator.edge_cloud_bandwidth_history.copy()
        )
        # 逐消息记账记录（[字节, 轮号]）——离线重放（scripts/rebill.py）的
        # 输入：换带宽/NTT/地板/口径的敏感性分析不用重跑 GPU。
        # getattr 兜底：测试替身可能只实现聚合口径。
        metrics["comm_trace_edge_cloud"] = getattr(
            comm_simulator, "edge_cloud_trace", []
        )
        _add_comm_accounting_metrics(metrics, self.args, comm_simulator)
        metrics["edge_cloud_topk_history"] = (
            comm_simulator.edge_cloud_topk_history.copy()
        )
        metrics["edge_cloud_draft_len_history"] = (
            comm_simulator.edge_cloud_draft_len_history.copy()
        )

        return prefix, metrics

    @Register.register_decoding("ceesd_w/o_arp")
    @Register.register_decoding("ceesd_without_arp")
    def ceesd_without_arp(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim: bool = False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 0,
        ntt_ms_edge_end: float = 0,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        # 不启用 adaptive head，但是要启用 rl agent（只修改 top-k），需要修改 rl agent 和 adapter 的互动行为，使之不报错
        batch_delay = self.args.batch_delay
        queuing_time = 0.0
        max_tokens = prefix.shape[1] + self.args.max_tokens
        little_device = self.get_model_input_device(self.little_model)
        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        # 使用 transfer_top_k 作为草稿模型的 top-k 压缩参数
        draft_top_k = (
            transfer_top_k
            if (transfer_top_k is not None and transfer_top_k > 0)
            else self.args.top_k
        )

        # B19：little/draft/target 都做多 token 验证前向（调度开销主体）⇒ 三个
        # 都开图并跨样本复用；此前只给 little/draft 单步图、无验证档位、且每
        # 样本重捕获（~534ms/样本把收益吃光）。target 此前完全没接线。
        graph_kw = _graph_mode_cache_kwargs(
            self.args,
            cap=int(getattr(self.args, "gamma1", 1))
            + int(getattr(self.args, "gamma2", 1))
            + 4,
        )
        caches = self._acquire_three_layer_caches(
            "_ceesd_without_arp_caches", graph_kw, draft_top_k
        )
        little_model_cache = caches["little"]
        draft_model_cache = caches["draft"]
        target_model_cache = caches["target"]

        if use_precise_comm_sim:
            comm_simulator: CommunicationSimulator = PreciseCommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=self.args.edge_end_bandwidth,
                bandwidth_cloud_end=self.args.cloud_end_bandwidth,
                transfer_top_k=transfer_top_k,
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=bool(getattr(self.args, "stochastic_ntt", False)),
            )

        # 统一计费口径（docs/protocol.md §3）：与 adaptive_tridecoding 同一套
        # 开关。CUHLM 系（uncertainty_decoding / cee_cuhlm）按决策暂不接。
        # 开关名必须出现在本方法代码里（protocols.py 运行时内省的依据）。
        comm_simulator.coalesce_rounds = (
            str(getattr(self.args, "comm_round_trip_mode", "per_transfer"))
            == "per_round"
        )
        charge_residual = bool(getattr(self.args, "charge_residual_payload", False))
        _topk_cap = int(getattr(self.args, "transfer_top_k_cap", 0) or 0)
        transfer_top_k = apply_transfer_top_k_cap(transfer_top_k, _topk_cap)
        comm_simulator.transfer_top_k = transfer_top_k

        # Metrics tracking
        little_model_forward_times = 0
        draft_model_forward_times = 0
        target_model_forward_times = 0
        total_little_model_generated_tokens = 0
        total_draft_model_generated_tokens = 0
        total_little_model_accepted_tokens = 0
        total_draft_model_accepted_tokens = 0
        wall_time = 0
        little_comp_time = 0.0
        draft_comp_time = 0.0
        target_comp_time = 0.0
        arp_overhead_time = 0.0
        dra_overhead_time = 0.0

        idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()  # 用于计算生成token数

        start_event.record(stream=torch.cuda.current_stream())

        comm_simulator.transfer(prefix, None, "edge_end")  # 将 prompt 传输到 edge

        _tri_prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点
        # D4/B20：RL 选出的草稿长度走方法局部变量——此前直接改写
        # self.args.gamma1/gamma2，动作残留在全局 Namespace 上污染后续方法
        gamma2 = int(self.args.gamma2)
        gamma1 = int(self.args.gamma1)
        while prefix.shape[1] < max_tokens:

            # B10：循环内 EOS 早停（与 dssd/adaptive_* 已接线方法语义对齐——
            # 计算与通信在 EOS 后立即停，时延口径跨方法可比）
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            idx += 1
            comm_simulator.set_round(idx)
            prefix_len = prefix.shape[1]
            current_proposal_top_k = proposal_top_k(transfer_top_k)
            little_stage_probs: Optional[torch.Tensor] = None
            draft_stage_probs: Optional[torch.Tensor] = None

            # 第一层 speculative
            edge_end_comm_start = comm_simulator.edge_end_comm_time
            step_start_time = time.time()

            x, little_rebuilt_probs, little_rebuilt_meta, q = self._generate_with_optional_rebuilt_proposal(
                little_model_cache,
                prefix.to(little_device),
                gamma2,
                current_proposal_top_k,
                need_topk_metadata=True,
            )

            if self.little_rl_adapter is not None:
                bandwidth = comm_simulator.bandwidth_edge_end_mbps
                latency = comm_simulator.ntt_edge_end_ms
                acc_probs = []  # No ARP head
                # 注意：ceesd_without_arp 不带 ARP adapter 且可能没开 top-k 压缩，
                # 此时 `_generate_with_optional_rebuilt_proposal` 会合法地返回 q=None
                # （它只在需要重建 top-k 概率时才返回 q）。熵因此必须走缓存里按原始
                # logits 算好的 last_entropy——原先的 assert 会让这个消融直接崩溃。
                entropy = state_entropy(q, little_model_cache)
                task_name = getattr(self, "task", "unknown")
                next_k, _ = self.little_rl_adapter.select_config(
                    bandwidth,
                    latency,
                    acc_probs,
                    entropy,
                    task_name,
                    training=not getattr(self.args, "disable_rl_update", False),
                )
                gamma2 = int(next_k)

            actual_gamma2 = x.shape[1] - prefix_len

            _ = draft_model_cache.generate(x.to(draft_device), 1)

            little_model_forward_times += actual_gamma2
            draft_model_forward_times += 1
            total_little_model_generated_tokens += actual_gamma2

            n1: int = prefix_len + actual_gamma2 - 1

            little_accepted_this_iter = 0
            if actual_gamma2 > 0:
                little_stage_probs = stage_prob_history(
                    little_model_cache,
                    prefix_len,
                    little_rebuilt_probs,
                )
                draft_tokens, draft_probs = collect_verification_payload(
                    little_stage_probs,
                    x,
                    prefix_len,
                    actual_gamma2,
                )
                comm_simulator.transfer(draft_tokens, draft_probs, "edge_end")
                little_stage_kwargs = {
                    "proposer_cache": little_model_cache,
                    "verifier_cache": draft_model_cache,
                    "x": x,
                    "prefix_len": prefix_len,
                    "gamma": actual_gamma2,
                    "output_device": little_device,
                    "draft_probs_override": cast(torch.Tensor, little_stage_probs),
                }
                little_topk_history = stage_topk_proposal_history(
                    little_rebuilt_meta,
                    actual_gamma2,
                )
                if little_topk_history is not None:
                    little_stage_kwargs["draft_topk_history"] = little_topk_history
                (
                    little_accepted_this_iter,
                    n1,
                    t,
                    little_all_accepted,
                ) = resolve_stage_verification(**little_stage_kwargs)
                if not little_all_accepted:
                    comm_simulator.send_reject_message("edge_end")
            else:
                t = sample_accept_token(
                    draft_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=little_device,
                )
                little_all_accepted = True

            total_little_model_accepted_tokens += little_accepted_this_iter

            if self.little_rl_adapter is not None:
                step_end_time = time.time()
                step_time = step_end_time - step_start_time
                step_comm_time = comm_simulator.edge_end_comm_time - edge_end_comm_start

                # 去掉分子 +1
                tps_part = little_accepted_this_iter / (
                    step_time + step_comm_time + 1e-9
                )
                reward = math.exp(min(tps_part, 100) / 20.0)

                # 平滑的幂次惩罚
                if actual_gamma2 > 1:
                    acc_rate = little_accepted_this_iter / actual_gamma2
                    reward *= acc_rate**2

                if not getattr(self.args, "disable_rl_update", False):
                    self.little_rl_adapter.step(reward)

            assert n1 >= prefix_len - 1, f"n1 {n1}, prefix_len {prefix_len}"
            prefix = x[:, : n1 + 1]

            if not little_all_accepted:
                # reject someone, sample from the pos n1
                little_stage_probs = stage_prob_history(
                    little_model_cache,
                    prefix_len,
                    little_rebuilt_probs,
                )
                comm_simulator.transfer(
                    None,
                    little_stage_probs[:, n1, : self.vocab_size],
                    "edge_end",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )
                if charge_residual:
                    # 统一口径（§3.4）：压缩行已含 k×(4+元素)，补尾部标量。
                    comm_simulator.simulate_transfer(
                        reject_tail_scalar_bytes(
                            little_stage_probs[:, n1, : self.vocab_size],
                            transfer_top_k,
                        ),
                        "edge_end",
                    )

            # 传输索引
            _send_downlink_token(comm_simulator, t, "edge_end")

            prefix = torch.cat((prefix, t), dim=1)
            new_generated_token = prefix[:, prefix_len:]

            # 第二层 speculative
            edge_cloud_comm_start = comm_simulator.edge_cloud_comm_time
            step_start_time = time.time()

            if idx == 1:
                comm_simulator.transfer(prefix, None, "edge_cloud")
            else:
                comm_simulator.transfer(new_generated_token, None, "edge_cloud")

            x, draft_rebuilt_probs, _, q = self._generate_with_optional_rebuilt_proposal(
                draft_model_cache,
                prefix.to(draft_device),
                gamma1,
                current_proposal_top_k,
            )

            if self.rl_adapter is not None:
                bandwidth = comm_simulator.bandwidth_edge_cloud_mbps
                latency = comm_simulator.ntt_edge_cloud_ms
                acc_probs = []
                # 同上：q 可能为 None（无 ARP adapter / 未开 top-k），熵来自缓存
                entropy = state_entropy(q, draft_model_cache)
                task_name = getattr(self, "task", "unknown")
                next_k, _ = self.rl_adapter.select_config(
                    bandwidth,
                    latency,
                    acc_probs,
                    entropy,
                    task_name,
                    training=not getattr(self.args, "disable_rl_update", False),
                )
                gamma1 = int(next_k)

            actual_gamma1 = x.shape[1] - prefix.shape[1]

            queuing_time += batch_delay
            _ = target_model_cache.generate(x.to(target_device), 1)

            draft_model_forward_times += actual_gamma1
            target_model_forward_times += 1
            total_draft_model_generated_tokens += (
                new_generated_token.shape[1] + actual_gamma1
            )

            n2: int = prefix_len + new_generated_token.shape[1] + actual_gamma1 - 1
            draft_accepted_this_iter = 0
            total_gamma_layer2 = new_generated_token.shape[1] + actual_gamma1
            if total_gamma_layer2 > 0:
                draft_stage_probs = stage_prob_history(
                    draft_model_cache,
                    prefix_len + new_generated_token.shape[1],
                    draft_rebuilt_probs,
                )
                draft_tokens_second, draft_probs_second = collect_verification_payload(
                    draft_stage_probs,
                    x,
                    prefix_len,
                    total_gamma_layer2,
                )
                comm_simulator.transfer(
                    draft_tokens_second,
                    draft_probs_second,
                    "edge_cloud",
                )
                (
                    draft_accepted_this_iter,
                    n2,
                    t,
                    draft_all_accepted,
                ) = resolve_stage_verification(
                    proposer_cache=draft_model_cache,
                    verifier_cache=target_model_cache,
                    x=x,
                    prefix_len=prefix_len,
                    gamma=total_gamma_layer2,
                    output_device=draft_device,
                    draft_probs_override=draft_stage_probs,
                )
                if not draft_all_accepted:
                    comm_simulator.send_reject_message("edge_cloud")
            else:
                t = sample_accept_token(
                    target_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=draft_device,
                )
                draft_all_accepted = True
            total_draft_model_accepted_tokens += draft_accepted_this_iter

            if self.rl_adapter is not None:
                step_end_time = time.time()
                step_time = step_end_time - step_start_time
                step_comm_time = (
                    comm_simulator.edge_cloud_comm_time - edge_cloud_comm_start
                )

                # 去掉分子 +1
                tps_part = draft_accepted_this_iter / (
                    step_time + step_comm_time + 1e-9
                )
                reward = math.exp(min(tps_part, 100) / 20.0)

                # 平滑的幂次惩罚
                if actual_gamma1 > 1:
                    acc_rate = draft_accepted_this_iter / actual_gamma1
                    reward *= acc_rate**2

                if not getattr(self.args, "disable_rl_update", False):
                    self.rl_adapter.step(reward)

            assert n2 >= prefix_len - 1, (
                f"n {n2} should be greater or equal than prefix_len {prefix_len}"
            )
            prefix = x[:, : n2 + 1]
            if n2 <= little_model_cache.current_length:
                little_model_cache.rollback(n2 + 1)
            if not draft_all_accepted:
                draft_stage_probs = stage_prob_history(
                    draft_model_cache,
                    prefix_len + new_generated_token.shape[1],
                    draft_rebuilt_probs,
                )
                comm_simulator.transfer(
                    None,
                    draft_stage_probs[:, n2, : self.vocab_size],
                    "edge_cloud",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )
                if charge_residual:
                    # 统一口径（§3.4）：同阶段一，补尾部标量。
                    comm_simulator.simulate_transfer(
                        reject_tail_scalar_bytes(
                            draft_stage_probs[:, n2, : self.vocab_size],
                            transfer_top_k,
                        ),
                        "edge_cloud",
                    )
                new_generated_token = prefix[:, prefix_len:]
            else:
                new_generated_token = prefix[:, prefix_len:]

            prefix = torch.cat((prefix, t), dim=1)
            # 传输索引
            _send_downlink_token(comm_simulator, t, "edge_cloud")
            _send_downlink_token(comm_simulator, t, "edge_end")

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        comm_simulator.flush_round()  # 结算最后一轮（按轮合并模式下必须）
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        wall_time += elapsed_time
        # 遵守 max_tokens：投机解码按整块追加，最后一轮可能多出若干 token
        # （F33 契约）。参照 dist_spec 的 remaining-1 截断，这里显式截断——
        # 否则会多拿 token，使配对质量比较与时延/吞吐统计都不公平。
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, :max_tokens]
        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        wall_time += (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )

        metrics = get_empty_metrics()
        metrics["little_forward_times"] = little_model_forward_times
        metrics["draft_forward_times"] = draft_model_forward_times
        metrics["target_forward_times"] = target_model_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["little_generated_tokens"] = total_little_model_generated_tokens
        metrics["draft_generated_tokens"] = total_draft_model_generated_tokens
        metrics["little_accepted_tokens"] = total_little_model_accepted_tokens
        metrics["draft_accepted_tokens"] = total_draft_model_accepted_tokens
        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = wall_time + queuing_time
        metrics["throughput"] = (
            metrics["generated_tokens"] / metrics["wall_time"]
            if metrics["wall_time"] > 0
            else 0
        )
        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time,
            queuing_time=queuing_time,
            little_comp_time=little_comp_time,
            draft_comp_time=draft_comp_time,
            target_comp_time=target_comp_time,
        )
        metrics["communication_time"] = (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times
        metrics["edge_cloud_bandwidth_history"] = (
            comm_simulator.edge_cloud_bandwidth_history.copy()
        )
        # 逐消息记账记录（[字节, 轮号]）——离线重放（scripts/rebill.py）的
        # 输入：换带宽/NTT/地板/口径的敏感性分析不用重跑 GPU。
        # getattr 兜底：测试替身可能只实现聚合口径。
        metrics["comm_trace_edge_cloud"] = getattr(
            comm_simulator, "edge_cloud_trace", []
        )
        _add_comm_accounting_metrics(metrics, self.args, comm_simulator)
        metrics["edge_cloud_topk_history"] = (
            comm_simulator.edge_cloud_topk_history.copy()
        )
        metrics["edge_cloud_draft_len_history"] = (
            comm_simulator.edge_cloud_draft_len_history.copy()
        )

        metrics["arp_overhead_time"] = arp_overhead_time
        metrics["dra_overhead_time"] = dra_overhead_time

        return prefix, metrics

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

    @Register.register_decoding("target_only")
    @torch.no_grad()
    def target_only(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim: bool = False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 0,
        ntt_ms_edge_end: float = 0,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """只用目标模型解码（无草稿、无投机、无 ARP、无通信）。

        这个模式承担两个不可替代的作用：
        1. **论文缺失的对照行**：加速比的分母只能是"不投机"的目标模型本身，
           而不是另一个投机配置（审稿人 R5 质疑的正是后者）。
        2. **无损性检验的基准**：在 temp=0（贪心）下，任何正确的投机解码
           都必须逐 token 复现目标模型的输出。把它与三级流水线的输出逐 token
           比对，就能把"加速是否以输出分布为代价"变成零噪声的二值判定
           （见 docs/rl_controller_diagnosis.md 的 E-A 实验）。
        """
        if prefix.dtype != torch.long:
            prefix = prefix.long()

        target_device = self.get_model_input_device(self.target_model)
        if use_precise_comm_sim or use_stochastic_comm:
            raise ValueError(
                "target_only has no communication stage; "
                "use_precise_comm_sim/use_stochastic_comm are not applicable."
            )

        # CUDA Graph **故意不接**（有据的口径决定，不是漏接）：
        # target_only 是两个不可替代作用的载体（见上面 docstring）——尤其是
        # "temp=0 下逐 token 复现目标模型"的无损性基准。图回放与 eager 有约 1%
        # 的 bf16 logits 漂移，实测会变成 40 样本里 1 个不同（docs/
        # graph_integration_status.md 旧 A/B）；且 13B 的图回放实测只快 1.03×
        # （13B 是 kernel/带宽瓶颈，不是 launch 瓶颈——同文件"对原调查结论的
        # 重要修正"）。用 1.03× 换掉基准的逐位可复现性不划算。
        # 因此 `--use_cuda_graph` 对本模式是**无操作**，这是设计而非遗漏。
        target_model_cache = KVCacheModel(
            self.target_model,
            self.args.temp,
            self.args.top_k,
            self.args.top_p,
        )
        target_model_cache.vocab_size = self.vocab_size

        max_new_tokens = int(self.args.max_tokens)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream=torch.cuda.current_stream())
        prompt_len = prefix.shape[1]
        generated = target_model_cache.generate(
            prefix.to(target_device), max_new_tokens
        )
        generated, _ = self._stop_at_eos(generated, prompt_len)
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        generated_tokens = generated.shape[1] - prefix.shape[1]
        metrics = get_empty_metrics()
        metrics["generated_tokens"] = generated_tokens
        metrics["target_forward_times"] = max(generated_tokens, 0)
        metrics["wall_time"] = elapsed_time
        metrics["throughput"] = (
            generated_tokens / elapsed_time if elapsed_time > 0 else 0.0
        )
        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=0.0,
            queuing_time=0.0,
        )
        metrics["communication_time"] = 0.0
        metrics["edge_cloud_data_bytes"] = 0
        metrics["comm_energy"] = 0.0

        return generated, metrics

    @Register.register_decoding("adaptive_decoding")
    @torch.no_grad()
    def adaptive_decoding(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim: bool = False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 0,
        ntt_ms_edge_end: float = 0,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        if use_precise_comm_sim:
            comm_simulator: CommunicationSimulator = PreciseCommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=float("inf"),
                bandwidth_cloud_end=float("inf"),
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=bool(getattr(self.args, "stochastic_ntt", False)),
            )
        self.color_print(f"Using transfer_top_k: {transfer_top_k}", 2)

        # 统一计费口径（docs/protocol.md §3）：与 adaptive_tridecoding 同一套
        # 开关。CUHLM 系（uncertainty_decoding / cee_cuhlm）按决策暂不接。
        # 开关名必须出现在本方法代码里（protocols.py 运行时内省的依据）。
        comm_simulator.coalesce_rounds = (
            str(getattr(self.args, "comm_round_trip_mode", "per_transfer"))
            == "per_round"
        )
        charge_residual = bool(getattr(self.args, "charge_residual_payload", False))
        _topk_cap = int(getattr(self.args, "transfer_top_k_cap", 0) or 0)
        transfer_top_k = apply_transfer_top_k_cap(transfer_top_k, _topk_cap)
        comm_simulator.transfer_top_k = transfer_top_k

        batch_delay = self.args.batch_delay
        queuing_time = 0.0

        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        # 使用 transfer_top_k 作为草稿模型的 top-k 压缩参数
        draft_top_k = (
            transfer_top_k
            if (transfer_top_k is not None and transfer_top_k > 0)
            else self.args.top_k
        )

        # CUDA Graph：草稿缓存跑 γ 步多 token 前向是图收益点；target 每步只做
        # 单 token 前向，开图无收益反而多占显存（与 dsd/dssd 的取舍一致）。
        # B19：此前 adaptive_decoding 完全没接线 —— 命令行开了 --use_cuda_graph
        # 对它毫无效果，而 exp.py 的扫描恰恰硬编码了 use_cuda_graph=True。
        _graph_kw = _graph_mode_cache_kwargs(
            self.args, cap=int(getattr(self.args, "gamma", 5)) + 4
        )
        approx_model_cache, target_model_cache = self._acquire_draft_target_caches(
            "_adaptive_decoding_caches",
            _graph_kw,
            draft_top_k,
            0,
            0.0,  # 目标模型不压缩
        )

        draft_forward_times = 0
        target_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        start_event.record(stream=torch.cuda.current_stream())

        idx: int = 0

        total_draft_steps = 0
        sum_draft_len = 0.0
        sum_top_k = 0.0

        prompt_len = prefix.shape[1]
        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]

            idx += 1
            comm_simulator.set_round(idx)

            step_start_time = time.time()
            step_comm_time_start = comm_simulator.edge_cloud_comm_time
            current_proposal_top_k = proposal_top_k(transfer_top_k)

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
                queuing_time += batch_delay
                _ = target_model_cache.generate(prefix.to(target_device), 1)
                target_forward_times += 1
                if self.accelerator.is_main_process:
                    self.target_forward_times += 1

                t = sample(
                    target_model_cache.prob_history[:, -1, : self.vocab_size]
                ).to(prefix.device)
                prefix = torch.cat((prefix, t), dim=1)
                self.num_acc_tokens.append(1)
                break

            step_start_time = time.time()
            step_comm_time_start = comm_simulator.edge_cloud_comm_time

            self.adapter.reset_step()
            x, rebuilt_draft_probs, _, q = self._generate_with_optional_rebuilt_proposal(
                approx_model_cache,
                prefix.to(draft_device),
                current_gamma,
                current_proposal_top_k,
                adapter=self.adapter,
            )

            if self.rl_adapter is not None:
                bandwidth = comm_simulator.bandwidth_edge_cloud_mbps
                latency = comm_simulator.ntt_edge_cloud_ms
                acc_probs = getattr(self.adapter, "step_acc_probs", [])

                # q 可能为 None（未开 top-k/无 adapter）；熵来自缓存的 last_entropy
                entropy = state_entropy(q, approx_model_cache)
                task_name = getattr(self, "task", "unknown")
                next_topk, next_threshold = self.rl_adapter.select_config(
                    bandwidth,
                    latency,
                    acc_probs,
                    entropy,
                    task_name,
                    training=not getattr(self.args, "disable_rl_update", False),
                )

                # 更新 top-k 压缩参数和 ARP 阈值
                transfer_top_k = next_topk
                # 统一口径：RL 选出的 top-k 同样受 cap 约束
                #（与 adaptive_tridecoding 阶段二的写法一致）
                if _topk_cap > 0 and transfer_top_k is not None:
                    transfer_top_k = min(int(transfer_top_k), _topk_cap)
                self.adapter.threshold = next_threshold

            actual_gamma = x.shape[1] - prefix_len  # 实际生成的token
            current_gamma = actual_gamma  # 更新current_gamma为实际生成的数量
            stage_probs = stage_prob_history(
                approx_model_cache,
                prefix_len,
                rebuilt_draft_probs,
            )

            total_draft_steps += 1
            sum_draft_len += current_gamma
            sum_top_k += (
                current_proposal_top_k if current_proposal_top_k is not None else 0
            )

            draft_forward_times += current_gamma
            total_drafted_tokens += current_gamma

            queuing_time += batch_delay
            _ = target_model_cache.generate(x.to(target_device), 1)

            target_forward_times += 1

            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            n = prefix_len + current_gamma - 1
            for i in range(current_gamma):
                # 检查索引是否合法
                draft_idx = prefix_len + i - 1
                target_idx = prefix_len + i - 1

                if draft_idx >= approx_model_cache.prob_history.shape[1]:
                    comm_simulator.send_reject_message("edge_cloud")
                    break
                if target_idx >= target_model_cache.prob_history.shape[1]:
                    comm_simulator.send_reject_message("edge_cloud")
                    break

                r = torch.rand(1, device=draft_device)
                j = x[:, prefix_len + i]

                # 传输 token id 和 prob 用于 rejection sampling
                comm_simulator.transfer(
                    j,
                    stage_probs[:, draft_idx, j],
                    "edge_cloud",
                )

                if (
                    r
                    > (
                        target_model_cache.prob_history.to(draft_device)[
                            :, target_idx, j
                        ]
                    )
                    / (stage_probs[:, draft_idx, j])
                ):
                    n = prefix_len + i - 1
                    comm_simulator.send_reject_message("edge_cloud")
                    break

            this_step_accepted_tokens = n - prefix_len + 1
            total_accepted_tokens += this_step_accepted_tokens

            self.num_acc_tokens.append(this_step_accepted_tokens)

            step_end_time = time.time()
            if self.rl_adapter is not None:
                step_time = step_end_time - step_start_time
                step_comm_time = (
                    comm_simulator.edge_cloud_comm_time - step_comm_time_start
                )

                # 去掉分子 +1：只有真正产生 accepted tokens 才有 TPS 基础奖励
                tps_part = this_step_accepted_tokens / (
                    step_time + step_comm_time + 1e-9
                )
                # 使用带上限的指数激励
                reward = math.exp(min(tps_part, 100) / 20.0)

                # 平滑的幂次惩罚：取代硬截断，让模型感知准确率的连续变化
                # 只有在预测长度 > 1 时才惩罚低命中率，鼓励模型在不确定时收缩长度
                if current_gamma > 1:
                    acc_rate = this_step_accepted_tokens / current_gamma
                    reward *= acc_rate**2

                if not getattr(self.args, "disable_rl_update", False):
                    self.rl_adapter.step(reward)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, : n + 1]

            approx_model_cache.rollback(n + 1)

            # 检查是否还有空间添加一个token
            if prefix.shape[1] >= max_tokens:
                break

            if n < prefix_len + current_gamma - 1:
                # reject someone, sample from the pos n

                # 发生拒绝，传输被拒绝的 token 的 full prob 用于采样
                comm_simulator.transfer(
                    None,
                    stage_probs[:, n, : self.vocab_size],
                    "edge_cloud",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )
                if charge_residual:
                    # 统一口径（§3.4）：压缩行已含 k×(4+元素)，补尾部标量。
                    comm_simulator.simulate_transfer(
                        reject_tail_scalar_bytes(
                            stage_probs[:, n, : self.vocab_size], transfer_top_k
                        ),
                        "edge_cloud",
                    )

                t = sample(
                    max_fn(
                        target_model_cache.prob_history[:, n, : self.vocab_size].to(
                            draft_device
                        )
                        - stage_probs[:, n, : self.vocab_size]
                    )
                )

                new_generated_tokens = prefix.shape[1] - current_tokens.shape[1] + 1

                target_model_cache.rollback(n + 1)
            else:
                # all approx model decoding accepted
                t = sample(
                    target_model_cache.prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                target_model_cache.rollback(n + 2)

                new_generated_tokens = prefix.shape[1] - current_tokens.shape[1] + 1

            # 最后检查添加token后是否会超出限制
            if prefix.shape[1] < max_tokens:
                t = t.to(prefix.device)
                prefix = torch.cat((prefix, t), dim=1)
                prefix, _eos_hit = self._stop_at_eos(prefix, prompt_len)
                if _eos_hit:
                    break
                # 逐迭代轨迹（APPEND_TRACE）：记录每一步确认下来的 token 与
                # 验证者的接受情况，用于把"输出与 target_only 分歧"定位到具体
                # 某一步的接受/重采样/上下文记账上。
                _append_trace = os.environ.get("APPEND_TRACE")
                if _append_trace:
                    try:
                        with open(_append_trace, "a") as _fh:
                            _fh.write(
                                json.dumps(
                                    {
                                        "iter": int(idx),
                                        "prefix_len_before": int(prefix_len),
                                        "gamma": int(current_gamma),
                                        "n": int(n),
                                        "accepted": int(this_step_accepted_tokens),
                                        "appended_token": int(t.item()),
                                        "prefix_len_after": int(prefix.shape[1]),
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

            # 传输新生成的 token id 与其位置索引（B17：合并为一次往返；
            # 此前只付了 INT_SIZE，token 本体没计）
            _send_downlink_token(comm_simulator, t, "edge_cloud")

        comm_simulator.flush_round()  # 结算最后一轮（按轮合并模式下必须）
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        prefix, _ = self._stop_at_eos(prefix, prompt_len)
        # 同上：按 max_tokens 截断，保证长度契约与基线一致
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, :max_tokens]
        generated_tokens = prefix.shape[1] - current_tokens.shape[1]

        metrics = get_empty_metrics()
        metrics["avg_top_k"] = (
            sum_top_k / total_draft_steps if total_draft_steps > 0 else 0
        )
        metrics["avg_draft_len"] = (
            sum_draft_len / total_draft_steps if total_draft_steps > 0 else 0
        )
        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = total_drafted_tokens
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = (
            elapsed_time + comm_simulator.edge_cloud_comm_time + queuing_time
        )
        metrics["throughput"] = (
            generated_tokens / metrics["wall_time"] if metrics["wall_time"] > 0 else 0
        )
        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time,
            queuing_time=queuing_time,
        )
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times

        if self.rl_adapter is not None:
            self.rl_adapter.save(metrics.get("throughput"))

        # 复制 edge-cloud 的带宽、top-k 和起草长度历史数据
        metrics["edge_cloud_bandwidth_history"] = (
            comm_simulator.edge_cloud_bandwidth_history.copy()
        )
        # 逐消息记账记录（[字节, 轮号]）——离线重放（scripts/rebill.py）的
        # 输入：换带宽/NTT/地板/口径的敏感性分析不用重跑 GPU。
        # getattr 兜底：测试替身可能只实现聚合口径。
        metrics["comm_trace_edge_cloud"] = getattr(
            comm_simulator, "edge_cloud_trace", []
        )
        _add_comm_accounting_metrics(metrics, self.args, comm_simulator)
        metrics["edge_cloud_topk_history"] = (
            comm_simulator.edge_cloud_topk_history.copy()
        )
        metrics["edge_cloud_draft_len_history"] = (
            comm_simulator.edge_cloud_draft_len_history.copy()
        )

        return prefix, metrics

    @Register.register_decoding("adaptive_tridecoding")
    @Register.register_decoding("cee_sd")
    @torch.no_grad()
    def adaptive_tridecoding(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud=10,
        ntt_ms_edge_end=1,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        _opportunistic_first_stage: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        if prefix.dtype != torch.long:
            prefix = prefix.long()

        # 新增开关（默认关闭，保证历史数字可复现）：
        #   force_full_vocab_transfer=True ⇒ 传完整词表分布（标准投机采样基线）。
        #   机制：proposal_top_k(vocab_size) 返回 None（不截断），且计费公式
        #   draft_len × vocab_size × (prob+index) 会如实 charge 全词表载荷。
        #   必须在方法入口改写，才能同时影响「真实截断」与「通信计费」两条路径。
        if bool(getattr(self.args, "force_full_vocab_transfer", False)):
            transfer_top_k = int(getattr(self, "vocab_size", 0) or 32000)

        batch_delay = self.args.batch_delay
        queuing_time = 0.0
        max_tokens = prefix.shape[1] + self.args.max_tokens
        little_device = self.get_model_input_device(self.little_model)
        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        # 使用 transfer_top_k 作为草稿模型的 top-k 压缩参数
        draft_top_k = (
            transfer_top_k
            if (transfer_top_k is not None and transfer_top_k > 0)
            else self.args.top_k
        )
        # F43：是否按"拒绝位置残差所需的分布载荷"计费，并改用精确的
        # top-k + 均匀尾表示（TopKProposalHistory）而不是整行。默认关闭以保持
        # 历史数字可复现；开启后通信记账才与真实分布式实现一致。
        charge_residual = bool(
            getattr(self.args, "charge_residual_payload", False)
        )
        probe_cache_max_length = getattr(self.args, "probe_cache_max_length", None)
        cache_kwargs = (
            {"max_length": probe_cache_max_length}
            if probe_cache_max_length is not None
            else {}
        )

        # ── CUDA Graph 接线（默认关，历史数字逐位可复现）────────────────────
        # 图模式三件事：
        #   1. 单步草稿 → (1,1) 单步图（原有能力）；
        #   2. 验证/resync 的多 token 合成前向 → (1,K) 定长 padding 验证图
        #      ——这是本轮修复的主体：13B/1.1B 每轮 84~92% 的时间是逐算子调度，
        #      且主体在多 token 前向上，单步图接不住；
        #   3. 缓存跨样本复用（StaticCache 原地 reset，图不重捕获）——
        #      否则每样本 ~534ms 的捕获开销把收益吃光。
        # 尺寸档位从 γ 上限推导：验证前向 k ≤ 接受数 + γ + 2，取安全帽
        # γ1+γ2+4；--graph_verify_sizes 可显式覆盖。
        _graph_enabled = bool(getattr(self.args, "use_cuda_graph", False))

        def _graph_cache_kwargs() -> Dict[str, object]:
            if not _graph_enabled:
                return dict(cache_kwargs)
            raw_sizes = getattr(self.args, "graph_verify_sizes", None) or ""
            cap = int(self.args.gamma1) + int(self.args.gamma2) + 4
            if isinstance(raw_sizes, str) and raw_sizes.strip():
                ladder = sorted(
                    {int(x) for x in str(raw_sizes).replace(" ", "").split(",") if x}
                )
            else:
                ladder = [4, 8, 16, 24, 32, 40, 48, 64]
            sizes = [s for s in ladder if 2 <= s <= cap]
            if not sizes or sizes[-1] < cap:
                sizes.append(min(((cap + 7) // 8) * 8, 128))
            return {
                **cache_kwargs,
                "use_cuda_graph": True,
                "verify_graph_sizes": sizes,
                "graph_len_budget": int(self.args.max_tokens) + 256,
            }

        # 跨样本复用：图模式的三个缓存挂在 self 上，样本间只 reset（图不重捕获）。
        # eager 模式保持每样本重建 ⇒ 行为与历史完全一致。
        _caches_attr = "_adaptive_tridecoding_caches"
        _reused_caches = getattr(self, _caches_attr, None) if _graph_enabled else None
        if _reused_caches is not None:
            little_model_cache = _reused_caches["little"]
            draft_model_cache = _reused_caches["draft"]
            target_model_cache = _reused_caches["target"]
            little_model_cache.reset_for_new_sample()
            draft_model_cache.reset_for_new_sample()
            target_model_cache.reset_for_new_sample()
        else:
            _little_kwargs = _graph_cache_kwargs()
            _draft_kwargs = _graph_cache_kwargs()
            _target_kwargs = _graph_cache_kwargs()
            little_model_cache = KVCacheModel(
                self.little_model,
                self.args.temp,
                draft_top_k,
                self.args.top_p,
                **_little_kwargs,
            )
            little_model_cache.vocab_size = self.vocab_size
            draft_model_cache = KVCacheModel(
                self.draft_model,
                self.args.temp,
                draft_top_k,
                self.args.top_p,
                **_draft_kwargs,
            )
            draft_model_cache.vocab_size = self.vocab_size
            target_model_cache = KVCacheModel(
                self.target_model,
                self.args.temp,
                0,
                0,  # 目标模型不压缩
                **_target_kwargs,
            )
            target_model_cache.vocab_size = self.vocab_size
            if _graph_enabled:
                setattr(
                    self,
                    _caches_attr,
                    {
                        "little": little_model_cache,
                        "draft": draft_model_cache,
                        "target": target_model_cache,
                    },
                )

        if use_precise_comm_sim:
            comm_simulator: CommunicationSimulator = PreciseCommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=self.args.edge_end_bandwidth,
                bandwidth_cloud_end=self.args.cloud_end_bandwidth,
                transfer_top_k=transfer_top_k,
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=bool(getattr(self.args, "stochastic_ntt", False)),
                mode=getattr(self.args, "comm_trace_mode", "static"),
            )

        # B 组改造（便宜且确定的收益）：
        #  · comm_round_trip_mode=per_round：同一轮内多条消息合成"每链路一次往返"，
        #    只付一次 NTT（原实现一轮要付 3.2+3.0 次）；
        #  · transfer_top_k_cap：给 RL 选出的 top-k 设上限，直接压低拒绝载荷字节。
        comm_simulator.coalesce_rounds = (
            str(getattr(self.args, "comm_round_trip_mode", "per_transfer")) == "per_round"
        )
        _topk_cap = int(getattr(self.args, "transfer_top_k_cap", 0) or 0)
        if bool(getattr(self.args, "force_full_vocab_transfer", False)):
            # 强制全词表：cap 不再覆盖，并把全词表规模告知 simulator 用于计费
            comm_simulator.transfer_top_k = transfer_top_k
        elif _topk_cap > 0:
            if (
                transfer_top_k is None
                or int(transfer_top_k) <= 0
                or int(transfer_top_k) > _topk_cap
            ):
                transfer_top_k = _topk_cap
            comm_simulator.transfer_top_k = transfer_top_k

        # Metrics tracking
        little_model_forward_times = 0
        draft_model_forward_times = 0
        target_model_forward_times = 0
        total_little_model_generated_tokens = 0
        total_draft_model_generated_tokens = 0
        total_little_model_accepted_tokens = 0
        total_draft_model_accepted_tokens = 0
        wall_time = 0
        arp_overhead_time = 0.0
        dra_overhead_time = 0.0
        little_entropy_history: List[float] = []
        draft_entropy_history: List[float] = []
        little_accept_rate_history: List[float] = []
        draft_accept_rate_history: List[float] = []
        little_accepted_vocab_rank_history: List[int] = []
        draft_accepted_vocab_rank_history: List[int] = []
        little_accepted_in_transfer_topk_history: List[bool] = []
        draft_accepted_in_transfer_topk_history: List[bool] = []
        little_accepted_transfer_topk_rank_history: List[int] = []
        draft_accepted_transfer_topk_rank_history: List[int] = []

        idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()  # 用于计算生成token数

        start_event.record(stream=torch.cuda.current_stream())

        comm_simulator.transfer(prefix, None, "edge_end")  # 将 prompt 传输到 edge

        cuhlm_uncertainty_sim: Optional[CUHLM] = None
        if _opportunistic_first_stage:
            _, little_uncertainty_threshold = self._select_cuhlm_stage_config(
                stage="little_to_draft",
                transfer_top_k=transfer_top_k,
                uncertainty_threshold=getattr(
                    self.args, "uncertainty_threshold", 0.8
                ),
            )
            cuhlm_uncertainty_sim = CUHLM(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=self.args.edge_end_bandwidth,
                bandwidth_cloud_end=self.args.cloud_end_bandwidth,
                uncertainty_threshold=little_uncertainty_threshold,
                vocab_size=self.vocab_size,
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=bool(getattr(self.args, "stochastic_ntt", False)),
            )

        little_comp_time = 0.0
        draft_comp_time = 0.0
        target_comp_time = 0.0

        _tri_prompt_len = prefix.shape[1]
        while prefix.shape[1] < max_tokens:
            idx += 1
            comm_simulator.set_round(idx)
            step_start_time = time.time()
            prefix_len = prefix.shape[1]
            current_proposal_top_k = proposal_top_k(transfer_top_k)
            little_stage_probs: Optional[torch.Tensor] = None
            draft_stage_probs: Optional[torch.Tensor] = None
            current_proposal_top_k = proposal_top_k(transfer_top_k)

            # 第一层 speculative
            edge_end_comm_start = comm_simulator.edge_end_comm_time
            edge_end_energy_start = comm_simulator.total_comm_energy

            self.small_draft_adapter.reset_step()
            adapter = self.small_draft_adapter
            assert adapter.device != torch.device("cpu")
            # 第一层的 draft 长度同理：动作空间含 gamma 时由策略决定。
            gamma2_used = getattr(self, "_next_gamma2", None) or self.args.gamma2
            little_gamma = 1 if _opportunistic_first_stage else gamma2_used
            t0 = time.time()
            x, little_rebuilt_probs, little_rebuilt_meta, q = (
                self._generate_with_optional_rebuilt_proposal(
                    little_model_cache,
                    _move_token_tensor(prefix, little_device),
                    little_gamma,
                    current_proposal_top_k,
                    adapter=adapter,
                    need_topk_metadata=charge_residual,
                )
            )
            little_comp_time += time.time() - t0

            # q 可能为 None（未开 top-k 压缩时不会重建概率）；熵取自缓存的
            # last_entropy（原始 logits、温度 1）——原实现在这里直接抛错/二次 softmax
            little_entropy = state_entropy(q, little_model_cache)
            little_entropy_history.append(little_entropy)

            if self.little_rl_adapter is not None:
                dra_start = time.time()
                bandwidth = comm_simulator.bandwidth_edge_end_mbps
                latency = comm_simulator.ntt_edge_end_ms
                acc_probs = getattr(self.small_draft_adapter, "step_acc_probs", [])

                task_name = getattr(self, "task", "unknown")
                next_topk, next_threshold = self.little_rl_adapter.select_config(
                    bandwidth,
                    latency,
                    acc_probs,
                    little_entropy,
                    task_name,
                    training=not getattr(self.args, "disable_rl_update", False),
                )
                # 小模型层面的 top-k 压缩（如果需要）和 ARP 阈值
                # transfer_top_k = next_topk  # edge-end 通常不压缩
                self.small_draft_adapter.threshold = next_threshold
                # 同主适配器：只有动作空间含 gamma 时才覆盖（本路径下第一级被
                # opportunistic 接管，gamma2 实际恒为 1，见诊断文档 F7）。
                if getattr(self.little_rl_adapter, "gamma_dim", 1) > 1:
                    self._next_gamma2 = int(
                        getattr(self.little_rl_adapter, "last_gamma", None) or gamma2_used
                    )
                dra_overhead_time += time.time() - dra_start

            actual_gamma2 = x.shape[1] - prefix_len

            little_model_forward_times += actual_gamma2
            total_little_model_generated_tokens += actual_gamma2

            n1: int = prefix_len + actual_gamma2 - 1
            little_accepted_this_iter = 0
            little_all_accepted = True

            skip_first_stage_draft = False
            if _opportunistic_first_stage and actual_gamma2 > 0:
                little_stage_probs = stage_prob_history(
                    little_model_cache,
                    prefix_len,
                    little_rebuilt_probs,
                )
                assert cuhlm_uncertainty_sim is not None
                if self.little_rl_adapter is not None:
                    cuhlm_uncertainty_sim.uncertainty_threshold = float(
                        self.small_draft_adapter.threshold
                    )
                little_token_id = int(x[:, prefix_len].item())
                if little_model_cache.logits_history is None:
                    raise ValueError(
                        "Little model logits history is required for "
                        "opportunistic CEE-SD"
                    )
                current_little_logits = little_model_cache.logits_history[
                    :, prefix_len - 1, : self.vocab_size
                ]
                uncertainty = cuhlm_uncertainty_sim.calculate_uncertainty(
                    current_little_logits,
                    M=20,
                    theta_max=2.0,
                    draft_token=little_token_id,
                )
                should_transfer, _ = cuhlm_uncertainty_sim.determine_transfer_strategy(
                    uncertainty,
                    little_stage_probs[:, prefix_len - 1, : self.vocab_size],
                )
                skip_first_stage_draft = not should_transfer

            if skip_first_stage_draft:
                little_accepted_this_iter = actual_gamma2
                n1 = prefix_len + actual_gamma2 - 1
                little_all_accepted = True
                t = torch.empty(
                    (prefix.shape[0], 0), dtype=torch.long, device=little_device
                )
                comm_simulator.simulate_transfer(8, "edge_end")
                comm_simulator.send_accept_message("edge_end")
            else:
                # Pre-launch draft verification on GPU (overlaps with CPU code below)
                t0 = time.time()
                _ = draft_model_cache.generate(
                    _move_token_tensor(x, draft_device), 1
                )
                draft_comp_time += time.time() - t0
                draft_model_forward_times += 1

                # 批量传输 draft tokens 和对应的 probabilities 以节省 RTT
                if actual_gamma2 > 0:
                    if little_stage_probs is None:
                        little_stage_probs = stage_prob_history(
                            little_model_cache,
                            prefix_len,
                            little_rebuilt_probs,
                        )
                    # B15：量化与计费原子绑定——第一级上行此前静默忽略位宽
                    # （下方 verify 的 override 复用同一变量，天然同源）
                    little_stage_probs, _little_bits = _uplink_prob_payload(
                        little_stage_probs, self.args
                    )
                    draft_tokens, draft_probs = collect_verification_payload(
                        little_stage_probs,
                        x,
                        prefix_len,
                        actual_gamma2,
                    )
                    comm_simulator.transfer(
                        draft_tokens,
                        draft_probs,
                        "edge_end",
                        prob_bits=_little_bits,
                    )

                if actual_gamma2 > 0:
                    (
                        little_accepted_this_iter,
                        n1,
                        t,
                        little_all_accepted,
                    ) = resolve_stage_verification(
                        proposer_cache=little_model_cache,
                        verifier_cache=draft_model_cache,
                        x=x,
                        prefix_len=prefix_len,
                        gamma=actual_gamma2,
                        output_device=little_device,
                        draft_probs_override=cast(torch.Tensor, little_stage_probs),
                        # F43：拒绝时要算残差 ⇒ 验证方需要提案分布。传精确的
                        # top-k+均匀尾表示，避免"整行 128 kB"被无声地省掉。
                        draft_topk_history=(
                            stage_topk_proposal_history(
                                little_rebuilt_meta, actual_gamma2
                            )
                            if charge_residual
                            else None
                        ),
                    )
                    if charge_residual and not little_all_accepted:
                        # 小模型→草稿链路：补记被拒位置残差载荷中**缺失的索引部分**
                        #（概率部分已由既有的 simulate_transfer 计费）
                        comm_simulator.simulate_transfer(
                            self._residual_payload_bytes(
                                little_stage_probs, current_proposal_top_k
                            ),
                            "edge_end",
                        )
                else:
                    t = sample_accept_token(
                        draft_model_cache.prob_history[:, -1, : self.vocab_size],
                        output_device=little_device,
                    )
                    little_all_accepted = True

            total_little_model_accepted_tokens += little_accepted_this_iter
            little_accept_rate_history.append(
                little_accepted_this_iter / actual_gamma2 if actual_gamma2 > 0 else 0.0
            )
            _record_accepted_token_ranks(
                stage_probs=little_stage_probs,
                x=x,
                prefix_len=prefix_len,
                accepted_count=little_accepted_this_iter,
                transfer_top_k=transfer_top_k,
                vocab_rank_history=little_accepted_vocab_rank_history,
                in_transfer_topk_history=little_accepted_in_transfer_topk_history,
                transfer_topk_rank_history=little_accepted_transfer_topk_rank_history,
            )

            if self.little_rl_adapter is not None:
                step_end_time = time.time()
                step_time = step_end_time - step_start_time
                step_comm_time = comm_simulator.edge_end_comm_time - edge_end_comm_start

                # 第一层（end->edge）奖励，走同一套奖励实现（见 src/rl_reward.py）。
                # 注意：当前版本的信用分配是"各层局部奖励"，两层强耦合时这是有偏的；
                # 团队奖励 / difference reward 是后续改进方向。
                reward = self.little_rl_adapter.compute_reward(
                    accepted=little_accepted_this_iter,
                    generated=actual_gamma2,
                    comm_s=step_comm_time,
                    compute_s=step_time,
                    energy_j=comm_simulator.total_comm_energy - edge_end_energy_start,
                    forward_counts={"little": float(actual_gamma2)},
                    opportunistic=_opportunistic_first_stage,
                )

                if getattr(self.args, "rl_team_reward", False):
                    # 团队奖励：两层共享同一个 iteration 级奖励。第一层先于第二层
                    # 决策，因此使用上一轮写入的团队奖励（延迟一步），这是 cooperative
                    # MARL 中处理时序耦合的标准做法；首轮退化为本地奖励。
                    team = getattr(self, "_team_reward_buffer", None)
                    if team is not None:
                        reward = team

                if not getattr(self.args, "disable_rl_update", False):
                    self.little_rl_adapter.step(reward)

            assert n1 >= prefix_len - 1, f"n1 {n1}, prefix_len {prefix_len}"
            prefix = x[:, : n1 + 1]

            prob_bytes = 0.0
            reject_overhead = 0.0

            if not little_all_accepted:
                # reject someone, sample from the pos n1
                # rebuild_probs = comm_simulator.rebuild_full_probs(
                #     little_model_cache.prob_history[:, n1, : self.vocab_size]
                # )
                # little_model_cache.prob_history[:, n1, : self.vocab_size] = (
                #     rebuild_probs
                # )

                # comm_simulator.transfer(
                #     None,
                #     little_model_cache.prob_history[:, n1, : self.vocab_size],
                #     "edge_end",
                #     transfer_top_k is not None and transfer_top_k > 0,
                #     transfer_top_k,
                # )

                prob_data = little_model_cache.prob_history[:, n1, : self.vocab_size]
                if actual_gamma2 > 0:
                    prob_data = cast(torch.Tensor, little_stage_probs)[
                        :, n1, : self.vocab_size
                    ]
                prob_bytes = prob_data.element_size() * prob_data.numel()
                if transfer_top_k is not None and transfer_top_k > 0:
                    prob_bytes = transfer_top_k * prob_data.element_size()
                # B15：位宽决策走原子 helper——此前只按 bits 计费却从不
                # 量化（"收 bits 钱、传全宽信息"的活跃实例）
                prob_data, _pb_bits = _uplink_prob_payload(prob_data, self.args)
                if _pb_bits is not None:
                    _n = (transfer_top_k if (transfer_top_k is not None and transfer_top_k > 0)
                          else prob_data.numel())
                    prob_bytes = int(_n) * _pb_bits / 8.0

                reject_overhead = 6.0

            # 传输索引和 token t (一次 RTT)
            # 包含了 rejection overhead (如果发生) 和 probs (如果发生)
            if not skip_first_stage_draft:
                total_bytes = (
                    INT_SIZE
                    + t.element_size() * t.numel()
                    + prob_bytes
                    + reject_overhead
                )
                comm_simulator.simulate_transfer(total_bytes, "edge_end")

            _validate_token_range(
                t,
                vocab_size=self.vocab_size,
                label="adaptive_tridecoding.edge_end.sampled_token",
            )
            prefix = torch.cat((prefix, t), dim=1)
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            _validate_token_range(
                prefix,
                vocab_size=self.vocab_size,
                label="adaptive_tridecoding.edge_end.prefix_after_concat",
            )
            new_generated_token = prefix[:, prefix_len:]

            # 第二层 speculative
            edge_cloud_comm_start = comm_simulator.edge_cloud_comm_time
            edge_cloud_energy_start = comm_simulator.total_comm_energy
            edge_cloud_bytes_start = comm_simulator.edge_cloud_data
            step_start_time = time.time()

            # draft 长度：默认用固定超参 args.gamma1；当 DRA 的动作空间包含 gamma 时
            # （--rl_action_space topk_thr_gamma），用策略上一轮选出的档位。gamma 决定
            # "每个 WAN 往返验证多少 token"，即往返次数（通信时间的 ~97% 来源）。
            gamma1_used = getattr(self, "_next_gamma1", None) or self.args.gamma1

            # Pre-launch GPU draft generation (overlaps with CPU comm sim below)
            self.draft_target_adapter.reset_step()
            adapter = self.draft_target_adapter
            assert adapter.device != torch.device("cpu")
            t0 = time.time()
            x, draft_rebuilt_probs, draft_rebuilt_meta, q = (
                self._generate_with_optional_rebuilt_proposal(
                    draft_model_cache,
                    _move_token_tensor(prefix, draft_device),
                    gamma1_used,
                    current_proposal_top_k,
                    adapter=adapter,
                    need_topk_metadata=charge_residual,
                )
            )
            draft_comp_time += time.time() - t0

            # Communication simulation (pure CPU): overlaps with draft.generate GPU above
            if idx == 1:
                comm_simulator.transfer(prefix, None, "edge_cloud")
            else:
                comm_simulator.transfer(new_generated_token, None, "edge_cloud")

            actual_gamma1 = x.shape[1] - prefix.shape[1]

            queuing_time += batch_delay
            # Pre-launch target forward on GPU (overlaps with RL/entropy CPU code below)
            t0 = time.time()
            _ = target_model_cache.generate(_move_token_tensor(x, target_device), 1)
            target_comp_time += time.time() - t0

            # 同上：这是 adaptive_tridecoding 主适配器实际使用的熵特征
            draft_entropy = state_entropy(q, draft_model_cache)
            draft_entropy_history.append(draft_entropy)

            if self.rl_adapter is not None:
                dra_start = time.time()
                bandwidth = comm_simulator.bandwidth_edge_cloud_mbps
                latency = comm_simulator.ntt_edge_cloud_ms
                acc_probs = getattr(self.draft_target_adapter, "step_acc_probs", [])

                # q 可能为 None（未开 top-k 压缩）；RL 的熵特征已在上面由
                # state_entropy(q, draft_model_cache) 取到，这里不需要 q。
                task_name = getattr(self, "task", "unknown")
                next_topk, next_threshold = self.rl_adapter.select_config(
                    bandwidth,
                    latency,
                    acc_probs,
                    draft_entropy,
                    task_name,
                    training=not getattr(self.args, "disable_rl_update", False),
                )
                # 更新 top-k 压缩参数和 ARP 阈值
                transfer_top_k = next_topk
                if _topk_cap > 0 and transfer_top_k is not None:
                    transfer_top_k = min(int(transfer_top_k), _topk_cap)
                self.draft_target_adapter.threshold = next_threshold
                # 只有动作空间真的包含 gamma 时才覆盖固定超参。legacy 适配器
                # （gamma_dim == 1）的 last_gamma 只是 gamma_candidates[0]（默认 2），
                # 无条件赋值会把 --gamma1 悄悄改掉 —— 这个 bug 曾让 A/B 的基线全部
                # 跑在 gamma=2 上，从而虚增了扩展动作空间的收益（诊断文档 F14）。
                if getattr(self.rl_adapter, "gamma_dim", 1) > 1:
                    self._next_gamma1 = int(
                        getattr(self.rl_adapter, "last_gamma", None) or gamma1_used
                    )
                dra_overhead_time += time.time() - dra_start

            draft_model_forward_times += actual_gamma1
            target_model_forward_times += 1
            total_draft_model_generated_tokens += (
                new_generated_token.shape[1] + actual_gamma1
            )

            total_gamma = new_generated_token.shape[1] + actual_gamma1
            n2: int = prefix_len + total_gamma - 1

            # F43：构造阶段二验证所需的提案分布历史（含 bonus 前缀位置）
            draft_stage_topk_history = None
            if charge_residual:
                _prefix_topk_history = None
                if new_generated_token.shape[1] > 0:
                    _prefix_prob_rows = draft_model_cache.prob_history[
                        :,
                        prefix_len - 1 : prefix_len - 1 + new_generated_token.shape[1],
                        :,
                    ]
                    _prefix_topk_history = build_stage_prefix_topk_history(
                        _prefix_prob_rows,
                        current_proposal_top_k,
                    )
                draft_stage_topk_history = stage_topk_proposal_history(
                    merge_stage_topk_histories(
                        _prefix_topk_history,
                        stage_topk_proposal_history(draft_rebuilt_meta, gamma1_used),
                    ),
                    total_gamma,
                )


            # 批量传输 draft tokens 和对应的 probabilities 以节省 RTT
            if actual_gamma1 > 0:
                draft_stage_probs = stage_prob_history(
                    draft_model_cache,
                    prefix_len + new_generated_token.shape[1],
                    draft_rebuilt_probs,
                )
                # B15：原子化——此点位本就"量化+计费+verify 复用"三对，
                # 收敛进 helper 消除门控样板；默认 16 行为不变
                draft_stage_probs, _draft_bits = _uplink_prob_payload(
                    draft_stage_probs, self.args
                )
                draft_tokens_second, draft_probs_second = collect_verification_payload(
                    draft_stage_probs,
                    x,
                    prefix_len,
                    total_gamma,
                )
                comm_simulator.transfer(
                    draft_tokens_second,
                    draft_probs_second,
                    "edge_cloud",
                    prob_bits=_draft_bits,
                )

            if actual_gamma1 > 0:
                # 缓存一致性自检（CACHE_CHECK_TRACE）：验证者 cache 里应当恰好有
                # prefix_len 个 token（= 已确认的上下文）。若实际长度与之不符，
                # 说明此前的 rollback 截断位置错了，后续前向会在错误的上下文上
                # 计算 logits —— 恒等检验里"模型把 prompt 又写一遍"的分歧就是
                # 这种失配的典型症状。
                _cache_trace = os.environ.get("CACHE_CHECK_TRACE")
                if _cache_trace:
                    try:
                        with open(_cache_trace, "a") as _fh:
                            _fh.write(
                                json.dumps(
                                    {
                                        "where": "pre_verify",
                                        "x_len": int(x.shape[1]),
                                        "prefix_len": int(prefix_len),
                                        "target_cache_len": int(
                                            target_model_cache.current_length
                                        ),
                                        "draft_cache_len": int(
                                            draft_model_cache.current_length
                                        ),
                                        "actual_gamma1": int(actual_gamma1),
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                (
                    draft_accepted_this_iter,
                    n2,
                    t,
                    draft_all_accepted,
                ) = resolve_stage_verification(
                    proposer_cache=draft_model_cache,
                    verifier_cache=target_model_cache,
                    x=x,
                    prefix_len=prefix_len,
                    gamma=total_gamma,
                    output_device=draft_device,
                    draft_probs_override=cast(torch.Tensor, draft_stage_probs),
                    # F43：同阶段一，拒绝时残差需要提案分布。注意阶段二的验证
                    # 窗口比本轮草稿长：前 `new_generated_token.shape[1]` 个位置是
                    # 上一轮带过来的 bonus token（分布取自草稿缓存的原始行），必须
                    # 用 merge_stage_topk_histories 补上，否则长度不匹配。
                    draft_topk_history=draft_stage_topk_history,
                )
                if charge_residual and not draft_all_accepted:
                    # 草稿→目标链路：同上，补记缺失的索引/尾部字节
                    comm_simulator.simulate_transfer(
                        self._residual_payload_bytes(
                            draft_stage_probs, current_proposal_top_k
                        ),
                        "edge_cloud",
                    )
            else:
                draft_accepted_this_iter = 0
                t = sample_accept_token(
                    target_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=draft_device,
                )
                draft_all_accepted = True
            # ARP_CALIB_TRACE: 逐位配对 (ARP预测, 真接受概率) —— 校准诊断专用。
            # 真值 = min(1, p_t(x_i)/p_d(x_i))（采样域）与 argmax 匹配（贪心域），
            # 两个都dump，分析端按温度取用。arp 行只取 draft_target 头（主级）。
            _calib_path = os.environ.get("ARP_CALIB_TRACE")
            _acc_probs_calib = list(
                getattr(self.draft_target_adapter, "step_acc_probs", []) or []
            )
            if _calib_path and _acc_probs_calib:
                try:
                    _g_calib = len(_acc_probs_calib)
                    _toks_calib = x[0, prefix_len : prefix_len + _g_calib].tolist()
                    _rows_calib = []
                    for _i, _ap in enumerate(_acc_probs_calib):
                        _tok = int(_toks_calib[_i])
                        _row_d = draft_model_cache.prob_history[
                            0, prefix_len - 1 + _i
                        ]
                        _row_t = target_model_cache.prob_history[
                            0, prefix_len - 1 + _i
                        ]
                        _pd = float(_row_d[_tok])
                        _pt = float(_row_t[_tok])
                        _rows_calib.append(
                            json.dumps(
                                {
                                    "arp": float(_ap),
                                    "tok": _tok,
                                    "pd": _pd,
                                    "pt": _pt,
                                    "ratio": (
                                        min(1.0, _pt / _pd) if _pd > 0 else 1.0
                                    ),
                                    "gmatch": int(
                                        int(_row_t.argmax().item()) == _tok
                                    ),
                                }
                            )
                        )
                    with open(_calib_path, "a") as _fh:
                        _fh.write("\n".join(_rows_calib) + "\n")
                except Exception:
                    pass
            # TOPK_TRACE: 逐位 dump 草稿分布形状（排序质量/熵/目标token秩）——
            # top-k 集中度诊断用。与 ARP_CALIB_TRACE 同点位同索引, 独立 env 门控。
            _topk_path = os.environ.get("TOPK_TRACE")
            if _topk_path and _acc_probs_calib:
                try:
                    _kg = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                    _rows_tk = []
                    for _i in range(len(_acc_probs_calib)):
                        _tok = int(x[0, prefix_len + _i].item())
                        # 优先取真实分布（logits→softmax）；prob_history 在该
                        # 路径可能已被重建/截断对象覆盖（传输侧视图），降级时标记。
                        _lh_tk = draft_model_cache.logits_history
                        if _lh_tk is not None:
                            _rd = torch.softmax(
                                _lh_tk[0, prefix_len - 1 + _i, : self.vocab_size]
                                .float(),
                                dim=-1,
                            )
                            _obj_tk = "true"
                        else:
                            _rd = draft_model_cache.prob_history[
                                0, prefix_len - 1 + _i
                            ]
                            _obj_tk = "rebuilt"
                        _rt = target_model_cache.prob_history[
                            0, prefix_len - 1 + _i
                        ]
                        _srt, _ord = _rd.sort(descending=True)
                        _cum = torch.cumsum(_srt, 0)[
                            [k - 1 for k in _kg if k <= _srt.numel()]
                        ]
                        _tok_rank = int((_rd > _rd[_tok]).sum().item()) + 1
                        _tam = int(_rt.argmax().item())
                        _tam_rank = int((_rd > _rd[_tam]).sum().item()) + 1
                        _ent = float(
                            -(_rd[_rd > 0] * _rd[_rd > 0].log()).sum().item()
                        )
                        _rows_tk.append(
                            json.dumps(
                                {
                                    "k_grid": _kg[: len(_cum)],
                                    "cum_mass": [round(float(v), 5) for v in _cum],
                                    "tok": _tok,
                                    "tok_rank": _tok_rank,
                                    "tok_pd": float(_rd[_tok]),
                                    "t_argmax": _tam,
                                    "t_argmax_rank": _tam_rank,
                                    "gmatch": int(_tam == _tok),
                                    "acc": int(_i < draft_accepted_this_iter),
                                    "obj": _obj_tk,
                                    "entropy": round(_ent, 4),
                                    "top20": [
                                        round(float(v), 5) for v in _srt[:20]
                                    ],
                                }
                            )
                        )
                    with open(_topk_path, "a") as _fh:
                        _fh.write("\n".join(_rows_tk) + "\n")
                except Exception:
                    pass
            total_draft_model_accepted_tokens += draft_accepted_this_iter
            draft_accept_rate_history.append(
                draft_accepted_this_iter / total_gamma if total_gamma > 0 else 0.0
            )
            _record_accepted_token_ranks(
                stage_probs=draft_stage_probs,
                x=x,
                prefix_len=prefix_len,
                accepted_count=draft_accepted_this_iter,
                transfer_top_k=transfer_top_k,
                vocab_rank_history=draft_accepted_vocab_rank_history,
                in_transfer_topk_history=draft_accepted_in_transfer_topk_history,
                transfer_topk_rank_history=draft_accepted_transfer_topk_rank_history,
            )

            if self.rl_adapter is not None:
                step_end_time = time.time()
                step_time = step_end_time - step_start_time
                step_comm_time = (
                    comm_simulator.edge_cloud_comm_time - edge_cloud_comm_start
                )

                # 奖励统一由 src/rl_reward.py 实现：默认 legacy 即 v1 原式
                # exp(min(N_acc/T,100)/20)*(N_acc/gamma)^2，可完全复现旧结果；
                # 其余模式为 N_acc - lambda*T 等线性形式，且可用与主机无关的
                # 计算时间模型（rl_compute_time_mode=model）。
                reward = self.rl_adapter.compute_reward(
                    accepted=draft_accepted_this_iter,
                    generated=actual_gamma1,
                    comm_s=step_comm_time,
                    compute_s=step_time,
                    energy_j=comm_simulator.total_comm_energy - edge_cloud_energy_start,
                    # 本区间 WAN 实际传输的字节数（--rl_byte_price 的输入，见 F17）
                    transferred_bytes=comm_simulator.edge_cloud_data
                    - edge_cloud_bytes_start,
                    forward_counts={
                        "little": float(actual_gamma2),
                        "draft": float(actual_gamma1),
                        "target": 1.0,
                    },
                    # 排队按每主决策一次 batch_delay 计（与 queuing_time 累计口径
                    # 一致）；是否计入 T 由 RewardConfig.charge_queue 决定。
                    queue_s=float(getattr(self.args, "batch_delay", 0.0)),
                )

                if getattr(self.args, "rl_team_reward", False):
                    # 供第一层下一轮取用的团队奖励（见 little 侧的说明）。
                    self._team_reward_buffer = reward

                if not getattr(self.args, "disable_rl_update", False):
                    self.rl_adapter.step(reward)

            assert n2 >= prefix_len - 1, (
                f"n {n2} should be greater or equal than prefix_len {prefix_len}"
            )
            prefix = x[:, : n2 + 1]
            if n2 <= little_model_cache.current_length:
                little_model_cache.rollback(n2 + 1)
            prob_bytes = 0.0
            reject_overhead = 0.0

            if not draft_all_accepted:
                # rebuild_probs = comm_simulator.rebuild_full_probs(
                #     draft_model_cache.prob_history[:, n2, : self.vocab_size]
                # )
                # draft_model_cache.prob_history[:, n2, : self.vocab_size] = (
                #     rebuild_probs
                # )

                # comm_simulator.transfer(
                #     None,
                #     draft_model_cache.prob_history[:, n2, : self.vocab_size],
                #     "edge_cloud",
                #     transfer_top_k is not None and transfer_top_k > 0,
                #     transfer_top_k,
                # )

                prob_data = cast(torch.Tensor, draft_stage_probs)[
                    :, n2, : self.vocab_size
                ]
                prob_bytes = prob_data.element_size() * prob_data.numel()
                if transfer_top_k is not None and transfer_top_k > 0:
                    prob_bytes = transfer_top_k * prob_data.element_size()
                # B15：位宽决策走原子 helper（同 stage-1 拒绝点）
                prob_data, _pb_bits = _uplink_prob_payload(prob_data, self.args)
                if _pb_bits is not None:
                    _n = (transfer_top_k if (transfer_top_k is not None and transfer_top_k > 0)
                          else prob_data.numel())
                    prob_bytes = int(_n) * _pb_bits / 8.0

                reject_overhead = 6.0
                new_generated_token = prefix[:, prefix_len:]

            else:
                new_generated_token = prefix[:, prefix_len:]

            prefix = torch.cat((prefix, t), dim=1)
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            # 传输索引和 token t (各链路一次 RTT)
            token_size = t.element_size() * t.numel()

            # Merged transfer for Edge-Cloud (Reject + Probs + Index + Token)
            total_bytes = INT_SIZE + token_size + prob_bytes + reject_overhead
            comm_simulator.simulate_transfer(
                total_bytes, "edge_cloud", topk=transfer_top_k, draft_len=total_gamma
            )

            _send_downlink_token(comm_simulator, t, "edge_end")

            # 同步

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        wall_time += elapsed_time
        # 遵守 max_tokens：投机解码按整块追加，最后一轮可能多出若干 token
        # （F33 契约）。参照 dist_spec 的 remaining-1 截断，这里显式截断——
        # 否则会多拿 token，使配对质量比较与时延/吞吐统计都不公平。
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, :max_tokens]
        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        wall_time += (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )

        metrics = get_empty_metrics()
        metrics["little_forward_times"] = little_model_forward_times
        metrics["draft_forward_times"] = draft_model_forward_times
        metrics["target_forward_times"] = target_model_forward_times
        metrics["little_computation_time"] = little_comp_time
        metrics["draft_computation_time"] = draft_comp_time
        metrics["target_computation_time"] = target_comp_time
        metrics["generated_tokens"] = generated_tokens
        metrics["little_generated_tokens"] = total_little_model_generated_tokens
        metrics["draft_generated_tokens"] = total_draft_model_generated_tokens
        metrics["little_accepted_tokens"] = total_little_model_accepted_tokens
        metrics["draft_accepted_tokens"] = total_draft_model_accepted_tokens
        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = wall_time + queuing_time
        metrics["throughput"] = (
            metrics["generated_tokens"] / metrics["wall_time"]
            if metrics["wall_time"] > 0
            else 0
        )
        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time,
            queuing_time=queuing_time,
            little_comp_time=little_comp_time,
            draft_comp_time=draft_comp_time,
            target_comp_time=target_comp_time,
        )
        comm_simulator.flush_round()  # 结算最后一轮（按轮合并模式下必须）
        metrics["communication_time"] = (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times
        metrics["arp_overhead_time"] = arp_overhead_time
        metrics["dra_overhead_time"] = dra_overhead_time

        dra_start = time.time()
        self._save_adaptive_rl_checkpoints(metrics["throughput"])
        metrics["dra_overhead_time"] += time.time() - dra_start

        # 复制 edge-cloud 的带宽、top-k 和起草长度历史数据
        metrics["edge_cloud_bandwidth_history"] = (
            comm_simulator.edge_cloud_bandwidth_history.copy()
        )
        # 逐消息记账记录（[字节, 轮号]）——离线重放（scripts/rebill.py）的
        # 输入：换带宽/NTT/地板/口径的敏感性分析不用重跑 GPU。
        # getattr 兜底：测试替身可能只实现聚合口径。
        metrics["comm_trace_edge_cloud"] = getattr(
            comm_simulator, "edge_cloud_trace", []
        )
        _add_comm_accounting_metrics(metrics, self.args, comm_simulator)
        metrics["edge_cloud_topk_history"] = (
            comm_simulator.edge_cloud_topk_history.copy()
        )
        metrics["edge_cloud_draft_len_history"] = (
            comm_simulator.edge_cloud_draft_len_history.copy()
        )

        return prefix, metrics

    @Register.register_decoding("cee_sd_opportunistic")
    @torch.no_grad()
    def cee_sd_opportunistic(
        self,
        prefix: torch.Tensor,
        transfer_top_k: int | None = 300,
        use_precise_comm_sim: bool = False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float = 10,
        ntt_ms_edge_end: float = 1,
        use_early_stopping: bool = False,
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, DecodingMetrics]:
        return self.adaptive_tridecoding(
            prefix,
            transfer_top_k=transfer_top_k,
            use_precise_comm_sim=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            use_early_stopping=use_early_stopping,
            stop_sequences=stop_sequences,
            _opportunistic_first_stage=True,
            **kwargs,
        )

    # Two-stage CUHLM variant using uncertainty-gated acceptance on both layers.
    @Register.register_decoding("cee_cuhlm")
    @torch.no_grad()
    def cee_cuhlm(
        self,
        prefix: torch.Tensor,
        max_tokens: int | None = None,
        # --- 传输控制 ---
        transfer_top_k: int | None = 300,
        use_precise_comm_sim: bool = True,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud: float | int = 0,
        ntt_ms_edge_end: float | int = 0,
        use_early_stopping: bool = False,
        stop_sequences: list[str] | None = None,
    ) -> tuple[torch.Tensor, DecodingMetrics]:
        if max_tokens is None:
            # 不放大预算：γ1+γ2+1 的放大会让本方法比基线多生成 token、
            # 高估吞吐（F33 契约）。越界由函数末尾的显式截断兜底。
            max_tokens = prefix.shape[1] + self.args.max_tokens
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream=torch.cuda.current_stream())

        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)
        little_device = self.get_model_input_device(self.little_model)

        # --- Communication Simulator ---
        if use_precise_comm_sim:
            # B45：channel_gain/noise_power_watt 为必填参数，此前缺省导致
            # cee_cuhlm 的精确仿真分支一进就 TypeError（对齐 1544 处
            # uncertainty_decoding 的完整调用）
            comm_simulator: CUHLM = PreciseCUHLM(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_hz=self.args.edge_cloud_bandwidth * 1e6,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                uncertainty_threshold=getattr(
                    self.args, "uncertainty_threshold", 0.8
                ),
            )
        else:
            comm_simulator = CUHLM(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=self.args.edge_end_bandwidth,
                bandwidth_cloud_end=self.args.cloud_end_bandwidth,
                uncertainty_threshold=getattr(self.args, "uncertainty_threshold", 0.8),
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=bool(getattr(self.args, "stochastic_ntt", False)),
            )

        batch_delay = self.args.batch_delay
        queuing_time = 0.0

        # 如果不使用精准通信模拟器，则应用推荐的初始阈值
        if not use_precise_comm_sim:
            _, uncertainty_threshold = self._select_cuhlm_stage_config(
                stage="shared",
                transfer_top_k=transfer_top_k,
                uncertainty_threshold=comm_simulator.uncertainty_threshold,
            )
            comm_simulator.uncertainty_threshold = uncertainty_threshold

        # --- KVCache Models ---
        # CUDA Graph：little/draft/target 都做多 token 验证前向（调度开销主体），
        # 三个都开图；图开启时挂 self 跨样本复用（与 adaptive_tridecoding 一致）。
        _graph_kw = _graph_mode_cache_kwargs(
            self.args,
            cap=int(getattr(self.args, "gamma1", 1))
            + int(getattr(self.args, "gamma2", 1))
            + 4,
        )
        _reused = getattr(self, "_cee_cuhlm_caches", None) if _graph_kw else None
        if _reused is not None:
            little_model_cache, draft_model_cache, target_model_cache = _reused
            little_model_cache.reset_for_new_sample()
            draft_model_cache.reset_for_new_sample()
            target_model_cache.reset_for_new_sample()
        else:
            little_model_cache = KVCacheModel(
                self.little_model, self.args.temp, 0, 0, **_graph_kw,
            )
            draft_model_cache = KVCacheModel(
                self.draft_model, self.args.temp, 0, 0, **_graph_kw,
            )
            target_model_cache = KVCacheModel(
                self.target_model, self.args.temp, 0, 0, **_graph_kw,
            )
            if _graph_kw:
                setattr(
                    self,
                    "_cee_cuhlm_caches",
                    (little_model_cache, draft_model_cache, target_model_cache),
                )
        little_model_cache.vocab_size = self.vocab_size
        draft_model_cache.vocab_size = self.vocab_size
        target_model_cache.vocab_size = self.vocab_size

        # Metrics tracking
        little_model_forward_times = 0
        draft_model_forward_times = 0
        target_model_forward_times = 0
        little_comp_time = 0.0
        draft_comp_time = 0.0
        target_comp_time = 0.0
        total_little_model_generated_tokens = 0
        total_draft_model_generated_tokens = 0
        total_little_model_accepted_tokens = 0
        total_draft_model_accepted_tokens = 0
        wall_time = 0
        arp_overhead_time = 0.0
        dra_overhead_time = 0.0

        idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        start_event.record(stream=torch.cuda.current_stream())

        comm_simulator.transfer(prefix, None, "edge_end")  # 将 prompt 传输到 edge

        _tri_prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点

        while prefix.shape[1] < max_tokens:

            # B10：循环内 EOS 早停（与 dssd/adaptive_* 已接线方法语义对齐——
            # 计算与通信在 EOS 后立即停，时延口径跨方法可比）
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            idx += 1
            step_start_time = time.time()
            prefix_len = prefix.shape[1]
            little_transfer_top_k, little_uncertainty_threshold = (
                self._select_cuhlm_stage_config(
                    stage="little_to_draft",
                    transfer_top_k=transfer_top_k,
                    uncertainty_threshold=comm_simulator.uncertainty_threshold,
                )
            )
            current_proposal_top_k = proposal_top_k(little_transfer_top_k)
            little_stage_probs: Optional[torch.Tensor] = None
            draft_stage_probs: Optional[torch.Tensor] = None

            # Stage 1 speculative (Little -> Draft) with rejection sampling
            # Replaces CUHLM so Draft's distribution governs what enters Stage 2,
            # improving distribution consistency for the Stage 2 CUHLM check.
            t0 = time.time()
            x, little_rebuilt_probs, little_rebuilt_meta, _ = self._generate_with_optional_rebuilt_proposal(
                little_model_cache,
                _move_token_tensor(prefix, little_device),
                self.args.gamma2,
                current_proposal_top_k,
                adapter=None,
                need_topk_metadata=False,
            )
            little_comp_time += time.time() - t0

            actual_gamma2 = x.shape[1] - prefix_len

            t0 = time.time()
            _ = draft_model_cache.generate(_move_token_tensor(x, draft_device), 1)
            draft_comp_time += time.time() - t0

            little_model_forward_times += actual_gamma2
            draft_model_forward_times += 1
            total_little_model_generated_tokens += actual_gamma2

            n1: int = prefix_len + actual_gamma2 - 1
            little_accepted_this_iter = 0
            little_all_accepted = True

            if actual_gamma2 > 0:
                little_stage_probs = stage_prob_history(
                    little_model_cache, prefix_len, little_rebuilt_probs,
                )
                # F-CUHLM（论文口径）：Stage-1 上行 = 论文式 (5) 的逐位置
                # 压缩分布——每个起草位置 k₁·(b_prob+b_index) bits
                # （b_prob=8、b_index=⌈log₂V⌉）；k₁ 未配置时按全词表计
                # （论文 vanilla HLM 的上行）。token 索引 negligible
                # （§II-B）。旧口径：gamma2 × V×4B 整行 fp32 + token 字节。
                comm_simulator.simulate_transfer(
                    actual_gamma2
                    * cuhlm_uplink_payload_bytes(
                        current_proposal_top_k, self.vocab_size
                    ),
                    "edge_end",
                    topk=(
                        int(current_proposal_top_k)
                        if current_proposal_top_k
                        else 0
                    ),
                    draft_len=int(actual_gamma2),
                )

                # Standard rejection sampling: Draft verifies Little's tokens
                verification_inputs = prepare_verification_inputs(
                    draft_model_cache=little_model_cache,
                    target_model_cache=draft_model_cache,
                    x=x, prefix_len=prefix_len, gamma=actual_gamma2,
                    draft_probs_override=little_stage_probs,
                )
                acceptance_result = compute_acceptance_result(verification_inputs)
                little_accepted_this_iter, n1, little_all_accepted = materialize_acceptance(
                    verification_inputs, acceptance_result
                )

                # F-CUHLM（论文口径）：accept/reject 判定不单独计费——验证
                # 结论随轮末响应回传（索引 negligible）；reject 在验证方
                # 基于已持有的分布重采样（论文式 17），无额外上行。
                # 旧口径：每接受 token 8B + accept 消息（6B+NTT）；reject
                # 再 8B + 二次计费的概率行 + reject 消息（6B+NTT）。
                if not little_all_accepted:
                    reject_pos = little_accepted_this_iter

                # Rollback and sample bonus/replacement token
                rollback_plan = build_rollback_plan(
                    prefix_len, verification_inputs.actual_gamma, n1
                )
                apply_rollback(little_model_cache, draft_model_cache, rollback_plan)

                if rollback_plan.all_accepted:
                    t = sample_accept_token(
                        draft_model_cache.prob_history[:, -1, : self.vocab_size],
                        output_device=little_device,
                    )
                else:
                    t = sample_reject_token(
                        verification_inputs.target_probs_batch[:, reject_pos, :],
                        verification_inputs.draft_probs_batch[:, reject_pos, :],
                        output_device=little_device,
                    )
            else:
                t = sample_accept_token(
                    draft_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=little_device,
                )

            total_little_model_accepted_tokens += little_accepted_this_iter

            assert n1 >= prefix_len - 1
            prefix = x[:, : n1 + 1]

            # F-CUHLM（论文口径）：Stage-1 轮末响应（edge→device）的 token
            # 索引 negligible（§II-B）——0 字节报文，保留一次 NTT。
            # 旧口径：INT_SIZE + token 字节 +（reject 时）二次计费的
            # prob_bytes + 魔数 reject_overhead=6B。
            _send_downlink_index_only(comm_simulator, "edge_end")

            prefix = torch.cat((prefix, t), dim=1)
            new_generated_token = prefix[:, prefix_len:]

            # 第二层 speculative (Draft -> Target) with CUHLM Uncertainty
            draft_transfer_top_k, draft_uncertainty_threshold = (
                self._select_cuhlm_stage_config(
                    stage="draft_to_target",
                    transfer_top_k=transfer_top_k,
                    uncertainty_threshold=comm_simulator.uncertainty_threshold,
                )
            )

            # F-CUHLM（论文口径）：首轮 prompt 上传保留（全仓一次性约定，
            # 论文不建模 prompt）；此后逐轮的 token 同步**不再单独上行**——
            # 新 token 索引随触发轮的上行捎带（negligible，§II-B），全跳过
            # 轮对云端零通信（旧口径：每轮一次 token 上行 = 每轮一个 NTT）。
            if idx == 1:
                comm_simulator.transfer(prefix, None, "edge_cloud")

            t0 = time.time()
            x, draft_rebuilt_probs, draft_rebuilt_meta, _ = self._generate_with_optional_rebuilt_proposal(
                draft_model_cache,
                _move_token_tensor(prefix, draft_device),
                self.args.gamma1,
                proposal_top_k(draft_transfer_top_k),
                adapter=None,
                need_topk_metadata=False,
            )
            draft_comp_time += time.time() - t0
            _validate_token_range(
                x,
                vocab_size=self.vocab_size,
                label="cee_cuhlm.draft_stage.proposal_x",
            )

            actual_gamma1 = x.shape[1] - prefix.shape[1]
            total_gamma = new_generated_token.shape[1] + actual_gamma1

            draft_model_forward_times += actual_gamma1
            total_draft_model_generated_tokens += total_gamma

            # Compute draft_stage_probs BEFORE target call (only needs draft_model_cache)
            draft_accepted_this_iter = 0
            draft_stage_probs = stage_prob_history(
                draft_model_cache,
                prefix_len + new_generated_token.shape[1],
                draft_rebuilt_probs,
            )

            # CUHLM Uncertainty Check Loop (BEFORE target call — as in the paper)
            original_threshold = comm_simulator.uncertainty_threshold
            comm_simulator.uncertainty_threshold = draft_uncertainty_threshold
            reject_offset: Optional[int] = None
            effective_gamma = min(
                total_gamma,
                draft_stage_probs.shape[1] - prefix_len,
            )
            n2 = prefix_len + effective_gamma - 1
            for i in range(effective_gamma):
                logit_idx = prefix_len + i - 1
                assert draft_model_cache.logits_history is not None, (
                    "Draft model logits history is None"
                )
                current_logit = draft_model_cache.logits_history[
                    :, logit_idx, : self.vocab_size
                ]
                current_token_id = int(x[:, prefix_len + i].item())

                uncertainty = comm_simulator.calculate_uncertainty(
                    current_logit, M=20, theta_max=2.0, draft_token=current_token_id
                )

                should_transfer, vocab_size = (
                    comm_simulator.determine_transfer_strategy(
                        uncertainty,
                        draft_stage_probs[:, prefix_len - 1 + i, : self.vocab_size],
                    )
                )

                if should_transfer:
                    # F-CUHLM（论文口径）：触发 = 一次上行，载荷 k(t)·(b_prob+
                    # b_index) bits（cuhlm_uplink_payload_bytes）；draft token
                    # 索引 negligible。旧口径：vocab_size×4+8 字节 + 独立
                    # reject 消息（6B + 一次 NTT）。
                    comm_simulator.simulate_transfer(
                        cuhlm_uplink_payload_bytes(int(vocab_size), self.vocab_size),
                        "edge_cloud",
                        topk=int(vocab_size),
                        draft_len=1,
                    )
                    reject_offset = i
                    n2 = prefix_len + i - 1
                    break
                else:
                    # F-CUHLM（论文口径）：跳过 = 零通信（旧口径：8B 上行 +
                    # accept 消息 6B+NTT，把论文的"跳过=免费"变成 2 RTT/token，
                    # 在 NTT 主导场景下系统性压死该基线）。
                    draft_accepted_this_iter += 1
            comm_simulator.uncertainty_threshold = original_threshold

            total_draft_model_accepted_tokens += draft_accepted_this_iter

            assert n2 >= prefix_len - 1
            prefix = x[:, : n2 + 1]

            if reject_offset is not None:
                # High uncertainty: run target model for verification
                queuing_time += batch_delay
                t0 = time.time()
                _ = target_model_cache.generate(
                    _move_token_tensor(x, target_device), 1
                )
                target_comp_time += time.time() - t0
                target_model_forward_times += 1

                verification_inputs = prepare_verification_inputs(
                    draft_model_cache=draft_model_cache,
                    target_model_cache=target_model_cache,
                    x=x,
                    prefix_len=prefix_len,
                    gamma=total_gamma,
                    draft_probs_override=draft_stage_probs,
                )

                # F-CUHLM（论文口径）：触发分布已在上面的一次上行里计费
                # （k(t)·(b_prob+b_index)），这里不再二次计费。旧口径：
                # prob_bytes 按全行/压缩行再计一次 + reject_overhead 魔数 6B。
                new_generated_token = prefix[:, prefix_len:]

                n2, t, _ = _finalize_cuhlm_verification(
                    proposer_cache=draft_model_cache,
                    verifier_cache=target_model_cache,
                    verification_inputs=verification_inputs,
                    x=x,
                    prefix_len=prefix_len,
                    accepted_count=draft_accepted_this_iter,
                    reject_offset=reject_offset,
                    output_device=draft_device,
                )
            else:
                # All tokens pass CUHLM check: truly skip target model
                # target_model_forward_times NOT incremented (target genuinely not called)
                new_generated_token = prefix[:, prefix_len:]

                # Sample bonus token from draft model (target not available)
                bonus_probs = draft_stage_probs[:, n2, : self.vocab_size]
                t = sample_accept_token(bonus_probs, output_device=draft_device)

                # Rollback draft model to n2+1 for proper KVCache continuation
                draft_model_cache.rollback(n2 + 1)

            prefix = x[:, : n2 + 1]
            if n2 <= little_model_cache.current_length:
                little_model_cache.rollback(n2 + 1)

            prefix = torch.cat((prefix, t), dim=1)
            _validate_token_range(
                prefix,
                vocab_size=self.vocab_size,
                label="cee_cuhlm.edge_cloud.prefix_after_concat",
            )

            if reject_offset is not None:
                # F-CUHLM（论文口径）：触发轮的云端响应（cloud→edge）是
                # negligible 的 token 索引——0 字节报文，保留一次 NTT。
                # 全跳过轮对云端零通信（旧口径：无条件 INT_SIZE+token+
                # 二次计费的 prob_bytes + 魔数 6B，且每轮一次 NTT）。
                _send_downlink_index_only(comm_simulator, "edge_cloud")
            # 轮末响应回传设备（edge→end）：token 索引 negligible（§II-B），
            # 0 字节报文，保留一次 NTT（旧口径：INT_SIZE + token 字节）。
            _send_downlink_index_only(comm_simulator, "edge_end")

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        wall_time += elapsed_time
        # 遵守 max_tokens：投机解码按整块追加，最后一轮可能多出若干 token
        # （F33 契约）。参照 dist_spec 的 remaining-1 截断，这里显式截断——
        # 否则会多拿 token，使配对质量比较与时延/吞吐统计都不公平。
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, :max_tokens]
        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        wall_time += (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )

        metrics = get_empty_metrics()
        metrics["little_forward_times"] = little_model_forward_times
        metrics["draft_forward_times"] = draft_model_forward_times
        metrics["target_forward_times"] = target_model_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["little_generated_tokens"] = total_little_model_generated_tokens
        metrics["draft_generated_tokens"] = total_draft_model_generated_tokens
        metrics["little_accepted_tokens"] = total_little_model_accepted_tokens
        metrics["draft_accepted_tokens"] = total_draft_model_accepted_tokens
        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = wall_time + queuing_time
        metrics["throughput"] = (
            metrics["generated_tokens"] / metrics["wall_time"]
            if metrics["wall_time"] > 0
            else 0
        )
        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time,
            queuing_time=queuing_time,
            little_comp_time=little_comp_time,
            draft_comp_time=draft_comp_time,
            target_comp_time=target_comp_time,
        )
        metrics["communication_time"] = (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times
        metrics["arp_overhead_time"] = arp_overhead_time
        metrics["dra_overhead_time"] = dra_overhead_time

        if self.rl_adapter is not None:
            dra_start = time.time()
            self.rl_adapter.save(metrics.get("throughput"))
            metrics["dra_overhead_time"] += time.time() - dra_start
        if self.little_rl_adapter is not None:
            dra_start = time.time()
            self.little_rl_adapter.save(metrics.get("throughput"))
            metrics["dra_overhead_time"] += time.time() - dra_start

        # 复制 edge-cloud 的带宽、top-k 和起草长度历史数据
        metrics["edge_cloud_bandwidth_history"] = (
            comm_simulator.edge_cloud_bandwidth_history.copy()
        )
        # 逐消息记账记录（[字节, 轮号]）——离线重放（scripts/rebill.py）的
        # 输入：换带宽/NTT/地板/口径的敏感性分析不用重跑 GPU。
        # getattr 兜底：测试替身可能只实现聚合口径。
        metrics["comm_trace_edge_cloud"] = getattr(
            comm_simulator, "edge_cloud_trace", []
        )
        _add_comm_accounting_metrics(metrics, self.args, comm_simulator)
        metrics["edge_cloud_topk_history"] = (
            comm_simulator.edge_cloud_topk_history.copy()
        )
        metrics["edge_cloud_draft_len_history"] = (
            comm_simulator.edge_cloud_draft_len_history.copy()
        )

        return prefix, metrics

    @Register.register_decoding("cee_dssd")
    @torch.no_grad()
    def cee_dssd(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud=10,
        ntt_ms_edge_end=1,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """
        CEE-DSSD: Combined Edge-End Distributed Split Speculative Decoding (3-layer).
        Uses serial verification similar to DSSD but in a 3-layer architecture.
        """
        max_tokens = prefix.shape[1] + self.args.max_tokens
        little_device = self.get_model_input_device(self.little_model)
        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        # 使用 transfer_top_k 作为草稿模型的 top-k 压缩参数
        draft_top_k = (
            transfer_top_k
            if (transfer_top_k is not None and transfer_top_k > 0)
            else self.args.top_k
        )

        # B19：little/draft/target 都做多 token 验证前向 ⇒ 三个都开图并跨样本
        # 复用；此前只给 little/draft 单步图、无验证档位、且每样本重捕获。
        graph_kw = _graph_mode_cache_kwargs(
            self.args,
            cap=int(getattr(self.args, "gamma1", 1))
            + int(getattr(self.args, "gamma2", 1))
            + 4,
        )
        caches = self._acquire_three_layer_caches(
            "_cee_dssd_caches", graph_kw, draft_top_k
        )
        little_model_cache = caches["little"]
        draft_model_cache = caches["draft"]
        target_model_cache = caches["target"]

        if use_precise_comm_sim:
            comm_simulator: CommunicationSimulator = PreciseCommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=self.args.edge_end_bandwidth,
                bandwidth_cloud_end=self.args.cloud_end_bandwidth,
                transfer_top_k=transfer_top_k,
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=bool(getattr(self.args, "stochastic_ntt", False)),
            )

        # 统一计费口径（docs/protocol.md §3）：与 adaptive_tridecoding 同一套
        # 开关。CUHLM 系（uncertainty_decoding / cee_cuhlm）按决策暂不接。
        # 开关名必须出现在本方法代码里（protocols.py 运行时内省的依据）。
        comm_simulator.coalesce_rounds = (
            str(getattr(self.args, "comm_round_trip_mode", "per_transfer"))
            == "per_round"
        )
        charge_residual = bool(getattr(self.args, "charge_residual_payload", False))
        _topk_cap = int(getattr(self.args, "transfer_top_k_cap", 0) or 0)
        transfer_top_k = apply_transfer_top_k_cap(transfer_top_k, _topk_cap)
        comm_simulator.transfer_top_k = transfer_top_k

        # Metrics tracking
        little_model_forward_times = 0
        draft_model_forward_times = 0
        target_model_forward_times = 0
        total_little_model_generated_tokens = 0
        total_draft_model_generated_tokens = 0
        total_little_model_accepted_tokens = 0
        total_draft_model_accepted_tokens = 0
        queuing_time = 0
        batch_delay = getattr(self.args, "batch_delay", 0)
        wall_time = 0

        idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        start_event.record(stream=torch.cuda.current_stream())

        comm_simulator.transfer(prefix, None, "edge_end")

        _tri_prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点

        while prefix.shape[1] < max_tokens:

            # B10：循环内 EOS 早停（与 dssd/adaptive_* 已接线方法语义对齐——
            # 计算与通信在 EOS 后立即停，时延口径跨方法可比）
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            idx += 1
            comm_simulator.set_round(idx)
            prefix_len = prefix.shape[1]
            current_proposal_top_k = proposal_top_k(transfer_top_k)
            little_stage_probs: Optional[torch.Tensor] = None
            draft_stage_probs: Optional[torch.Tensor] = None

            # --- Layer 1: Little -> Draft (Serial) ---
            x, little_rebuilt_probs, little_rebuilt_meta, _ = self._generate_with_optional_rebuilt_proposal(
                little_model_cache,
                prefix.to(little_device),
                self.args.gamma2,
                current_proposal_top_k,
                need_topk_metadata=True,
            )
            # Sync with Draft Model
            _ = draft_model_cache.generate(x.to(draft_device), 1)

            little_model_forward_times += self.args.gamma2
            draft_model_forward_times += 1
            total_little_model_generated_tokens += self.args.gamma2

            n1: int = prefix_len + self.args.gamma2 - 1

            if self.args.gamma2 > 0:
                little_stage_probs = stage_prob_history(
                    little_model_cache,
                    prefix_len,
                    little_rebuilt_probs,
                )
                draft_tokens, draft_probs = collect_verification_payload(
                    little_stage_probs,
                    x,
                    prefix_len,
                    self.args.gamma2,
                )
                comm_simulator.transfer(draft_tokens, draft_probs, "edge_end")
                little_stage_kwargs = {
                    "proposer_cache": little_model_cache,
                    "verifier_cache": draft_model_cache,
                    "x": x,
                    "prefix_len": prefix_len,
                    "gamma": self.args.gamma2,
                    "output_device": little_device,
                    "draft_probs_override": cast(torch.Tensor, little_stage_probs),
                }
                little_topk_history = stage_topk_proposal_history(
                    little_rebuilt_meta,
                    self.args.gamma2,
                )
                if little_topk_history is not None:
                    little_stage_kwargs["draft_topk_history"] = little_topk_history
                (
                    little_accepted_this_iter,
                    n1,
                    t,
                    little_all_accepted,
                ) = resolve_stage_verification(**little_stage_kwargs)
                if not little_all_accepted:
                    comm_simulator.send_reject_message("edge_end")
            else:
                little_accepted_this_iter = 0
                t = sample_accept_token(
                    draft_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=little_device,
                )
                little_all_accepted = True

            total_little_model_accepted_tokens += little_accepted_this_iter

            assert n1 >= prefix_len - 1
            prefix = x[:, : n1 + 1]

            if not little_all_accepted:
                # Reject, resample from Draft
                comm_simulator.transfer(
                    None,
                    cast(torch.Tensor, little_stage_probs)[:, n1, : self.vocab_size],
                    "edge_end",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )
                if charge_residual:
                    # 统一口径（§3.4）：压缩行已含 k×(4+元素)，补尾部标量。
                    comm_simulator.simulate_transfer(
                        reject_tail_scalar_bytes(
                            cast(torch.Tensor, little_stage_probs)[
                                :, n1, : self.vocab_size
                            ],
                            transfer_top_k,
                        ),
                        "edge_end",
                    )

            # Transfer sampled token index back
            _send_downlink_token(comm_simulator, t, "edge_end")

            prefix = torch.cat((prefix, t), dim=1)

            # New tokens generated in Layer 1
            new_generated_token_layer1 = prefix[:, prefix_len:]

            # --- Layer 2: Draft -> Target (Serial) ---
            # Pre-launch GPU work (draft + target) to overlap with CPU comm simulation

            x, draft_rebuilt_probs, draft_rebuilt_meta, _ = self._generate_with_optional_rebuilt_proposal(
                draft_model_cache,
                prefix.to(draft_device),
                self.args.gamma1,
                current_proposal_top_k,
                need_topk_metadata=True,
            )

            queuing_time += batch_delay
            # Sync with Target Model (runs on GPU, overlaps with CPU comm below)
            _ = target_model_cache.generate(x.to(target_device), 1)

            # Communication simulation (pure CPU): overlaps with target GPU forward above
            if idx == 1:
                comm_simulator.transfer(prefix, None, "edge_cloud")
            else:
                comm_simulator.transfer(new_generated_token_layer1, None, "edge_cloud")

            draft_model_forward_times += self.args.gamma1
            target_model_forward_times += 1
            total_draft_model_generated_tokens += (
                new_generated_token_layer1.shape[1] + self.args.gamma1
            )

            n2: int = (
                prefix_len + new_generated_token_layer1.shape[1] + self.args.gamma1 - 1
            )
            draft_accepted_this_iter = 0

            # Iterate over both new tokens from Layer 1 and Draft speculation
            total_gamma_layer2 = new_generated_token_layer1.shape[1] + self.args.gamma1

            if total_gamma_layer2 > 0:
                draft_stage_probs = stage_prob_history(
                    draft_model_cache,
                    prefix_len + new_generated_token_layer1.shape[1],
                    draft_rebuilt_probs,
                )
                draft_tokens_second, draft_probs_second = collect_verification_payload(
                    draft_stage_probs,
                    x,
                    prefix_len,
                    total_gamma_layer2,
                )
                comm_simulator.transfer(
                    draft_tokens_second,
                    draft_probs_second,
                    "edge_cloud",
                )
                draft_stage_kwargs = {
                    "proposer_cache": draft_model_cache,
                    "verifier_cache": target_model_cache,
                    "x": x,
                    "prefix_len": prefix_len,
                    "gamma": total_gamma_layer2,
                    "output_device": draft_device,
                    "draft_probs_override": cast(torch.Tensor, draft_stage_probs),
                }
                prefix_topk_history_dssd = None
                if new_generated_token_layer1.shape[1] > 0:
                    prefix_prob_rows = draft_model_cache.prob_history[
                        :,
                        prefix_len - 1 : prefix_len - 1 + new_generated_token_layer1.shape[1],
                        :,
                    ]
                    prefix_topk_history_dssd = build_stage_prefix_topk_history(
                        prefix_prob_rows,
                        current_proposal_top_k,
                    )
                draft_stage_topk_history = merge_stage_topk_histories(
                    prefix_topk_history_dssd,
                    stage_topk_proposal_history(
                        draft_rebuilt_meta,
                        self.args.gamma1,
                    ),
                )
                draft_stage_topk_history = stage_topk_proposal_history(
                    draft_stage_topk_history,
                    total_gamma_layer2,
                )
                if draft_stage_topk_history is not None:
                    draft_stage_kwargs["draft_topk_history"] = draft_stage_topk_history
                (
                    draft_accepted_this_iter,
                    n2,
                    t,
                    draft_all_accepted,
                ) = resolve_stage_verification(**draft_stage_kwargs)
                if not draft_all_accepted:
                    comm_simulator.send_reject_message("edge_cloud")
            else:
                draft_accepted_this_iter = 0
                t = sample_accept_token(
                    target_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=draft_device,
                )
                draft_all_accepted = True

            total_draft_model_accepted_tokens += draft_accepted_this_iter

            assert n2 >= prefix_len - 1
            prefix = x[:, : n2 + 1]
            if n2 <= little_model_cache.current_length:
                little_model_cache.rollback(n2 + 1)

            if not draft_all_accepted:
                # Reject, resample from Target
                comm_simulator.transfer(
                    None,
                    cast(torch.Tensor, draft_stage_probs)[:, n2, : self.vocab_size],
                    "edge_cloud",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )
                if charge_residual:
                    # 统一口径（§3.4）：同阶段一，补尾部标量。
                    comm_simulator.simulate_transfer(
                        reject_tail_scalar_bytes(
                            cast(torch.Tensor, draft_stage_probs)[
                                :, n2, : self.vocab_size
                            ],
                            transfer_top_k,
                        ),
                        "edge_cloud",
                    )

            prefix = torch.cat((prefix, t), dim=1)

            # Transfer index back
            _send_downlink_token(comm_simulator, t, "edge_cloud")
            _send_downlink_token(comm_simulator, t, "edge_end")

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        comm_simulator.flush_round()  # 结算最后一轮（按轮合并模式下必须）
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        wall_time += elapsed_time
        # 遵守 max_tokens：投机解码按整块追加，最后一轮可能多出若干 token
        # （F33 契约）。参照 dist_spec 的 remaining-1 截断，这里显式截断——
        # 否则会多拿 token，使配对质量比较与时延/吞吐统计都不公平。
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, :max_tokens]
        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        wall_time += (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )

        metrics = get_empty_metrics()
        metrics["little_forward_times"] = little_model_forward_times
        metrics["draft_forward_times"] = draft_model_forward_times
        metrics["target_forward_times"] = target_model_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["little_generated_tokens"] = total_little_model_generated_tokens
        metrics["draft_generated_tokens"] = total_draft_model_generated_tokens
        metrics["little_accepted_tokens"] = total_little_model_accepted_tokens
        metrics["draft_accepted_tokens"] = total_draft_model_accepted_tokens
        metrics["wall_time"] = wall_time
        metrics["throughput"] = (
            metrics["generated_tokens"] / wall_time if wall_time > 0 else 0
        )
        metrics["communication_time"] = (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times

        metrics["queuing_time"] = target_model_forward_times * batch_delay
        metrics["wall_time"] += metrics["queuing_time"]
        if metrics["wall_time"] > 0:
            metrics["throughput"] = metrics["generated_tokens"] / metrics["wall_time"]

        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time,
            queuing_time=queuing_time,
        )

        # 复制 edge-cloud 的带宽、top-k 和起草长度历史数据
        metrics["edge_cloud_bandwidth_history"] = (
            comm_simulator.edge_cloud_bandwidth_history.copy()
        )
        # 逐消息记账记录（[字节, 轮号]）——离线重放（scripts/rebill.py）的
        # 输入：换带宽/NTT/地板/口径的敏感性分析不用重跑 GPU。
        # getattr 兜底：测试替身可能只实现聚合口径。
        metrics["comm_trace_edge_cloud"] = getattr(
            comm_simulator, "edge_cloud_trace", []
        )
        _add_comm_accounting_metrics(metrics, self.args, comm_simulator)
        metrics["edge_cloud_topk_history"] = (
            comm_simulator.edge_cloud_topk_history.copy()
        )
        metrics["edge_cloud_draft_len_history"] = (
            comm_simulator.edge_cloud_draft_len_history.copy()
        )

        return prefix, metrics

    @Register.register_decoding("cee_dsd")
    @torch.no_grad()
    def cee_dsd(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        use_stochastic_comm: bool = False,
        ntt_ms_edge_cloud=10,
        ntt_ms_edge_end=1,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """
        CEE-DSD: Combined Edge-End Distributed Speculative Decoding (3-layer).
        Uses parallel/batch verification (send all probs at once) similar to DSD.
        """
        max_tokens = prefix.shape[1] + self.args.max_tokens
        little_device = self.get_model_input_device(self.little_model)
        draft_device = self.get_model_input_device(self.draft_model)
        target_device = self.get_model_input_device(self.target_model)

        draft_top_k = (
            transfer_top_k
            if (transfer_top_k is not None and transfer_top_k > 0)
            else self.args.top_k
        )

        # B19：little/draft/target 都做多 token 验证前向 ⇒ 三个都开图并跨样本
        # 复用；此前只给 little/draft 单步图、无验证档位、且每样本重捕获。
        graph_kw = _graph_mode_cache_kwargs(
            self.args,
            cap=int(getattr(self.args, "gamma1", 1))
            + int(getattr(self.args, "gamma2", 1))
            + 4,
        )
        caches = self._acquire_three_layer_caches(
            "_cee_dsd_caches", graph_kw, draft_top_k
        )
        little_model_cache = caches["little"]
        draft_model_cache = caches["draft"]
        target_model_cache = caches["target"]

        if use_precise_comm_sim:
            comm_simulator: CommunicationSimulator = PreciseCommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CommunicationSimulator(
                min_bandwidth_mbps=getattr(self.args, "min_bandwidth_mbps", 5.0),
                bw_model=str(getattr(self.args, "comm_bw_model", "instant")),
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=self.args.edge_end_bandwidth,
                bandwidth_cloud_end=self.args.cloud_end_bandwidth,
                transfer_top_k=transfer_top_k,
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
                use_stochastic=use_stochastic_comm,
                stochastic_ntt=bool(getattr(self.args, "stochastic_ntt", False)),
            )

        # 统一计费口径（docs/protocol.md §3）：与 adaptive_tridecoding 同一套
        # 开关。CUHLM 系（uncertainty_decoding / cee_cuhlm）按决策暂不接。
        # 开关名必须出现在本方法代码里（protocols.py 运行时内省的依据）。
        comm_simulator.coalesce_rounds = (
            str(getattr(self.args, "comm_round_trip_mode", "per_transfer"))
            == "per_round"
        )
        charge_residual = bool(getattr(self.args, "charge_residual_payload", False))
        _topk_cap = int(getattr(self.args, "transfer_top_k_cap", 0) or 0)
        transfer_top_k = apply_transfer_top_k_cap(transfer_top_k, _topk_cap)
        comm_simulator.transfer_top_k = transfer_top_k

        # Metrics tracking
        little_model_forward_times = 0
        draft_model_forward_times = 0
        target_model_forward_times = 0
        total_little_model_generated_tokens = 0
        total_draft_model_generated_tokens = 0
        total_little_model_accepted_tokens = 0
        total_draft_model_accepted_tokens = 0
        queuing_time = 0
        batch_delay = getattr(self.args, "batch_delay", 0)
        wall_time = 0

        idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        # Pre-allocate prefix buffer to avoid O(n²) torch.cat during generation
        # x can grow up to max_tokens + gamma2 + gamma1 + 1 inside the loop
        buffer_size = max_tokens + self.args.gamma1 + self.args.gamma2 + 1
        prefix_buffer = torch.empty(
            1, buffer_size, dtype=prefix.dtype, device=prefix.device
        )
        prefix_len_tracker = prefix.shape[1]
        prefix_buffer[:, :prefix_len_tracker] = prefix
        prefix = prefix_buffer[:, :prefix_len_tracker]

        start_event.record(stream=torch.cuda.current_stream())

        comm_simulator.transfer(prefix, None, "edge_end")

        _tri_prompt_len = prefix.shape[1]  # B10：EOS 早停的生成段起点

        while prefix.shape[1] < max_tokens:

            # B10：循环内 EOS 早停（与 dssd/adaptive_* 已接线方法语义对齐——
            # 计算与通信在 EOS 后立即停，时延口径跨方法可比）
            prefix, _eos_hit = self._stop_at_eos(prefix, _tri_prompt_len)
            if _eos_hit:
                break
            idx += 1
            comm_simulator.set_round(idx)
            prefix_len = prefix.shape[1]
            current_proposal_top_k = proposal_top_k(transfer_top_k)

            # --- Layer 1: Little -> Draft (Parallel) ---
            if current_proposal_top_k is not None and hasattr(
                little_model_cache, "generate_with_topk_metadata_only"
            ):
                x, little_topk_history = little_model_cache.generate_with_topk_metadata_only(
                    prefix.to(little_device),
                    self.args.gamma2,
                    current_proposal_top_k,
                )
            else:
                x, little_rebuilt_probs = self._generate_with_optional_rebuilt_proposal(
                    little_model_cache,
                    prefix.to(little_device),
                    self.args.gamma2,
                    current_proposal_top_k,
                    need_topk_metadata=False,
                )[:2]
                little_topk_history = None

            # Launch draft verification on GPU immediately (overlaps with CPU comm below)
            _ = draft_model_cache.generate(x.to(draft_device), 1)

            # Communication simulation (pure CPU): overlaps with draft.generate GPU above
            comm_simulator.transfer(x, None, "edge_end")

            little_model_forward_times += self.args.gamma2
            draft_model_forward_times += 1
            total_little_model_generated_tokens += self.args.gamma2

            n1: int = prefix_len + self.args.gamma2 - 1

            little_accepted_this_iter = 0
            if self.args.gamma2 > 0:
                l1_actual_gamma = min(self.args.gamma2, x.shape[1] - prefix_len)
                if l1_actual_gamma <= 0:
                    little_accepted_this_iter = 0
                    n1 = prefix_len - 1
                    t = sample(
                        draft_model_cache.prob_history[
                            :, -1, : draft_model_cache.vocab_size
                        ]
                    ).to(little_device)
                    little_all_accepted = True
                    little_model_cache.rollback(n1 + 1)
                    draft_model_cache.rollback(n1 + 2)
                else:
                    _simulate_topk_prob_transfer(
                        comm_simulator,
                        link_type="edge_end",
                        draft_len=l1_actual_gamma,
                        transfer_top_k=transfer_top_k,
                        prob_dtype=draft_model_cache.prob_history.dtype,
                    )
                    little_stage_kwargs = {
                        "proposer_cache": little_model_cache,
                        "verifier_cache": draft_model_cache,
                        "x": x,
                        "prefix_len": prefix_len,
                        "gamma": self.args.gamma2,
                        "output_device": little_device,
                    }
                    if little_topk_history is not None:
                        little_stage_kwargs["draft_topk_history"] = little_topk_history
                    (
                        little_accepted_this_iter,
                        n1,
                        t,
                        little_all_accepted,
                    ) = resolve_stage_verification(**little_stage_kwargs)
                    if not little_all_accepted:
                        comm_simulator.send_reject_message("edge_end")
                        if charge_residual:
                            # 统一口径（§3.4）：legacy 只发 6B 拒绝信号，拒绝
                            # 位置的残差分布被无声省掉；honest 补
                            # k*(4+元素)+元素（行取验证方 draft 在 n1 的分布）。
                            comm_simulator.simulate_transfer(
                                reject_residual_payload_bytes(
                                    draft_model_cache.prob_history[
                                        :, n1, : self.vocab_size
                                    ],
                                    transfer_top_k,
                                ),
                                "edge_end",
                            )
            else:
                t = sample_accept_token(
                    draft_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=little_device,
                )
                little_all_accepted = True

            total_little_model_accepted_tokens += little_accepted_this_iter

            assert n1 >= prefix_len - 1
            n1_plus1 = n1 + 1
            prefix_buffer[:, :n1_plus1] = x[:, :n1_plus1]
            prefix_len_tracker = n1_plus1

            # Transfer sampled token index + token back (merged)
            _send_downlink_token(comm_simulator, t, "edge_end")

            prefix_buffer[:, prefix_len_tracker : prefix_len_tracker + t.shape[1]] = t
            prefix_len_tracker += t.shape[1]
            prefix = prefix_buffer[:, :prefix_len_tracker]

            new_generated_token_layer1 = prefix[:, prefix_len:]

            # --- Layer 2: Draft -> Target (Parallel) ---
            # Pre-launch GPU work (draft) to overlap with CPU comm simulation

            if current_proposal_top_k is not None and hasattr(
                draft_model_cache, "generate_with_topk_metadata_only"
            ):
                x, draft_topk_history = draft_model_cache.generate_with_topk_metadata_only(
                    prefix.to(draft_device),
                    self.args.gamma1,
                    current_proposal_top_k,
                )
            else:
                x, draft_rebuilt_probs = self._generate_with_optional_rebuilt_proposal(
                    draft_model_cache,
                    prefix.to(draft_device),
                    self.args.gamma1,
                    current_proposal_top_k,
                    need_topk_metadata=False,
                )[:2]
                draft_topk_history = None

            # Transfer generated tokens by draft model (CPU: overlaps with GPU above)
            speculated_tokens = x[:, -self.args.gamma1 :]

            # Communication simulation (pure CPU): overlaps with draft.generate GPU above
            if idx == 1:
                comm_simulator.transfer(prefix, None, "edge_cloud")
            else:
                comm_simulator.transfer(new_generated_token_layer1, None, "edge_cloud")
            comm_simulator.transfer(speculated_tokens, None, "edge_cloud")

            queuing_time += batch_delay
            # Sync with Target Model (runs on GPU while CPU does verification prep below)
            _ = target_model_cache.generate(x.to(target_device), 1)

            draft_model_forward_times += self.args.gamma1
            target_model_forward_times += 1
            total_draft_model_generated_tokens += (
                new_generated_token_layer1.shape[1] + self.args.gamma1
            )

            total_gamma_layer2 = new_generated_token_layer1.shape[1] + self.args.gamma1
            n2: int = prefix_len + total_gamma_layer2 - 1

            draft_accepted_this_iter = 0

            if total_gamma_layer2 > 0:
                l2_actual_gamma = min(total_gamma_layer2, x.shape[1] - prefix_len)
                if l2_actual_gamma <= 0:
                    draft_accepted_this_iter = 0
                    n2 = prefix_len - 1
                    t = sample(
                        target_model_cache.prob_history[
                            :, -1, : target_model_cache.vocab_size
                        ]
                    ).to(draft_device)
                    draft_all_accepted = True
                    draft_model_cache.rollback(n2 + 1)
                    target_model_cache.rollback(n2 + 2)
                else:
                    _simulate_topk_prob_transfer(
                        comm_simulator,
                        link_type="edge_cloud",
                        draft_len=l2_actual_gamma,
                        transfer_top_k=transfer_top_k,
                        prob_dtype=target_model_cache.prob_history.dtype,
                    )
                    prefix_topk_history = None
                    if new_generated_token_layer1.shape[1] > 0:
                        prefix_prob_rows = draft_model_cache.prob_history[
                            :,
                            prefix_len - 1 : prefix_len - 1 + new_generated_token_layer1.shape[1],
                            :,
                        ]
                        prefix_topk_history = build_stage_prefix_topk_history(
                            prefix_prob_rows,
                            current_proposal_top_k,
                        )
                    draft_stage_topk_history = merge_stage_topk_histories(
                        prefix_topk_history,
                        draft_topk_history,
                    )
                    draft_stage_topk_history = stage_topk_proposal_history(
                        draft_stage_topk_history,
                        total_gamma_layer2,
                    )
                    draft_stage_kwargs = {
                        "proposer_cache": draft_model_cache,
                        "verifier_cache": target_model_cache,
                        "x": x,
                        "prefix_len": prefix_len,
                        "gamma": total_gamma_layer2,
                        "output_device": draft_device,
                    }
                    if draft_stage_topk_history is not None:
                        draft_stage_kwargs["draft_topk_history"] = draft_stage_topk_history
                    (
                        draft_accepted_this_iter,
                        n2,
                        t,
                        draft_all_accepted,
                    ) = resolve_stage_verification(**draft_stage_kwargs)
                    if not draft_all_accepted:
                        comm_simulator.send_reject_message("edge_cloud")
                        if charge_residual:
                            # 统一口径（§3.4）：同阶段一，残差取验证方 target
                            # 在 n2 的分布。
                            comm_simulator.simulate_transfer(
                                reject_residual_payload_bytes(
                                    target_model_cache.prob_history[
                                        :, n2, : self.vocab_size
                                    ],
                                    transfer_top_k,
                                ),
                                "edge_cloud",
                            )
            else:
                t = sample_accept_token(
                    target_model_cache.prob_history[:, -1, : self.vocab_size],
                    output_device=draft_device,
                )
                draft_all_accepted = True

            total_draft_model_accepted_tokens += draft_accepted_this_iter

            assert n2 >= prefix_len - 1
            n2_plus1 = n2 + 1
            prefix_buffer[:, :n2_plus1] = x[:, :n2_plus1]
            prefix_len_tracker = n2_plus1
            if n2 <= little_model_cache.current_length:
                little_model_cache.rollback(n2_plus1)

            prefix_buffer[:, prefix_len_tracker : prefix_len_tracker + t.shape[1]] = t
            prefix_len_tracker += t.shape[1]
            prefix = prefix_buffer[:, :prefix_len_tracker]

            # Transfer index + token back to both links (merged)
            _send_downlink_token(comm_simulator, t, "edge_cloud")
            _send_downlink_token(comm_simulator, t, "edge_end")

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        comm_simulator.flush_round()  # 结算最后一轮（按轮合并模式下必须）
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        wall_time += elapsed_time
        # 遵守 max_tokens：投机解码按整块追加，最后一轮可能多出若干 token
        # （F33 契约）。参照 dist_spec 的 remaining-1 截断，这里显式截断——
        # 否则会多拿 token，使配对质量比较与时延/吞吐统计都不公平。
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, :max_tokens]
        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        wall_time += (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )

        metrics = get_empty_metrics()
        metrics["little_forward_times"] = little_model_forward_times
        metrics["draft_forward_times"] = draft_model_forward_times
        metrics["target_forward_times"] = target_model_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["little_generated_tokens"] = total_little_model_generated_tokens
        metrics["draft_generated_tokens"] = total_draft_model_generated_tokens
        metrics["little_accepted_tokens"] = total_little_model_accepted_tokens
        metrics["draft_accepted_tokens"] = total_draft_model_accepted_tokens
        metrics["wall_time"] = wall_time
        metrics["throughput"] = (
            metrics["generated_tokens"] / wall_time if wall_time > 0 else 0
        )
        metrics["communication_time"] = (
            comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time
        )
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times

        metrics["queuing_time"] = target_model_forward_times * batch_delay
        metrics["wall_time"] += metrics["queuing_time"]
        if metrics["wall_time"] > 0:
            metrics["throughput"] = metrics["generated_tokens"] / metrics["wall_time"]

        _add_per_model_wall_time(
            metrics,
            elapsed_time=elapsed_time,
            comm_time=comm_simulator.edge_cloud_comm_time + comm_simulator.edge_end_comm_time,
            queuing_time=queuing_time,
        )

        # 复制 edge-cloud 的带宽、top-k 和起草长度历史数据
        metrics["edge_cloud_bandwidth_history"] = (
            comm_simulator.edge_cloud_bandwidth_history.copy()
        )
        # 逐消息记账记录（[字节, 轮号]）——离线重放（scripts/rebill.py）的
        # 输入：换带宽/NTT/地板/口径的敏感性分析不用重跑 GPU。
        # getattr 兜底：测试替身可能只实现聚合口径。
        metrics["comm_trace_edge_cloud"] = getattr(
            comm_simulator, "edge_cloud_trace", []
        )
        _add_comm_accounting_metrics(metrics, self.args, comm_simulator)
        metrics["edge_cloud_topk_history"] = (
            comm_simulator.edge_cloud_topk_history.copy()
        )
        metrics["edge_cloud_draft_len_history"] = (
            comm_simulator.edge_cloud_draft_len_history.copy()
        )

        return prefix, metrics
