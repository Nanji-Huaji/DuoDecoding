"""从 baselines.py 抽出的模块级 helper（S1）。

只允许依赖叶子模块；不 import src.baselines。baselines.py 通过
``from .baselines_helpers import (...)`` re-export，保持 src.baselines.X 兼容。
"""

from __future__ import annotations

import json
import math
import os
from typing import Callable, List, Optional, Tuple, cast

import torch

from .SpecDec_pp.specdec_pp.wrap_model import AcceptancePredictionHead
from .acc_head_registry import anchor_repo_path, is_usable_acc_head_dir
from .comm_accounting import CommEnv
from .communication import (
    PROB_QUANT_PASS_THROUGH_BITS,
    CommunicationSimulator,
    PreciseCommunicationSimulator,
)
from .decoding_ops import (
    apply_rollback,
    build_rollback_plan,
    sample_accept_token,
    sample_reject_token,
)
from .metrics import INT_SIZE, DecodingMetrics
from .model_gpu import KVCacheModel
from .utils import sample, skip_token_validation

def _add_comm_accounting_metrics(metrics: dict, args, comm_simulator) -> None:
    """L1：把 edge-cloud 动态 NTT 观测历史挂进 per-sample metrics。

    列表字段会被 eval 层用 += 拼接成全运行历史；口径标签等运行级常量由
    eval/utils.py 的 get_save_dict 从 args 写入最终 json（单一事实源）。
    """
    metrics["edge_cloud_ntt_history"] = list(
        getattr(comm_simulator, "ntt_edge_cloud_history", [])
    )
def _finalize_comm_metrics(
    metrics: dict, comm_simulator: CommunicationSimulator, args
) -> None:
    """S7 公共尾巴：带宽/top-k/草稿历史 + 逐消息账 + 口径标签。

    11 个解码方法此前各写一遍同一段（10 个逐字相同 + ceesd_without_arp
    的无注释变体）；抽成 helper 后方法体只留各自的 forward/token/耗时字段。
    调用点必须在 flush_round 之后、return 之前，顺序与抽取前逐事件一致
    （逻辑指纹要求）。
    """
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
    _add_comm_accounting_metrics(metrics, args, comm_simulator)
    metrics["edge_cloud_topk_history"] = (
        comm_simulator.edge_cloud_topk_history.copy()
    )
    metrics["edge_cloud_draft_len_history"] = (
        comm_simulator.edge_cloud_draft_len_history.copy()
    )
def _reward_comm_s(args, comm_simulator, link_type: str) -> float:
    """RL 奖励的通信成本（--rl_comm_cost_mode）。

    - nominal（默认）：NTT + 本轮未结算字节/带宽（nominal_comm_s，
      轮内即时可读——fluid+per_round 口径下真实计费到 flush_round 才
      入账，差值口径轮内恒 0，v2 重训的奖励因此长期无通信项、策略
      系统性偏好短草稿（根因①，见
      experiment_results/ceesd_gamma_threshold_probe.md）。
    - off：恒 0。

    历史注记：settled 口径（flush 差值）已于 Phase 1 删除（2026-10-11，
    golden 闭环后）——fluid+per_round 下轮内差值恒 0 属坏观测，零脚本
    引用；历史训练复现走 commit 锚定（结果工件 git_sha）。
    """
    mode = getattr(args, "rl_comm_cost_mode", "nominal")
    if mode == "nominal":
        return float(comm_simulator.nominal_comm_s(link_type))
    if mode == "off":
        return 0.0
    raise ValueError(
        f"未知 rl_comm_cost_mode {mode!r}（可用: nominal, off；"
        f"settled 已删除，历史复现请 checkout 删除前 commit）"
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
def _rl_train_paths(output_dir, role: str, model_path: str, best_model_path: str):
    """训练 checkpoint 的落盘路径。

    ``--rl_output_dir`` 为空时沿用生产路径；设置后写到
    ``<output_dir>/<role>/<pair>/{latest,best}.pth``，由
    ``tools/promote_checkpoint.py`` 显式晋升，训练过程不再改生产文件。
    """
    if not output_dir:
        return model_path, best_model_path
    pair = os.path.basename(os.path.dirname(str(model_path)))
    base = os.path.join(str(output_dir), role, pair)
    return (
        os.path.join(base, os.path.basename(str(model_path))),
        os.path.join(base, os.path.basename(str(best_model_path))),
    )
