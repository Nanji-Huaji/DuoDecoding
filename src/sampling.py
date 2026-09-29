"""token 分布运算（D3：从 utils 拆分，utils 再导出保兼容）。

包含：top-k/top-p 过滤、温度归一化、采样、top-k 重建、状态熵、max_fn。
"""

import torch
import torch.nn.functional as F


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensor): Tensor with shape (batch, vocab) or (batch, seq_len, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        # Avoid out of bounds if top_k > vocab_size
        k = min(top_k, logits.size(-1))
        # Support multi-dimensional tensor (e.g. 3D: [batch, seq_len, vocab])
        filter_value = torch.topk(logits, k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(logits < filter_value, float("-inf"))

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Determine elements to remove
        filter_mask = cumulative_probs > top_p

        # Shift mask to the right to keep the first token that exceeds top_p
        filter_mask[..., 1:] = filter_mask[..., :-1].clone()
        filter_mask[..., 0] = 0

        # Scatter the mask back to the original index positions
        indices_to_remove = filter_mask.scatter(-1, sorted_indices, filter_mask)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


def norm_logits(
    logits: torch.Tensor, temperature: float, top_k: float, top_p: float
) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (batch, vocab) or (batch, seq_len, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: probs with same shape as logits
    """
    if temperature == 0:
        idx = logits.argmax(dim=-1, keepdim=True)
        new_logits = torch.zeros_like(logits, device=logits.device)
        new_logits.scatter_(-1, idx, 1)
        return new_logits.float()

    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=int(top_k), top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return probs


def sample(probs: torch.Tensor, num_samples: int = 1):
    probs = probs.float()
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs.clamp_min(0.0)

    probs_sum = probs.sum(dim=-1, keepdim=True)
    invalid_rows = probs_sum.squeeze(-1) <= 0

    # 无分支化：invalid 检查不再用 .any()（host 同步）逐次打断 CPU/GPU 流水线
    # （投机解码热循环里每次采样都要付这个代价）。torch.where 在 GPU 上等价
    # 完成：有效行归一化，无效行（clamp 后即全零行 ⇒ 和 ≤ 0）回退 argmax
    # one-hot —— 与原分支逐位一致（含 argmax 平局取 0 的行为），multinomial
    # 的 RNG 消耗不变。invalid_rows 仅保留给调试断言。
    del invalid_rows
    tiny = torch.finfo(probs.dtype).tiny
    normalized = probs / probs_sum.clamp_min(tiny)
    fallback = torch.zeros_like(probs).scatter_(
        -1, probs.argmax(dim=-1, keepdim=True), 1.0
    )
    probs = torch.where(probs_sum > 0, normalized, fallback)

    idx_next = torch.multinomial(probs, num_samples=num_samples)
    return idx_next


def rebuild_topk_probs(
    probs: torch.Tensor,
    top_k: int | None,
    strategy: str = "uniform",
) -> torch.Tensor:
    if strategy != "uniform":
        raise ValueError(f"Unsupported top-k rebuild strategy: {strategy}")

    if top_k is None or top_k <= 0 or probs.numel() == 0 or top_k >= probs.shape[-1]:
        return probs

    top_k_values, top_k_indices = torch.topk(probs, top_k, dim=-1, sorted=True)
    compressed_probs = torch.zeros_like(probs)
    compressed_probs.scatter_(-1, top_k_indices, top_k_values)

    top_k_sum = compressed_probs.sum(dim=-1, keepdim=True)
    residual_mass = (1.0 - top_k_sum).clamp_min(0.0)
    zero_mask = compressed_probs == 0
    zero_count = zero_mask.sum(dim=-1, keepdim=True)
    uniform_prob = torch.where(
        zero_count > 0,
        residual_mass / zero_count,
        torch.zeros_like(residual_mass),
    )
    rebuilt_probs = torch.where(zero_mask, uniform_prob, compressed_probs)
    rebuilt_sum = rebuilt_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return rebuilt_probs / rebuilt_sum


def rebuild_topk_uniform_probs(
    probs: torch.Tensor,
    top_k: int | None,
) -> torch.Tensor:
    return rebuild_topk_probs(probs, top_k, strategy="uniform")


def state_entropy(probs: "torch.Tensor | None", cache=None) -> float:
    """RL 控制器的 entropy 状态特征（**不要**对概率再 softmax 一次）。

    背景（真实 bug）：缓存的 `_forward_with_kvcache` 返回的是 `norm_logits` 归一化
    之后的**概率**。原实现在此处 `torch.softmax(q)` 再算熵，等于对概率分布再做一次
    softmax —— 得到近似均匀分布，熵恒为 ln(vocab)≈10.3735，经 `min(entropy/10,1)`
    归一化后饱和成常数 1.0，该特征从未携带信息（实测 300 步只有一个取值）。
    另外 `--temp 0.0` 时 `norm_logits` 直接返回 one-hot，从返回的概率算熵同样恒为 0。

    因此优先取缓存里按**原始 logits（温度 1）**算好的 `last_entropy`；回退时才从
    传入张量算，并按"已是概率"处理（仅当出现负值才认为传的是 logits）。
    """
    if cache is not None:
        value = getattr(cache, "last_entropy", None)
        if value is not None:
            return float(value)
    if probs is None:
        return 0.0
    p = probs.float()
    if float(p.min()) < 0:
        p = torch.softmax(p, dim=-1)
    p = p.clamp_min(0)
    total = p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p = p / total
    return float(-(p * torch.log(p + 1e-9)).sum(dim=-1).mean().item())


def max_fn(x):
    """
    norm(max (x, 0))
    """
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)

    valid_rows = x_max_sum.squeeze(-1) > 0
    result = torch.zeros_like(x_max)

    if valid_rows.any():
        result[valid_rows] = x_max[valid_rows] / x_max_sum[valid_rows]

    if (~valid_rows).any():
        fallback = torch.zeros_like(x_max[~valid_rows])
        fallback.scatter_(
            -1,
            x[~valid_rows].argmax(dim=-1, keepdim=True),
            1.0,
        )
        result[~valid_rows] = fallback

    return result
