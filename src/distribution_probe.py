from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


class MetricsJson(TypedDict):
    total_variation: float
    jensen_shannon: float
    maximum_absolute_difference: float


class TokenProbabilityJson(TypedDict):
    token_id: int
    token: str
    probability: float


@dataclass(frozen=True, slots=True)
class FirstTokenDistribution:
    counts: torch.Tensor
    probabilities: torch.Tensor


@dataclass(frozen=True, slots=True)
class MetricComparisons:
    cee_sd_vs_ar: CategoricalMetrics
    cee_sd_vs_target_exact: CategoricalMetrics
    ar_vs_target_exact: CategoricalMetrics


@dataclass(frozen=True, slots=True)
class CategoricalMetrics:
    total_variation: float
    jensen_shannon: float
    maximum_absolute_difference: float

    def to_json(self) -> MetricsJson:
        return MetricsJson(
            total_variation=self.total_variation,
            jensen_shannon=self.jensen_shannon,
            maximum_absolute_difference=self.maximum_absolute_difference,
        )


def categorical_metrics(
    cee_sd_probs: torch.Tensor, ar_probs: torch.Tensor
) -> CategoricalMetrics:
    if cee_sd_probs.ndim != 1 or ar_probs.ndim != 1:
        raise ValueError("Expected one-dimensional categorical distributions")
    if cee_sd_probs.shape != ar_probs.shape:
        raise ValueError("Categorical distributions must have matching shapes")

    cee_sd = cee_sd_probs.detach().to(dtype=torch.float64)
    ar = ar_probs.detach().to(dtype=torch.float64)
    midpoint = (cee_sd + ar) / 2.0
    total_variation = 0.5 * torch.abs(cee_sd - ar).sum()
    maximum_absolute_difference = torch.abs(cee_sd - ar).max()
    cee_sd_kl = _kl_divergence(cee_sd, midpoint)
    ar_kl = _kl_divergence(ar, midpoint)
    jensen_shannon = 0.5 * (cee_sd_kl + ar_kl)
    return CategoricalMetrics(
        total_variation=float(total_variation.item()),
        jensen_shannon=float(jensen_shannon.item()),
        maximum_absolute_difference=float(maximum_absolute_difference.item()),
    )


def first_token_distribution(
    full_outputs: torch.Tensor, prefix_len: int, vocab_size: int
) -> FirstTokenDistribution:
    if full_outputs.ndim != 2:
        raise ValueError(
            "Expected returned decoding outputs with shape [batch, sequence]"
        )
    if full_outputs.shape[1] <= prefix_len:
        raise ValueError("Returned decoding output did not append a token")
    first_tokens = full_outputs[:, prefix_len]
    counts = torch.bincount(first_tokens, minlength=vocab_size)[:vocab_size]
    probabilities = counts.to(dtype=torch.float64) / first_tokens.numel()
    return FirstTokenDistribution(counts=counts, probabilities=probabilities)


def metric_comparisons(
    cee_empirical: torch.Tensor,
    ar_empirical: torch.Tensor,
    target_exact: torch.Tensor,
) -> MetricComparisons:
    return MetricComparisons(
        cee_sd_vs_ar=categorical_metrics(cee_empirical, ar_empirical),
        cee_sd_vs_target_exact=categorical_metrics(cee_empirical, target_exact),
        ar_vs_target_exact=categorical_metrics(ar_empirical, target_exact),
    )


def top_token_probabilities(
    probabilities: torch.Tensor, tokenizer: TokenDecoder, count: int
) -> list[TokenProbabilityJson]:
    values, token_ids = torch.topk(probabilities.detach().float().cpu(), k=count)
    return [
        TokenProbabilityJson(
            token_id=int(token_id.item()),
            token=tokenizer.decode([int(token_id.item())]),
            probability=float(value.item()),
        )
        for value, token_id in zip(values, token_ids, strict=True)
    ]


class TokenDecoder:
    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError


def _kl_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    positive = left > 0
    return torch.where(
        positive,
        left * (torch.log(left) - torch.log(right)),
        torch.zeros_like(left),
    ).sum()
