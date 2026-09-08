from __future__ import annotations

import gc
import hashlib
import random
from argparse import Namespace
from typing import Protocol, TypedDict

import numpy as np
import torch
from transformers import AutoTokenizer

from src.acc_head_registry import resolve_acc_head_path
from src.distribution_probe import (
    MetricsJson,
    TokenProbabilityJson,
    first_token_distribution,
    metric_comparisons,
    top_token_probabilities,
)
from src.fixed_prefix_probe_runner import FixedPrefixProbeRunner
from src.model_gpu import KVCacheModel
from src.utils import model_zoo


class ProbeSettings(Protocol):
    little_model: str
    draft_model: str
    target_model: str
    target_quantization: str
    temperatures: tuple[float, ...]
    prompts: tuple[str, ...]
    trials: int
    seed: int
    top_k_output: int


class PrefixJson(TypedDict):
    identity: str
    text: str
    token_count: int


class MeasuredResult(TypedDict):
    temperature: float
    prefix: PrefixJson
    trials: int
    cee_sd_counts: list[int]
    ar_counts: list[int]
    cee_sd_vs_ar: MetricsJson
    cee_sd_vs_target_exact: MetricsJson
    ar_vs_target_exact: MetricsJson
    cee_sd_top_tokens: list[TokenProbabilityJson]
    ar_top_tokens: list[TokenProbabilityJson]
    target_exact_top_tokens: list[TokenProbabilityJson]
    cee_sd_average_appended_length: float


def measure(config: ProbeSettings) -> list[MeasuredResult]:
    tokenizer = AutoTokenizer.from_pretrained(
        _target_model_path(config), trust_remote_code=True
    )
    results: list[MeasuredResult] = []
    for temperature in config.temperatures:
        decoder = _new_decoder(config, temperature)
        try:
            for prompt in config.prompts:
                prefix = tokenizer(prompt, return_tensors="pt").input_ids
                target_probs = _target_probabilities(decoder, prefix, temperature)
                cee_outputs, cee_mean_length = _cee_trial_outputs(
                    decoder, config, prefix
                )
                ar_outputs = _ar_trial_outputs(decoder, config, prefix, temperature)
                cee = first_token_distribution(
                    cee_outputs, prefix.shape[1], target_probs.numel()
                )
                ar = first_token_distribution(
                    ar_outputs, prefix.shape[1], target_probs.numel()
                )
                comparisons = metric_comparisons(
                    cee.probabilities, ar.probabilities, target_probs
                )
                results.append(
                    MeasuredResult(
                        temperature=temperature,
                        prefix=PrefixJson(
                            identity=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                            text=prompt,
                            token_count=int(prefix.shape[1]),
                        ),
                        trials=config.trials,
                        cee_sd_counts=cee.counts.cpu().tolist(),
                        ar_counts=ar.counts.cpu().tolist(),
                        cee_sd_vs_ar=comparisons.cee_sd_vs_ar.to_json(),
                        cee_sd_vs_target_exact=comparisons.cee_sd_vs_target_exact.to_json(),
                        ar_vs_target_exact=comparisons.ar_vs_target_exact.to_json(),
                        cee_sd_top_tokens=top_token_probabilities(
                            cee.probabilities, tokenizer, config.top_k_output
                        ),
                        ar_top_tokens=top_token_probabilities(
                            ar.probabilities, tokenizer, config.top_k_output
                        ),
                        target_exact_top_tokens=top_token_probabilities(
                            target_probs, tokenizer, config.top_k_output
                        ),
                        cee_sd_average_appended_length=cee_mean_length,
                    )
                )
        finally:
            del decoder
            _release_cuda_resources()
    return results


def _target_probabilities(
    decoder: FixedPrefixProbeRunner, prefix: torch.Tensor, temperature: float
) -> torch.Tensor:
    target = KVCacheModel(
        decoder.target_model,
        temperature=temperature,
        top_k=0,
        top_p=0,
        max_length=prefix.shape[1] + 1,
    )
    device = decoder.get_model_input_device(decoder.target_model)
    return target._forward_with_kvcache(prefix.to(device))[0].squeeze(0).cpu()


def _cee_trial_outputs(
    decoder: FixedPrefixProbeRunner, config: ProbeSettings, prefix: torch.Tensor
) -> tuple[torch.Tensor, float]:
    decoder.args.probe_cache_max_length = (
        prefix.shape[1]
        + decoder.args.max_tokens
        + decoder.args.gamma1
        + decoder.args.gamma2
    )
    first_tokens: list[torch.Tensor] = []
    appended_lengths: list[int] = []
    for trial in range(config.trials):
        _seed_everything(_cee_seed(config.seed, trial))
        output, _ = decoder.get_decoding_method()(
            prefix.clone(), transfer_top_k=decoder.args.transfer_top_k
        )
        _require_appended_token(output, prefix.shape[1])
        appended_lengths.append(output.shape[1] - prefix.shape[1])
        first_tokens.append(output[:, prefix.shape[1] : prefix.shape[1] + 1].cpu())
    outputs = torch.cat(
        (prefix.cpu().expand(config.trials, -1), torch.cat(first_tokens)), dim=1
    )
    return outputs, float(sum(appended_lengths) / len(appended_lengths))


def _ar_trial_outputs(
    decoder: FixedPrefixProbeRunner,
    config: ProbeSettings,
    prefix: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for trial in range(config.trials):
        _seed_everything(_ar_seed(config.seed, trial))
        target = KVCacheModel(
            decoder.target_model,
            temperature=temperature,
            top_k=0,
            top_p=0,
            max_length=prefix.shape[1] + 1,
        )
        output = target.generate(
            prefix.to(decoder.get_model_input_device(decoder.target_model)), 1
        )
        _require_appended_token(output, prefix.shape[1])
        outputs.append(output.cpu())
    return torch.cat(outputs, dim=0)


def _new_decoder(config: ProbeSettings, temperature: float) -> FixedPrefixProbeRunner:
    args = Namespace(
        little_model=config.little_model,
        draft_model=config.draft_model,
        target_model=config.target_model,
        target_quantization=config.target_quantization,
        eval_mode="cee_sd",
        temp=temperature,
        top_k=0,
        top_p=0.0,
        gamma1=1,
        gamma2=1,
        max_tokens=1,
        seed=config.seed,
        transfer_top_k=300,
        use_rl_adapter=False,
        disable_rl_update=True,
        use_stochastic_comm=False,
        use_precise=False,
        ntt_ms_edge_cloud=10.0,
        ntt_ms_edge_end=1.0,
        edge_cloud_bandwidth=20.0,
        edge_end_bandwidth=100.0,
        cloud_end_bandwidth=100.0,
        batch_delay=0.0,
        use_early_stopping=False,
        dump_network_stats=False,
        exp_name="distribution_probe",
        eval_dataset="distribution_probe",
        small_draft_threshold=0.8,
        draft_target_threshold=0.8,
        small_draft_acc_head_path=resolve_acc_head_path(
            config.little_model, config.draft_model
        ),
        draft_target_acc_head_path=resolve_acc_head_path(
            config.draft_model, config.target_model
        ),
    )
    model_zoo(args)
    _seed_everything(config.seed)
    decoder = FixedPrefixProbeRunner(args)
    decoder.load_model()
    decoder.vocab_size = decoder._get_model_embedding_vocab_size(decoder.target_model)
    return decoder


def _release_cuda_resources() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _target_model_path(config: ProbeSettings) -> str:
    args = Namespace(
        little_model=config.little_model,
        draft_model=config.draft_model,
        target_model=config.target_model,
    )
    model_zoo(args)
    return args.target_model


def _require_appended_token(output: torch.Tensor, prefix_len: int) -> None:
    if output.shape[1] <= prefix_len:
        raise RuntimeError("CEE-SD or AR returned no token after the fixed prefix")


def _cee_seed(seed: int, trial: int) -> int:
    return seed * 2 + trial * 2


def _ar_seed(seed: int, trial: int) -> int:
    return seed * 2 + trial * 2 + 1


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
