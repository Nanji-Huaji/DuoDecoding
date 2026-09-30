import argparse
import json
import logging
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

from src.protocols import (
    PROTOCOLS,
    apply_protocol,
    emit_effective_report,
)
from src.mode_features import MODE_FEATURES
from src.model_zoo import get_vocab_size, model_zoo  # noqa: F401 D3 拆分后再导出
from src.sampling import (  # noqa: F401 D3 拆分后再导出
    max_fn,
    norm_logits,
    rebuild_topk_probs,
    rebuild_topk_uniform_probs,
    sample,
    state_entropy,
    top_k_top_p_filter,
)
from src.trace_io import read_trace_file, return_closest_mean_index  # noqa: F401
from src.acc_head_registry import resolve_acc_head_path
from src.rl_agent_registry import (
    ROLE_LITTLE,
    ROLE_MAIN,
    resolve_rl_agent_paths,
)


logger = logging.getLogger(__name__)
_LIMITED_WARNING_COUNTS: dict[str, int] = {}


def parse_range_spec(spec: str) -> tuple[float, float]:
    """Parse a "low,high" range string into a (low, high) float tuple."""
    try:
        low_s, high_s = spec.split(",")
        low, high = float(low_s.strip()), float(high_s.strip())
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid range spec {spec!r}, expected 'low,high'") from exc
    if low > high:
        raise ValueError(f"Range low ({low}) must not exceed high ({high})")
    return low, high


def sample_curriculum_condition(
    step: int,
    total_steps: int,
    bw_start: tuple[float, float],
    bw_end: tuple[float, float],
    ntt_start: tuple[float, float],
    ntt_end: tuple[float, float],
    sampling: str = "uniform",
) -> tuple[float, float]:
    """Sample the network condition for one RL training step.

    Implements a smooth curriculum: the (bandwidth, latency) sampling ranges
    are interpolated from the start ranges (easy conditions) to the end
    ranges (hard conditions, e.g. constrained links) as training progresses.

    - Bandwidth bounds are interpolated geometrically, and the bandwidth is
      sampled log-uniformly within the bounds when sampling="loguniform"
      (bandwidth perception is roughly logarithmic: 1->2 Mbps matters more
      than 40->41 Mbps).
    - Latency bounds are interpolated linearly, sampled uniformly.

    Returns (bandwidth_mbps, ntt_ms).
    """
    progress = step / max(1, total_steps - 1)

    def interp_bw_bound(start: float, end: float) -> float:
        if start > 0 and end > 0:
            return start * (end / start) ** progress
        return start + (end - start) * progress

    bw_low = interp_bw_bound(bw_start[0], bw_end[0])
    bw_high = interp_bw_bound(bw_start[1], bw_end[1])
    if sampling == "loguniform" and bw_low > 0 and bw_high > 0:
        bw = math.exp(random.uniform(math.log(bw_low), math.log(bw_high)))
    else:
        bw = random.uniform(bw_low, bw_high)

    ntt_low = ntt_start[0] + (ntt_end[0] - ntt_start[0]) * progress
    ntt_high = ntt_start[1] + (ntt_end[1] - ntt_start[1]) * progress
    ntt = random.uniform(ntt_low, ntt_high)
    return bw, ntt


def _env_flag_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


def numeric_debug_checks_enabled() -> bool:
    return _env_flag_enabled("DUODEC_DEBUG_NUMERICS")


def skip_token_validation() -> bool:
    return _env_flag_enabled("DUODEC_SKIP_VALIDATE")


def _log_limited_warning(label: str, message: str, max_warnings: int = 5) -> None:
    count = _LIMITED_WARNING_COUNTS.get(label, 0)
    _LIMITED_WARNING_COUNTS[label] = count + 1

    if count < max_warnings:
        logger.warning(message)
    elif count == max_warnings:
        logger.warning("[%s] additional warnings suppressed", label)


def log_prob_tensor_if_invalid(
    probs: torch.Tensor | None,
    label: str,
    *,
    expected_sum: float = 1.0,
    atol: float = 1e-3,
    max_warnings: int = 5,
) -> bool:
    if not numeric_debug_checks_enabled():
        return False

    if probs is None or probs.numel() == 0:
        return False

    probs_float = probs.detach().float()
    if probs_float.dim() == 0:
        probs_float = probs_float.reshape(1, 1)
    elif probs_float.dim() == 1:
        probs_float = probs_float.unsqueeze(0)
    else:
        probs_float = probs_float.reshape(-1, probs_float.shape[-1])

    has_nan = torch.isnan(probs_float).any().item()
    has_posinf = torch.isposinf(probs_float).any().item()
    has_neginf = torch.isneginf(probs_float).any().item()
    negative_count = int((probs_float < -atol).sum().item())

    row_sums = probs_float.sum(dim=-1)
    finite_sum_mask = torch.isfinite(row_sums)
    bad_sum_mask = finite_sum_mask & ((row_sums - expected_sum).abs() > atol)

    if not (
        has_nan
        or has_posinf
        or has_neginf
        or negative_count > 0
        or bad_sum_mask.any().item()
    ):
        return False

    finite_values = probs_float[torch.isfinite(probs_float)]
    if finite_values.numel() > 0:
        min_prob = float(finite_values.min().item())
        max_prob = float(finite_values.max().item())
    else:
        min_prob = float("nan")
        max_prob = float("nan")

    finite_row_sums = row_sums[finite_sum_mask]
    if finite_row_sums.numel() > 0:
        min_sum = float(finite_row_sums.min().item())
        max_sum = float(finite_row_sums.max().item())
    else:
        min_sum = float("nan")
        max_sum = float("nan")

    _log_limited_warning(
        label,
        (
            f"[{label}] invalid probability tensor detected: "
            f"shape={tuple(probs.shape)}, "
            f"nan={bool(has_nan)}, posinf={bool(has_posinf)}, neginf={bool(has_neginf)}, "
            f"negative_count={negative_count}, "
            f"bad_sum_rows={int(bad_sum_mask.sum().item())}/{row_sums.numel()}, "
            f"sum_range=[{min_sum:.6g}, {max_sum:.6g}], "
            f"value_range=[{min_prob:.6g}, {max_prob:.6g}]"
        ),
        max_warnings=max_warnings,
    )
    return True


def log_ratio_if_invalid(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    label: str,
    *,
    atol: float = 1e-12,
    max_warnings: int = 5,
) -> bool:
    if not numeric_debug_checks_enabled():
        return False

    numerator = numerator.detach().float()
    denominator = denominator.detach().float()
    ratio = numerator / denominator

    invalid_numerator = ~torch.isfinite(numerator) | (numerator < -atol)
    invalid_denominator = ~torch.isfinite(denominator) | (denominator <= atol)
    invalid_ratio = ~torch.isfinite(ratio) | (ratio < -atol)
    issue_mask = invalid_numerator | invalid_denominator | invalid_ratio

    if not issue_mask.any().item():
        return False

    finite_ratio = ratio[torch.isfinite(ratio)]
    if finite_ratio.numel() > 0:
        min_ratio = float(finite_ratio.min().item())
        max_ratio = float(finite_ratio.max().item())
    else:
        min_ratio = float("nan")
        max_ratio = float("nan")

    _log_limited_warning(
        label,
        (
            f"[{label}] invalid acceptance ratio detected: "
            f"shape={tuple(ratio.shape)}, "
            f"invalid_numerator={int(invalid_numerator.sum().item())}, "
            f"invalid_denominator={int(invalid_denominator.sum().item())}, "
            f"invalid_ratio={int(invalid_ratio.sum().item())}, "
            f"finite_ratio_range=[{min_ratio:.6g}, {max_ratio:.6g}]"
        ),
        max_warnings=max_warnings,
    )
    return True


def seed_everything(seed: int):
    "set all random seed for reproducible results."
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # B28：deterministic 与 benchmark 互斥——benchmark=True 会为输入形状选
    # "最快但不一定确定"的实现，上面的 deterministic 沦为伪保证。复现优先；
    # 解码负载形状固定，benchmark 的加速收益本就有限。
    torch.backends.cudnn.benchmark = False


def _dest_to_flags(parser: argparse.ArgumentParser) -> dict[str, list[str]]:
    """dest → 该选项的全部 CLI 拼写，用于判定"是否被命令行显式设置"。"""
    out: dict[str, list[str]] = {}
    for action in parser._actions:
        if action.dest and action.option_strings:
            out.setdefault(action.dest, []).extend(action.option_strings)
    return out


def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description="args for this file")

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/",
    )

    parser.add_argument(
        "--draft_model",
        type=str,
        default=None,
        help="必填。原默认 codellama-7b 不可解析已移除（B24）",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default=None,
        help="必填。原默认 codellama-70b 不可解析已移除（B24）",
    )

    parser.add_argument(
        "--exp_name",
        "-e",
        type=str,
        default="test",
        help="folder name for storing results.",
    )
    parser.add_argument("--eval_mode", type=str, default="small", help="eval mode.")
    parser.add_argument(
        "--num_samples_per_task",
        "-n",
        type=int,
        default=1,
        help="num_samples for a task (prompt) in humaneval dataset.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=1234,
        help="set a random seed, which can makes the result reproducible",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="max token number generated."
    )
    parser.add_argument(
        "--temp", type=float, default=0.2, help="temperature for generating new tokens."
    )
    parser.add_argument(
        "--top_k", type=int, default=0, help="top_k for ungreedy sampling strategy."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="top_p for ungreedy sampling strategy.",
    )
    parser.add_argument("--gamma", type=int, default=4, help="guess time.")
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true",
        help=(
            "把定长 decode 前向捕获成 CUDA Graph 回放，绕开 per-op 启动开销："
            "单步草稿走 (1,1) 单步图，验证/resync 的多 token 合成前向走 "
            "(1,K) 定长 padding 验证图（主体收益：13B/1.1B 验证前向 84~92%% "
            "是逐算子调度开销）。图与 KV 缓存跨样本复用（StaticCache 原地 "
            "reset，不重捕获）。默认关闭以保证与历史实验逐位可复现；图模式会"
            "按 prompt+max_tokens+余量 预分配 KV 缓存。"
        ),
    )
    parser.add_argument(
        "--graph_verify_sizes",
        type=str,
        default="",
        help=(
            "验证图的捕获档位（逗号分隔升序，如 '4,8,16,32'）。留空则自动："
            "从 {4,8,16,24,32,40,48,64} 里取 ≤ γ1+γ2+4 的档位并补齐上限。"
            "回放时选 ≥k 的最小档位 padding，pad 行 KV 随回滚作废。"
        ),
    )
    parser.add_argument(
        "--eval_data_num",
        type=int,
        default=80,
        help="number of samples to evaluate.",
    )
    parser.add_argument(
        "--run_full_dataset",
        action="store_true",
        help="Evaluate the full dataset instead of truncating to eval_data_num.",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Randomly sample eval_data_num examples instead of taking the first examples.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=1234,
        help="Random seed used when --random_sample is enabled.",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=0,
        help="number of shots for few-shot evaluation.",
    )
    parser.add_argument(
        "--sub_domain",
        type=str,
        default="math_reasoning",
        help="sub domain in specbench.",
        choices=[
            "math_reasoning",
            "mt-bench",
            "qa",
            "rag",
            "summarization",
            "translation",
        ],
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default="unknown",
        help="Task name for RL adapter context (e.g., mt_bench, humaneval).",
    )

    # for lookahead decoding（--level/--guess 已删除：全仓无消费者）
    parser.add_argument(
        "--window",
        type=int,
        default=10,
    )
    # end for lookahead decoding

    # for rest（--max-token-span/--num-draft 已删除：全仓无消费者）
    parser.add_argument(
        "--datastore-path",
        type=str,
        default="datastore/",
        help="The path of the datastore for retrival.",
    )
    # end for rest
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API Key for MT-Bench Judge",
    )
    parser.add_argument(
        "--openai_api_base",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI API Base for MT-Bench Judge",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=os.environ.get("JUDGE_MODEL", "deepseek-v3.1"),
        help="Judge model for MT-Bench",
    )

    parser.add_argument(
        "--little_model",
        type=str,
        default="vicuna-68m",
        help="The little model for decoding.",
    )
    parser.add_argument(
        "--gamma1",
        type=int,
        default=4,
        help="The number of guesses for the first draft model.",
    )
    parser.add_argument(
        "--gamma2",
        type=int,
        default=4,
        help="The number of guesses for the second draft model.",
    )
    parser.add_argument(
        "--edge_cloud_bandwidth",
        type=float,
        default=20.0,
        help="The bandwidth between edge and cloud in Mbps.",
    )
    parser.add_argument(
        "--edge_end_bandwidth",
        type=float,
        default=100.0,
        help="The bandwidth between edge and end device in Mbps.",
    )
    parser.add_argument(
        "--cloud_end_bandwidth",
        type=float,
        default=100.0,
        help="The bandwidth between cloud and end device in Mbps.",
    )
    parser.add_argument(
        "--uncertainty_threshold",
        type=float,
        default=0.8,
        help="The uncertainty threshold for uncertainty-based decoding.",
    )
    parser.add_argument(
        "--transfer_top_k",
        type=int,
        default=300,
        help="The top k probs to transfer during communication.",
    )
    parser.add_argument(
        "--use_precise",
        action="store_true",
        help="Use the physics level to simulate the communication.",
    )
    parser.add_argument(
        "--ntt_ms_edge_end",
        type=float,
        default=20.0,
        help="The network time delay between edge and end device in ms.",
    )
    parser.add_argument(
        "--ntt_ms_edge_cloud",
        type=float,
        default=200.0,
        help="The network time delay between edge and cloud in ms.",
    )
    parser.add_argument(
        "--acc_head_path",
        type=str,
        default=None,  # 后置解析：见 parse_arguments 末尾按实际模型对 resolve
        help="The path of the accuracy head model.",
    )
    parser.add_argument(
        "--small_draft_acc_head_path",
        type=str,
        default=resolve_acc_head_path("llama-68m", "tiny-llama-1.1b"),
        help="The path of the small draft accuracy head model.",
    )
    parser.add_argument(
        "--draft_target_acc_head_path",
        type=str,
        default=None,  # 后置解析：见 parse_arguments 末尾按实际模型对 resolve
        help="The path of the draft-target accuracy head model.",
    )
    parser.add_argument(
        "--small_draft_threshold",
        type=float,
        default=0.8,
        help="The threshold for the small draft model for adaptive tri-decoding. Default is 0.8.",
    )
    parser.add_argument(
        "--draft_target_threshold",
        type=float,
        default=0.8,
        help="The threshold for the draft-target model for adaptive decoding. Default is 0.8.",
    )
    parser.add_argument(
        "--comm_trace_mode",
        choices=["static", "driving", "walking"],
        default="static",
        help=("随机通信 trace 的移动模式（配合 --use_stochastic_comm）："
              "driving=5G mmWave 车载轨迹，波动最剧烈；static=静止场景（历史默认）。"),
    )
    parser.add_argument(
        "--use_stochastic_comm",
        action="store_true",
        help="Whether to use stochastic communication simulator.",
    )
    parser.add_argument(
        "--ntt_trace_file",
        type=str,
        default="",
        help=(
            "真实 RTT trace 回放（sigcomm ping/ 实测，与 throughput trace 同 Campaign）。"
            "设置后 edge-cloud NTT 逐次传输从 trace 采样（优先级高于 --stochastic_ntt "
            "拥塞模型与固定基值）；与带宽 trace 独立推进。空 = 不回放。"
        ),
    )
    parser.add_argument(
        "--ntt_trace_scale",
        type=float,
        default=1.0,
        help="RTT trace 回放缩放（1.0=原样；ping 数据为往返毫秒值）。",
    )
    parser.add_argument(
        "--stochastic_ntt",
        action="store_true",
        help=(
            "L1：edge-cloud NTT 动态化（要求 --use_stochastic_comm）。拥塞相关 RTT 模型，"
            "与带宽 trace 同源同索引、确定性可复现：ntt_t = ntt_base * (1 + (mean_bw/bw_t "
            "- 1)^+)，带宽跌到 trace 均值一半时 RTT 翻倍（排队延迟），带宽充足时保持基值。"
            "默认关闭（历史数字逐位可复现）；开启后 RL 的网络状态输入才有时延维度的变化。"
            "edge-end 链路（LAN）保持固定 NTT。"
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=["none", *PROTOCOLS],
        default="none",
        help=(
            "命名口径（唯一真源: src/protocols.py；定义见 docs/protocol.md）。"
            "none（默认）= 不写入任何参数，数字与历史完全一致。给出协议名时只填"
            "命令行未显式设置的项；被命令行覆盖的项会在启动自述里告警，该 run "
            "据此不再属于该协议。"
        ),
    )
    parser.add_argument(
        "--comm_accounting",
        choices=["honest", "legacy"],
        default=None,
        help=(
            "L1 口径收敛总开关：honest = 残差计费 + per_round + top-k cap16（论文协议，"
            "= 仓库默认）；legacy = --no-charge_residual_payload + per_transfer + cap0"
            "（复现 2026-09-24 前历史序列：ab/vg/vg2/t5a/q_ours_hist）。显式给出时覆盖"
            "三个子开关并在日志回显；不给出时子开关独立生效。metrics json 会记录实际口径。"
        ),
    )
    parser.add_argument(
        "--min_bandwidth_mbps",
        type=float,
        default=5.0,
        help=(
            "Minimum bandwidth floor (Mbps) applied inside the communication "
            "simulator; bandwidth below this value is clamped. Set to 0 to "
            "disable the floor (e.g. to study links weaker than 5 Mbps)."
        ),
    )
    parser.add_argument(
        "--curriculum_bw_start",
        type=str,
        default="20,50",
        help=(
            "Curriculum bandwidth range (Mbps) at the start of RL training, "
            "format 'low,high'. Sampled per step; interpolates towards "
            "--curriculum_bw_end as training progresses."
        ),
    )
    parser.add_argument(
        "--curriculum_bw_end",
        type=str,
        default="20,50",
        help="Curriculum bandwidth range (Mbps) at the end of RL training, format 'low,high'.",
    )
    parser.add_argument(
        "--curriculum_ntt_start",
        type=str,
        default="0,5",
        help=(
            "Curriculum edge-cloud latency range (ms) at the start of RL "
            "training, format 'low,high'."
        ),
    )
    parser.add_argument(
        "--curriculum_ntt_end",
        type=str,
        default="0,5",
        help="Curriculum edge-cloud latency range (ms) at the end of RL training, format 'low,high'.",
    )
    parser.add_argument(
        "--curriculum_sampling",
        type=str,
        choices=["uniform", "loguniform"],
        default="uniform",
        help=(
            "How bandwidth is sampled within the curriculum range: 'uniform' "
            "(legacy behavior) or 'loguniform' (recommended; bandwidth "
            "perception is roughly logarithmic)."
        ),
    )
    parser.add_argument(
        "--state_bw_scaling",
        type=str,
        choices=["linear", "log"],
        default="linear",
        help=(
            "How the bandwidth state feature is scaled before it enters the RL "
            "network: 'linear' (legacy bw/1000, which squeezes the whole "
            "0.5-50 Mbps range into [0.0005, 0.05] and makes the policy "
            "effectively bandwidth-blind) or 'log' "
            "(log10(bw+1)/log10(1000+1), spreads it over [0.0, 0.57])."
        ),
    )
    parser.add_argument(
        "--state_latency_scaling",
        type=str,
        choices=["linear", "centi", "log"],
        default="linear",
        help=(
            "How the latency state feature is scaled: 'linear' (legacy "
            "ntt/500 -> only 0-0.2 for a 0-100 ms curriculum), 'centi' "
            "(ntt/100, uses the full range) or 'log'."
        ),
    )
    parser.add_argument(
        "--rl_reward_mode",
        type=str,
        choices=["legacy", "linear", "lagrangian", "slo", "energy"],
        default="legacy",
        help=(
            "Reward for the RL adapters. 'legacy' = the v1 hand-tuned reward "
            "exp(min(N_acc/T,100)/20) * (N_acc/gamma)^2 with the wall-clock T "
            "(kept as default so earlier results stay reproducible). "
            "'linear' = N_acc/T. 'lagrangian' = N_acc - lambda*T, the Lagrangian "
            "relaxation of the ratio objective E[N]/E[T] (recommended). "
            "'slo' = deadline-aware variant, 'energy' = adds a comm-energy term."
        ),
    )
    parser.add_argument(
        "--rl_charge_queue",
        action="store_true",
        help=(
            "Reward 修复：把每轮排队时延（batch_delay）计入 lagrangian reward 的 "
            "T。t5a 轮预算里排队占 27%%，与 NTT 一样被长草稿摊薄——不计价会让策略"
            "系统性偏好短草稿（v2 重训 3.58 vs 钉死0.4 的 4.43 的主因之一）。"
            "默认关闭保持历史 reward 逐位可复现。"
        ),
    )
    parser.add_argument(
        "--rl_reward_lambda",
        type=float,
        default=0.0,
        help=(
            "Shadow price of time (tokens/second) for --rl_reward_mode "
            "lagrangian/energy. 0 (default) tracks the policy's observed "
            "tokens/second with an EMA; set a fixed value for reproducible "
            "ablations (e.g. the pilot run's mean throughput)."
        ),
    )
    parser.add_argument(
        "--rl_reward_deadline_ms",
        type=float,
        default=0.0,
        help="Per-decision deadline (ms) for --rl_reward_mode slo.",
    )
    parser.add_argument(
        "--rl_reward_deadline_penalty",
        type=float,
        default=1.0,
        help="Penalty per second of deadline overrun for --rl_reward_mode slo.",
    )
    parser.add_argument(
        "--rl_reward_energy_weight",
        type=float,
        default=0.0,
        help="Weight mu of the communication-energy term for --rl_reward_mode energy.",
    )
    parser.add_argument(
        "--rl_force_threshold_little",
        type=float,
        default=None,
        help=(
            "Ablation: pin the *little* (edge-end) stage threshold, separately from "
            "--rl_force_threshold. Needed because the little stage is opportunistic "
            "and its early stop changes how many unverified tokens are accepted, "
            "which moves accuracy -- mixing the two confounds the main-stage "
            "threshold ablation."
        ),
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help=(
            "Compute dtype for all models (and bnb_4bit_compute_dtype). Default "
            "bf16 reproduces the historical behaviour. This exists to test whether "
            "argmax flips between batched speculative verification and sequential "
            "decoding come from low-precision near-ties: bf16 has 8 mantissa bits "
            "(~0.2%% relative), fp16 10, fp32 24."
        ),
    )
    parser.add_argument(
        "--disable_eos_stop",
        action="store_true",
        help=(
            "Disable EOS early stopping (restores the historical behaviour of "
            "always generating max_tokens). EOS stopping is ON by default because "
            "without it task quality cannot be measured at all: GSM8K never emits "
            "'#### <answer>' and HumanEval code is truncated. Note that enabling it "
            "changes all throughput/latency numbers, so baselines must be re-run."
        ),
    )
    parser.add_argument(
        "--comm_round_trip_mode",
        choices=["per_transfer", "per_round"],
        default="per_round",
        help=("通信往返口径：per_round=同一轮内每条链路合并为一次往返（成批实现的"
              "真实情形；默认=论文协议 ours_full）；per_transfer=每次消息各付一次"
              " NTT（遗留口径，复现 2026-09-24 前的历史数字时用）。"),
    )
    parser.add_argument(
        "--transfer_top_k_cap",
        type=int,
        default=16,
        help="给（含 RL 选出的）transfer_top_k 设上限，压低拒绝载荷字节"
             "（默认 16=论文协议）；0 = 不设上限（遗留口径）。",
    )
    parser.add_argument(
        "--force_full_vocab_transfer",
        action="store_true",
        help="强制传输完整词表分布（不做 top-k 稀疏化）。用于构造"
             "标准投机采样的通信基线；默认关闭，保证历史数字可复现。",
    )
    parser.add_argument(
        "--prob_payload_bits",
        type=int,
        default=16,
        help="传输概率载荷的位宽（16=与历史一致；8=int8 量化；4=上界数据点）。"
             "量化在对数域进行并重归一化、保持序关系；作用于验证路径与计费。",
    )
    parser.add_argument(
        "--charge_residual_payload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "F43：如实计入拒绝位置残差采样所需的提案分布载荷。开启（默认=论文协议 "
            "ours_full）时解码改用精确的 top-k + 均匀尾表示（TopKProposalHistory），"
            "并按 k*(4+元素大小)+元素大小 字节/拒绝位置/链路计费；"
            "--no-charge_residual_payload 回到只计标量的遗留口径（复现 2026-09-24 "
            "前的历史数字，即 ours_hist）。"
        ),
    )
    parser.add_argument(
        "--dump_outputs",
        type=str,
        default=None,
        help=(
            "Write one JSON line per evaluated sample (task, item, prompt_len, "
            "generated token ids, decoded text) to this path. Needed for the "
            "losslessness identity test (temp=0: speculative output must be "
            "token-identical to target_only) and for offline task-quality scoring."
        ),
    )
    parser.add_argument(
        "--arp_stop_mode",
        type=str,
        choices=["cumulative", "per_token"],
        default="cumulative",
        help=(
            "Acceptance-prediction early-stop rule. 'cumulative' (default) is the "
            "historical 1 - prod_i p_i > threshold, which saturates after ~2 "
            "drafted tokens and makes the threshold dimension inert. 'per_token' "
            "stops when 1 - p_last > threshold, which is monotone in the threshold."
        ),
    )
    parser.add_argument(
        "--rl_force_topk",
        type=int,
        default=None,
        help=(
            "Ablation: pin the uplink top-k to the nearest candidate and let the "
            "policy choose threshold (and gamma, if present) within the pinned "
            "slice. Required for static-k ladders: the policy action otherwise "
            "overrides --transfer_top_k, making ladder rungs identical."
        ),
    )
    parser.add_argument(
        "--rl_force_threshold",
        type=float,
        default=None,
        help=(
            "Ablation: pin the ARP early-stop threshold to the nearest candidate "
            "and let the policy choose top-k (and gamma, if in the action space) "
            "around it.  Used to split the end-to-end effect into the dimension "
            "that actually controls how long the draft runs."
        ),
    )
    parser.add_argument(
        "--rl_byte_price",
        type=float,
        default=0.0,
        help=(
            "Price of transferred bytes, expressed as equivalent seconds per "
            "megabyte, added to the reward's time term. 0 (default) keeps the "
            "historical behaviour, where bytes are only priced through the "
            "physical transmission time inside comm_s (3-9%% of comm at 0.5-1 "
            "Mbps) -- which is why the policy happily trades bytes for round "
            "trips. 16 s/MB is the physical transmission cost at 0.5 Mbps; use "
            "larger values to emulate metered or more expensive links."
        ),
    )
    parser.add_argument(
        "--rl_compute_time_mode",
        type=str,
        choices=["wall", "model"],
        default="wall",
        help=(
            "Where the reward's compute time comes from: 'wall' (measured wall "
            "clock, host-load dependent -- this is what v1 used) or 'model' "
            "(reconstructed from forward counts and --rl_compute_cost_json, "
            "making the reward independent of host load)."
        ),
    )
    parser.add_argument(
        "--rl_compute_cost_json",
        type=str,
        default=None,
        help=(
            "JSON with seconds per forward pass, e.g. "
            '{"little": 0.002, "draft": 0.006, "target": 0.020}; produced by '
            "scripts/calibrate_compute_model.py."
        ),
    )
    parser.add_argument(
        "--rl_reward_no_alpha2",
        action="store_true",
        help="Drop the (N_acc/gamma)^2 factor from the legacy reward (ablation).",
    )
    parser.add_argument(
        "--rl_action_space",
        type=str,
        choices=["topk_thr", "topk_thr_gamma"],
        default="topk_thr",
        help=(
            "Action space of the RL adapters. 'topk_thr' = (top-k, ARP threshold) "
            "= 88 actions (legacy, keeps old checkpoints loadable). "
            "'topk_thr_gamma' additionally selects the draft length per round: the "
            "communication time is ~97%% round-trip time and the number of WAN "
            "round trips is ~tokens/gamma, so gamma is the lever that decides the "
            "dominant term (measured: gamma 4 -> 16 gives -17.7%% round trips, "
            "-11.9%% compute and +22.1%% throughput at 0.5 Mbps)."
        ),
    )
    parser.add_argument(
        "--rl_gamma_candidates",
        type=str,
        default="2,4,8,16",
        help="Comma-separated draft lengths for --rl_action_space topk_thr_gamma.",
    )
    parser.add_argument(
        "--rl_force_gamma",
        type=int,
        default=None,
        help=(
            "Ablation: pin the draft length gamma to this value while top-k and the "
            "ARP threshold are still chosen by the loaded policy. Lets an A/B gain be "
            "split into the gamma contribution and the top-k/threshold contribution "
            "(otherwise 'the joint agent wins because it learned to use a large "
            "top-k' cannot be ruled out)."
        ),
    )
    parser.add_argument(
        "--rl_factored_q",
        action="store_true",
        help=(
            "Use a branching (factored) dueling Q-network for the joint "
            "topk x threshold x gamma action space: Q(s,a) = V(s) + sum_h A_h(s,a_h) "
            "with per-dimension centering. argmax stays exact because Q is additive, "
            "and every decision trains all three heads, which fixes the credit "
            "assignment problem of a flat 352-way head (each combination was visited "
            "only ~27 times in a 400-sample run, so the agent never learned that "
            "larger gamma is worth +0.83 reward per decision)."
        ),
    )
    parser.add_argument(
        "--rl_include_gamma_in_state",
        action="store_true",
        help="Append the last chosen draft length to the RL state vector.",
    )
    parser.add_argument(
        "--rl_buffer_size",
        type=int,
        default=5000,
        help="Replay-buffer size of the DDQN agents (larger for bigger action spaces).",
    )
    parser.add_argument(
        "--rl_team_reward",
        action="store_true",
        help=(
            "Give both adapters the same iteration-level team reward instead of "
            "each one's local reward. The first (end->edge) stage decides before "
            "the second one, so it consumes the previous iteration's team reward "
            "(one-step delay), which is the standard cooperative-MARL treatment."
        ),
    )
    parser.add_argument(
        "--rl_reward_log_window",
        type=int,
        default=200,
        help="Window of the adapter's windowed reward log.",
    )
    parser.add_argument(
        "--use_rl_adapter",
        action="store_true",
        help="Whether to use RL adapter for dynamic k selection.",
    )
    parser.add_argument(
        "--main_rl_path",
        type=str,
        default=None,
        help="The path of the main RL adapter model.",
    )
    parser.add_argument(
        "--main_rl_best_path",
        type=str,
        default=None,
        help="The path of the best main RL adapter model.",
    )
    parser.add_argument(
        "--little_rl_path",
        type=str,
        default=None,
        help="The path of the little RL adapter model.",
    )
    parser.add_argument(
        "--little_rl_best_path",
        type=str,
        default=None,
        help="The path of the best little RL adapter model.",
    )
    parser.add_argument(
        "--rl_checkpoint_root",
        type=str,
        default="checkpoints/rl_agents",
        help="Root directory used to resolve pair-specific RL checkpoints.",
    )
    parser.add_argument(
        "--rl_init_seed",
        type=int,
        default=None,
        help="Seed used for deterministic RL network initialization and exploration.",
    )
    parser.add_argument(
        "--rl_init_strategy",
        choices=["fresh", "resume"],
        default="resume",
        help="Initialize new RL agents or resume existing dedicated checkpoints.",
    )
    parser.add_argument(
        "--rl_epsilon_decay",
        type=float,
        default=None,
        help="Override the mode-specific RL epsilon decay default.",
    )
    parser.add_argument(
        "--rl_reward_scale",
        type=float,
        default=None,
        help="Override the mode-specific RL reward scale default.",
    )
    parser.add_argument(
        "--rl_batch_size",
        type=int,
        default=None,
        help="Override the mode-specific RL batch size default.",
    )
    parser.add_argument(
        "--disable_rl_update",
        action="store_true",
        help="Whether to disable RL adapter update (training).",
    )
    parser.add_argument(
        "--batch_delay",
        type=float,
        default=50e-3,  # 50 ms
        help="The delay time added to each batch in seconds.",
    )
    parser.add_argument(
        "--use_early_stopping",
        action="store_true",
        help="Whether to use early stopping during decoding.",
    )
    parser.add_argument(
        "--dump_network_stats",
        action="store_true",
        help="Whether to dump network statistics during decoding.",
    )
    parser.add_argument(
        "--draft_quantization",
        type=str,
        choices=["auto", "4bit", "none"],
        default="auto",
        help="Quantization mode for the draft model.",
    )
    parser.add_argument(
        "--target_quantization",
        type=str,
        choices=["auto", "4bit", "none"],
        default="auto",
        help="Quantization mode for the target model.",
    )
    parser.add_argument(
        "--keep_target_on_single_gpu",
        action="store_true",
        help="Keep the DSD target model on its selected GPU instead of auto-sharding.",
    )
    parser.add_argument(
        "--little_quantization",
        type=str,
        choices=["auto", "4bit", "none"],
        default="auto",
        help="Quantization mode for the little model.",
    )

    cli_args = sys.argv[1:]
    args = parser.parse_args()

    # 口径协议：只填命令行未显式设置的项（默认 none ⇒ 不改变任何数字）。
    # 放在校验/后处理之前，让后续逻辑（如 --comm_accounting）看到协议值。
    _protocol_app = apply_protocol(
        args,
        getattr(args, "protocol", "none"),
        cli_args,
        _dest_to_flags(parser),
    )

    # B24：必填检查必须先于 acc-head/RL 路径解析（它们在 model_zoo 之前
    # 运行，draft_model=None 会让 canonicalize 以 AttributeError 崩）。
    if not getattr(args, "draft_model", None) or not getattr(
        args, "target_model", None
    ):
        parser.error(
            "--draft_model 与 --target_model 必须显式指定"
            "（原默认 codellama-7b/codellama-70b 不可解析，已移除。"
            "例: --draft_model tiny-llama-1.1b --target_model llama-2-13b）"
        )

    # B15：--prob_payload_bits < 16 只在 tri 系双投机阶段协议中接线。
    # 此前其它模式静默忽略该参数——用户以为设了位宽压缩，实际全宽传输。
    # 现在显式拒绝，杜绝"参数无效却无告警"。
    _pb_val = int(getattr(args, "prob_payload_bits", 16) or 16)
    if _pb_val < 16:
        _mf_spec = MODE_FEATURES.get(args.eval_mode)
        if _mf_spec is None or not _mf_spec.supports_prob_bits:
            parser.error(
                f"--prob_payload_bits < 16 仅在 tridecoding / "
                f"adaptive_tridecoding 协议中生效；当前 --eval_mode "
                f"{args.eval_mode!r} 的上行载荷不参与位宽压缩，"
                f"如需全宽传输请使用默认值 16"
            )

    # L1 口径收敛：显式给出 --comm_accounting 时统一覆盖三个子开关。
    # 解决 Table V 对齐时发现的"口径不可辨"问题：以后每个 run 的口径都有唯一标签。
    if getattr(args, "comm_accounting", None):
        if args.comm_accounting == "honest":
            args.charge_residual_payload = True
            args.comm_round_trip_mode = "per_round"
            args.transfer_top_k_cap = 16
        else:  # legacy
            args.charge_residual_payload = False
            args.comm_round_trip_mode = "per_transfer"
            args.transfer_top_k_cap = 0
        print(
            f"[comm-accounting] 口径={args.comm_accounting} ⇒ "
            f"charge_residual={args.charge_residual_payload}, "
            f"round_trip={args.comm_round_trip_mode}, "
            f"topk_cap={args.transfer_top_k_cap}"
        )

    if args.run_full_dataset:
        args.eval_data_num = None

    explicit_small_draft_acc_head = "--small_draft_acc_head_path" in cli_args
    explicit_draft_target_acc_head = "--draft_target_acc_head_path" in cli_args
    explicit_acc_head = "--acc_head_path" in cli_args

    if (
        not explicit_small_draft_acc_head
        and getattr(args, "little_model", None) is not None
    ):
        args.small_draft_acc_head_path = resolve_acc_head_path(
            args.little_model, args.draft_model
        )
    if not explicit_draft_target_acc_head:
        args.draft_target_acc_head_path = resolve_acc_head_path(
            args.draft_model, args.target_model
        )
    if not explicit_acc_head:
        args.acc_head_path = args.draft_target_acc_head_path

    # D3：默认路径解析单点化（原两处四块复制逻辑下沉 registry）
    args.main_rl_path, args.main_rl_best_path = resolve_rl_agent_paths(
        ROLE_MAIN,
        little_model=getattr(args, "little_model", None),
        draft_model=args.draft_model,
        target_model=args.target_model,
        latest=args.main_rl_path,
        best=getattr(args, "main_rl_best_path", None),
        checkpoint_root=args.rl_checkpoint_root,
    )

    if (
        getattr(args, "little_model", None) is not None
        and bool(
            (_mf_spec := MODE_FEATURES.get(args.eval_mode)) is None
            or _mf_spec.uses_little_rl
        )
    ):
        args.little_rl_path, args.little_rl_best_path = resolve_rl_agent_paths(
            ROLE_LITTLE,
            little_model=args.little_model,
            draft_model=args.draft_model,
            target_model=args.target_model,
            latest=args.little_rl_path,
            best=getattr(args, "little_rl_best_path", None),
            checkpoint_root=args.rl_checkpoint_root,
        )

    args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    # 生效口径自述：此处后处理（协议 / comm_accounting / 路径解析）均已生效。
    emit_effective_report(args, _protocol_app)

    model_zoo(args)
    return args
