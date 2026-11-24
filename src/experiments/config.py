"""One experiment's plain-dict configuration, including resource paths and run name."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypedDict

from src.acc_head_registry import resolve_acc_head_path
from src.mode_features import specedge_candidate_limit
from src.rl_agent_registry import ROLE_LITTLE, ROLE_MAIN, resolve_rl_agent_paths


class EvalDataset(str, Enum):
    mt_bench = "eval/eval_mt_bench.py"
    humaneval = "eval/eval_humaneval.py"
    cnndm = "eval/eval_cnndm.py"
    xsum = "eval/eval_xsum.py"
    gsm8k = "eval/eval_gsm8k.py"
    mt_bench_noeval = "eval/eval_mt_bench_noeval.py"


class EvalMode(str, Enum):
    autoregression = "large"
    dssd = "dist_split_spec"
    dsd = "dist_spec"
    cuhlm = "uncertainty_decoding"
    tk_slt = "tk_slt"
    specedge = "specedge"
    tridecoding = "tridecoding"
    ceesd = "adaptive_tridecoding"
    cee_cuhlm = "cee_cuhlm"
    cee_dssd = "cee_dssd"
    cee_dsd = "cee_dsd"
    adaptive_decoding = "adaptive_decoding"


class _ExplicitProtocol(TypedDict, total=False):
    """Extra CLI settings pinned by the current-code paper rerun, not old sweeps."""

    protocol: str
    seed: int
    model_source: str
    model_dtype: str
    little_quantization: str
    draft_quantization: str
    target_quantization: str
    comm_trace_mode: str
    humaneval_generate_only: bool
    specedge_lookahead: int


class ExpConfig(_ExplicitProtocol):
    CUDA_VISIBLE_DEVICES: str | int
    eval_mode: str | EvalMode
    edge_end_bandwidth: int | float
    edge_cloud_bandwidth: int | float
    cloud_end_bandwidth: int | float
    min_bandwidth_mbps: float
    comm_bw_model: str
    transfer_top_k: int
    num_shots: int
    num_samples_per_task: int
    eval_data_num: int | None
    exp_name: str
    use_precise: bool
    use_stochastic_comm: bool
    use_rl_adapter: bool
    disable_rl_update: bool
    small_draft_threshold: float
    draft_target_threshold: float
    uncertainty_threshold: float
    ntt_ms_edge_cloud: int | float
    ntt_ms_edge_end: int | float
    eval_dataset: str | EvalDataset
    run_full_dataset: bool
    random_sample: bool
    sample_seed: int
    draft_model: str
    target_model: str
    little_model: str
    acc_head_path: str
    small_draft_acc_head_path: str
    draft_target_acc_head_path: str
    main_rl_path: str
    little_rl_path: str
    main_rl_best_path: str
    little_rl_best_path: str
    max_tokens: int
    gamma: int
    gamma1: int
    gamma2: int
    use_early_stopping: bool
    dump_network_stats: bool
    batch_delay: int | float
    use_cuda_graph: bool


def _head_path(override: str | None, source: str, target: str) -> str:
    # An explicit empty path is distinct from asking for a registry default.
    return resolve_acc_head_path(source, target) if override is None else override


def create_config(
    eval_mode: str | EvalMode,
    ntt_ms_edge_cloud: int | float = 0,
    ntt_ms_edge_end: int | float = 0,
    batch_delay: int | float = 50e-3,
    use_precise: bool = True,
    use_stochastic_comm: bool = False,
    CUDA_VISIBLE_DEVICES: str | int = "0",
    use_rl_adapter: bool = False,
    disable_rl_update: bool = False,
    edge_end_bandwidth: int | float = 100,
    edge_cloud_bandwidth: int | float = 100,
    cloud_end_bandwidth: int | float = 100,
    min_bandwidth_mbps: float = 5.0,
    comm_bw_model: str = "instant",
    small_draft_threshold: float = 0.3,
    draft_target_threshold: float = 0.9,
    uncertainty_threshold: float = 0.8,
    transfer_top_k: int = 300,
    gamma: int = 5,
    gamma1: int = 5,
    gamma2: int = 5,
    num_shots: int = 3,
    num_samples_per_task: int = 1,
    eval_data_num: int | None = 80,
    max_tokens: int = 128,
    use_early_stopping: bool = False,
    eval_dataset: str | EvalDataset = EvalDataset.mt_bench,
    run_full_dataset: bool = False,
    random_sample: bool = False,
    sample_seed: int = 1234,
    draft_model: str = "tiny-vicuna-1b",
    target_model: str = "vicuna-13b-v1.5",
    little_model: str = "vicuna-68m",
    acc_head_path: str | None = None,
    small_draft_acc_head_path: str | None = None,
    draft_target_acc_head_path: str | None = None,
    main_rl_path: str | None = None,
    little_rl_path: str | None = None,
    main_rl_best_path: str | None = None,
    little_rl_best_path: str | None = None,
    dump_network_stats: bool = False,
    use_cuda_graph: bool = False,
    specedge_lookahead: int | None = None,
) -> ExpConfig:
    """Normalize a config for JSON/logging; only resolve resources its mode can use."""
    mode = eval_mode.value if isinstance(eval_mode, EvalMode) else eval_mode
    if isinstance(eval_dataset, EvalDataset):
        dataset, dataset_name = eval_dataset.value, eval_dataset.name
    else:
        dataset, dataset_name = (
            eval_dataset,
            Path(eval_dataset).stem.removeprefix("eval_"),
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_delay_ms = int(round(batch_delay * 1000))
    config = ExpConfig(
        eval_dataset=dataset,
        run_full_dataset=run_full_dataset,
        random_sample=random_sample,
        sample_seed=sample_seed,
        CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
        eval_mode=mode,
        edge_end_bandwidth=edge_end_bandwidth,
        edge_cloud_bandwidth=edge_cloud_bandwidth,
        cloud_end_bandwidth=cloud_end_bandwidth,
        min_bandwidth_mbps=min_bandwidth_mbps,
        comm_bw_model=comm_bw_model,
        transfer_top_k=transfer_top_k,
        gamma=gamma,
        gamma1=gamma1,
        gamma2=gamma2,
        num_shots=num_shots,
        num_samples_per_task=num_samples_per_task,
        eval_data_num=eval_data_num,
        max_tokens=max_tokens,
        small_draft_threshold=small_draft_threshold,
        draft_target_threshold=draft_target_threshold,
        uncertainty_threshold=uncertainty_threshold,
        exp_name=(
            f"{mode}/{dataset_name}/{mode}_{num_shots}shot_g1{gamma1}_g2{gamma2}"
            f"_batchdelay{batch_delay_ms}ms{'_cudagraph' if use_cuda_graph else ''}"
            f"_bw{edge_cloud_bandwidth:g}_{timestamp}"
        ),
        use_precise=use_precise,
        use_stochastic_comm=use_stochastic_comm,
        ntt_ms_edge_cloud=ntt_ms_edge_cloud,
        ntt_ms_edge_end=ntt_ms_edge_end,
        batch_delay=batch_delay,
        use_rl_adapter=use_rl_adapter,
        disable_rl_update=disable_rl_update,
        draft_model=draft_model,
        target_model=target_model,
        little_model=little_model,
        use_early_stopping=use_early_stopping,
        acc_head_path="",
        small_draft_acc_head_path="",
        draft_target_acc_head_path="",
        main_rl_path="",
        little_rl_path="",
        main_rl_best_path="",
        little_rl_best_path="",
        dump_network_stats=dump_network_stats,
        use_cuda_graph=use_cuda_graph,
    )

    if mode == EvalMode.specedge:
        config["specedge_lookahead"] = specedge_candidate_limit(
            gamma, specedge_lookahead
        )

    paired = mode in (
        EvalMode.ceesd,
        EvalMode.cee_cuhlm,
        EvalMode.cee_dsd,
        EvalMode.cee_dssd,
    )
    if paired:
        config["small_draft_acc_head_path"] = _head_path(
            small_draft_acc_head_path, little_model, draft_model
        )
        config["draft_target_acc_head_path"] = _head_path(
            draft_target_acc_head_path, draft_model, target_model
        )
    elif mode == EvalMode.adaptive_decoding:
        config["acc_head_path"] = _head_path(acc_head_path, draft_model, target_model)
    else:
        return config

    config["main_rl_path"], config["main_rl_best_path"] = resolve_rl_agent_paths(
        ROLE_MAIN,
        little_model=little_model if paired else None,
        draft_model=draft_model,
        target_model=target_model,
        latest=main_rl_path,
        best=main_rl_best_path,
    )
    if paired:
        config["little_rl_path"], config["little_rl_best_path"] = (
            resolve_rl_agent_paths(
                ROLE_LITTLE,
                little_model=little_model,
                draft_model=draft_model,
                target_model=target_model,
                latest=little_rl_path,
                best=little_rl_best_path,
            )
        )
    return config
