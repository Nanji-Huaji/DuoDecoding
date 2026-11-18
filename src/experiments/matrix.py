"""Table V experiment matrix. Importing this module does not build any configs."""

import os
from itertools import product

from .config import EvalDataset, EvalMode, ExpConfig, create_config

# Ordered axes: series (little, draft, target), dataset, mode, WAN bandwidth, delay.
llama_series = ("llama-68m", "tiny-llama-1.1b", "llama-2-13b")
qwen_1_5_series = (
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-7B-Chat",
)
qwen_series = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-14B")
paper_table5_series = (llama_series, qwen_1_5_series, qwen_series)
paper_table5_datasets = (
    EvalDataset.mt_bench_noeval,
    EvalDataset.gsm8k,
    EvalDataset.humaneval,
)
paper_table5_modes = (
    EvalMode.dsd,
    EvalMode.dssd,
    EvalMode.cuhlm,
    EvalMode.tk_slt,
    EvalMode.ceesd,
)

# docs/protocol.md §2: 46 Mbps matches Table II; 10 and 5 are constrained links.
edge_cloud_bandwidth = [46.0, 10.0, 5.0]
batch_delay_values = [50e-3]  # seconds per cloud request
NTT_MS_EDGE_CLOUD = 50
NTT_MS_EDGE_END = 0.317

# The comparison is at comparable accuracy; 0.08 is CUHLM's Llama/GSM8K frontier.
UNCERTAINTY_THRESHOLD_CUHLM = 0.08
UNCERTAINTY_THRESHOLD_DEFAULT = 0.8
# DSD and DSSD transmit full rows. TK-SLT uses its Fig. 4 optimum; Static uses
# the fixed top-k of Table VIII. CUHLM does not consume this config field.
TRANSFER_TOP_K_PAPER = 0
TRANSFER_TOP_K_OURS = 300
TRANSFER_TOP_K_TKSLT = 320
TRANSFER_TOP_K_STATIC = 1024
_TOP_K_BY_MODE = {
    EvalMode.ceesd: TRANSFER_TOP_K_OURS,
    EvalMode.cuhlm: TRANSFER_TOP_K_OURS,
    EvalMode.tk_slt: TRANSFER_TOP_K_TKSLT,
    EvalMode.tridecoding: TRANSFER_TOP_K_STATIC,
}
# DSD/CUHLM/TK-SLT draft on the end device. DSSD and CEE use the middle model.
_USE_MIDDLE_DRAFT = {
    EvalMode.dssd,
    EvalMode.ceesd,
    EvalMode.tridecoding,
    EvalMode.adaptive_decoding,
    EvalMode.cee_cuhlm,
    EvalMode.cee_dsd,
    EvalMode.cee_dssd,
}

# docs/protocol.md §4: the original matrix uses uniform γ=3. by_mode opts in
# to the method × bandwidth sweep; an unmeasured cell keeps the legacy window.
GAMMA_SINGLE = 3
GAMMA_BY_MODE_BANDWIDTH = {
    "dist_spec": {46.0: 3, 10.0: 1, 5.0: 1},
    "dist_split_spec": {46.0: 8, 10.0: 8, 5.0: 8},
    "tk_slt": {46.0: 8, 10.0: 8, 5.0: 8},
    "uncertainty_decoding": {46.0: 3, 10.0: 3, 5.0: 3},
}
_SERIES_GAMMA_SUFFIX = {
    llama_series[-1]: "LLAMA",
    qwen_1_5_series[-1]: "QWEN15",
    qwen_series[-1]: "QWEN3",
}


def gamma_for_matrix(mode_value: str, edge_cloud_bw: float) -> int:
    """Baseline γ for a mode and WAN bandwidth, using the current environment."""
    schedule = os.environ.get("EXP_GAMMA_SCHEDULE", "legacy").strip().lower()
    if schedule == "legacy":
        return GAMMA_SINGLE
    if schedule == "by_mode":
        return GAMMA_BY_MODE_BANDWIDTH.get(mode_value, {}).get(
            float(edge_cloud_bw), GAMMA_SINGLE
        )
    raise ValueError(f"EXP_GAMMA_SCHEDULE 只接受 legacy / by_mode，收到 {schedule!r}")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} 需要整数，收到 {value!r}") from None


def _tri_windows(target: str) -> tuple[int, int]:
    """A family override has priority over CEESD_GAMMA1/2 (both default to 5)."""
    suffix = next(
        (s for prefix, s in _SERIES_GAMMA_SUFFIX.items() if target.startswith(prefix)),
        None,
    )

    def window(which: int) -> int:
        key = f"CEESD_GAMMA{which}"
        default = _env_int(key, 5)
        return _env_int(f"{key}_{suffix}", default) if suffix else default

    return window(1), window(2)


def _tokens(csv: str | None) -> list[str]:
    return [token for item in (csv or "").split(",") if (token := item.strip())]


def _matrix_modes(extra_modes: str | None) -> list[EvalMode]:
    modes = list(paper_table5_modes)
    for token in _tokens(extra_modes):
        try:
            mode = EvalMode.__members__.get(token) or EvalMode(token)
        except ValueError:
            raise ValueError(
                f"--extra-modes 里有未知方法 {token!r}；"
                f"可用: {sorted(EvalMode.__members__)}"
            ) from None
        if mode in modes:
            raise ValueError(f"--extra-modes 里 {token!r} 重复或已属主表")
        modes.append(mode)
    return modes


def build_matrix(extra_modes: str | None = None) -> list[ExpConfig]:
    """Build fresh Table V configs in paper order, appending any requested methods."""
    modes = _matrix_modes(extra_modes)
    configs = []
    for little, middle, target in paper_table5_series:
        tri_windows = _tri_windows(target)
        for dataset, mode, bw, delay in product(
            paper_table5_datasets, modes, edge_cloud_bandwidth, batch_delay_values
        ):
            # The Table VIII Static ablation has the same two windows as CEE-SD,
            # but only CEE-SD uses the (frozen) RL adapter (protocol §2, item 12).
            gamma1, gamma2 = (
                tri_windows
                if mode in (EvalMode.ceesd, EvalMode.tridecoding)
                else (GAMMA_SINGLE, GAMMA_SINGLE)
            )
            configs.append(
                create_config(
                    eval_mode=mode,
                    eval_dataset=dataset,
                    little_model=little,
                    draft_model=middle if mode in _USE_MIDDLE_DRAFT else little,
                    target_model=target,
                    gamma=gamma_for_matrix(mode.value, bw),
                    gamma1=gamma1,
                    gamma2=gamma2,
                    transfer_top_k=_TOP_K_BY_MODE.get(mode, TRANSFER_TOP_K_PAPER),
                    uncertainty_threshold=UNCERTAINTY_THRESHOLD_CUHLM
                    if mode == EvalMode.cuhlm
                    else UNCERTAINTY_THRESHOLD_DEFAULT,
                    small_draft_threshold=0.6,
                    draft_target_threshold=0.7,
                    use_rl_adapter=mode == EvalMode.ceesd,
                    disable_rl_update=mode == EvalMode.ceesd,
                    ntt_ms_edge_cloud=NTT_MS_EDGE_CLOUD,
                    ntt_ms_edge_end=NTT_MS_EDGE_END,
                    batch_delay=delay,
                    edge_end_bandwidth=563,
                    edge_cloud_bandwidth=bw,
                    cloud_end_bandwidth=bw,
                    # Scale the trace floor so low-bandwidth cells retain fluctuations.
                    min_bandwidth_mbps=max(0.5, bw / 10),
                    comm_bw_model="fluid",
                    use_stochastic_comm=True,
                    use_precise=False,
                    max_tokens=128,
                    num_shots=3,
                    eval_data_num=80,
                    run_full_dataset=False,
                    random_sample=True,
                    sample_seed=1234,
                    use_early_stopping=False,
                    use_cuda_graph=True,
                )
            )
    return configs


def filter_configs_by_mode(
    configs: list[ExpConfig], only_modes: str | None = None
) -> list[ExpConfig]:
    """Select from this matrix by enum name or value, preserving its order."""
    if not only_modes:
        return list(configs)
    wanted = set(_tokens(only_modes))
    aliases_by_value = {mode.value: {mode.name, mode.value} for mode in EvalMode}
    available = set()
    selected = []
    for config in configs:
        value = config["eval_mode"]
        # Unknown string modes still match their literal value.
        aliases = aliases_by_value.get(value, {value})
        available.update(aliases)
        if not aliases.isdisjoint(wanted):
            selected.append(config)
    unknown = wanted - available
    if unknown:
        raise ValueError(
            f"--only-modes 里有未知方法 {sorted(unknown)}；可用: {sorted(available)}"
        )
    return selected
