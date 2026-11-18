import json
import os
import math
import time
import warnings
from typing import Callable, Dict as Dict, List, Optional, Tuple, cast

import torch
import transformers

from ..decoding_types import RollbackPlan as _CeeRollbackPlan
from ._cee_state import (
    CeeDecodeStats as _CeeDecodeStats,
    CeeProposal as _CeeProposal,
    CeeVerification as _CeeVerification,
)

from ..SpecDec_pp.specdec_pp.wrap_model import AcceptancePredictionHead

transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")

from ..adapter import DecodingAdapter
from ..comm_accounting import CommEnv
from ..communication import (
    CUHLM,
    CommunicationSimulator,
    PreciseCommunicationSimulator as PreciseCommunicationSimulator,
    PreciseCUHLM,
    PROB_QUANT_PASS_THROUGH_BITS,
    cuhlm_uplink_payload_bytes,
    tk_slt_uplink_payload_bytes,  # noqa: F401
)
from ..cee_prefix import CEETriPrefix
from ..checkpoint_provenance import (
    describe_dir,
    describe_path,
    record as record_checkpoint,
    summary_line,
)
from ..decoding_ops import (
    _prepare_legacy_verification_inputs,
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
from ..acc_head_registry import anchor_repo_path, is_usable_acc_head_dir
from ..engine import Decoding
from ..graph_decode import (  # B19：CUDA Graph 接线的单一入口
    acquire_graph_caches,
    graph_mode_cache_kwargs as _graph_mode_cache_kwargs,
    kv_cache_construction_key as _kv_cache_construction_key,
)
from ..metrics import INT_SIZE, DecodingMetrics, get_empty_metrics
from ..model_gpu import KVCacheModel
from ..mode_features import MODE_FEATURES
from ..proposal_utils import (
    apply_transfer_top_k_cap,
    build_stage_prefix_topk_history,
    build_draft_probs_override,  # noqa: F401
    build_topk_proposal_history_step,
    concat_topk_proposal_history,
    merge_stage_topk_histories,
    proposal_top_k,
    stage_topk_proposal_history,
    stage_prob_history,
)
from ..register import Register
from ..rl_agent_registry import (
    ROLE_LITTLE,
    ROLE_MAIN,
    get_rl_agent_spec,
    resolve_legacy_rl_agent_load_path,
)
from ..rl_adapter import RLNetworkAdapter
from ..utils import (
    max_fn,
    norm_logits,
    rebuild_topk_uniform_probs,
    sample,
    skip_token_validation,
    state_entropy,
)


from ..baselines_helpers import (
    _add_comm_accounting_metrics,
    _add_per_model_wall_time,
    _compute_token_vocab_rank,
    _compute_transfer_topk_rank,
    _ensure_token_shape,
    _finalize_comm_metrics,
    _lambert_w_minus_one,  # noqa: F401
    _last_edge_cloud_tx_seconds,  # noqa: F401
    _move_token_tensor,
    _quantize_probs_fp16,  # noqa: F401
    _quantize_probs_logspace,
    _record_accepted_token_ranks as _record_accepted_token_ranks,
    _reward_comm_s,
    _rl_train_paths,
    _sample_token_from_probs,  # noqa: F401
    _send_downlink_index_only,
    _send_downlink_token,
    _simulate_topk_prob_transfer,
    _validate_token_range,
    compute_stage_reward,
    get_decoding_fn,
    load_acceptance_prediction_head,
    tk_slt_optimal_draft_length,  # noqa: F401
    tk_slt_select_speculative,  # noqa: F401
    tk_slt_speedup_ratio,  # noqa: F401
)


from ._adaptive_state import (
    AdaptiveDecodeStats as _AdaptiveDecodeStats,
    AdaptiveProposal as _AdaptiveProposal,
)

# 私有 re-export：实现已移到 simulators.py；保留 core 全局名，
# 让既有 src.baselines 导入和调用方的方法体继续解析到同一个函数。
from .simulators import (
    _make_comm_simulator,
    _make_symmetric_comm_simulator,
)


# Compatibility re-export: the implementation belongs to its protocol module.
from .cee_cuhlm import _finalize_cuhlm_verification


# A2：算法族实现已迁到真实定义模块；这里只做组合与有限兼容 re-export。
# import 别名保持 core 的星号导出面不变，避免把 mixin 名称变成新公共 API。
from .adaptive import AdaptiveDecodingMixin as _AdaptiveDecodingMixin
from .cee import CeeSdMixin as _CeeSdMixin
from .cee_cuhlm import CeeCuhlmMixin as _CeeCuhlmMixin
from .cee_distributed import CeeDistributedMixin as _CeeDistributedMixin
from .cuhlm import CuhlmMixin as _CuhlmMixin
from .dual import DualDecodingMixin as _DualDecodingMixin
from .operations import SharedOperations as _SharedOperations
from .operations import _uplink_prob_payload
from . import runtime as _runtime
from .runtime import _build_cache
from .tk_slt import (
    _TK_SLT_ODLD_GAMMA_CAP,  # noqa: F401
    TkSltMixin as _TkSltMixin,
)


class Baselines(
    _AdaptiveDecodingMixin,
    _CuhlmMixin,
    _TkSltMixin,
    _CeeSdMixin,
    _CeeCuhlmMixin,
    _CeeDistributedMixin,
    _DualDecodingMixin,
    _SharedOperations,
    Decoding,
):
    """A fresh run's policies, heads and sample caches over resident weights."""

    _acquire_three_layer_caches = _runtime._acquire_three_layer_caches
    _acquire_draft_target_caches = _runtime._acquire_draft_target_caches
    build_adaptive_tridecoding_caches = _runtime.build_adaptive_tridecoding_caches
    _save_adaptive_rl_checkpoints = _runtime._save_adaptive_rl_checkpoints

    def __init__(self, args):
        super().__init__(args)
        _runtime._configure_ntt_trace(args)
        self.rl_adapter, self.little_rl_adapter = _runtime._build_rl_adapters(args)
        self.task = "unknown"

    def load_model(self):
        self.load_weights()
        self.load_acc_head()

    def load_acc_head(self):
        """Build this run's ARP heads after weights are loaded or reattached."""
        resources = _runtime._build_acc_head_resources(
            self.args,
            little_model=getattr(self, "little_model", None),
            draft_model=getattr(self, "draft_model", None),
        )
        for name, value in resources.items():
            setattr(self, name, value)


    @staticmethod
    def _collect_dssd_uplink_payload(
        prob_history: torch.Tensor,
        x: torch.Tensor,
        prefix_len: int,
        gamma: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return collect_verification_payload(prob_history, x, prefix_len, gamma)


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
        """Legacy 实现（旧验证/回滚血缘），已移到 ``src/legacy/ceesd_without_arp.py``。

        仅用于复现历史 run；新的无 ARP 消融请用 ``cee_sd_no_arp``。
        """
        from ..legacy.ceesd_without_arp import run as _legacy_run

        # 兼容协议内省：实现已移到 legacy 模块，但 protocols.mode_consumption
        # 通过 co_names 判断本模式消费哪些计费/深度开关。这里保留只读引用，
        # 让声明与运行时观测一致（不改变行为）。
        _wiring = (
            self.comm_config.comm_round_trip_mode,
            self.comm_config.charge_residual_payload,
            self.comm_config.transfer_top_k_cap,
            self.args.gamma1,
            self.args.gamma2,
        )
        del _wiring

        return _legacy_run(
            self,
            prefix,
            transfer_top_k=transfer_top_k,
            use_precise_comm_sim=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
            use_early_stopping=use_early_stopping,
            stop_sequences=stop_sequences,
            **kwargs,
        )


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
