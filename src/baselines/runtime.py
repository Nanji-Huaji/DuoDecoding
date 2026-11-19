"""Per-run policy/head factories and per-sample caches for Baselines.

RL depends only on run settings. ARP is built after the language models are
available so its heads land on the correct devices. Graph caches live on the
run that captured them; eager caches are fresh for each sample.
"""

import os
from typing import Optional, Tuple

from ..adapter import DecodingAdapter
from ..baselines_helpers import (
    _rl_train_paths,
    load_acceptance_prediction_head,
)
from ..checkpoint_provenance import (
    describe_dir,
    describe_path,
    summary_line,
)
from ..checkpoint_provenance import (
    record as record_checkpoint,
)
from ..comm_accounting import CommEnv
from ..graph_decode import (
    acquire_graph_caches,
    kv_cache_construction_key,
)
from ..graph_decode import (
    graph_mode_cache_kwargs as _graph_mode_cache_kwargs,
)
from ..mode_features import MODE_FEATURES
from ..model_gpu import KVCacheModel
from ..rl_adapter import RLNetworkAdapter
from ..rl_agent_registry import (
    ROLE_LITTLE,
    ROLE_MAIN,
    get_rl_agent_spec,
    resolve_legacy_rl_agent_load_path,
)


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


def _configure_ntt_trace(args):
    # 真实 RTT trace 回放：模块级一次性配置（所有模拟器实例共享，含基线）。
    # 收显式 args 而不是 decoder：测试会用 patched Decoding.__init__ 跳过
    # 基类初始化（没有 self.args），但 trace 配置仍须可测。
    env = CommEnv.from_args(args)
    ntt_trace_file = env.ntt_trace_file
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
            scale=env.ntt_trace_scale,
            src=os.path.basename(ntt_trace_file),
        )


def _build_rl_adapters(args) -> tuple:
    """按 args 构造 main/little 两个 RL 策略适配器并返回（不落任何 decoder 状态）。

    策略参数、checkpoint 解析与 provenance 记录（record_checkpoint/print）
    与历史行为逐字一致；``Baselines.__init__`` 负责把返回值挂到
    ``self.rl_adapter`` / ``self.little_rl_adapter``。
    """
    # D2：模式能力查询单点化（原两处硬编码集合移入 mode_features）
    eval_mode = getattr(args, "eval_mode", "")
    _spec = MODE_FEATURES.get(eval_mode)
    uses_main_rl = bool(_spec and _spec.uses_main_rl)
    uses_little_rl = bool(_spec and _spec.uses_little_rl)

    rl_adapter = None
    little_rl_adapter = None
    if getattr(args, "use_rl_adapter", False):
        checkpoint_root = getattr(
            args, "rl_checkpoint_root", "checkpoints/rl_agents"
        )
        init_seed = getattr(args, "rl_init_seed", None)
        init_strategy = getattr(args, "rl_init_strategy", "resume")
        rl_output_dir = getattr(args, "rl_output_dir", None)
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
            # 阈值候选格：CLI 显式指定 > registry 默认。默认格顶 0.4 是
            # v2 策略学偏的根因②（深草稿不可达）；扩格训练时传
            # --rl_threshold_candidates，且格会随 checkpoint 持久化、
            # 评测加载时自动还原（不会重演训练/评测格错配）。
            _main_thr_grid = main_spec.threshold_candidates
            if getattr(args, "rl_threshold_candidates", None):
                _main_thr_grid = [
                    float(x)
                    for x in str(args.rl_threshold_candidates).split(",")
                    if x.strip()
                ]
                if not _main_thr_grid:
                    raise ValueError(
                        "--rl_threshold_candidates 解析为空格: "
                        f"{args.rl_threshold_candidates!r}"
                    )
            if max(float(x) for x in _main_thr_grid) < 0.5:
                print(
                    "Warning: main 阈值候选格上限 "
                    f"{max(float(x) for x in _main_thr_grid)} < 0.5 —— 深草稿"
                    "不可达（评测最优 ~0.8，见"
                    " experiment_results/ceesd_gamma_threshold_probe.md）。"
                    "训练请考虑 --rl_threshold_candidates 扩格。"
                )
            _main_model_path = (
                getattr(args, "main_rl_path", None) or main_spec.latest_path
            )
            _main_best_path = (
                getattr(args, "main_rl_best_path", None) or main_spec.best_path
            )
            _main_model_path, _main_best_path = _rl_train_paths(
                rl_output_dir, ROLE_MAIN, _main_model_path, _main_best_path
            )
            _info = describe_path(_main_model_path)
            record_checkpoint(args, "rl_main", _info)
            print(summary_line("rl_main", _info))
            if os.path.exists(_main_best_path):
                record_checkpoint(
                    args, "rl_main_best", describe_path(_main_best_path)
                )
            rl_adapter = RLNetworkAdapter(
                args,
                model_path=_main_model_path,
                best_model_path=_main_best_path,
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
                threshold_candidates=_main_thr_grid,
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

        if uses_little_rl:
            little_spec = get_rl_agent_spec(
                ROLE_LITTLE,
                little_model=args.little_model,
                draft_model=args.draft_model,
                target_model=args.target_model,
                checkpoint_root=checkpoint_root,
            )
            _little_model_path = (
                getattr(args, "little_rl_path", None) or little_spec.latest_path
            )
            _little_best_path = (
                getattr(args, "little_rl_best_path", None)
                or little_spec.best_path
            )
            _little_model_path, _little_best_path = _rl_train_paths(
                rl_output_dir, ROLE_LITTLE, _little_model_path, _little_best_path
            )
            _info = describe_path(_little_model_path)
            record_checkpoint(args, "rl_little", _info)
            print(summary_line("rl_little", _info))
            if os.path.exists(_little_best_path):
                record_checkpoint(
                    args, "rl_little_best", describe_path(_little_best_path)
                )
            little_rl_adapter = RLNetworkAdapter(
                args,
                model_path=_little_model_path,
                best_model_path=_little_best_path,
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
    return rl_adapter, little_rl_adapter


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
    *,
    cache_kwargs: Optional[dict] = None,
) -> dict[str, KVCacheModel]:
    """little/draft/target 三层缓存的统一入口（B19）。

    三个模型每轮都做多 token 验证前向（little 验证 draft、draft 验证 little
    的接受段、target 验证 draft —— 这是调度开销的主体），所以**三个都开图**；
    图开启时经 acquire_graph_caches 跨样本复用（StaticCache 原地 reset ⇒ 图
    不重捕获）。little/draft 用 draft_top_k 压缩、target 不压缩，与各方法
    历史取值一致。cache_kwargs 只设置构造参数，不开启图模式或跨样本复用。

    ``self`` 是缓存复用的存储点（即 decoder 本身）：图缓存挂在它的属性上，
    才能在同一 decoder 的样本之间复用。
    """
    builder_kw = {**(cache_kwargs or {}), **graph_kw}
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
                **builder_kw,
            ),
            "draft": lambda: _build_cache(
                self.draft_model,
                temperature=self.args.temp,
                top_k=draft_top_k,
                top_p=self.args.top_p,
                vocab_size=self.vocab_size,
                **builder_kw,
            ),
            "target": lambda: _build_cache(
                self.target_model,
                temperature=self.args.temp,
                top_k=0,
                top_p=0,
                vocab_size=self.vocab_size,
                **builder_kw,
            ),
        },
        cache_key={
            role: kv_cache_construction_key(
                model,
                temperature=self.args.temp,
                top_k=top_k,
                top_p=top_p,
                vocab_size=self.vocab_size,
                **builder_kw,
            )
            for role, model, top_k, top_p in (
                ("little", self.little_model, draft_top_k, self.args.top_p),
                ("draft", self.draft_model, draft_top_k, self.args.top_p),
                ("target", self.target_model, 0, 0),
            )
        } if graph_kw else None,
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
    ``self``（decoder）是跨样本复用的存储点。
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
        cache_key={
            "draft": kv_cache_construction_key(
                self.draft_model,
                temperature=self.args.temp,
                top_k=draft_top_k,
                top_p=self.args.top_p,
                vocab_size=self.vocab_size,
                **graph_kw,
            ),
            "target": kv_cache_construction_key(
                self.target_model,
                temperature=self.args.temp,
                top_k=target_top_k,
                top_p=target_top_p,
                vocab_size=self.vocab_size,
            ),
        } if graph_kw else None,
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


def _load_acc_head(args, key: str, path, device_model):
    """加载单个验收头：provenance 记录 + eval + 搬到配套模型所在设备。"""
    _info = describe_dir(path)
    record_checkpoint(args, key, _info)
    print(summary_line(key, _info))
    head = load_acceptance_prediction_head(path)
    head.eval()
    if device_model is not None:
        head.to(device_model.device)
    return head


def _build_acc_head_resources(
    args,
    *,
    little_model=None,
    draft_model=None,
) -> dict:
    """按 args 构造本次运行的 ARP head/adapter，返回 decoder 应挂载的属性表。

    必须在权重就位之后调用（含 ModelStore 复用已加载权重的路径）：
    head 需要搬到 little/draft 模型所在的设备。返回 {属性名: 值}；
    模式不需要验收头时返回空表（与历史"不设置任何属性"一致），由
    ``Baselines.load_acc_head`` 统一 setattr，属性只存在一份。
    """
    _spec = MODE_FEATURES.get(args.eval_mode)
    _acc = _spec.acc_head if _spec else "none"
    if _acc == "draft_target":
        draft_target_threshold: float | int = args.draft_target_threshold
        acc_head_path = args.acc_head_path
        acc_head = _load_acc_head(args, "acc_head", acc_head_path, draft_model)
        return {
            "acc_head_path": acc_head_path,
            "acc_head": acc_head,
            "adapter": DecodingAdapter(
                acc_head,
                draft_target_threshold,
                stop_mode=getattr(args, "arp_stop_mode", "cumulative"),
            ),
        }
    if _acc == "both":
        small_draft_threshold: float | int = args.small_draft_threshold
        draft_target_threshold: float | int = args.draft_target_threshold
        small_draft_acc_head_path = args.small_draft_acc_head_path
        draft_target_acc_head_path = args.draft_target_acc_head_path
        small_draft_acc_head = _load_acc_head(
            args, "acc_head_small_draft", small_draft_acc_head_path, little_model
        )
        draft_target_acc_head = _load_acc_head(
            args, "acc_head_draft_target", draft_target_acc_head_path, draft_model
        )
        return {
            "small_draft_acc_head_path": small_draft_acc_head_path,
            "small_draft_acc_head": small_draft_acc_head,
            "draft_target_acc_head_path": draft_target_acc_head_path,
            "draft_target_acc_head": draft_target_acc_head,
            "small_draft_adapter": DecodingAdapter(
                small_draft_acc_head,
                small_draft_threshold,
                stop_mode=getattr(args, "arp_stop_mode", "cumulative"),
            ),
            "draft_target_adapter": DecodingAdapter(
                draft_target_acc_head,
                draft_target_threshold,
                stop_mode=getattr(args, "arp_stop_mode", "cumulative"),
            ),
        }
    return {}
