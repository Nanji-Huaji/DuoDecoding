"""D2：eval_mode 能力表单点化回归测试。

Historical bug：新增模式要同步改 ≥5 处（装饰器注册、engine.load_model
两处分支列表、baselines 的 uses_main_rl/uses_little_rl 两集合、
load_acc_head 第三集合、utils 的 RL 路径特判），漏改即静默走错路径。
本测试锁定：能力表与注册表双向完备、字段与迁移前的硬编码集合逐一相等、
注册器拒绝重名/非法名、get_decoding_method 不再反射兜底。
"""

from argparse import Namespace

import pytest

from src.mode_features import MODE_FEATURES, get_mode_spec
from src.register import Register

# ---- 迁移前的硬编码集合（迁移等价性的ground truth，勿随手改）----
# 2026-10 追加 tk_slt/tkslt（TK-SLT 基线，WCSP'25 Zheng & Yang）：双模型
# 拓扑（68M 草稿 + 大模型验证，同 dsd/cuhlm），并入 dual ground truth。
OLD_DUAL = {
    "sd", "dsd", "dssd", "dist_spec", "dist_split_spec",
    "uncertainty_decoding", "cuhlm",
    "tk_slt", "tkslt",
    "speculative_decoding_with_bandwidth",
}
OLD_TRI = {
    "tridecoding", "adaptive_tridecoding", "target_only",
    "cee_sd", "cee_sd_opportunistic", "ceesd_without_arp",
    "ceesd_w/o_arp", "cee_cuhlm", "cee_dsd", "cee_dssd",
}
OLD_USES_MAIN_RL = {
    "adaptive_decoding", "adaptive_tridecoding", "cee_sd",
    "cee_sd_opportunistic", "cee_cuhlm", "cee_dsd", "cee_dssd",
    "ceesd_without_arp", "ceesd_w/o_arp",
}
OLD_USES_LITTLE_RL = OLD_USES_MAIN_RL - {"adaptive_decoding"}
OLD_ACC_BOTH = {
    "adaptive_tridecoding", "cee_sd", "cee_cuhlm", "cee_dsd",
    "cee_dssd", "cee_sd_opportunistic",
}


class TestModeFeaturesTable:
    def test_every_registered_mode_has_spec(self):
        import src.baselines  # noqa: F401 触发装饰器注册
        import src.engine  # noqa: F401

        missing = set(Register._DECODING_REGISTRY) - set(MODE_FEATURES)
        assert not missing, f"注册了但能力表缺条目: {missing}"

    def test_every_spec_mode_is_registered(self):
        import src.baselines  # noqa: F401 触发装饰器注册
        import src.engine  # noqa: F401

        # 幽灵模式 speculative_decoding_with_bandwidth_full_prob 已删除：
        # 无实现注册；该能力属 --force_full_vocab_transfer 开关，不是 eval_mode。
        unregistered = set(MODE_FEATURES) - set(Register._DECODING_REGISTRY)
        assert not unregistered, f"能力表有但未注册: {unregistered}"

    def test_ghost_full_prob_mode_removed(self):
        # 幽灵模式已从能力表删除，防止未来被误加回
        assert (
            "speculative_decoding_with_bandwidth_full_prob" not in MODE_FEATURES
        )

    def test_model_topology_matches_old_branches(self):
        dual = {m for m, s in MODE_FEATURES.items() if s.models == "dual"}
        tri = {m for m, s in MODE_FEATURES.items() if s.models == "tri"}
        assert dual == OLD_DUAL | {"adaptive_decoding"}
        assert tri == OLD_TRI

    def test_rl_flags_match_old_sets(self):
        main_rl = {m for m, s in MODE_FEATURES.items() if s.uses_main_rl}
        little_rl = {m for m, s in MODE_FEATURES.items() if s.uses_little_rl}
        assert main_rl == OLD_USES_MAIN_RL
        assert little_rl == OLD_USES_LITTLE_RL

    def test_acc_head_tiers_match_old_sets(self):
        both = {m for m, s in MODE_FEATURES.items() if s.acc_head == "both"}
        draft_target = {
            m for m, s in MODE_FEATURES.items() if s.acc_head == "draft_target"
        }
        assert both == OLD_ACC_BOTH
        assert draft_target == {"adaptive_decoding"}
        # 无 ARP 变体：用 RL 但不带头
        assert MODE_FEATURES["ceesd_without_arp"].acc_head == "none"

    def test_unknown_mode_raises_with_available_list(self):
        with pytest.raises(ValueError, match="未知 eval_mode"):
            get_mode_spec("no_such_mode")


class TestRegisterHardening:
    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="重复注册"):

            @Register.register_decoding("sd")
            def _dup(self):
                pass

    def test_illegal_name_rejected(self):
        with pytest.raises(ValueError, match="非法"):
            Register.register_decoding("__init__")
        with pytest.raises(ValueError, match="非法"):
            Register.register_decoding("has space")

    def test_no_reflection_fallback(self):
        # hasattr 兜底已删：任意属性名（如 __init__）必须报错而非被当方法返回
        reg = Register(Namespace(eval_mode="__init__"))
        with pytest.raises(NotImplementedError, match="not found"):
            reg.get_decoding_method()

    def test_registered_mode_resolves(self):
        import src.baselines  # noqa: F401

        reg = Register(Namespace(eval_mode="sd"))
        method = reg.get_decoding_method()
        assert callable(method)
