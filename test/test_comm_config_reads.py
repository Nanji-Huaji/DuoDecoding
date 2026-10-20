"""CommConfig 接线守卫：5 个计费开关不得再从 args 直读。

背景：迁移前 29 个读取点散在 13 个方法里各自 ``getattr(args, ...)``，
2026-04 曾出现"标签对全表生效、实际只接线 tri 系"的口径错配
（src/protocols.py 模块头）。迁移后统一经 ``Baselines.comm_config``
（或 ``_comm = self.comm_config`` 局部别名）；本测试防止回退。

扫描两个注册解码方法所在文件：``src/baselines.py`` 与 ``src/engine.py``
（历史方法 ``speculative_decoding_with_bandwidth`` 曾漏审，2026-10-19
补入）。同时守 ``CommEnv`` 的 9 个通信数值键：迁移前它们在 ~20 处
构造点各自 ``getattr(self.args, ..., 默认)``，现在只许经 ``self.comm_env``。``src/utils.py`` 里 ``--prob_payload_bits`` 的 CLI 合法性校验
（B15 模式门）不是解码期读取，必须保留。
"""

import ast
from pathlib import Path

#: 通信数值层：迁移前在 ~20 处构造点各自 getattr，现已收进 CommEnv。
ENV_SWITCHES = {
    "edge_cloud_bandwidth",
    "edge_end_bandwidth",
    "cloud_end_bandwidth",
    "min_bandwidth_mbps",
    "comm_bw_model",
    "stochastic_ntt",
    "batch_delay",
    "ntt_trace_file",
    "ntt_trace_scale",
}

SWITCHES = {
    "comm_round_trip_mode",
    "charge_residual_payload",
    "transfer_top_k_cap",
    "force_full_vocab_transfer",
    "prob_payload_bits",
}

SRC = Path(__file__).resolve().parents[1] / "src"
# 所有注册解码方法所在文件：Baselines 一族 + engine.py 的历史方法
# （speculative_decoding_with_bandwidth 曾漏在审计外，2026-10-19 补上）。
SOURCES = (SRC / "baselines.py", SRC / "engine.py")


def _trees() -> dict[Path, ast.Module]:
    return {path: ast.parse(path.read_text(encoding="utf-8")) for path in SOURCES}


def _tree() -> ast.Module:
    return ast.parse((SRC / "baselines.py").read_text(encoding="utf-8"))


def _is_self_args(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "args"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def test_no_direct_args_reads_of_accounting_switches():
    offenders = []
    for path, tree in _trees().items():
      for node in ast.walk(tree):
        # getattr(self.args, "<switch>", ...)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and _is_self_args(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in SWITCHES
        ):
            offenders.append((path.name, node.lineno, node.args[1].value))
        # self.args.<switch>
        if (
            isinstance(node, ast.Attribute)
            and node.attr in SWITCHES
            and _is_self_args(node.value)
        ):
            offenders.append((path.name, node.lineno, node.attr))
    assert not offenders, (
        "计费开关仍从 args 直读（应改读 self.comm_config / _comm）："
        f"{offenders}"
    )


def test_every_switch_is_still_read_somewhere():
    attrs = set()
    for tree in _trees().values():
        attrs |= {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
    missing = SWITCHES - attrs
    assert not missing, (
        "以下开关在 src/baselines.py 里完全找不到属性读取，"
        f"可能是被别名隐藏或整段删掉了：{missing}"
    )


def test_uplink_prob_payload_takes_bits_not_args():
    fn = next(
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.FunctionDef) and node.name == "_uplink_prob_payload"
    )
    names = [a.arg for a in fn.args.args]
    assert names == ["probs", "bits"], (
        "_uplink_prob_payload 必须收 (probs, bits)，"
        f"由调用点传 self.comm_config.prob_payload_bits，实际: {names}"
    )


def test_no_direct_args_reads_of_env_knobs():
    """通信数值层也只能经 self.comm_env 读取（单一默认值来源）。"""
    offenders = []
    for path, tree in _trees().items():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and _is_self_args(node.args[0])
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in ENV_SWITCHES
            ):
                offenders.append((path.name, node.lineno, node.args[1].value))
            if (
                isinstance(node, ast.Attribute)
                and node.attr in ENV_SWITCHES
                and _is_self_args(node.value)
            ):
                offenders.append((path.name, node.lineno, node.attr))
    assert not offenders, (
        "通信数值仍从 args 直读（应改读 self.comm_env / CommEnv）："
        f"{offenders}"
    )


def test_comm_env_is_used_at_construction_sites():
    attrs = {
        node.attr
        for tree in _trees().values()
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert "comm_env" in attrs or "comm_env" in {
        node.id for tree in _trees().values() for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
