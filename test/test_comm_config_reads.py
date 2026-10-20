"""CommConfig 接线守卫：5 个计费开关不得再从 args 直读。

背景：迁移前 29 个读取点散在 13 个方法里各自 ``getattr(args, ...)``，
2026-04 曾出现"标签对全表生效、实际只接线 tri 系"的口径错配
（src/protocols.py 模块头）。迁移后统一经 ``Baselines.comm_config``
（或 ``_comm = self.comm_config`` 局部别名）；本测试防止回退。

只约束解码方法所在文件 ``src/baselines.py``：``src/utils.py`` 里
``--prob_payload_bits`` 的 CLI 合法性校验（B15 模式门）不是解码期读取，
必须保留。
"""

import ast
from pathlib import Path

SWITCHES = {
    "comm_round_trip_mode",
    "charge_residual_payload",
    "transfer_top_k_cap",
    "force_full_vocab_transfer",
    "prob_payload_bits",
}

BASELINES = Path(__file__).resolve().parents[1] / "src" / "baselines.py"


def _tree() -> ast.Module:
    return ast.parse(BASELINES.read_text(encoding="utf-8"))


def _is_self_args(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "args"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def test_no_direct_args_reads_of_accounting_switches():
    offenders = []
    for node in ast.walk(_tree()):
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
            offenders.append((node.lineno, node.args[1].value))
        # self.args.<switch>
        if (
            isinstance(node, ast.Attribute)
            and node.attr in SWITCHES
            and _is_self_args(node.value)
        ):
            offenders.append((node.lineno, node.attr))
    assert not offenders, (
        "计费开关仍从 args 直读（应改读 self.comm_config / _comm）："
        f"{offenders}"
    )


def test_every_switch_is_still_read_somewhere():
    attrs = {
        node.attr for node in ast.walk(_tree()) if isinstance(node, ast.Attribute)
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
