"""口径机制层（`src/protocols.py`）与 CLI 收敛的回归测试。

背景：参数一度有三个来源（CLI 默认 / `exp.py` 扫描配置 / `cmd_temp` 字面量），
且部分开关只被个别方法消费，而 metrics 只记录**标称**口径——于是 run 的标签
可能与真实行为不一致（分析见 `docs/param_ledger.md` §2，最终口径见
`docs/protocol.md`）。这里锁住机制层的四件事：

1. 消费表的快照与**运行时内省**一致（防止手维护的表漂移）；
2. 注册表里每个解码方法都在快照中登记（新方法必须显式登记消费情况）；
3. 协议应用只填命令行未显式设置的项，命令行覆盖会被记成"偏离"；
4. `--protocol none`（默认）不写入任何参数，且 15 个无消费者参数已从 CLI 删除
   （其中 `--task_name` 有 17 处 call site，须一并清掉）。
"""

import ast
import re
import unittest
from argparse import Namespace
from pathlib import Path

from src.baselines import Baselines  # noqa: F401  触发解码方法注册
from src.engine import Decoding  # noqa: F401  触发 engine 侧解码方法注册
from src.protocols import (
    PROTOCOLS,
    _ACCOUNTING_CONSUMERS,
    _STATIC_DEPTH_KEYS,
    apply_protocol,
    live_mode_consumption,
    mode_consumption,
    render_effective_report,
)
from src.register import Register

ROOT = Path(__file__).resolve().parents[1]
UTILS_PY = ROOT / "src" / "utils.py"

#: 全仓无消费者、已从 CLI 删除的参数（docs/protocol.md §6）
REMOVED_ARGS = (
    "level",
    "guess",
    "window",
    "max_token_span",
    "num_draft",
    "dtype_comm",
    "adaptive_debug_log",
    "datastore_path",
    "task_name",
    "controlled_eval_task",
    "controlled_topk_values",
    "controlled_topk_step",
    "controlled_entropy_quantile",
    "controlled_entropy_threshold",
    "controlled_max_high_entropy_states",
)

#: 上一组里**曾被 exp.py / scripts/ / cmds/ 调用**的那些。删掉定义后，遗留的
#: call site 会变成 argparse 的 "unrecognized arguments" 当场报错，所以必须一并
#: 清掉；而脚本只有在真正跑批时才暴露，代价高——这里静态锁死。
REMOVED_ARGS_WITH_CALL_SITES = ("task_name",)

#: 自带独立 parser 的 vendored 包：其中的同名 flag 与本项目的 CLI 无关
_VENDORED = ("SpecDec_pp/", "src/model/rest/")


def _python_docstring_lines(path: Path) -> set[int]:
    """该 .py 里所有 docstring 覆盖的行号（1-based）。

    docstring 是字符串而非注释，静态扫描会把它当代码——本测试自己的说明文字
    就提到过 `--task_name`，所以必须显式排除，否则测试自伤。
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None) or []
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


def _tracked_files() -> list[str]:
    """受版本控制的 .py/.sh（排除 vendored 包），供全仓静态扫描用。

    不用 `os.walk`：仓库里有 `data/`、`.venv/`、`SpecDec_pp/.venv`，遍历会慢到超时。
    """
    files = _git_ls_files()
    return [
        f
        for f in files
        if f.endswith((".py", ".sh")) and not f.startswith(_VENDORED)
    ]


def _git_ls_files() -> list[str]:
    import subprocess

    proc = subprocess.run(
        ["git", "ls-files", "*.py", "*.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in proc.stdout.split("\n") if line]


def _arg_records() -> list[tuple[str, tuple[str, ...], int]]:
    """从 `src/utils.py` 的 `add_argument` 调用提取 (dest, 选项拼写, 行号)。"""
    tree = ast.parse(UTILS_PY.read_text(encoding="utf-8"))
    out: list[tuple[str, tuple[str, ...], int]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        opts = tuple(
            a.value
            for a in node.args
            if isinstance(a, ast.Constant)
            and isinstance(a.value, str)
            and a.value.startswith("-")
        )
        long_opt = next((o for o in opts if o.startswith("--")), None)
        if long_opt is None:
            continue
        # BooleanOptionalAction 由 argparse 自动生成 --no- 拼写（源码里只写了正形式）
        action_kw = next((kw for kw in node.keywords if kw.arg == "action"), None)
        if (
            action_kw is not None
            and isinstance(action_kw.value, ast.Attribute)
            and action_kw.value.attr == "BooleanOptionalAction"
        ):
            opts = opts + (f"--no-{long_opt[2:]}",)
        out.append((long_opt[2:].replace("-", "_"), opts, node.lineno))
    return out


def _dest_to_flags() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for dest, opts, _ in _arg_records():
        out.setdefault(dest, []).extend(opts)
    return out


def _base_args(**overrides) -> Namespace:
    """模拟默认解析结果（用于纯函数测试，不加载模型）。"""
    base = dict(
        eval_mode="dsd",
        ntt_ms_edge_cloud=76.3,
        ntt_ms_edge_end=0.317,
        edge_end_bandwidth=941,
        edge_cloud_bandwidth=46,
        cloud_end_bandwidth=46,
        comm_round_trip_mode="per_transfer",
        charge_residual_payload=True,
        transfer_top_k_cap=16,
        force_full_vocab_transfer=False,
        transfer_top_k=300,
        temp=0.2,
        max_tokens=1024,
        num_shots=0,
        eval_data_num=80,
        random_sample=False,
        sample_seed=1234,
        num_samples_per_task=1,
        small_draft_threshold=0.8,
        draft_target_threshold=0.8,
        uncertainty_threshold=0.8,
        use_early_stopping=False,
        use_stochastic_comm=False,
        use_cuda_graph=False,
        gamma=4,
        gamma1=5,
        gamma2=5,
        batch_delay=0.05,
    )
    base.update(overrides)
    return Namespace(**base)


class TestConsumptionTruth(unittest.TestCase):
    """消费表必须是"运行时真值"，快照只是它的离线替身。"""

    def test_every_registered_method_is_in_snapshot(self):
        missing = set(Register._DECODING_REGISTRY) - set(_STATIC_DEPTH_KEYS)
        self.assertFalse(
            missing,
            f"新注册的解码方法必须在 _STATIC_DEPTH_KEYS 登记消费情况: {sorted(missing)}",
        )

    def test_snapshot_matches_runtime(self):
        for mode in sorted(_STATIC_DEPTH_KEYS):
            live = live_mode_consumption(mode)
            if live is None:
                continue  # 未注册（例如能力表里的 phantom 模式）
            with self.subTest(mode=mode):
                snap = mode_consumption(mode)
                self.assertEqual(live.accounting, snap.accounting)
                self.assertEqual(live.depth_keys, snap.depth_keys)
                self.assertEqual(snap.source, "runtime")

    def test_accounting_consumers_are_exactly_the_adaptive_family(self):
        consumers = {
            m for m in _STATIC_DEPTH_KEYS if mode_consumption(m).consumes_accounting
        }
        self.assertEqual(consumers, set(_ACCOUNTING_CONSUMERS))
        for mode in ("adaptive_tridecoding", "cee_sd", "cee_sd_opportunistic"):
            self.assertTrue(mode_consumption(mode).consumes_accounting, mode)

    def test_baselines_do_not_consume_accounting(self):
        for mode in (
            "dsd",
            "dssd",
            "cuhlm",
            "dist_spec",
            "dist_split_spec",
            "uncertainty_decoding",
            "ceesd_without_arp",
            "cee_dsd",
            "cee_dssd",
            "cee_cuhlm",
            "adaptive_decoding",
            "tridecoding",
        ):
            with self.subTest(mode=mode):
                self.assertFalse(mode_consumption(mode).consumes_accounting)

    def test_depth_keys_split_between_single_and_tri(self):
        self.assertEqual(mode_consumption("dsd").depth_keys, ("gamma",))
        self.assertEqual(mode_consumption("dssd").depth_keys, ("gamma",))
        self.assertEqual(mode_consumption("cuhlm").depth_keys, ("gamma",))
        self.assertEqual(
            mode_consumption("ceesd_without_arp").depth_keys, ("gamma1", "gamma2")
        )
        self.assertEqual(
            mode_consumption("adaptive_tridecoding").depth_keys, ("gamma1", "gamma2")
        )
        for mode in ("small", "large", "target_only"):
            self.assertEqual(mode_consumption(mode).depth_keys, (), mode)


class TestProtocolApplication(unittest.TestCase):
    def test_none_changes_nothing(self):
        args = _base_args()
        before = dict(vars(args))
        app = apply_protocol(args, "none", [])
        self.assertEqual(app.applied, [])
        self.assertFalse(app.is_named)
        self.assertEqual(vars(args), before)

    def test_fills_only_unset_values(self):
        args = _base_args()
        app = apply_protocol(args, "paper_table5", [], _dest_to_flags())
        self.assertEqual(args.ntt_ms_edge_cloud, 50)
        self.assertEqual(args.edge_end_bandwidth, 563)
        self.assertEqual(args.edge_cloud_bandwidth, 46)
        self.assertEqual(args.comm_round_trip_mode, "per_round")
        self.assertEqual(args.temp, 0.0)
        self.assertEqual(args.max_tokens, 128)
        self.assertTrue(args.use_stochastic_comm)
        self.assertIn("ntt_ms_edge_cloud", app.applied)
        self.assertEqual(app.deviations, [])
        self.assertTrue(app.clean)

    def test_cli_override_wins_and_is_reported_as_deviation(self):
        args = _base_args(ntt_ms_edge_cloud=76.3)
        app = apply_protocol(
            args, "paper_table5", ["--ntt_ms_edge_cloud", "76.3"], _dest_to_flags()
        )
        self.assertEqual(args.ntt_ms_edge_cloud, 76.3, "显式 CLI 值不得被协议覆盖")
        self.assertTrue(app.deviations)
        self.assertIn("ntt_ms_edge_cloud", app.deviations[0])
        self.assertFalse(app.clean)

    def test_boolean_optional_action_override_detected(self):
        args = _base_args(charge_residual_payload=False)
        app = apply_protocol(
            args,
            "paper_table5",
            ["--no-charge_residual_payload"],
            _dest_to_flags(),
        )
        self.assertFalse(args.charge_residual_payload)
        self.assertTrue(any("charge_residual_payload" in d for d in app.deviations))

    def test_legacy_protocol_values(self):
        args = _base_args()
        apply_protocol(args, "legacy", [], _dest_to_flags())
        self.assertFalse(args.charge_residual_payload)
        self.assertEqual(args.comm_round_trip_mode, "per_transfer")
        self.assertEqual(args.transfer_top_k_cap, 0)


class TestEffectiveReport(unittest.TestCase):
    def _report(self, mode: str, name: str = "paper_table5", cli=()):
        args = _base_args(eval_mode=mode)
        app = apply_protocol(args, name, list(cli), _dest_to_flags())
        return render_effective_report(args, app)

    def test_baseline_mode_is_flagged_as_not_consuming(self):
        report = self._report("dsd")
        self.assertIn("不消费", report)
        self.assertIn("读取键=gamma", report)

    def test_adaptive_mode_is_not_flagged(self):
        report = self._report("adaptive_tridecoding")
        self.assertNotIn("不消费", report)
        self.assertIn("gamma1", report)

    def test_gamma_rule_flagged_undecided(self):
        self.assertIn("规则未定", self._report("dsd"))

    def test_unnamed_protocol_reported(self):
        report = self._report("dsd", name="none")
        self.assertIn("未命名口径", report)

    def test_deviation_is_called_out(self):
        report = self._report(
            "dsd", cli=["--ntt_ms_edge_cloud", "76.3"]
        )
        self.assertIn("不属于该协议", report)

    def test_autoregressive_mode_has_no_gamma(self):
        report = self._report("small")
        self.assertIn("不读任何 γ", report)


class TestCliSurface(unittest.TestCase):
    """CLI 面：死参数确实删除、协议键都是真实 dest。"""

    def test_removed_args_are_gone(self):
        dests = {dest for dest, _, _ in _arg_records()}
        for name in REMOVED_ARGS:
            with self.subTest(arg=name):
                self.assertNotIn(name, dests, f"--{name} 应已删除（全仓无消费者）")

    def test_removed_args_have_no_call_sites(self):
        """被传过的死参数：call site 必须一起清掉，否则脚本当场报错。

        `--task_name` 曾被 `exp.py`（每条扫描命令）和 12 个 `scripts/*.sh`、
        `cmds/train_rl.sh` 传着——删掉定义后这些命令会 argparse 报错。这里按
        "非注释行里出现该 flag" 扫描，钉死清理结果。
        """
        tracked = _tracked_files()
        self.assertGreater(len(tracked), 50, "扫描面太小，git ls-files 可能失败了")
        offenders: list[str] = []
        for rel in tracked:
            path = ROOT / rel
            skip = _python_docstring_lines(path) if rel.endswith(".py") else set()
            for i, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if i in skip or line.lstrip().startswith("#"):
                    continue  # 注释/docstring 里提到参数名是允许的
                for name in REMOVED_ARGS_WITH_CALL_SITES:
                    if re.search(rf"--{name.replace('_', '[-_]')}\b", line) or re.search(
                        rf"add_args\([^)]*[\"']{name}[\"']", line
                    ):
                        offenders.append(f"{rel}:{i}: {line.strip()[:70]}")
        self.assertFalse(
            offenders,
            "仍有 call site 传递已删除的参数（运行时会 unrecognized arguments）:\n"
            + "\n".join(offenders),
        )

    def test_protocol_option_exists_with_all_protocols(self):
        records = {dest: opts for dest, opts, _ in _arg_records()}
        self.assertIn("protocol", records)
        tree = ast.parse(UTILS_PY.read_text(encoding="utf-8"))
        choices: list[str] = []
        starred: list[str] = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and getattr(node.args[0], "value", None) == "--protocol"
            ):
                continue
            for kw in node.keywords:
                if kw.arg == "choices":
                    choices = [
                        e.value for e in kw.value.elts if isinstance(e, ast.Constant)
                    ]
                    starred = [
                        e.value.id
                        for e in kw.value.elts
                        if isinstance(e, ast.Starred)
                        and isinstance(e.value, ast.Name)
                    ]
        # 协议名由 *PROTOCOLS 展开，静态解析只能看到 "none" + starred 名
        self.assertIn("none", choices)
        self.assertEqual(starred, ["PROTOCOLS"])
        for name in ("paper_table5", "honest", "legacy", "smoke"):
            self.assertIn(name, PROTOCOLS)

    def test_protocol_keys_are_real_dests(self):
        dests = {dest for dest, _, _ in _arg_records()}
        for name, spec in PROTOCOLS.items():
            for key in spec.values:
                with self.subTest(protocol=name, key=key):
                    self.assertIn(key, dests, f"协议 {name} 引用了不存在的参数 {key}")

    def test_default_protocol_is_none(self):
        tree = ast.parse(UTILS_PY.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and getattr(node.args[0], "value", None) == "--protocol"
            ):
                defaults = [
                    kw.value.value
                    for kw in node.keywords
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant)
                ]
                self.assertEqual(defaults, ["none"], "默认必须是 none（不改动任何数字）")

    def test_help_strings_escape_percent(self):
        """help 里的字面 % 必须写成 %%：argparse 会对 help 做 %-格式化。

        否则 `--help` 直接抛 ValueError（未转义 % 后面跟多字节字符时
        "unsupported format character"）——110 个参数的 CLI 连帮助都打不出来，
        这正是"参数太多却看不清"的一个隐藏原因（本次修复前 --help 退出码为 1）。
        """
        tree = ast.parse(UTILS_PY.read_text(encoding="utf-8"))
        offenders: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            for kw in node.keywords:
                if kw.arg != "help" or not isinstance(kw.value, ast.Constant):
                    continue
                text = kw.value.value
                if isinstance(text, str) and re.search(r"(?<!%)%(?!%)", text):
                    offenders.append((node.lineno, text[:48]))
        self.assertFalse(offenders, f"help 字符串含未转义的 %: {offenders}")


if __name__ == "__main__":
    unittest.main()
