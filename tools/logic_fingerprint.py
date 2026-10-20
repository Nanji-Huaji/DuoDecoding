"""抽取不敏感的逻辑指纹。

`golden` 只能证明「这 80 条输入上结果没变」，证明不了「所有输入的判断规则
没变」。本工具补中间那一层：把方法的**可观测副作用序列**抽出来逐事件比对。

    叶子调用 · RNG 抽取 · comm 计费操作 · args 口径读取
    self 赋值 · 循环/异常结构 · return/break/continue/raise

关键设计是**展开局部 helper**：把同一模块内被调用的函数/方法原地展开，
于是「把一段代码搬进新方法并原地调用」这种抽取不改变指纹；而少一次计费、
换一次 RNG、挪一个 return，都会立刻改变。

    before: run():  a(); b(); c()
    after:  run():  _step()          → 展开 _step 后事件序列完全一致
            _step(): a(); b(); c()

这是**门禁**，不是定理证明：它看不见条件表达式内部、参数取值与数据流。
真正可以证明的「内联等价」见 docs/verification_gates.md。

用法（仓库根目录）：

    # 快照当前指纹
    python tools/logic_fingerprint.py dump src/baselines.py --out /tmp/fp.json

    # PR 前后对比（旧版本从 git 读）
    python tools/logic_fingerprint.py diff src/baselines.py --base HEAD

退出码：diff 有差异 = 1；无差异 = 0。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: 随机抽样入口——出现次数与顺序变化意味着随机数流改变
RNG_CALLS = frozenset(
    {
        "sample",
        "sample_accept_token",
        "sample_reject_token",
        "sample_reject_token_from_topk_proposal",
        "sample_token_from_probs",
        "multinomial",
        "rand",
        "randn",
        "randint",
    }
)

#: 通信计费入口（作用于 comm_simulator 变量）
COMM_ATTRS = frozenset(
    {
        "transfer",
        "simulate_transfer",
        "send_reject_message",
        "send_accept_message",
        "flush_round",
        "set_round",
        "rebuild_full_probs",
        "determine_transfer_strategy",
        "calculate_uncertainty",
        "pending_bytes",
    }
)

COMM_RECEIVER = "comm_simulator"

#: 局部 helper 展开深度上限（防病态递归；正常代码 1-3 层足够）
MAX_EXPAND_DEPTH = 6


@dataclass
class Index:
    funcs: dict[str, list[ast.AST]] = field(default_factory=dict)
    methods: dict[str, list[ast.AST]] = field(default_factory=dict)


def build_index(trees: list[ast.Module]) -> Index:
    """多文件联合索引：同名的跨文件定义视为歧义，不展开。"""
    index = Index()
    for tree in trees:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                index.funcs.setdefault(node.name, []).append(node)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        index.methods.setdefault(item.name, []).append(item)
    return index


def _resolve_call(func: ast.AST, index: Index) -> tuple[str, ast.AST] | None:
    """把被调方解析到本模块内的 def；歧义或外部函数返回 None。"""
    if isinstance(func, ast.Name):
        candidates = index.funcs.get(func.id, [])
        if len(candidates) == 1:
            return func.id, candidates[0]
        return None
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in ("self", "cls"):
            candidates = index.methods.get(func.attr, [])
            if len(candidates) == 1:
                return func.attr, candidates[0]
    return None


def _receiver_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Attribute):
        base = func.value
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            return base.attr
    return None


def _is_args(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "args"
        and isinstance(node.value, ast.Name)
        and node.value.id in ("self", "args_owner")
    ) or (isinstance(node, ast.Name) and node.id == "args")


def _flag_read(node: ast.AST) -> str | None:
    """self.args.x / args.x"""
    if isinstance(node, ast.Attribute) and _is_args(node.value):
        return node.attr
    return None


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


@dataclass
class Ctx:
    index: Index
    events: list[list[str]] = field(default_factory=list)
    stack: tuple[str, ...] = ()
    depth: int = 0

    def emit(self, kind: str, detail: str = "") -> None:
        self.events.append([kind, detail])


def _walk(node: ast.AST, ctx: Ctx) -> None:
    if isinstance(node, ast.Call):
        _walk_call(node, ctx)
        return
    if isinstance(node, ast.Assign):
        _walk(node.value, ctx)
        for target in node.targets:
            _record_assign(target, ctx)
            _walk(target, ctx)
        return
    if isinstance(node, ast.AnnAssign):
        if node.value is not None:
            _walk(node.value, ctx)
        _record_assign(node.target, ctx)
        _walk(node.target, ctx)
        return
    if isinstance(node, ast.AugAssign):
        _walk(node.target, ctx)
        _walk(node.value, ctx)
        return
    if isinstance(node, ast.Return):
        ctx.emit("return")
    elif isinstance(node, ast.Break):
        ctx.emit("break")
    elif isinstance(node, ast.Continue):
        ctx.emit("continue")
    elif isinstance(node, ast.Raise):
        ctx.emit("raise")
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        ctx.emit("loop", type(node).__name__.lower())
    elif isinstance(node, ast.Try):
        ctx.emit("try")
    elif isinstance(node, ast.ExceptHandler):
        ctx.emit("except")
    elif isinstance(node, ast.BoolOp):
        ctx.emit("boolop", type(node.op).__name__)

    flag = _flag_read(node)
    if flag is not None:
        ctx.emit("flag", flag)

    for child in ast.iter_child_nodes(node):
        _walk(child, ctx)


def _record_assign(target: ast.AST, ctx: Ctx) -> None:
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        if target.value.id == "self":
            ctx.emit("assign_self", target.attr)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            _record_assign(item, ctx)


def _walk_call(node: ast.Call, ctx: Ctx) -> None:
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)

    # 口径读取：getattr(self.args, "x", default)
    if isinstance(func, ast.Name) and func.id == "getattr":
        target = node.args[0] if node.args else None
        if target is not None and _is_args(target):
            literal = _const_str(node.args[1]) if len(node.args) > 1 else None
            ctx.emit("flag_getattr", literal or "?")
        for arg in node.args[1:]:
            _walk(arg, ctx)
        return

    # 参数先于被调方执行
    for arg in node.args:
        _walk(arg, ctx)
    for kw in node.keywords:
        _walk(kw.value, ctx)

    resolved = _resolve_call(func, ctx.index)
    if (
        resolved is not None
        and _expandable(resolved[1])
        and ctx.depth < MAX_EXPAND_DEPTH
        and resolved[0] not in ctx.stack
    ):
        callee_name, callee = resolved
        child = Ctx(
            index=ctx.index,
            events=ctx.events,
            stack=ctx.stack + (callee_name,),
            depth=ctx.depth + 1,
        )
        for stmt in getattr(callee, "body", []):
            _walk(stmt, child)
        return

    receiver = _receiver_name(func)
    if name in COMM_ATTRS and receiver == COMM_RECEIVER:
        ctx.emit("comm", name)
        return
    if name in RNG_CALLS:
        ctx.emit("rng", name or "?")
        return
    ctx.emit("call", name or "?")


def _expandable(node: ast.AST) -> bool:
    """只展开"纯语句块"helper。

    含 `return` 的 helper 不展开：把带 return 的语句块抽成函数会改变控制流
    （旧代码 return 退出方法，新代码 return 只退出 helper）。不展开意味着
    这种抽取会表现为 "return" → "call"，被 diff 抓住。
    """
    return not any(isinstance(item, ast.Return) for item in ast.walk(node))


def events_for(node: ast.AST, index: Index) -> list[list[str]]:
    ctx = Ctx(index=index)
    body = getattr(node, "body", [])
    for stmt in body:
        _walk(stmt, ctx)
    return ctx.events


def iter_entries(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    entries: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entries.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    entries.append((f"{node.name}.{item.name}", item))
    return entries


def digest(events: list[list[str]]) -> str:
    blob = json.dumps(events, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def surface(events: list[list[str]]) -> dict[str, object]:
    grouped: dict[str, dict[str, int]] = {}
    for kind, detail in events:
        bucket = grouped.setdefault(kind, {})
        bucket[detail] = bucket.get(detail, 0) + 1
    return {kind: dict(sorted(counts.items())) for kind, counts in sorted(grouped.items())}


def summarize(name: str, events: list[list[str]]) -> dict[str, object]:
    return {
        "events": len(events),
        "sha256_16": digest(events),
        "surface": surface(events),
    }


def read_source(path: str, rev: str | None) -> str:
    if rev:
        proc = subprocess.run(
            ["git", "show", f"{rev}:{path}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(f"git show {rev}:{path} 失败：{proc.stderr.strip()}")
        return proc.stdout
    return Path(path).read_text(encoding="utf-8")


def load_entries(sources: dict[str, str]) -> dict[str, list[list[str]]]:
    """sources: {快照内路径: 源码}；第一个文件是入口所在的被测文件。"""
    trees = {path: ast.parse(text) for path, text in sources.items()}
    index = build_index(list(trees.values()))
    entries: dict[str, list[list[str]]] = {}
    for tree in trees.values():
        for name, node in iter_entries(tree):
            entries.setdefault(name, events_for(node, index))
    return entries


def selected(all_names: list[str], spec: str | None) -> list[str]:
    if not spec:
        return all_names
    wanted = [item.strip() for item in spec.split(",") if item.strip()]
    missing = [item for item in wanted if item not in all_names]
    if missing:
        raise SystemExit(f"入口不存在: {missing}")
    return wanted


def _read_sources(path: str, includes: list[str], rev: str | None) -> dict[str, str]:
    sources = {path: read_source(path, rev)}
    for extra in includes:
        try:
            sources[extra] = read_source(extra, rev)
        except SystemExit:
            if rev is None:
                raise
    return sources


def cmd_dump(args: argparse.Namespace) -> int:
    sources = _read_sources(args.path, args.include, args.rev)
    entries = load_entries(sources)
    names = selected(sorted(entries), args.entries)
    payload = {
        "path": args.path,
        "include": args.include,
        "rev": args.rev or "WORKTREE",
        "entries": {name: summarize(name, entries[name]) for name in names},
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def _first_divergence(old: list[list[str]], new: list[list[str]]) -> int | None:
    for idx, (left, right) in enumerate(zip(old, new)):
        if left != right:
            return idx
    return min(len(old), len(new)) if len(old) != len(new) else None


def cmd_diff(args: argparse.Namespace) -> int:
    old_entries = load_entries(_read_sources(args.path, args.include, args.base))
    new_entries = load_entries(_read_sources(args.path, args.include, None))
    names = selected(sorted(set(old_entries) | set(new_entries)), args.entries)

    failed = 0
    for name in names:
        old = old_entries.get(name)
        new = new_entries.get(name)
        if old is None:
            print(f"  + 新增入口 {name}（{len(new or [])} 事件）")
            continue
        if new is None:
            print(f"  - 删除入口 {name}（{len(old)} 事件）")
            failed += 1
            continue
        if old == new:
            continue
        failed += 1
        where = _first_divergence(old, new)
        print(f"  ✗ {name}: {len(old)} → {len(new)} 事件，首个分叉 @{where}")
        if where is not None:
            lo = max(0, where - 2)
            for label, seq in (("base", old), ("new ", new)):
                for offset, event in enumerate(seq[lo:where]):
                    print(f"      {label} [{lo + offset}] {event}")
            if where < len(old) and where < len(new):
                print(f"      base [{where}] {old[where]}")
                print(f"      new  [{where}] {new[where]}")

    if failed:
        print(f"\n  判定: FAIL（{failed} 个入口的逻辑足迹变了）")
        return 1
    print(f"  判定: PASS（{len(names)} 个入口，事件序列一致）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dump = sub.add_parser("dump", help="输出快照指纹")
    dump.add_argument("path")
    dump.add_argument("--rev", default=None, help="从 git 读（默认工作区）")
    dump.add_argument("--include", action="append", default=[], help="额外参与展开的文件")
    dump.add_argument("--entries", default=None, help="逗号分隔的入口（类方法名）")
    dump.add_argument("--out", default=None)
    dump.set_defaults(func=cmd_dump)

    diff = sub.add_parser("diff", help="对比 base 与工作区的逻辑足迹")
    diff.add_argument("path")
    diff.add_argument("--base", required=True, help="git 修订，如 HEAD / main")
    diff.add_argument("--include", action="append", default=[], help="额外参与展开的文件")
    diff.add_argument("--entries", default=None)
    diff.set_defaults(func=cmd_diff)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
