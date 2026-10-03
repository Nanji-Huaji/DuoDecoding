"""口径决定：哪些基线路径**故意不接** CUDA Graph。

不是所有"没接 use_cuda_graph"都是漏接。这里把已有据的例外锁成断言，
避免后人照着 B19 的清单把有意的取舍当 bug 改掉。
依据：docs/verify_graph_fix.md 的干净窗口配对 A/B（2026-09-24）。
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "baselines.py"


def _method(name: str):
    src = SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return src, node
    raise AssertionError(f"src/baselines.py 里找不到方法 {name}")


def test_target_only_deliberately_skips_cuda_graph():
    """target_only 是无损性检验的基准，13B 上图只快 1.03% 且有 bf16 漂移。"""
    src, node = _method("target_only")
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]

    graph_kwargs = [
        c
        for c in calls
        if isinstance(c.func, ast.Name) and c.func.id == "graph_mode_cache_kwargs"
    ]
    assert not graph_kwargs, "target_only 不该接图（理由写在方法内的注释里）"

    caches = [
        c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "KVCacheModel"
    ]
    assert caches, "target_only 应当直接构造 KVCacheModel"
    for c in caches:
        assert "use_cuda_graph" not in {k.arg for k in c.keywords}


def test_target_only_records_why_it_skips_graph():
    """理由必须留在方法体里，否则下一个人只会看到'没接'。"""
    src, node = _method("target_only")
    body = ast.get_source_segment(src, node) or ""
    assert "故意不接" in body
    assert "1.03" in body or "漂移" in body
