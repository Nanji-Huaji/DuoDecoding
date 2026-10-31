"""多文件指纹语料：跨文件 helper 应保持事件序列。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import logic_fingerprint as lf  # noqa: E402


def _events(sources: dict[str, str], name: str):
    entries = lf.load_entries(sources)
    return lf.summarize(name, entries[name])


def test_cross_file_helper_expands_same_as_inline():
    inline = """\
def helper(x):
    return x + 1
def run(x):
    return helper(x)
"""
    split = {
        "main.py": "def run(x):\n    return helper(x)\n",
        "helpers.py": "def helper(x):\n    return x + 1\n",
    }
    assert _events({"main.py": inline}, "run") == _events(split, "run")


def test_cross_file_ambiguity_degrades_to_call():
    sources = {
        "main.py": "def run(x):\n    return helper(x)\n",
        "a.py": "def helper(x):\n    return x + 1\n",
        "b.py": "def helper(x):\n    return x - 1\n",
    }
    entries = lf.load_entries(sources)
    events = entries["run"]
    assert ["call", "helper"] in events
