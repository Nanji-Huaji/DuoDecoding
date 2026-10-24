"""文档改写不变量检查器的离线测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_docs_invariants as chk  # noqa: E402

BASE = """# 标题 A

正文有 `inline` 和 https://example.com/x 链接。

```bash
python x.py --flag 1
```

| a | b |
|---|---|
| 1 | 2 |
"""


def test_identical_is_ok():
    assert chk.compare(BASE, BASE) == []


def test_prose_rewrite_is_ok():
    new = BASE.replace("正文有", "这段文字有")
    assert chk.compare(BASE, new) == []


def test_code_change_is_flagged():
    new = BASE.replace("--flag 1", "--flag 2")
    assert any("code_blocks" in p for p in chk.compare(BASE, new))


def test_table_and_url_change_flagged():
    new = BASE.replace("https://example.com/x", "https://example.com/y")
    assert any("urls" in p for p in chk.compare(BASE, new))
    new2 = BASE.replace("| 1 | 2 |", "| 1 | 3 |")
    assert any("tables" in p for p in chk.compare(BASE, new2))
