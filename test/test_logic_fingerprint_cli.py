"""CLI 门禁：包化改名后 `--base-path` 只切换旧侧读取路径。

`tools/logic_fingerprint.py diff` 的比较对象是「git base 修订」与「工作区」。
`src/baselines.py` 迁到 `src/baselines/core.py` 之后，base 上只有旧路径，
工作区只有新路径；因此旧侧必须能单独指定路径，否则改名后的 diff 无法运行。
同时，路径/修订错误必须显式失败——不能静默给出 PASS。

这些用例在临时 git 仓库里跑真实子进程，不依赖本仓库当前工作树。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "logic_fingerprint.py"

OLD_PATH = "src/baselines.py"
NEW_PATH = "src/baselines/core.py"
HELPER_PATH = "src/baselines_helpers.py"

CALLER_SOURCE = """\
class Runner:
    def run(self):
        self._step()
"""

HELPER_SOURCE = """\
class Helper:
    def _step(self):
        comm_simulator.transfer(1)
        sample(1)
"""

# 只交换两件事的顺序：足迹必须判 FAIL（而不是因为改了文件名就重生成基线）。
DRIFTED_SOURCE = """\
class Runner:
    def run(self):
        sample(1)
        comm_simulator.transfer(1)
"""

# 入口不变，只动 include 文件；用于锁“旧侧 include 必须从 git base 读”。
DRIFTED_HELPER_SOURCE = """\
class Helper:
    def _step(self):
        sample(1)
        comm_simulator.transfer(1)
"""


def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "logic-fingerprint@example.invalid")
    _git(root, "config", "user.name", "Logic Fingerprint Test")
    _git(root, "config", "commit.gpgsign", "false")
    _write(root, OLD_PATH, CALLER_SOURCE)
    _write(root, HELPER_PATH, HELPER_SOURCE)
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _rename_entry(root: Path) -> None:
    old = root / OLD_PATH
    new = root / NEW_PATH
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "diff",
            *args,
            "--include",
            HELPER_PATH,
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )


def test_same_path_default_passes(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    result = _run(root, OLD_PATH, "--base", "HEAD")
    assert result.returncode == 0, result.stderr
    assert "判定: PASS" in result.stdout


def test_renamed_path_with_base_path_passes(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    _rename_entry(root)
    result = _run(root, NEW_PATH, "--base", "HEAD", "--base-path", OLD_PATH)
    assert result.returncode == 0, result.stderr
    assert "判定: PASS" in result.stdout


def test_renamed_path_without_base_path_fails_loudly(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    _rename_entry(root)
    result = _run(root, NEW_PATH, "--base", "HEAD")
    assert result.returncode != 0
    assert "判定: PASS" not in result.stdout
    assert NEW_PATH in result.stderr
    assert "失败" in result.stderr


def test_renamed_path_with_behavior_drift_fails(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    _rename_entry(root)
    _write(root, NEW_PATH, DRIFTED_SOURCE)
    result = _run(root, NEW_PATH, "--base", "HEAD", "--base-path", OLD_PATH)
    assert result.returncode == 1
    assert "判定: FAIL" in result.stdout
    assert "Runner.run" in result.stdout


def test_renamed_path_reads_include_from_base_revision(tmp_path: Path) -> None:
    """入口与旧侧同名，只有 include 文件在工作区漂移。

    旧侧必须从 ``HEAD:src/baselines_helpers.py`` 读 helper，新侧才读到工作区
    的交换版；否则两侧都读工作区，会错误 PASS。这个用例专门锁住这一点。
    """
    root = _make_repo(tmp_path)
    _rename_entry(root)
    _write(root, HELPER_PATH, DRIFTED_HELPER_SOURCE)
    result = _run(root, NEW_PATH, "--base", "HEAD", "--base-path", OLD_PATH)
    assert result.returncode == 1
    assert "判定: FAIL" in result.stdout
    assert "Runner.run" in result.stdout


def test_missing_base_path_does_not_pass(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    _rename_entry(root)
    missing = "src/baselines/does_not_exist.py"
    result = _run(root, NEW_PATH, "--base", "HEAD", "--base-path", missing)
    assert result.returncode != 0
    assert "判定: PASS" not in result.stdout
    assert missing in result.stderr
    assert "失败" in result.stderr


def test_wrong_base_path_with_disjoint_entries_does_not_pass(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    _write(
        root,
        "src/other.py",
        "class Other:\n    def run(self):\n        sample(1)\n",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "other")
    _rename_entry(root)
    result = _run(root, NEW_PATH, "--base", "HEAD", "--base-path", "src/other.py")
    assert result.returncode != 0
    assert "判定: PASS" not in result.stdout
    assert "没有任何同名条目" in result.stderr


def test_missing_base_revision_does_not_pass(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    result = _run(root, OLD_PATH, "--base", "no-such-revision")
    assert result.returncode != 0
    assert "判定: PASS" not in result.stdout
    assert "失败" in result.stderr
