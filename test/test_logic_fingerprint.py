"""逻辑足迹门禁。

本轮重构的判据不是「测试还绿」——golden 只能证明那 80 条输入上结果没变，
证明不了判定规则没变。这里冻结的是**可观测副作用序列**：

    叶子调用 · RNG 抽取 · comm 计费操作 · args 口径读取
    self 赋值 · 循环/异常结构 · return/break/continue/raise

两条性质：
1. 抽取（把代码搬进同文件/跨文件的新方法，原地调用）不改变足迹；
2. 重排、增删计费点、改变 RNG 次数会改变足迹。

因此「足迹不变」是重构的必要条件。若 PR 有意改变口径，必须同时更新
`test/golden/logic/*.json` 并在 PR 里说明——那是评审，不是自我实现。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import logic_fingerprint as lf  # noqa: E402

FIXTURES = {
    "src/baselines.py": ROOT / "test/golden/logic/baselines.json",
    "src/engine.py": ROOT / "test/golden/logic/engine.json",
    "src/cee_prefix.py": ROOT / "test/golden/logic/cee_prefix.json",
}


def _entries(relative: str) -> dict:
    source = (ROOT / relative).read_text(encoding="utf-8")
    return lf.load_entries({relative: source})


@pytest.mark.parametrize("path", sorted(FIXTURES))
def test_frozen_fingerprint_matches(path: str) -> None:
    frozen = json.loads(FIXTURES[path].read_text(encoding="utf-8"))
    entries = _entries(path)
    for name, expected in frozen["entries"].items():
        assert name in entries, f"{path}:{name} 不见了——入口被改名或删除"
        current = lf.summarize(name, entries[name])
        assert current == expected, (
            f"{path}:{name} 的逻辑足迹变了。抽取应当不改变它；"
            "如果这是有意的口径变更，同 PR 更新 test/golden/logic/*.json 并说明理由。"
        )


def test_fingerprint_is_deterministic() -> None:
    first = lf.load_entries({"src/baselines.py": (ROOT / "src/baselines.py").read_text(encoding="utf-8")})
    second = lf.load_entries({"src/baselines.py": (ROOT / "src/baselines.py").read_text(encoding="utf-8")})
    assert first == second


EXTRACTED_IN_PLACE = """
class Runner:
    def run(self):
        self._step()

    def _step(self):
        comm_simulator.transfer(1)
        comm_simulator.flush_round()
        for _ in range(2):
            sample(1)
        if getattr(self.args, "charge_residual_payload", False):
            comm_simulator.simulate_transfer(2)
"""

INLINE = """
class Runner:
    def run(self):
        comm_simulator.transfer(1)
        comm_simulator.flush_round()
        for _ in range(2):
            sample(1)
        if getattr(self.args, "charge_residual_payload", False):
            comm_simulator.simulate_transfer(2)
"""

CROSS_FILE_CALLER = """
class Runner:
    def run(self):
        self._step()
"""

CROSS_FILE_HELPER = """
class Other:
    def _step(self):
        comm_simulator.transfer(1)
        comm_simulator.flush_round()
        for _ in range(2):
            sample(1)
        if getattr(self.args, "charge_residual_payload", False):
            comm_simulator.simulate_transfer(2)
"""

REORDERED = """
class Runner:
    def run(self):
        comm_simulator.flush_round()
        comm_simulator.transfer(1)
        for _ in range(2):
            sample(1)
        if getattr(self.args, "charge_residual_payload", False):
            comm_simulator.simulate_transfer(2)
"""

DROPPED_BILLING = """
class Runner:
    def run(self):
        comm_simulator.transfer(1)
        comm_simulator.flush_round()
        for _ in range(2):
            sample(1)
"""

RETURN_HELPER = """
class Runner:
    def run(self):
        self._step()

    def _step(self):
        if self.args.stop:
            return
        sample(1)
"""

RETURN_INLINE = """
class Runner:
    def run(self):
        if self.args.stop:
            return
        sample(1)
"""


def test_extraction_keeps_fingerprint() -> None:
    inline = lf.load_entries({"m.py": INLINE})["Runner.run"]
    extracted = lf.load_entries({"m.py": EXTRACTED_IN_PLACE})["Runner.run"]
    assert inline == extracted
    assert inline == lf.load_entries({"m.py": INLINE})["Runner.run"]


def test_cross_file_extraction_keeps_fingerprint() -> None:
    inline = lf.load_entries({"m.py": INLINE})["Runner.run"]
    split = lf.load_entries({"m.py": CROSS_FILE_CALLER, "n.py": CROSS_FILE_HELPER})["Runner.run"]
    assert inline == split


def test_reorder_is_detected() -> None:
    inline = lf.load_entries({"m.py": INLINE})["Runner.run"]
    reordered = lf.load_entries({"m.py": REORDERED})["Runner.run"]
    assert inline != reordered


def test_dropped_billing_is_detected() -> None:
    inline = lf.load_entries({"m.py": INLINE})["Runner.run"]
    dropped = lf.load_entries({"m.py": DROPPED_BILLING})["Runner.run"]
    assert inline != dropped


VALUE_INLINE = """
class Runner:
    def run(self):
        sim = precise_sim() if self.args.precise else sim_a(1, 2)
        sim.flush_round()
"""

VALUE_EXTRACTED = """
def _make_sim(flag):
    if flag:
        sim = precise_sim()
    else:
        sim = sim_a(1, 2)
    return sim

class Runner:
    def run(self):
        sim = _make_sim(self.args.precise)
        sim.flush_round()
"""


def test_value_factory_extraction_keeps_fingerprint() -> None:
    """尾返回的值工厂展开后应与内联代码同足迹（H2 式抽取）。"""
    inline = lf.load_entries({"m.py": VALUE_INLINE})["Runner.run"]
    extracted = lf.load_entries({"m.py": VALUE_EXTRACTED})["Runner.run"]
    assert inline == extracted


def test_early_return_is_not_silently_absorbed() -> None:
    """含 return 的 helper 不展开：`return` 退出谁，是语义差异。"""
    inline = lf.load_entries({"m.py": RETURN_INLINE})["Runner.run"]
    extracted = lf.load_entries({"m.py": RETURN_HELPER})["Runner.run"]
    assert inline != extracted
    assert any(event[0] == "return" for event in inline)


SINGLE_ARM_RETURN_INLINE = """
class Runner:
    def run(self):
        if self.args.stop:
            return
        sample(1)
"""

SINGLE_ARM_RETURN_EXTRACTED = """
class Runner:
    def run(self):
        self._step()
        sample(1)

    def _step(self):
        if self.args.stop:
            return
"""


def test_single_arm_return_extraction_is_detected() -> None:
    """单臂 if 的 return 位于"尾位置"，但抽成 helper 会改变控制流：

    内联版 `if stop: return` 会跳过后面的 sample(1)；抽取版只退出 helper，
    sample(1) 仍执行。指纹必须判 FAIL，不能因为 _is_tail 是位置判断而假 PASS。
    """
    inline = lf.load_entries({"m.py": SINGLE_ARM_RETURN_INLINE})["Runner.run"]
    extracted = lf.load_entries({"m.py": SINGLE_ARM_RETURN_EXTRACTED})["Runner.run"]
    assert inline != extracted
