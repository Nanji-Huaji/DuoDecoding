"""B 组 harness 修复的回归测试（B29/B31/B32/B34/B35）。

B29（warmup 次数）、B31（异常样本剔除）、B32（partial 转发）都位于需要真实
模型的 eval 主循环，无法在单测里跑通；这里锁住可离线验证的两项：

- B35：``read_results(skip_lines=...)`` 只覆盖本次运行写入的行；
- B34：chat 模板模型族判定（模板自带 BOS，encode 需关 add_special_tokens）。

两个 eval 入口把仓库根与 ``eval/`` 都当作 import 根（脚本内用
``from few_shot_examples import ...``），因此这里按 ROOT 在前、EVAL 在后的顺序
补齐 sys.path，避免 ``eval`` 包被 ``eval/eval.py`` 顶掉。
"""

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
# ROOT 必须在 EVAL 之前：EVAL 在前会让 `eval` 解析成 eval/eval.py，顶掉
# eval 包，`from eval.model_ids import ...` 随即失败。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EVAL) not in sys.path:
    sys.path.append(str(EVAL))

from eval import eval_mt_bench as mt_bench_module  # noqa: E402
from eval.eval_humaneval import EvalHumaneval  # noqa: E402
from eval.eval_mixed import EvalMixed  # noqa: E402
from eval.eval_mt_bench import read_results as read_mt_bench  # noqa: E402
from eval.eval_mt_bench_noeval import read_results as read_noeval  # noqa: E402


def _record(category, wall_time, num_token):
    return {
        "category": category,
        "choices": [{"wall_time": [wall_time], "num_token": [num_token]}],
    }


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_mt_bench_read_results_skips_previous_runs(tmp_path):
    path = tmp_path / "mt_bench.jsonl"
    _write_jsonl(
        path,
        [
            _record("old", 10.0, 100),
            _record("old", 10.0, 100),
            _record("new", 1.0, 50),
        ],
    )

    whole = read_mt_bench(str(path))
    assert sum(whole["old"]["num_token"]) == 200

    current = read_mt_bench(str(path), skip_lines=2)
    assert "old" not in current
    assert sum(current["new"]["num_token"]) == 50
    assert sum(current["new"]["wall_time"]) == pytest.approx(1.0)


def test_noeval_read_results_skips_previous_runs(tmp_path):
    path = tmp_path / "mt_bench.jsonl"
    _write_jsonl(path, [_record("c", 5.0, 25), _record("c", 2.0, 40)])

    current = read_noeval(str(path), skip_lines=1)
    assert sum(current["c"]["num_token"]) == 40
    assert sum(current["c"]["wall_time"]) == pytest.approx(2.0)


def _stub(cls, model_id):
    inst = object.__new__(cls)
    inst.model_id = model_id
    return inst


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("llama-3.1", True),
        ("llama-3.2", True),
        ("llama-3", True),
        ("qwen", True),
        ("gemma", True),
        ("vicuna", False),
        ("llama-2-chat", False),
    ],
)
def test_humaneval_chat_template_detection(model_id, expected):
    assert _stub(EvalHumaneval, model_id)._uses_chat_template() is expected


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("llama-3.1", True),
        ("qwen", True),
        ("vicuna", False),
        ("llama-2-chat", False),
    ],
)
def test_mixed_chat_template_detection(model_id, expected):
    assert _stub(EvalMixed, model_id)._uses_chat_template() is expected


class _FakeJudgeClient:
    """假判分客户端：create 返回指定文本，或抛出指定异常。"""

    def __init__(self, content=None, error=None):
        async def create(**_kwargs):
            if error is not None:
                raise error
            message = types.SimpleNamespace(content=content)
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        )


@pytest.mark.parametrize(
    "content,error,counter",
    [
        ("判分没有给出评分格式", None, "unparsed"),
        (None, RuntimeError("rate limited"), "api"),
    ],
)
def test_mt_bench_grade_failure_is_counted_but_still_scores_zero(
    monkeypatch, content, error, counter
):
    """判分失败仍按 0 分计入（口径不变），但必须落在对应计数器上。"""
    mt_bench_module._MT_BENCH_GRADE_FAILURES.update({"api": 0, "unparsed": 0})
    monkeypatch.setattr(
        mt_bench_module,
        "AsyncOpenAI",
        lambda **kwargs: _FakeJudgeClient(content=content, error=error),
    )

    score = asyncio.run(
        mt_bench_module.grade_mt_bench_async(
            "问题", "回答", "gpt-4", "sk-test", "http://localhost"
        )
    )

    assert score == 0
    assert mt_bench_module._MT_BENCH_GRADE_FAILURES[counter] == 1
    other = "api" if counter == "unparsed" else "unparsed"
    assert mt_bench_module._MT_BENCH_GRADE_FAILURES[other] == 0


def test_mt_bench_grade_success_leaves_counters_empty(monkeypatch):
    mt_bench_module._MT_BENCH_GRADE_FAILURES.update({"api": 0, "unparsed": 0})
    monkeypatch.setattr(
        mt_bench_module,
        "AsyncOpenAI",
        lambda **kwargs: _FakeJudgeClient(content="评分：[[7]]"),
    )

    score = asyncio.run(
        mt_bench_module.grade_mt_bench_async("q", "a", "gpt-4", "sk-test", None)
    )

    assert score == 7.0
    assert mt_bench_module._MT_BENCH_GRADE_FAILURES == {"api": 0, "unparsed": 0}


def test_mt_bench_grade_without_api_key_is_not_a_failure():
    """未配置 key 时判分整体不启用，不应污染失败计数。"""
    mt_bench_module._MT_BENCH_GRADE_FAILURES.update({"api": 0, "unparsed": 0})

    score = asyncio.run(
        mt_bench_module.grade_mt_bench_async("q", "a", "gpt-4", None, None)
    )

    assert score == 0
    assert mt_bench_module._MT_BENCH_GRADE_FAILURES == {"api": 0, "unparsed": 0}
