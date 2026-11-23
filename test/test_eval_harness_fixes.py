"""B 组 harness 修复的回归测试（B29/B31/B32/B34/B35）。

覆盖 read_results 的兼容接口、chat 模板选择和 MT-Bench 判分失败口径。
"""

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import eval_mt_bench as mt_bench_module  # noqa: E402
from eval.eval_humaneval import HumanevalTask  # noqa: E402
from eval.eval_mixed import MixedTask  # noqa: E402
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
    assert _stub(HumanevalTask, model_id)._uses_chat_template() is expected


@pytest.mark.parametrize("speed_only", [True, False])
def test_humaneval_speed_only_never_executes_completions(monkeypatch, speed_only):
    import torch

    args = types.SimpleNamespace(
        draft_model="draft",
        target_model="target",
        eval_mode="large",
        humaneval_generate_only=speed_only,
    )
    tokenizer = types.SimpleNamespace(
        bos_token=None, eos_token=None, decode=lambda _: "xy"
    )
    task = HumanevalTask(args, tokenizer)
    rows = []
    run = types.SimpleNamespace(
        args=args,
        decoder=types.SimpleNamespace(accelerator=None),
        next_seed=lambda: None,
        generate=lambda *a, **kw: (torch.tensor([[1, 2]]), None, 0.25),
        write=rows.append,
        metrics={},
        is_main_process=True,
    )

    def fake_grade(*_):
        if speed_only:
            pytest.fail("Speed matrix executed a model-generated program")
        return 1

    monkeypatch.setattr("eval.eval_humaneval.check_correctness", fake_grade)
    item = dict(
        input_ids=torch.tensor([[1]]),
        input_text="x",
        test="",
        entry_point="",
        task_id="id",
    )
    task.run(run, [item, item])
    assert [row["passed"] for row in rows] == [None if speed_only else 1] * 2
    assert run.metrics.get("accuracy") == (None if speed_only else 1)


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
    assert _stub(MixedTask, model_id)._uses_chat_template() is expected


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
    """判分失败仍按 0 分计入（口径不变），但必须落在 run-local 计数器上。"""
    monkeypatch.setattr(
        mt_bench_module,
        "_load_async_openai",
        lambda: lambda **kwargs: _FakeJudgeClient(content=content, error=error),
    )
    failures = {"api": 0, "unparsed": 0}

    score = asyncio.run(
        mt_bench_module.grade_mt_bench_async(
            "问题", "回答", "gpt-4", "sk-test", "http://localhost", failures
        )
    )

    assert score == 0
    assert failures[counter] == 1
    other = "api" if counter == "unparsed" else "unparsed"
    assert failures[other] == 0


def test_mt_bench_grade_success_leaves_counters_empty(monkeypatch):
    monkeypatch.setattr(
        mt_bench_module,
        "_load_async_openai",
        lambda: lambda **kwargs: _FakeJudgeClient(content="评分：[[7]]"),
    )
    failures = {"api": 0, "unparsed": 0}

    score = asyncio.run(
        mt_bench_module.grade_mt_bench_async(
            "q", "a", "gpt-4", "sk-test", None, failures
        )
    )

    assert score == 7.0
    assert failures == {"api": 0, "unparsed": 0}


def test_mt_bench_grade_without_api_key_is_not_a_failure(monkeypatch):
    """未配置 key 时判分整体不启用，不应污染失败计数或触碰判分依赖。"""
    monkeypatch.setattr(
        mt_bench_module,
        "_load_async_openai",
        lambda: pytest.fail("未配置 key 时不应导入/实例化 judge 客户端"),
    )
    failures = {"api": 0, "unparsed": 0}

    score = asyncio.run(
        mt_bench_module.grade_mt_bench_async("q", "a", "gpt-4", None, None, failures)
    )

    assert score == 0
    assert failures == {"api": 0, "unparsed": 0}
