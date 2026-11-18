"""MT-Bench：生成多轮回答并调用 judge 评分。

本入口只额外提供判分；对话、解码、速度与 IO 全部复用
``eval.mt_bench_common``。openai 为可选依赖，只在真正需要判分时导入，
因此 ``eval.eval_mt_bench_noeval`` 不接触 API/凭证。
"""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import re

from eval.mt_bench_common import (  # noqa: E402, F401
    MTBenchTaskBase,
    read_results,  # 兼容旧的 ``eval.eval_mt_bench.read_results`` 导入
)
from eval.runner import Evaluation  # noqa: E402
from src.utils import parse_arguments  # noqa: E402

# 可选依赖占位：测试/调用方可以显式替换；只有真正判分时才填充。
AsyncOpenAI = None


def _load_async_openai():
    """延迟导入可选判分依赖，导入本模块本身不要求安装/配置 openai。"""
    global AsyncOpenAI
    if AsyncOpenAI is None:
        from openai import AsyncOpenAI as client_cls

        AsyncOpenAI = client_cls
    return AsyncOpenAI


async def grade_mt_bench_async(
    question,
    answer,
    judge_model="gpt-4",
    api_key=None,
    api_base=None,
    failures=None,
):
    """给一个 turn 打分；``failures`` 是调用方持有的 run-local 计数字典。

    判分失败仍按 0 分计入（与旧口径一致），但失败次数只落在显式传入的
    ``failures`` 上，不再使用模块级全局字典。
    """
    if not api_key:
        return 0

    prompt_template = (
        "\n[Instruction]\n"
        "Please act as an impartial judge and evaluate the quality of the "
        "response provided by an AI assistant to the user question displayed "
        "below. Your evaluation should consider factors such as helpfulness, "
        "relevance, accuracy, depth, creativity, and level of detail. Begin "
        "your evaluation by providing a short explanation. Be as objective as "
        "possible. After providing your explanation, you must rate the "
        "response on a scale of 1 to 10 by strictly following this format: "
        '"[[rating]]", for example: "Rating: [[5]]".\n'
        "\n[Question]\n"
        "{question}\n"
        "\n[The Start of Assistant's Answer]\n"
        "{answer}\n"
        "[The End of Assistant's Answer]\n"
    )

    prompt = prompt_template.format(question=question, answer=answer)

    try:
        client = _load_async_openai()(api_key=api_key, base_url=api_base)
        response = await client.chat.completions.create(
            model=judge_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful and precise assistant for checking "
                        "the quality of the answer."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        content = response.choices[0].message.content
        match = re.search(r"\[\[(\d+)\]\]", content if content is not None else "")
        if match:
            return float(match.group(1))

        # 判分失败与"评分为 0"必须可区分：否则 judge API 限流/超时会把
        # 基础设施故障显示成"模型答得差"。口径不变，只统计次数。
        if failures is not None:
            failures["unparsed"] = failures.get("unparsed", 0) + 1
        return 0
    except Exception as e:
        if failures is not None:
            failures["api"] = failures.get("api", 0) + 1
        print(f"Error in grading: {e}")
        return 0


async def _gather_scores(grading_tasks):
    return await asyncio.gather(*grading_tasks)


class MTBenchTask(MTBenchTaskBase):
    """judge 入口：与 noeval 相同流程，额外对每个 choice 判分。"""

    name = "mt_bench"
    record_name = "mt_bench"
    model_id_task = "mt_bench"
    # 旧 judge warmup：10 个问题，每个问题按数据的全部轮数生成。
    warmup_questions = 10
    warmup_turns = None
    enable_judge = True

    def _grade_choices(self, run, question, choices, scores, failures) -> None:
        args = run.args
        if not getattr(args, "openai_api_key", None):
            return
        judge_model = getattr(args, "judge_model", "gpt-4")
        api_base = getattr(args, "openai_api_base", None)

        for choice in choices:
            grading_tasks = [
                grade_mt_bench_async(
                    question["turns"][turn_idx],
                    turn,
                    judge_model,
                    args.openai_api_key,
                    api_base,
                    failures=failures,
                )
                for turn_idx, turn in enumerate(choice["turns"])
            ]

            turn_scores = asyncio.run(_gather_scores(grading_tasks))
            choice["score"] = sum(turn_scores) / len(turn_scores) if turn_scores else 0
            scores.append(choice["score"])


class EvalMTBench(Evaluation):
    task_type = MTBenchTask


if __name__ == "__main__":
    args = parse_arguments()
    EvalMTBench(args).eval()
