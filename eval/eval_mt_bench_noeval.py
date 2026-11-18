"""MT-Bench noeval：只生成并统计速度，不判分、不需要 API/凭证。

对话、解码、速度与 IO 全部复用 ``eval.mt_bench_common``；与 judge 入口
的差异仅在 warmup 规模/轮数（2 个问题、只跑第一轮）和关闭判分。
"""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.mt_bench_common import (  # noqa: E402, F401
    MTBenchTaskBase,
    _token_debug_checks_enabled,  # 保留旧 noeval 调试开关的导入面
    read_results,  # 兼容旧的 ``eval.eval_mt_bench_noeval.read_results``
)
from eval.runner import Evaluation  # noqa: E402
from src.utils import parse_arguments  # noqa: E402


class MTBenchNoEvalTask(MTBenchTaskBase):
    name = "mt_bench"
    record_name = "mt_bench"
    model_id_task = "mt_bench_noeval"
    # 旧 noeval warmup：2 个问题，且只热身第一轮以加快速度。
    warmup_questions = 2
    warmup_turns = 1
    enable_judge = False
    log_warmup_speed = True
    debug_token_checks = True


class EvalMTBench(Evaluation):
    task_type = MTBenchNoEvalTask


if __name__ == "__main__":
    args = parse_arguments()
    EvalMTBench(args).eval()
