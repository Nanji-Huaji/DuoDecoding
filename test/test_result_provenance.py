"""Phase 0.4：结果工件的 git 溯源（Step 0，口径删除的解锁条件）。

get_save_dict 产出的每个 metrics json 必须携带 git_sha + git_dirty，
使"复现历史数字"锚定到 commit 而非代码内 legacy 分支。
"""

import re
from types import SimpleNamespace

from eval.utils import ExpPrint, _git_provenance


def _minimal_args() -> SimpleNamespace:
    return SimpleNamespace(
        dump_network_stats=False,
        little_model="little",
        draft_model="draft",
        target_model="target",
        eval_mode="adaptive_tridecoding",
        gamma=3,
        gamma1=5,
        gamma2=5,
    )


def _minimal_metrics() -> dict:
    return {"wall_time": 1.0, "generated_tokens": 10, "communication_time": 0.5}


def test_git_provenance_shape():
    prov = _git_provenance()
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", prov["git_sha"])
    assert prov["git_dirty"] in (True, False, None)


def test_save_dict_carries_provenance():
    result = ExpPrint(_minimal_args()).get_save_dict(_minimal_metrics())
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", result["git_sha"]), (
        "结果工件缺少合法 git_sha——Step 0 的核心产出"
    )
    assert result["git_dirty"] in (True, False, None)


def test_provenance_cached_per_process():
    """同一进程内重复调用不重复 spawn git（get_save_dict 每样本都会调）。"""
    first = _git_provenance()
    second = _git_provenance()
    assert first == second
