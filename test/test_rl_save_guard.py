"""评测不得写 RL checkpoint：`--disable_rl_update` 必须被所有保存点尊重。

背景（2026-10-19）：adaptive_decoding / adaptive_tridecoding 的收尾各自
直接调 `rl_adapter.save()`，绕过了带 `disable_rl_update` 守卫的
`_save_adaptive_rl_checkpoints`。结果是评测跑完会把 worktree 里的
checkpoint 重写（golden 的 checkpoint 指纹因此漂移，且覆盖用户训练状态）。
"""

from argparse import Namespace
from unittest.mock import MagicMock

from src.baselines import Baselines


class _Probe(Baselines):
    """只为实例化抽象基类；保存逻辑不依赖解码入口。"""

    def load_data(self):
        return None

    def preprocess(self, input_text):
        return input_text

    def postprocess(self, input_text, output_text):
        return output_text

    def eval(self):
        return None


def _bare(disable: bool):
    obj = object.__new__(_Probe)
    obj.args = Namespace(disable_rl_update=disable)
    obj.rl_adapter = MagicMock(name="main")
    obj.little_rl_adapter = MagicMock(name="little")
    return obj


def test_eval_mode_never_saves_rl_checkpoints():
    obj = _bare(disable=True)
    obj._save_adaptive_rl_checkpoints(12.0)
    obj.rl_adapter.save.assert_not_called()
    obj.little_rl_adapter.save.assert_not_called()


def test_training_mode_saves_both_agents():
    obj = _bare(disable=False)
    obj._save_adaptive_rl_checkpoints(12.0)
    obj.rl_adapter.save.assert_called_once_with(12.0)
    obj.little_rl_adapter.save.assert_called_once_with(12.0)
