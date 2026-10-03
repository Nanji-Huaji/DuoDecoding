"""幽灵模式清理的注册表完备性回归测试。

历史问题：``MODE_FEATURES`` 里列了
``speculative_decoding_with_bandwidth_full_prob``，但没有任何解码实现
注册它——它能通过能力表校验，却在 ``Register.get_decoding_method()``
抛 ``NotImplementedError``。它想表达的"强制传完整词表概率"已由 CLI 开关
``--force_full_vocab_transfer`` 在任何模式上提供，故该幽灵条目删除。

本测试锁定两件事：
1. 能力表里每个模式都能在 ``Register._DECODING_REGISTRY`` 找到实现；
2. 幽灵模式不会复活（既不在能力表，也不在注册表）。
"""

from src.mode_features import MODE_FEATURES
from src.register import Register

GHOST_MODE = "speculative_decoding_with_bandwidth_full_prob"


def _trigger_decoding_registration() -> None:
    """import 触发装饰器注册；无 import 链时注册表是空的。"""
    import src.baselines  # noqa: F401
    import src.engine  # noqa: F401


class TestModeFeaturesRegistry:
    def test_every_mode_feature_is_registered(self):
        _trigger_decoding_registration()

        unregistered = set(MODE_FEATURES) - set(Register._DECODING_REGISTRY)
        assert not unregistered, (
            f"MODE_FEATURES 有但未注册解码方法: {unregistered}"
        )

    def test_ghost_mode_is_not_in_features(self):
        assert GHOST_MODE not in MODE_FEATURES

    def test_ghost_mode_is_not_registered(self):
        _trigger_decoding_registration()

        assert GHOST_MODE not in Register._DECODING_REGISTRY
