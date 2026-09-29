from argparse import Namespace

# 历史遗留的非标识符模式名（含 "/"），保留兼容
_LEGACY_NAMES = {"ceesd_w/o_arp"}


class Register:
    # 这里的字典存储的是 { "mode_name": UnboundFunction }
    _DECODING_REGISTRY = {}

    def __init__(self, args: Namespace):
        self.args = args

    @classmethod
    def register_decoding(cls, name: str):
        # D2：名字校验在装饰器工厂调用时立即执行（不等应用到函数）
        if name.startswith("_") or not (
            name.isidentifier() or name in _LEGACY_NAMES
        ):
            # 防御把任意属性名（如 __init__）当解码方法注册
            raise ValueError(f"非法解码方法名 {name!r}")

        def decorator(func):
            # D2：异函数同名注册显式拒绝。此前的静默覆盖会掩盖 import
            # 副作用（例如 profile_cee_dsd.py 的同名插桩类覆盖生产实现）。
            # 同一函数挂多个名字（别名，如 dist_split_spec/dssd）合法。
            existing = cls._DECODING_REGISTRY.get(name)
            if existing is not None and existing.__code__ is not func.__code__:
                raise ValueError(
                    f"解码方法 {name!r} 重复注册："
                    f"已注册于 {getattr(existing, '__module__', '?')}"
                    f".{getattr(existing, '__qualname__', '?')}，"
                    f"再次注册于 {getattr(func, '__module__', '?')}"
                    f".{getattr(func, '__qualname__', '?')}"
                )
            cls._DECODING_REGISTRY[name] = func
            return func

        return decorator

    def get_decoding_method(self):
        mode = self.args.eval_mode

        func = self._DECODING_REGISTRY.get(mode, None)

        if func is not None:
            # 手动将函数绑定到当前实例 (self)
            # 这相当于把 func 变成了 self.func
            return func.__get__(self, self.__class__)

        # D2：删除 hasattr 反射兜底——它会把任意属性名（如 __init__）
        # 当解码方法返回，掩盖拼写错误。未注册的模式直接报错并列出可用名。
        raise NotImplementedError(
            f"Decoding method {mode!r} not found. "
            f"Available: {sorted(self._DECODING_REGISTRY)}"
        )
