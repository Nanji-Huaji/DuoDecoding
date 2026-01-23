from argparse import Namespace

class Register:
    # 这里的字典存储的是 { "mode_name": UnboundFunction }
    _DECODING_REGISTRY = {}

    def __init__(self, args: Namespace):
        self.args = args

    @classmethod
    def register_decoding(cls, name: str):
        def decorator(func):
            # 将函数注册到解码方法注册表中
            cls._DECODING_REGISTRY[name] = func
            return func

        return decorator

    def get_decoding_method(self) -> callable:
        mode = self.args.eval_mode

        # 1. 尝试从注册表中获取
        func = self._DECODING_REGISTRY.get(mode, None)

        if func is not None:
            # 手动将函数绑定到当前实例 (self)
            # 这相当于把 func 变成了 self.func
            return func.__get__(self, self.__class__)

        # 2. 兜底逻辑：尝试反射 (getattr 会自动处理绑定)
        if hasattr(self, mode):
            return getattr(self, mode)
        else:
            raise NotImplementedError(f"Decoding method {mode} not found.")
