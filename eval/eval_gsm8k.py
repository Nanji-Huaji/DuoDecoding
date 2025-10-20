from eval.eval import Eval
from typing import Callable, Tuple, Dict, Any


class EvalGSM8K(Eval):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

