from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Type
from src.baselines import Baselines
import inspect

class Eval(Baselines):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.load_tokenizer()
        self.load_model()

        if "Llama-2" in str(self.args.draft_model) and "Llama-2" in str(
            self.args.target_model
        ):
            self.model_id = "llama-2-chat"
        elif "Llama-2" in str(self.args.target_model):
            self.model_id = "vicuna"
        elif "vicuna" in str(self.args.draft_model) and "vicuna" in str(
            self.args.target_model
        ):
            self.model_id = "vicuna"
        elif "Llama-3.1" in str(self.args.draft_model) and "Llama-3.1" in str(
            self.args.target_model
        ):
            self.model_id = "llama-3.1"
        elif "llama" in str(self.args.draft_model):
            self.model_id = "vicuna"
        elif "Qwen" in str(self.args.target_model) or "qwen" in str(self.args.target_model):
            self.model_id = "qwen"
        else:
            raise NotImplementedError
        
    @staticmethod
    def get_class_methods(target_class: Type[Baselines]) -> List[str]:
        """获取类中定义的所有方法（不包括继承的方法和__init__）"""
        methods = []
        for name, method in inspect.getmembers(target_class, predicate=inspect.isfunction):
            # 检查方法是否在当前类中定义（而非继承）
            if name in target_class.__dict__ and name != "__init__":
                methods.append(name)
        return methods

    @abstractmethod
    def warmup(self) -> None:
        pass

    def get_decoding_fn(self) -> Callable:
        decoding: Callable
        if self.args.eval_mode == "small" or self.args.eval_mode == "large":
            decoding = self.autoregressive_sampling
        elif self.args.eval_mode == "sd":
            decoding = self.speculative_decoding
        elif self.args.eval_mode == "para_sd":
            decoding = self.parallel_speculative_decoding
        elif self.args.eval_mode == "duodec":
            decoding = self.duodecoding
        elif self.args.eval_mode == "pld":
            decoding = self.pld_forward
        elif self.args.eval_mode == "lade":
            decoding = self.lookahead_forward
        elif self.args.eval_mode == "rest":
            decoding = self.rest_forward
        elif self.args.eval_mode == "tridecoding":
            decoding = self.tridecoding
        elif self.args.eval_mode == "tridecoding_with_bandwidth":
            decoding = self.tridecoding_with_bandwidth
        elif self.args.eval_mode == "uncertainty_decoding":
            decoding = self.uncertainty_decoding
        elif self.args.eval_mode == "speculative_decoding_with_bandwidth":
            decoding = self.speculative_decoding_with_bandwidth
        elif (
            self.args.eval_mode
            == "speculative_decoding_with_bandwidth_full_prob"
        ):
            decoding = self.speculative_decoding_with_bandwidth_full_prob
        elif self.args.eval_mode in self.get_class_methods(Baselines):
            decoding = getattr(self, self.args.eval_mode)
        else:
            try:
                methods = getattr(self, self.args.eval_mode)
                if callable(methods):
                    decoding = methods
                else:
                    self.color_print("Unknown eval mode", 3)
                    raise NotImplementedError
            except AttributeError:
                self.color_print("Unknown eval mode", 3)
                raise NotImplementedError
            print(f"Unknown eval mode: {self.args.eval_mode}")
            raise NotImplementedError
        return decoding

    @abstractmethod
    def load_data(self) -> None:
        """
        Should load data to self.data
        """
        pass

    @abstractmethod
    def eval(self) -> None:
        """
        Main evaluation function
        """
        pass

    @abstractmethod
    def wrap_prompt(self, input_text: str) -> str:
        """
        Wrap the input_text into a model-specific prompt
        """
        pass
    
    