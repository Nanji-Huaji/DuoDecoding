from eval.eval import Eval
from typing import Callable, Tuple, Dict, Any
import transformers
import fastchat
from datasets import load_dataset
import accelerate
import tqdm

PROMPT = {
    "system_prompt": "You are a helpful assistant that helps people find information.",
}

class EvalGSM8K(Eval):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.load_tokenizer()
        self.load_data()
        self.load_model()

    def warmup(self) -> None:
        decoding = self.get_decoding_fn()
        for question in tqdm.tqdm(
            self.data,
            total=len(self.data),
            disable=not self.accelerator.is_main_process,
            ncols=50,
        ):
            pass

    def load_data(self) -> None:
        """
        load a jsonl data from huggingface.
        the data will be stored in self.data as a list of dict.
        dataset format:
        {
            "question": str,
            "answer": str,
        }
        """
        self.color_print(f"Loading GSM8K data...", 3)
        try:
            dataset = load_dataset("gsm8k", "main", split="test")
            self.data = [item for item in dataset]
            self.color_print(f"Loaded {len(self.data)} samples.", 3)
        except Exception as e:
            self.color_print(f"Error loading GSM8K data: {e}", 1)
