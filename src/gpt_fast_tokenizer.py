import os
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import sentencepiece as spm
from pathlib import Path
from typing import Union, List, Dict


class TokenizerInterface:
    def __init__(self, model_path):
        self.model_path = model_path

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")


class SentencePieceWrapper:
    """SentencePiece tokenizer wrapper"""

    def __init__(self, model_path: Path):
        try:
            import sentencepiece as smp

            self.processor = smp.SentencePieceProcessor()
            self.processor.load(str(model_path))
        except ImportError:
            raise ImportError("sentencepiece is required for tokenizer functionality")

        # 根据 Llama 模型的标准设置 special_tokens_map
        # 只包含实际存在的特殊 token，避免 None 值
        special_tokens = {}

        # 检查并添加 BOS token
        if self.processor.bos_id() != -1:
            special_tokens["bos_token"] = self.processor.id_to_piece(self.processor.bos_id())

        # 检查并添加 EOS token
        if self.processor.eos_id() != -1:
            special_tokens["eos_token"] = self.processor.id_to_piece(self.processor.eos_id())

        # 检查并添加 UNK token
        if self.processor.unk_id() != -1:
            special_tokens["unk_token"] = self.processor.id_to_piece(self.processor.unk_id())

        # Llama 模型通常没有专用的 PAD token，使用 EOS token 作为 PAD
        # 如果有专门的 PAD token ID，则添加
        if self.processor.pad_id() != -1:
            special_tokens["pad_token"] = self.processor.id_to_piece(self.processor.pad_id())
        elif self.processor.eos_id() != -1:
            # 如果没有专门的 PAD token，使用 EOS token
            special_tokens["pad_token"] = self.processor.id_to_piece(self.processor.eos_id())

        self.special_tokens_map = special_tokens

    def encode(self, text: str, **kwargs) -> list:
        """编码文本为 token IDs"""
        return self.processor.encode_as_ids(text)

    def decode(self, ids, skip_special_tokens=True, spaces_between_special_tokens=False, **kwargs) -> str:
        """解码 token IDs 为文本，兼容 HuggingFace 接口"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        elif not isinstance(ids, list):
            ids = list(ids)

        # SentencePiece 的 decode 方法
        return self.processor.decode_ids(ids)

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    @property
    def pad_token_id(self) -> int:
        # Llama 通常使用 EOS token 作为 PAD token
        pad_id = self.processor.pad_id()
        if pad_id != -1:
            return pad_id
        else:
            # 如果没有专门的 PAD token，返回 EOS token ID
            return self.processor.eos_id()

    @property
    def eos_token_id(self) -> int:
        return self.processor.eos_id()

    @property
    def bos_token_id(self) -> int:
        return self.processor.bos_id()

    @property
    def unk_token_id(self) -> int:
        return self.processor.unk_id()

    def __call__(self, text, **kwargs):
        """支持直接调用进行编码"""
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        elif isinstance(text, list):
            return {"input_ids": [self.encode(t) for t in text]}
        else:
            raise ValueError(f"Unsupported input type: {type(text)}")


class TiktokenWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path):
        super().__init__(model_path)
        assert os.path.isfile(model_path), str(model_path)
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, self.num_reserved_special_tokens - 5)]
        self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]

    def encode(self, text):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id


def get_tokenizer(model_path: Path, model_name: str) -> SentencePieceWrapper:
    """Get tokenizer for the model"""
    # model_path 现在指向 model.pth，我们需要在其父目录中查找分词器
    model_dir = model_path.parent

    # 查找分词器文件的可能位置
    possible_tokenizer_paths = [
        model_dir / "tokenizer.model",
        model_dir / "tokenizer.sp",
        model_dir / "spiece.model",
        model_dir / f"{model_name}.model",
    ]

    tokenizer_path = None
    for path in possible_tokenizer_paths:
        if path.exists():
            tokenizer_path = path
            break

    if tokenizer_path is None:
        raise FileNotFoundError(
            f"Tokenizer file not found in {model_dir}. Searched for: {[p.name for p in possible_tokenizer_paths]}"
        )

    return SentencePieceWrapper(tokenizer_path)
