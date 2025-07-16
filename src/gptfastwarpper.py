import torch
from torch import nn
from .gpt_fast_model import Transformer
from torch.nn.attention.flex_attention import create_block_mask


def causal_mask(b, h, q_idx, kv_idx):
    """Causal mask function for attention"""
    return q_idx >= kv_idx


class GPTFastWrapper(nn.Module):
    """包装 GPT-Fast Transformer 模型，使其兼容现有接口"""

    def __init__(self, transformer: Transformer):
        super().__init__()
        self.transformer = transformer
        self._caches_initialized = False

    def _ensure_caches_initialized(self, batch_size: int, seq_len: int):
        """确保缓存已初始化"""
        if not self._caches_initialized:
            # 设置一个合理的最大序列长度
            max_seq_len = max(seq_len + 1024, 2048)  # 给生成留出空间
            self.transformer.setup_caches(batch_size, max_seq_len)
            self._caches_initialized = True

    def forward(self, input_ids, **kwargs):
        """兼容 HuggingFace 接口的 forward 方法"""
        batch_size, seq_len = input_ids.shape

        # 获取模型设备
        model_device = self.device

        # 确保输入在正确设备上
        input_ids = input_ids.to(model_device)

        # 确保缓存已初始化
        self._ensure_caches_initialized(batch_size, seq_len)

        # 获取实际的 KV 缓存长度
        kv_length = self.transformer.max_seq_length

        # 创建 causal mask - 使用 seq_len 作为查询长度，kv_length 作为键值长度
        mask = create_block_mask(causal_mask, batch_size, None, seq_len, kv_length)

        # 创建 input_pos，确保在正确设备上
        input_pos = torch.arange(0, seq_len, device=model_device)

        # 调用 transformer
        logits = self.transformer.forward(mask, input_ids, input_pos)

        # 返回与 HuggingFace 兼容的输出格式
        class Output:
            def __init__(self, logits):
                self.logits = logits
                # 添加 past_key_values 属性，设置为 None（因为 GPT-Fast 内部管理 KV 缓存）
                self.past_key_values = None

        return Output(logits)

    def generate(self, input_ids, max_new_tokens=1, **kwargs):
        """兼容生成接口"""
        batch_size, seq_len = input_ids.shape

        # 确保输入在正确设备上
        input_ids = input_ids.to(self.device)  # 修复这里

        self._ensure_caches_initialized(batch_size, seq_len + max_new_tokens)
        return self.transformer.generate(input_ids, max_new_tokens)

    def eval(self):
        """设置为评估模式"""
        self.transformer.eval()
        return self

    def to(self, device):
        """移动到设备"""
        # 重置缓存标志，因为设备改变了
        self._caches_initialized = False
        self.transformer = self.transformer.to(device)
        return self

    @property
    def device(self):
        """获取设备"""
        return next(self.transformer.parameters()).device

    def setup_caches(self, max_batch_size, max_seq_length):
        # 获取模型设备
        model_device = next(self.parameters()).device

        # 确保 freqs_cis 在正确设备上
        if hasattr(self, "freqs_cis") and self.freqs_cis is not None:
            self.freqs_cis = self.freqs_cis.to(model_device)
