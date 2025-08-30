import numpy as np
from typing import TypedDict, List, Tuple, Literal, Union, Optional, Any
import torch
import math

import warnings

class TransferUnit(TypedDict):
    data_size_bytes: int
    transfer_time: float


class Statistics(TypedDict):
    edge_cloud: List[TransferUnit]
    edge_end: List[TransferUnit]
    cloud_end: List[TransferUnit]


LinkType = Literal["edge_cloud", "edge_end", "cloud_end"]


class CommunicationSimulator:

    def __init__(self, bandwidth_edge_cloud, bandwidth_edge_end, bandwidth_cloud_end, protocol_overhead_bytes: int = 0, transfer_top_k: Optional[int] = None):
        self.bandwidth_edge_cloud = bandwidth_edge_cloud
        self.bandwidth_edge_end = bandwidth_edge_end
        self.bandwidth_cloud_end = bandwidth_cloud_end
        self.protocol_overhead_bytes = protocol_overhead_bytes
        self.stats = Statistics(
            edge_cloud=[],
            edge_end=[],
            cloud_end=[],
        )
        # TODO: 实现top-k压缩
        self.transfer_top_k = transfer_top_k 

    @property
    def edge_cloud_comm_time(self):
        return sum(self.stats["edge_cloud"][i]["transfer_time"] for i in range(len(self.stats["edge_cloud"])))

    @property
    def edge_end_comm_time(self):
        return sum(self.stats["edge_end"][i]["transfer_time"] for i in range(len(self.stats["edge_end"])))

    @property
    def cloud_end_comm_time(self):
        return sum(self.stats["cloud_end"][i]["transfer_time"] for i in range(len(self.stats["cloud_end"])))

    def simulate_transfer(self, data_size_bytes, link_type, add_to_stats=True) -> float:
        if link_type == "edge_cloud":
            bandwidth = self.bandwidth_edge_cloud
        elif link_type == "edge_end":
            bandwidth = self.bandwidth_edge_end
        elif link_type == "cloud_end":
            bandwidth = self.bandwidth_cloud_end
        else:
            raise ValueError(f"Unknown link type: {link_type}")

        transfer_time = data_size_bytes / bandwidth

        if add_to_stats:
            transfer_unit = TransferUnit(data_size_bytes=data_size_bytes, transfer_time=transfer_time)
            self.stats[link_type].append(transfer_unit)

        return transfer_time
    
    @staticmethod
    def _apply_top_k_compression(probs: torch.Tensor | None, k: int) -> torch.Tensor:
        """
        probs: torch.Tensor，当前词表概率分布，形状为(..., V)
        k: int，保留的top-k数量
        返回压缩后的概率分布，形状与输入相同，但仅保留top-k概率，其他位置为0，值得注意的是，这不是真正的压缩，但我们假设传输时只传输非零部分
        """
        if probs is None or probs.numel() == 0:
            return torch.empty(0)

        if k >= len(probs):
            return probs

        # 获取top-k indices
        top_k_values, top_k_indices = torch.topk(probs, k, sorted=True)

        # 创建压缩的概率分布
        compressed_probs = torch.zeros_like(probs)
        compressed_probs[top_k_indices] = top_k_values

        return compressed_probs

    def transfer(self, tokens: torch.Tensor, prob: torch.Tensor|None, link_type: LinkType) -> float:
        token_bytes = 0
        prob_bytes = 0

        # Token data size (int32 or int64)
        if tokens is not None and tokens.numel() > 0:
            token_bytes = tokens.element_size() * tokens.numel()

        # Probability history data size (float32 or float16)
        if prob is not None and prob.numel() > 0:
            prob_bytes = prob.element_size() * prob.numel()

        total_bytes = token_bytes + prob_bytes

        total_bytes += self.protocol_overhead_bytes

        return self.simulate_transfer(total_bytes, link_type)

    def __call__(
        self,
        tokens: torch.Tensor,
        prob_history: Optional[torch.Tensor] = None,
        link_type: LinkType = "edge_cloud",
        prob_history_dtype=torch.float16,
        description="",
    ) -> Tuple[float, str]:
        if link_type not in ["edge_cloud", "edge_end", "cloud_end"]:
            raise ValueError(f"Unknown link type: {link_type}")

        token_bytes = 0
        prob_bytes = 0
        total_bytes = 0

        if tokens is not None and tokens.numel() > 0:
            token_bytes = tokens.element_size() * tokens.numel()
            total_bytes += token_bytes

        if prob_history is not None and prob_history.numel() > 0:
            if prob_history_dtype is not None:
                prob_history = prob_history.to(prob_history_dtype)
            prob_bytes = prob_history.element_size() * prob_history.numel()
            total_bytes += prob_bytes

        transfer_time = self.simulate_transfer(total_bytes, link_type)

        return transfer_time, link_type


class CUHLM(CommunicationSimulator):
    DEFAULT_COMPRESSED_VOCAB_SIZE = 300

    def __init__(
        self,
        bandwidth_edge_cloud,
        bandwidth_edge_end = float('inf'),
        bandwidth_cloud_end = float('inf'),
        uncertainty_threshold: float = 0.8,
        vocab_size: int = 32000,
    ):
        # 除了edge-cloud链路，其他链路假设无限带宽，因为不传输数据
        super().__init__(bandwidth_edge_cloud, bandwidth_edge_end, bandwidth_cloud_end)
        self.uncertainty_threshold = uncertainty_threshold
        self.vocab_size = vocab_size

    @staticmethod
    def calculate_uncertainty(
        logits: torch.Tensor | None, M: int = 20, theta_max: float = 2.0, draft_token: Optional[int] = None
    ) -> float:
        if logits is None or logits.numel() == 0:
            warnings.warn("警告：logits为空，无法计算不确定度，默认返回1.0")
            return 1.0
        if logits.dim() > 1:
            logits = logits[0]
        if draft_token is None:
            warnings.warn("警告：draft_token未提供，默认使用最高概率的token")
            draft_token = int(torch.argmax(logits).item())
        # 向量化温度采样
        temperatures = torch.rand(M, device=logits.device) * theta_max
        temperatures = torch.clamp(temperatures, min=1e-6)

        # 向量化扰动分布计算
        perturbed_logits = logits.unsqueeze(0) / temperatures.unsqueeze(1)  # [M, vocab_size]
        perturbed_logits = perturbed_logits - perturbed_logits.max(dim=1, keepdim=True)[0]
        perturbed_probs = torch.softmax(perturbed_logits, dim=-1)

        # 批量采样
        perturbed_tokens = torch.multinomial(perturbed_probs, 1).squeeze(1)  # [M]

        # 计算分歧
        disagreements = (perturbed_tokens != draft_token).sum().item()

        return disagreements / M

    @staticmethod
    def _get_current_probs(prob_history: Optional[torch.Tensor]) -> torch.Tensor:
        """
        返回当前时间步的概率分布，为一维张量，(...,vocab_size)
        """
        if prob_history is None or prob_history.numel() == 0:
            warnings.warn("警告：prob_history为空，无法获取当前概率分布")
            return torch.empty(0)
            # prob_history形状: (1, seq_len, 32000)
        if prob_history.dim() == 3:
            current_probs = prob_history[0, -1, :]  
        elif prob_history.dim() == 2:
            current_probs = prob_history[-1, :]  
        elif prob_history.dim() == 1:
            current_probs = prob_history
        else:
            raise ValueError("prob_history维度不支持")

        return current_probs  

    @staticmethod
    def rebuild_full_probs(compressed_probs: torch.Tensor) -> torch.Tensor:
        """
        接收一个稀疏的概率分布张量(compressed_probs)，并重建为完整的概率分布。
        compressed_probs: torch.Tensor，形状为(..., vocab_size)
        重建后形状同样为(..., vocab_size)
        """
        if compressed_probs is None or compressed_probs.numel() == 0:
            warnings.warn("警告：compressed_probs为空，无法重建完整概率分布")
            return torch.empty(0)

        rebuilt_probs = compressed_probs.clone()

        # 处理最后一个维度（vocab_size）
        # 找到非零位置（top-k位置）
        nonzero_mask = compressed_probs > 0  

        # 计算每个位置的top-k概率总和
        top_k_sum = compressed_probs.sum(dim=-1, keepdim=True)  

        # 计算剩余概率质量
        residual_mass = 1.0 - top_k_sum  

        # 计算零位置的数量
        zero_mask = compressed_probs == 0  
        zero_count = zero_mask.sum(dim=-1, keepdim=True)  

        # 避免除零：如果没有零位置，则不需要重建
        uniform_prob = torch.where(
            zero_count > 0, residual_mass / zero_count, torch.zeros_like(residual_mass)
        )  
        # 将均匀概率分配到零位置
        rebuilt_probs = torch.where(zero_mask, uniform_prob, rebuilt_probs)

        return rebuilt_probs

    def determine_transfer_strategy(self, uncertainty: float, current_probs: torch.Tensor | None) -> Tuple[bool, int]:
        "返回是否传输以及传输的词汇表大小"
        if current_probs is None or current_probs.numel() == 0: 
            warnings.warn("警告：current_probs为空，无法决定传输策略，默认不传输概率分布")
            return False, 0
        if uncertainty >= self.uncertainty_threshold: # 不确定度高，传输
            vocab_size = max(1, self._calculate_compressed_vocab_size(uncertainty, current_probs))
            return True, vocab_size
        else: # 大模型直接接受当前输出
            return False, 0

    @staticmethod
    def _apply_top_k_compression(probs: torch.Tensor | None, k: int) -> torch.Tensor:
        """
        probs: torch.Tensor，当前词表概率分布，形状为(..., V)
        k: int，保留的top-k数量
        返回压缩后的概率分布，形状与输入相同，但仅保留top-k概率，其他位置为0，值得注意的是，这不是真正的压缩，但我们假设传输时只传输非零部分
        """
        if probs is None or probs.numel() == 0:
            return torch.empty(0)

        if k >= len(probs):
            return probs

        # 获取top-k indices
        top_k_values, top_k_indices = torch.topk(probs, k, sorted=True)

        # 创建压缩的概率分布
        compressed_probs = torch.zeros_like(probs)
        compressed_probs[top_k_indices] = top_k_values

        return compressed_probs

    @staticmethod
    def softplus(z, eta=1.0):
        return torch.log(1 + torch.exp(eta * z)) / eta

    def _calculate_compressed_vocab_size(
        self, uncertainty: float, current_probs: torch.Tensor, theta: float = 0.1, draft_token: Optional[int] = None
    ) -> int:
        """严格按照论文公式(24)实现：k(t)* = arg min {k(t) | U_TV(au(t) + b) ≤ θ}"""

        if current_probs is None or current_probs.numel() == 0:
            return 0

        # 确保输入是完整的词汇表概率分布
        if len(current_probs) != self.vocab_size:
            warnings.warn(f"警告：概率分布长度({len(current_probs)})与词汇表大小({self.vocab_size})不匹配")
            return max(1, min(300, self.vocab_size // 100))

        # Step 1: 计算rejection probability
        a, b = 0.815, -0.066
        beta_d = max(0, min(1, a * uncertainty + b))

        # Step 2: 对概率分布排序
        sorted_probs, sorted_indices = torch.sort(current_probs, descending=True)

        # Step 3: 获取draft token概率
        if draft_token is None:
            x_d = sorted_probs[0].item()
        else:
            # draft_token是索引，获取对应概率
            if 0 <= draft_token < len(current_probs):
                x_d = current_probs[draft_token].item()
            else:
                warnings.warn(f"警告：draft_token索引({draft_token})超出范围")
                x_d = sorted_probs[0].item()

        # Step 4: 计算softplus函数
        eta = 1.0

        l_neg_1 = self.softplus(torch.tensor(-1.0), eta).item()
        l_neg_beta = self.softplus(torch.tensor(-beta_d), eta).item()

        # Step 5: 计算分母
        denominator = (1 - x_d) * l_neg_1 + x_d * l_neg_beta
        if denominator <= 0:
            return 30

        # Step 6: 搜索最小的k
        for k in range(1, self.vocab_size):  # 确保k < vocab_size
            # 计算top-k概率累积和
            top_k_sum = torch.sum(sorted_probs[:k]).item()
            residual_mass = 1.0 - top_k_sum

            # 计算均匀概率（避免除零）
            if k >= self.vocab_size or residual_mass <= 0:
                uniform_prob = 0
            else:
                uniform_prob = residual_mass / (self.vocab_size - k)

            # 计算分子：Σ(i=k+1 to |V|) |x_i(t) - x̂_i(t)|
            numerator = 0.0
            for i in range(k, len(sorted_probs)):
                original_prob = sorted_probs[i].item()
                numerator += abs(original_prob - uniform_prob)

            # 计算上界 U_TV(β_d(t))
            u_tv = numerator / denominator

            # 检查约束条件
            if u_tv <= theta:
                return k

        # 如果没找到满足条件的k，返回保守值
        return min(self.DEFAULT_COMPRESSED_VOCAB_SIZE, self.vocab_size // 100)

    def terminal_prob(self, current_probs: torch.Tensor, logits: Optional[torch.Tensor]) -> torch.Tensor:
        """
        返回先经过压缩再重建的终端概率分布
        形状为(vocab_size,)
        """
        if current_probs is None and logits is None:
            warnings.warn("警告：current_probs和logits均为空，无法获取终端概率分布")
            return torch.empty(0)
        
        if logits is None:
            # 按照贪心解码重建logits
            probs = torch.clamp(current_probs, min=1e-8)
            log_probs = torch.log(probs)
            # 减去最大值，使最大概率对应的logit为0
            if current_probs.dim() == 1:
                logits = log_probs - torch.max(log_probs)
            else:
                logits = log_probs - torch.max(log_probs, dim=-1, keepdim=True)[0]

        uncertainty = self.calculate_uncertainty(logits)
        should_transfer_prob, vocab_size = self.determine_transfer_strategy(uncertainty, current_probs)
        if not should_transfer_prob:
            return current_probs
        if vocab_size < self.vocab_size:
            compressed_probs = self._apply_top_k_compression(current_probs, vocab_size)
            rebuilt_probs = self.rebuild_full_probs(compressed_probs)
            return rebuilt_probs
        else:
            return current_probs


    def transfer(
        self, tokens: Optional[torch.Tensor], prob_history: Optional[torch.Tensor] = None, logits: Optional[torch.Tensor] = None, link_type: LinkType = "edge_cloud"
    ) -> float:
        if link_type != "edge_cloud":
            # 其他链路不传输数据
            return 0.0

        if tokens is None:
            tokens = torch.empty(0)

        current_probs = self._get_current_probs(prob_history)
        # 如果 prob_history 为空，current_probs 也是空张量，后续逻辑会在 determine_transfer_strategy 中处理这种情况（不传输概率分布，只传输 tokens）
        if logits is None and current_probs.numel() > 0:
            # 假设为贪心解码
            probs = torch.clamp(current_probs, min=1e-8)
            log_probs = torch.log(probs)
            # 减去最大值，使最大概率对应的logit为0
            if current_probs.dim() == 1:
                logits = log_probs - torch.max(log_probs)
            else:
                logits = log_probs - torch.max(log_probs, dim=-1, keepdim=True)[0]

        if (logits is None or logits.numel() == 0) and current_probs.numel() == 0:
            # logits和prob_history都为空，无法计算不确定度，直接传输tokens
            return super().transfer(tokens, None, link_type)


        uncertainty = self.calculate_uncertainty(logits)
        should_transfer_prob, vocab_size = self.determine_transfer_strategy(uncertainty, current_probs)
        if not should_transfer_prob:
            return super().transfer(tokens, None, link_type)

        if prob_history is None:
            prob_history = torch.empty(0)

        if tokens is None:
            tokens = torch.empty(0)

        if vocab_size < self.vocab_size:
            compressed_current_probs = self._apply_top_k_compression(current_probs, vocab_size)
            prob_history = prob_history.clone()
            if prob_history.dim() == 3:
                prob_history[0, -1, :] = compressed_current_probs
            else:
                raise ValueError("prob_history维度不支持")

        prob_size = vocab_size * prob_history.element_size() if prob_history.numel() > 0 else 0 # 假设传递稀疏词表时只传递非0
        token_size = tokens.element_size() * tokens.numel() if tokens.numel() > 0 else 0

        return super().simulate_transfer(token_size + prob_size, link_type)