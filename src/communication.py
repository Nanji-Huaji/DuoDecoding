import numpy as np
from typing import TypedDict, List, Tuple, Literal, Union, Optional, Any
import torch
import math

import warnings

import logging


class TransferUnit(TypedDict):
    data_size_bytes: int | float
    transfer_time: float


class Statistics(TypedDict):
    edge_cloud: List[TransferUnit]
    edge_end: List[TransferUnit]
    cloud_end: List[TransferUnit]


LinkType = Literal["edge_cloud", "edge_end", "cloud_end"]

Dimension = Literal["Mbps", "MBps", "bps", "Bps"]


def _convert_to_bytes_per_second(
    bandwidth: float, dimension: Dimension
) -> float:
    """
    将带宽转换为bytes/second
    """
    if dimension == "Mbps":
        return bandwidth * 1e6 / 8
    elif dimension == "MBps":
        return bandwidth * 1e6
    elif dimension == "bps":
        return bandwidth / 8
    elif dimension == "Bps":
        return bandwidth
    else:
        raise ValueError(f"Unknown dimension: {dimension}")


class CommunicationSimulator:
    """
    用于模拟通信的类
    - 带宽单位：bytes/second
    - protocol_overhead_bytes: 每次传输的协议开销，单位bytes
    - transfer_top_k: Optional[int], 如果设置了top-k压缩，则在传输概率分布时只传输top-k概率，其他位置为0，假设传输时只传输非零部分
    - 统计信息保存在self.stats中
    - 统计信息包括每次传输的数据大小和传输时间
    - 统计信息分为三类链路：edge-cloud, edge-end, cloud-end
    """

    def __init__(
        self,
        bandwidth_edge_cloud,
        bandwidth_edge_end,
        bandwidth_cloud_end,
        protocol_overhead_bytes: int = 0,
        transfer_top_k: Optional[int] = None,
        dimension: Dimension = "Mbps",
        ntt_ms_edge_end: float = 20,
        ntt_ms_edge_cloud: float = 200
    ):
        self.bandwidth_edge_cloud = _convert_to_bytes_per_second(
            bandwidth_edge_cloud, dimension
        )
        self.bandwidth_edge_end = _convert_to_bytes_per_second(
            bandwidth_edge_end, dimension
        )
        self.bandwidth_cloud_end = _convert_to_bytes_per_second(
            bandwidth_cloud_end, dimension
        )
        self.protocol_overhead_bytes = protocol_overhead_bytes
        self.stats = Statistics(
            edge_cloud=[],
            edge_end=[],
            cloud_end=[],
        )
        self.transfer_top_k = transfer_top_k

        self.ntt_edge_end = ntt_ms_edge_end / 1000  # 转换为秒
        self.ntt_edge_cloud = ntt_ms_edge_cloud / 1000  # 转换为秒

        self.connect_times = {
            "edge_end": 0, 
            "cloud_end": 0,
            "edge_cloud": 0
        }

    @property
    def edge_cloud_comm_time(self):
        return sum(
            self.stats["edge_cloud"][i]["transfer_time"]
            for i in range(len(self.stats["edge_cloud"]))
        )

    @property
    def edge_end_comm_time(self):
        return sum(
            self.stats["edge_end"][i]["transfer_time"]
            for i in range(len(self.stats["edge_end"]))
        )

    @property
    def cloud_end_comm_time(self):
        return sum(
            self.stats["cloud_end"][i]["transfer_time"]
            for i in range(len(self.stats["cloud_end"]))
        )

    @property
    def edge_cloud_data(self):
        return sum(
            self.stats["edge_cloud"][i]["data_size_bytes"]
            for i in range(len(self.stats["edge_cloud"]))
        )

    @property
    def edge_end_data(self):
        return sum(
            self.stats["edge_end"][i]["data_size_bytes"]
            for i in range(len(self.stats["edge_end"]))
        )

    @property
    def cloud_end_data(self):
        return sum(
            self.stats["cloud_end"][i]["data_size_bytes"]
            for i in range(len(self.stats["cloud_end"]))
        )

    @property
    def get_connect_times(self) -> dict:
        return self.connect_times


    def simulate_transfer(
        self,
        data_size_bytes: int | float,
        link_type: Literal["edge_cloud", "edge_end", "cloud_end"],
        add_to_stats=True,
    ) -> float:
        """
        执行一次传输模拟。
        - data_size_bytes: 传输的数据大小，单位bytes
        - link_type: 传输链路类型，"edge_cloud", "edge_end", "cloud_end"
        """
        if link_type == "edge_cloud":
            bandwidth = self.bandwidth_edge_cloud
        elif link_type == "edge_end":
            bandwidth = self.bandwidth_edge_end
        elif link_type == "cloud_end":
            bandwidth = self.bandwidth_cloud_end
        else:
            raise ValueError(f"Unknown link type: {link_type}")

        transfer_time = data_size_bytes / bandwidth

        if link_type == "edge_end":
            ntt = self.ntt_edge_end
        elif link_type == "edge_cloud":
            ntt = self.ntt_edge_cloud
        elif link_type == "cloud_end":
            ntt = self.ntt_edge_cloud + self.ntt_edge_end

        self.connect_times[link_type] += 1

        transfer_time += ntt

        if add_to_stats:
            transfer_unit = TransferUnit(
                data_size_bytes=data_size_bytes, transfer_time=transfer_time
            )
            self.stats[link_type].append(transfer_unit)

        return transfer_time

    @staticmethod
    def _apply_top_k_compression(
        probs: torch.Tensor | None, k: int
    ) -> torch.Tensor:
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
            zero_count > 0,
            residual_mass / zero_count,
            torch.zeros_like(residual_mass),
        )
        # 将均匀概率分配到零位置
        rebuilt_probs = torch.where(zero_mask, uniform_prob, rebuilt_probs)

        return rebuilt_probs

    @staticmethod
    def compress_rebuild_probs(probs: torch.Tensor, k: int) -> torch.Tensor:
        """
        Args:
            probs: 形状：(batch_size, seq_len, vocab_size)
            k: int, top-k数量
        Returns:
            rebuilt_probs: 形状：(batch_size, seq_len, vocab_size)
        """
        if probs is None or probs.numel() == 0:
            warnings.warn("警告：probs为空，无法进行压缩重建")
            return torch.empty(0)

        if probs.dim() != 3:
            raise ValueError(
                f"probs维度应为3，实际为{probs.dim()}，无法进行压缩重建"
            )

        if k >= probs.shape[-1]:
            return probs  # 无需压缩

        batch_size, seq_len, vocab_size = probs.shape

        # 向量化处理：reshape为(batch_size * seq_len, vocab_size)
        flat_probs = probs.view(-1, vocab_size)

        # 批量获取top-k
        top_k_values, top_k_indices = torch.topk(
            flat_probs, k, dim=-1, sorted=True
        )

        # 创建压缩的概率分布
        compressed_probs = torch.zeros_like(flat_probs)
        batch_indices = (
            torch.arange(flat_probs.shape[0]).unsqueeze(1).expand(-1, k)
        )
        compressed_probs[batch_indices, top_k_indices] = top_k_values

        # 重建概率分布
        top_k_sum = compressed_probs.sum(dim=-1, keepdim=True)
        residual_mass = 1.0 - top_k_sum
        zero_mask = compressed_probs == 0
        zero_count = zero_mask.sum(dim=-1, keepdim=True)

        uniform_prob = torch.where(
            zero_count > 0,
            residual_mass / zero_count,
            torch.zeros_like(residual_mass),
        )

        rebuilt_flat_probs = torch.where(
            zero_mask, uniform_prob, compressed_probs
        )

        # 恢复原始形状
        return rebuilt_flat_probs.view(batch_size, seq_len, vocab_size)

    def transfer(
        self,
        tokens: torch.Tensor | None,
        prob: torch.Tensor | None,
        link_type: Literal["edge_cloud", "edge_end", "cloud_end"],
        is_compressed: bool = False,
        compressed_k: Optional[int] = 300,
    ) -> float:
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

        if (
            is_compressed
            and prob is not None
            and prob.numel() > 0
            and compressed_k is not None
        ):
            # 计算压缩后的概率分布大小，假设传输时只传输非零部分
            if prob.dim() == 3:
                seq_length = prob.shape[1]
            else:
                seq_length = 1
            prob_size = compressed_k * prob.element_size() * seq_length
            total_bytes = token_bytes + prob_size + self.protocol_overhead_bytes

        return self.simulate_transfer(total_bytes, link_type)

    def send_reject_message(
        self, linktype: Literal["edge_cloud", "edge_end", "cloud_end"]
    ) -> None:
        self.simulate_transfer(6, linktype)

    def send_accept_message(
        self, linktype: Literal["edge_cloud", "edge_end", "cloud_end"]
    ) -> None:
        self.simulate_transfer(6, linktype)

    def __call__(
        self,
        tokens: torch.Tensor,
        prob_history: Optional[torch.Tensor] = None,
        link_type: Literal[
            "edge_cloud", "edge_end", "cloud_end"
        ] = "edge_cloud",
        prob_history_dtype=torch.float16,
        is_compressed: bool = False,
        compressed_k: Optional[int] = 300,
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

    @property
    def total_comm_energy(self) -> float:
        """
        仅在子类中实现，用于占位，表示通信能耗，单位焦耳
        """
        return 0.0


class CUHLM(CommunicationSimulator):
    """
    基于不确定性进行机会传输的对比实验方法。

    超参数：
    - uncertainty_threshold: float, 不确定度阈值，默认0.8
    - M: int, 温度扰动采样数量，默认20
    - theta_max: float, 最大温度扰动，默认2.0
    """

    DEFAULT_COMPRESSED_VOCAB_SIZE = 300

    def __init__(
        self,
        bandwidth_edge_cloud,
        bandwidth_edge_end=float("inf"),
        bandwidth_cloud_end=float("inf"),
        uncertainty_threshold: float = 0.8,
        vocab_size: int = 32000,
        dimension: Dimension = "Mbps",
        ntt_ms_edge_end: float = 20,
        ntt_ms_edge_cloud: float = 200,
    ):
        # 除了edge-cloud链路，其他链路假设无限带宽，因为不传输数据
        super().__init__(
            bandwidth_edge_cloud,
            bandwidth_edge_end,
            bandwidth_cloud_end,
            dimension=dimension,
            ntt_ms_edge_end=ntt_ms_edge_end,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
        )
        self.uncertainty_threshold = uncertainty_threshold
        self.vocab_size = vocab_size

    @staticmethod
    def calculate_uncertainty(
        logits: torch.Tensor | None,
        M: int = 20,
        theta_max: float = 2.0,
        draft_token: Optional[int] = None,
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
        perturbed_logits = logits.unsqueeze(0) / temperatures.unsqueeze(
            1
        )  # [M, vocab_size]
        perturbed_logits = (
            perturbed_logits - perturbed_logits.max(dim=1, keepdim=True)[0]
        )
        perturbed_probs = torch.softmax(perturbed_logits, dim=-1)

        # 批量采样
        perturbed_tokens = torch.multinomial(perturbed_probs, 1).squeeze(
            1
        )  # [M]

        # 计算分歧
        disagreements = (perturbed_tokens != draft_token).sum().item()

        return disagreements / M

    @staticmethod
    def _get_current_probs(
        prob_history: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
            zero_count > 0,
            residual_mass / zero_count,
            torch.zeros_like(residual_mass),
        )
        # 将均匀概率分配到零位置
        rebuilt_probs = torch.where(zero_mask, uniform_prob, rebuilt_probs)

        return rebuilt_probs

    def determine_transfer_strategy(
        self, uncertainty: float, current_probs: torch.Tensor | None
    ) -> Tuple[bool, int]:
        "返回是否传输以及传输的词汇表大小"
        if current_probs is None or current_probs.numel() == 0:
            warnings.warn(
                "警告：current_probs为空，无法决定传输策略，默认不传输概率分布"
            )
            return False, 0
        if uncertainty >= self.uncertainty_threshold:  # 不确定度高，传输
            vocab_size = max(
                1,
                self._calculate_compressed_vocab_size(
                    uncertainty, current_probs
                ),
            )
            return True, vocab_size
        else:  # 大模型直接接受当前输出
            return False, 0

    @staticmethod
    def _apply_top_k_compression(
        probs: torch.Tensor | None, k: int
    ) -> torch.Tensor:
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
        self,
        uncertainty: float,
        current_probs: torch.Tensor,
        theta: float = 0.1,
        draft_token: Optional[int] = None,
    ) -> int:
        """严格按照论文公式(24)实现：k(t)* = arg min {k(t) | U_TV(au(t) + b) ≤ θ}"""

        if current_probs is None or current_probs.numel() == 0:
            return 0

        # 确保输入是完整的词汇表概率分布
        if len(current_probs) != self.vocab_size:
            warnings.warn(
                f"警告：概率分布长度({len(current_probs)})与词汇表大小({self.vocab_size})不匹配"
            )
            return max(1, min(300, self.vocab_size // 100))

        # Step 1: 计算rejection probability
        a, b = 0.815, -0.066
        beta_d = max(0, min(1, a * uncertainty + b))

        # Step 2: 对概率分布排序
        sorted_probs, sorted_indices = torch.sort(
            current_probs, descending=True
        )

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

    def terminal_prob(
        self, current_probs: torch.Tensor, logits: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        返回先经过压缩再重建的终端概率分布
        形状为(vocab_size,)
        """
        if current_probs is None and logits is None:
            warnings.warn(
                "警告：current_probs和logits均为空，无法获取终端概率分布"
            )
            return torch.empty(0)

        if logits is None:
            # 按照贪心解码重建logits
            probs = torch.clamp(current_probs, min=1e-8)
            log_probs = torch.log(probs)
            # 减去最大值，使最大概率对应的logit为0
            if current_probs.dim() == 1:
                logits = log_probs - torch.max(log_probs)
            else:
                logits = (
                    log_probs - torch.max(log_probs, dim=-1, keepdim=True)[0]
                )

        uncertainty = self.calculate_uncertainty(logits)
        should_transfer_prob, vocab_size = self.determine_transfer_strategy(
            uncertainty, current_probs
        )
        if not should_transfer_prob:
            return current_probs
        if vocab_size < self.vocab_size:
            compressed_probs = self._apply_top_k_compression(
                current_probs, vocab_size
            )
            rebuilt_probs = self.rebuild_full_probs(compressed_probs)
            return rebuilt_probs
        else:
            return current_probs


class PreciseCommunicationSimulator(CommunicationSimulator):
    """
    用于基于香农信道容量计算通信参数的通信模拟器
    Args:
        bandwidth_hz: 信道带宽，单位Hz
        channel_gain: 信道增益
        send_power_watt: 发送功率，单位瓦特
        noise_power_watt: 噪声功率，单位瓦特
    """

    def __init__(
        self,
        bandwidth_hz: int | float,
        channel_gain: float,
        send_power_watt: float,
        noise_power_watt: float,
        ntt_ms_edge_end: float = 20,
        ntt_ms_edge_cloud: float = 200,
        edge_cloud_args: dict | None = None,
        edge_end_args: dict | None = None
    ):
        SNR = channel_gain * send_power_watt / noise_power_watt
        channel_capacity_bps = bandwidth_hz * math.log2(1 + SNR)
        logging.info(
            f"信道容量: {channel_capacity_bps/1e6:.2f} Mbps, 以 {channel_capacity_bps / 10} bps, {channel_capacity_bps} bps, {channel_capacity_bps / 10} bps 初始化 "
        )

        if edge_cloud_args is None:
            edge_cloud_bandwidth = channel_capacity_bps / 10
        else:
            try:
                edge_cloud_SNR = edge_cloud_args["channel_gain"] * edge_cloud_args["send_power_watt"] / edge_cloud_args["noise_power_watt"]
                edge_cloud_bandwidth = edge_cloud_args["bandwidth_hz"] * math.log2(1 + edge_cloud_SNR)
            except KeyError:
                edge_cloud_bandwidth = channel_capacity_bps / 10

        if edge_end_args is None:
            edge_end_bandwidth = channel_capacity_bps / 10
        else:
            try:
                edge_end_SNR = edge_end_args["channel_gain"] * edge_end_args["send_power_watt"] / edge_end_args["noise_power_watt"]
                edge_end_bandwidth = edge_end_args["bandwidth_hz"] * math.log2(1 + edge_end_SNR)
            except KeyError:
                edge_end_bandwidth = channel_capacity_bps / 10

        super().__init__(
            edge_cloud_bandwidth,
            channel_capacity_bps,
            edge_end_bandwidth,
            dimension="bps",
            ntt_ms_edge_end = ntt_ms_edge_end,
            ntt_ms_edge_cloud = ntt_ms_edge_cloud,
        )  # 假设云端链路和边缘端链路带宽均为信道容量的十分之一

        self.comm_energy = 0.0  # 通信能耗，单位焦耳
        self.send_power_watt = send_power_watt
        self.noise_power_watt = noise_power_watt
        self.bandwidth_hz = bandwidth_hz
        self.channel_gain = channel_gain

    @property
    def total_comm_energy(self):
        energy = 0.0
        for link_type in ["edge_cloud", "edge_end", "cloud_end"]:
            for unit in self.stats[link_type]:
                energy += unit["transfer_time"] * self.send_power_watt
        return energy


class PreciseCUHLM(CUHLM):
    """
    CUHLM的复杂建模版本，基于香农信道容量计算实际通信参数

    参数：
    - bandwidth_hz: 信道带宽，单位Hz
    - channel_gain: 信道增益
    - send_power_watt: 发送功率，单位瓦特
    - noise_power_watt: 噪声功率，单位瓦特
    - uncertainty_threshold: 不确定度阈值，默认0.8
    - vocab_size: 词汇表大小，默认32000
    """

    def __init__(
        self,
        bandwidth_hz: int | float,
        channel_gain: float,
        send_power_watt: float,
        noise_power_watt: float,
        uncertainty_threshold: float = 0.8,
        vocab_size: int = 32000,
        ntt_ms_edge_cloud: float = 200,
        ntt_ms_edge_end: float = 20,
    ):
        # 计算信噪比
        SNR = channel_gain * send_power_watt / noise_power_watt
        # 根据香农公式计算信道容量（bits/second）
        channel_capacity_bps = bandwidth_hz * math.log2(1 + SNR)

        # 初始化CUHLM，使用计算得到的信道容量
        # edge-cloud使用完整信道容量，其他链路假设为容量的十分之一

        print(
            f"信道容量: {channel_capacity_bps/1e6:.2f} Mbps, 以 {channel_capacity_bps / 10} bps, {channel_capacity_bps} bps, {channel_capacity_bps / 10} bps 初始化 "
        )

        super().__init__(
            bandwidth_edge_cloud=channel_capacity_bps,
            bandwidth_edge_end=channel_capacity_bps / 10,
            bandwidth_cloud_end=channel_capacity_bps / 10,
            uncertainty_threshold=uncertainty_threshold,
            vocab_size=vocab_size,
            dimension="bps",
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
        )

        # 存储通信物理参数
        self.bandwidth_hz = bandwidth_hz
        self.channel_gain = channel_gain
        self.send_power_watt = send_power_watt
        self.noise_power_watt = noise_power_watt
        self.SNR = SNR
        self.channel_capacity_bps = channel_capacity_bps

        # 通信能耗统计，单位焦耳
        self.comm_energy = 0.0

    @property
    def total_comm_energy(self) -> float:
        """计算总通信能耗（焦耳）"""
        energy = 0.0
        for link_type in ["edge_cloud", "edge_end", "cloud_end"]:
            for unit in self.stats[link_type]:
                energy += unit["transfer_time"] * self.send_power_watt
        return energy

class StochasticCommunication(CommunicationSimulator):
    """
    基于随机信道模型的通信模拟器
    """

    def __init__(
        self,
        **kwargs):
        super().__init__(**kwargs)
        raise NotImplementedError("StochasticCommunication尚未实现")