"""CEE 三层方法的 prefix 容器（H5c 第一步）。

两种语义：
- buffer 模式（cee_dsd）：预分配固定容量，原地写 token，避免 O(n²) cat；
- cat 模式（cee_dssd）：每次追加用 torch.cat，保持原实现的存储行为。

对外只暴露 ``tensor`` / ``assign(x, length)`` / ``append(t)``；行为与原先
手工维护的 prefix_buffer / torch.cat 逐语句一致。
"""

from __future__ import annotations

import torch


class CEETriPrefix:
    def __init__(self, prefix: torch.Tensor, capacity: int | None = None):
        self._capacity = capacity
        self._buffer = None
        if capacity is None:
            self._tensor = prefix
            self._length = prefix.shape[1]
        else:
            self._buffer = torch.empty(
                1, capacity, dtype=prefix.dtype, device=prefix.device
            )
            self._length = prefix.shape[1]
            self._buffer[:, : self._length] = prefix
            self._tensor = self._buffer[:, : self._length]

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    def assign(self, x: torch.Tensor, length: int) -> torch.Tensor:
        if self._buffer is None:
            self._tensor = x[:, :length]
            self._length = length
        else:
            self._buffer[:, :length] = x[:, :length]
            self._length = length
            self._tensor = self._buffer[:, : self._length]
        return self._tensor

    def append(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._buffer is None:
            self._tensor = torch.cat((self._tensor, tokens), dim=1)
            self._length = self._tensor.shape[1]
        else:
            n = self._length
            self._buffer[:, n : n + tokens.shape[1]] = tokens
            self._length = n + tokens.shape[1]
            self._tensor = self._buffer[:, : self._length]
        return self._tensor
