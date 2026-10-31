"""CEE 三层方法的 prefix 容器（H5c 第一步）。

两种语义：
- buffer 模式（cee_dsd）：预分配固定容量，原地写 token，避免 O(n²) cat；
- cat 模式（cee_dssd）：每次追加用 torch.cat，保持原实现的存储行为。

对外只暴露 ``tensor`` / ``assign(tokens, num_tokens)`` / ``append(tokens)``；
行为与原先手工维护的 prefix_buffer / torch.cat 逐语句一致。

别名契约：buffer 模式返回的是 ``_buffer`` 的切片视图（后续写入可见），
cat 模式返回的是当次拼接后的独立张量。调用方只应逐次消费返回值，不要
长期持有旧句柄；生产代码两种模式各自沿用原语义。

不变式：buffer 模式只支持 batch=1（与原 prefix_buffer 一致）；调用方在
``append`` 前应先 ``assign`` 或使用构造时的 ``tensor``。
"""

from __future__ import annotations

import torch


class CEETriPrefix:
    def __init__(self, prefix: torch.Tensor, capacity: int | None = None):
        self._buffered = capacity is not None
        self._capacity = capacity
        self._buffer = None
        if not self._buffered:
            self._tensor = prefix
        else:
            if prefix.shape[0] != 1:
                raise ValueError(
                    "CEETriPrefix buffer mode only supports batch=1; "
                    f"got shape {tuple(prefix.shape)}"
                )
            self._buffer = torch.empty(
                1, capacity, dtype=prefix.dtype, device=prefix.device
            )
            self._buffer[:, : prefix.shape[1]] = prefix
            self._tensor = self._buffer[:, : prefix.shape[1]]
        self._length = prefix.shape[1]

    @property
    def tensor(self) -> torch.Tensor:
        """当前 prefix；buffer 模式下是 _buffer 的视图（见模块 docstring）。"""
        return self._tensor

    def _check_capacity(self, required: int) -> None:
        if self._capacity is not None and required > self._capacity:
            raise ValueError(
                "CEETriPrefix buffer overflow: "
                f"mode=buffer capacity={self._capacity} required={required}"
            )

    def assign(self, tokens: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if num_tokens > tokens.shape[1]:
            raise ValueError(
                "CEETriPrefix.assign: num_tokens larger than provided tokens; "
                f"num_tokens={num_tokens} tokens.shape={tuple(tokens.shape)}"
            )
        self._check_capacity(num_tokens)
        if not self._buffered:
            self._tensor = tokens[:, :num_tokens]
        else:
            self._buffer[:, :num_tokens] = tokens[:, :num_tokens]
            self._tensor = self._buffer[:, :num_tokens]
        self._length = num_tokens
        return self._tensor

    def append(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self._buffered:
            self._tensor = torch.cat((self._tensor, tokens), dim=1)
            self._length = self._tensor.shape[1]
        else:
            n = self._length
            self._check_capacity(n + tokens.shape[1])
            self._buffer[:, n : n + tokens.shape[1]] = tokens
            self._length = n + tokens.shape[1]
            self._tensor = self._buffer[:, : self._length]
        return self._tensor
