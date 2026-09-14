"""把 (1,1) 的单步 decode 捕获成 CUDA Graph，绕开 per-op 启动开销。

为什么值得做
------------
`KVCacheModel._generate_with_kvcache` 的草稿循环是 γ 次**单 token 前向**，
每次前向本身只有十几毫秒级的计算量，但由几十上百个 kernel 组成，每个 kernel
约 70µs 的启动延迟成为主导（实测 1.1B 只跑到峰值的约 16%，68M 约 2%）。
实测收益（scripts/bench_cuda_graph.py）：68M 12.85×、1.1B 4.13×。

关键约束（每一条都在 scripts/test_graph_decode.py 里验证过）
----------------------------------------------------------
1. **图冻结的是地址**，不是数值 ⇒ 所有输入必须是预先分配好的固定缓冲区，
   回放前用 `copy_` / `fill_` 改内容；
2. **attention_mask 必须定长** ⇒ 用 max_len 长的缓冲区，每步把前 n 个槽位
   置 1（不能用 `torch.ones` 现造，那会改变地址）；
3. **prefill 是变长的，不能进图** ⇒ prefill 走 eager，只捕获之后的单步；
4. **必须一并产出 hidden state** ⇒ acc_head（接受率预测头）需要
   `hidden_states[-1]`（见 src/baselines.py 的 adapter.predict 调用）。
   是否产出由**模型配置**决定（model_loading.py 用 load_kwargs 打开），
   本模块不显式传参，以保证与 eager 路径行为完全一致；模型返回了就
   用 `copy_` 写进固定缓冲区。
5. **回滚 = 移动 Python 侧指针** ⇒ 无需 crop，后续写入自然覆盖旧槽位。

等价性验证结论（temp 0，argmax 确定性）：
    68M   逐 token 100% 一致，|Δlogits| 相对幅度 0.000%，回滚后继续一致
    1.1B  逐 token 100% 一致，|Δlogits| 相对幅度 0.938%（bf16 舍入级），回滚后继续一致
"""
from __future__ import annotations

import torch
from transformers import StaticCache


class GraphDecodeRunner:
    """单步 decode 的 CUDA Graph 执行器。

    生命周期：构造 → `prefill(prompt)`（内含图捕获）→ 反复 `step(token)`，
    中途可 `rollback(pos)`。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_len: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        capture_hidden: bool = True,
    ) -> None:
        self.model = model
        self.max_len = int(max_len)
        self.device = torch.device(device)
        self.dtype = dtype
        self.capture_hidden = capture_hidden

        self.cache = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=self.max_len,
            device=self.device,
            dtype=dtype,
        )

        hidden = int(getattr(model.config, "hidden_size", 0)) or None
        self.hidden_size = hidden

        # ── 固定地址缓冲区（图冻结的就是这些地址）────────────────────────
        self.step_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.step_pos = torch.zeros((1,), dtype=torch.long, device=self.device)
        self.mask = torch.zeros((1, self.max_len), dtype=torch.long, device=self.device)
        self.logits_buf: torch.Tensor | None = None
        self.hidden_buf: torch.Tensor | None = None

        self.graph: torch.cuda.CUDAGraph | None = None
        self.prefill_logits: torch.Tensor | None = None
        self.nnz = 0  # 逻辑长度（回滚只改它）

    # ────────────────────────────────────────── 内部
    def _forward_step(self):
        """一次单步前向；捕获与回放共用，保证两者走完全相同的算子。"""
        outputs = self.model(
            self.step_ids,
            past_key_values=self.cache,
            cache_position=self.step_pos,
            attention_mask=self.mask,
            use_cache=True,
        )
        logits = outputs.logits[:, -1]
        if self.capture_hidden:
            hs = getattr(outputs, "hidden_states", None)
            if hs is not None and self.hidden_buf is not None:
                # 原地写入 ⇒ 图内合法，且地址不变
                self.hidden_buf.copy_(hs[-1][:, -1:, :].to(self.hidden_buf.dtype))
        return logits

    def _set_step_inputs(self, token: torch.Tensor | None, pos: int) -> None:
        if token is not None:
            self.step_ids.copy_(token)
        self.step_pos.fill_(pos)
        # mask 的"有效长度" = 已写入的 token 数
        self.mask.zero_()
        self.mask[:, : pos + 1] = 1

    # ────────────────────────────────────────── 对外
    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """变长 prefill（eager，不进图），随后捕获单步 decode 图。

        返回最后一个位置的**原始 logits**，形状 (1, vocab)。
        """
        seq = int(input_ids.shape[1])
        if seq >= self.max_len:
            raise RuntimeError(
                f"prompt 长度 {seq} 超过图模式 max_len {self.max_len}；"
                "调大 KVCacheModel 的 max_length 或关闭 CUDA Graph"
            )
        self.mask.zero_()
        self.mask[:, :seq] = 1
        outputs = self.model(
            input_ids,
            past_key_values=self.cache,
            cache_position=torch.arange(seq, dtype=torch.long, device=self.device),
            attention_mask=self.mask,
            use_cache=True,
        )
        logits = outputs.logits
        if logits is None:
            raise RuntimeError("Model returned logits=None in graph prefill")

        self.prefill_logits = logits  # (1, seq, V) 完整 prefill logits，供调用方写历史缓冲
        self.logits_buf = torch.empty(
            (1, logits.shape[-1]), dtype=logits.dtype, device=logits.device
        )
        hs = getattr(outputs, "hidden_states", None)
        if self.capture_hidden and hs is not None:
            self.hidden_buf = torch.empty(
                (1, 1, hs[-1].shape[-1]), dtype=hs[-1].dtype, device=hs[-1].device
            )
            self.hidden_buf.copy_(hs[-1][:, -1:, :])

        self.nnz = seq

        # 捕获单步图
        self.step_ids.fill_(0)
        self._set_step_inputs(None, seq)
        with torch.inference_mode():
            for _ in range(3):  # 热身：把 lazy init / 分配挡在图外
                self._forward_step()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            captured = self._forward_step()
            self.logits_buf.copy_(captured)  # 图内写固定缓冲区
        torch.cuda.synchronize()

        return logits[:, -1].clone()

    @torch.inference_mode()
    def step(self, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """写入一个 token 并前进一步。

        返回 (原始 logits (1,vocab), 最后一层 hidden (1,1,H) 或 None)。
        """
        if self.graph is None or self.logits_buf is None:
            raise RuntimeError("step() 前必须先调用 prefill()")
        pos = self.nnz
        if pos >= self.max_len:
            raise RuntimeError(f"超出图模式 max_len {self.max_len}")
        self._set_step_inputs(token, pos)
        self.graph.replay()
        self.nnz = pos + 1
        return self.logits_buf, self.hidden_buf

    def rollback(self, end_pos: int) -> None:
        """回滚 = 只改指针；旧槽位会在后续写入时被覆盖。"""
        self.nnz = max(0, min(int(end_pos), self.nnz))
