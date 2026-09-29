"""把 decode 前向捕获成 CUDA Graph，绕开 per-op 启动开销。

为什么值得做
------------
投机解码每轮的前向由几十上百个 kernel 组成，每个 kernel 约 70µs 的启动延迟
在 bs=1 时成为主导（实测 1.1B 只跑到峰值的约 16%，68M 约 2%）。
实测收益（scripts/bench_cuda_graph.py）：68M 12.85×、1.1B 4.13×（单步）。

2026-09 扩展：**定长 padding 验证图**（`(1, K)` 多 token 前向）
----------------------------------------------------------
真实管线（adaptive_tridecoding）里每个模型每轮的主体**不是** γ 次单步，而是
「上轮接受 token 的 resync + 本轮草稿 token」合成的**一个多 token 前向**：
draft 验证 little（k ≈ 接受+γ2）、target 验证 draft（k ≈ 接受+γ1）。
这些前向 `new_len > 1`，单步图接不住，此前走 eager 回退（还踩在 StaticCache
上，慢 1.84×）。

把多 token 前向**拆成单步**是负优化：每步都整读一遍权重（13B bf16 单步地板
34ms ⇒ 21 步 = 714ms，比整段 eager 的 101ms 还慢 7×）。正确做法是 vLLM 式
**定长 padding**：为每个捕获尺寸 K 预分配 (1,K) 输入缓冲，回放前把真实 k 个
token 拷进去、尾部 padding，回放后取前 k 行 logits。

padding 为什么是安全的（因果链，每条都依赖现成机制）：
1. **causal 注意心**：query i 只看 key ≤ pos+i ⇒ 前 k 行 logits 与真实 k-token
   前向一致（bf16 数值差异除外，与单步图同级）；
2. **pad 行的输出是垃圾但被切掉**：只拷贝前 k 行进 prob/logits 历史；
3. **pad 行写进的 KV 槽位随回滚作废**：rollback 本来就是"只移 nnz 指针，后续
   写入覆盖"（见下方 rollback），只是从 1 个槽位变成 K 个；
4. **垃圾 KV 不会产生 NaN 传染**：StaticCache 初始化为 0，pad 行读到的是
   有限值，且 pad 行结果根本不被读。

跨样本复用
----------
KVCacheModel 每样本重建 ⇒ 图每样本重捕获（13B 实测 ~534ms/样本，把收益吃光）。
`StaticLayer.reset()` 是 `zero_()` **原地**清零（transformers 4.57 验证过），
张量地址不变 ⇒ 捕获的图在新样本上仍然有效。`prefill()` 检测到图已存在时只做
eager prefill 进同一个 StaticCache，不重捕获。

2026-09-24 再扩展：**前后处理进图**（消除回放前后的逐 op 派发）
----------------------------------------------------------
图回放收走模型内部 kernel 后，小模型每次前向仍有 ~15-18 个 Python/torch op：
回放前的 `arange` 现造 + `mask.zero_` + 切片置 1，回放后的 logits 历史写入、
熵（5-6 个 op）、`norm_logits`（temp 0 = argmax+scatter）、prob 历史写入。
68M 实测 0.68ms/前向里 GPU 真算只有 ~0.15ms，其余全是这层 CPU 派发。

解法（vLLM 同思路）：把**定长**的前后处理也编进同一张图：
* 回放前 CPU 只剩 2 个 op：拷 k 个真实 token、填一个 `start_pos` 标量；
  `cache_position = start_pos + range_buf[:K]` 与 `mask = range_buf < start_pos+K`
  在图内由标量算出；
* 回放后熵（每行）、norm_logits(temp 0 的 argmax one-hot）、logits/prob 历史写入
  （`index_copy_` 按 `start_pos + range_buf` 的行索引）全部在图内 —— pad 行
  落在 `current_seq_len` 之外，读端按逻辑长度切片永不可见（与 KV 槽同一套
  回滚语义）。

开启条件：`attach_history_buffers(..., capture_post=True)`（temp==0 时
KVCacheModel 自动开启）；temp>0 的 norm 走 top-k/top-p 分支，保守起见仍留在
图外（回退到旧行为，从 `verify_logits` 缓冲做 eager 后处理）。

关键约束（每一条都在 scripts/test_graph_decode.py / test_graph_verify.py 验证）
--------------------------------------------------------------------------
1. **图冻结的是地址**，不是数值 ⇒ 所有输入必须是预先分配好的固定缓冲区，
   回放前用 `copy_` / `fill_` / `arange(out=)` 改内容；
2. **attention_mask 必须定长** ⇒ 用 max_len 长的缓冲区，图内由标量重算；
3. **prefill 是变长的，不能进图** ⇒ prefill 走 eager，只捕获之后的定长前向；
4. **必须一并产出 hidden state** ⇒ acc_head（接受率预测头）需要
   `hidden_states[-1][:, -1]`（见 src/adapter.py 的 predict）。
   是否产出由**模型配置**决定（model_loading.py 用 load_kwargs 打开），本模块
   不显式传参，以保证与 eager 路径行为完全一致；模型返回了就用 `copy_` 写进
   固定缓冲区；
5. **回滚 = 移动 Python 侧指针** ⇒ 无需 crop，后续写入自然覆盖旧槽位；
6. **历史缓冲地址在捕获时冻结** ⇒ `_ensure_buffer_size` 的扩容分支在图存在时
   必须不可达（KVCacheModel 的桶长公式保证 end_pos < 缓冲长度；违反即断言报错，
   宁可炸也不写野地址）。

等价性验证结论（temp 0，argmax 确定性）：
    68M   逐 token 100% 一致，|Δlogits| 相对幅度 0.000%，回滚后继续一致
    1.1B  逐 token 100% 一致，|Δlogits| 相对幅度 0.938%（bf16 舍入级），回滚后继续一致
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import StaticCache


def graph_mode_cache_kwargs(args, cap: int) -> dict:
    """CUDA Graph 模式下 KVCacheModel 的公共 kwargs（**单一接线点**）。

    静默漏接的教训：构造 KVCacheModel 时若不透传 use_cuda_graph，命令行开了
    --use_cuda_graph 也毫无效果（uncertainty_decoding / cee_cuhlm / tridecoding
    都曾漏接，靠 model_gpu.py 的 "[cuda-graph] 已启用图回放" 金丝雀日志才发现）。
    cap 为验证前向 k 的上界（调用方按 γ 推导），档位阶梯与 --graph_verify_sizes
    语义一致。graph_len_budget 是**生成预算**（KVCacheModel 按 prompt+budget 定桶，
    见 model_gpu.py），故这里取 max_tokens + 256 余量。
    """
    if not bool(getattr(args, "use_cuda_graph", False)):
        return {}
    raw_sizes = getattr(args, "graph_verify_sizes", None) or ""
    if isinstance(raw_sizes, str) and raw_sizes.strip():
        ladder = sorted(
            {int(x) for x in str(raw_sizes).replace(" ", "").split(",") if x}
        )
    else:
        ladder = [4, 8, 16, 24, 32, 40, 48, 64]
    sizes = [s for s in ladder if 2 <= s <= cap]
    if not sizes or sizes[-1] < cap:
        sizes.append(min(((cap + 7) // 8) * 8, 128))
    return {
        "use_cuda_graph": True,
        "verify_graph_sizes": sizes,
        "graph_len_budget": int(getattr(args, "max_tokens", 128)) + 256,
    }


def acquire_graph_caches(holder, attr: str, graph_kw: dict, builders: dict) -> dict:
    """图模式下跨样本复用缓存；eager 模式每次重建（历史行为逐位不变）。

    `builders` 是 {名字: 零参可调用}，只在未命中复用时调用。命中时只对已有缓存
    做原地 `reset_for_new_sample()`（StaticCache `zero_()`，地址不变 ⇒ 图不重
    捕获），避免每样本 ~534ms 的重捕获把收益吃光。`attr` 各方法独立，避免不同
    top-k 配置的缓存互相串用。
    """
    reused = getattr(holder, attr, None) if graph_kw else None
    if reused is not None:
        for cache in reused.values():
            cache.reset_for_new_sample()
        return reused
    caches = {name: build() for name, build in builders.items()}
    if graph_kw:
        setattr(holder, attr, caches)
    return caches


class GraphDecodeRunner:
    """定长 decode 前向的 CUDA Graph 执行器。

    生命周期：构造 → `attach_history_buffers()`（可选，开启前后处理进图）→
    `prefill(prompt)`（首次内含图捕获）→ 反复 `step(token)`（单步图）/
    `verify(tokens)`（(1,K) padding 验证图），中途可 `rollback(pos)`；新样本
    直接再调 `prefill()`（内部 reset 复用图）。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_len: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        capture_hidden: bool = True,
        verify_sizes: Sequence[int] = (),
    ) -> None:
        self.model = model
        self.max_len = int(max_len)
        self.device = torch.device(device)
        self.dtype = dtype
        self.capture_hidden = capture_hidden
        # 捕获尺寸阶梯（去重、排序、过滤非法值）。回放时取 ≥k 的最小档。
        self.verify_sizes: list[int] = sorted(
            {int(s) for s in verify_sizes if int(s) >= 2}
        )

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

        # 前后处理进图：唯一需要 host 写的标量 + 预造的行号表
        self.start_pos = torch.zeros((), dtype=torch.long, device=self.device)
        self.range_buf = torch.arange(self.max_len, dtype=torch.long, device=self.device)
        self.capture_post = False
        self.vocab_limit: int | None = None
        self.hist_logits_buf: torch.Tensor | None = None
        self.hist_prob_buf: torch.Tensor | None = None
        self.step_entropy: torch.Tensor | None = None
        self.verify_entropy: dict[int, torch.Tensor] = {}
        self._last_ent_rows: torch.Tensor | None = None

        # (1,K) 验证图：ids / pos / logits / hidden 四个固定缓冲 per K
        self.verify_ids: dict[int, torch.Tensor] = {}
        self.verify_pos: dict[int, torch.Tensor] = {}
        self.verify_logits: dict[int, torch.Tensor] = {}
        self.verify_hidden: dict[int, torch.Tensor] = {}
        self.verify_graphs: dict[int, torch.cuda.CUDAGraph] = {}

        self.graph: torch.cuda.CUDAGraph | None = None
        self.prefill_logits: torch.Tensor | None = None
        self.nnz = 0  # 逻辑长度（回滚只改它；不等同于 StaticCache 内部计数）
        self.capture_count = 0  # 重捕获计数（跨样本复用后应恒为 1）

    # ────────────────────────────────────────── 前后处理进图
    def attach_history_buffers(
        self,
        logits_buf: torch.Tensor,
        prob_buf: torch.Tensor,
        vocab_size: int | None = None,
        capture_post: bool = True,
    ) -> None:
        """把 KVCacheModel 的 logits/prob 历史缓冲接进图（捕获前调用一次）。

        图内通过 `index_copy_(1, start_pos + range_buf[:K], ...)` 写入 —— 行索引
        由标量在图内算出，地址冻结在此刻的缓冲上。因此**捕获之后缓冲不得再
        扩容/换址**（KVCacheModel 的图模式桶长公式保证；违反由
        `_ensure_buffer_size` 的断言兜底）。

        capture_post=False 时退回旧行为（回放前 Python 填 pos/mask、回放后
        eager 后处理），供 temp>0 等不满足捕获条件的配置使用。
        """
        if self.graph is not None:
            same = (
                self.hist_logits_buf is logits_buf
                and self.hist_prob_buf is prob_buf
                and self.capture_post == capture_post
                and self.vocab_limit == vocab_size
            )
            if not same:
                raise RuntimeError(
                    "attach_history_buffers 与已捕获图不一致；图模式下历史缓冲"
                    "不支持捕获后更换（需重建 runner）"
                )
            return
        self.hist_logits_buf = logits_buf
        self.hist_prob_buf = prob_buf
        self.vocab_limit = vocab_size
        self.capture_post = capture_post

    def _pre_forward_setup(self, size: int) -> None:
        """图内：由 start_pos 标量计算 cache_position 与 attention_mask。

        原来回放前要 4~5 个 Python op（arange 现造+copy、mask 清零+切片置 1）；
        现在全部编进图，host 只剩「拷 token + 填一个标量」。
        """
        pos_ids = self.start_pos + self.range_buf[:size]
        if size == 1:
            self.step_pos.copy_(pos_ids)
        else:
            self.verify_pos[size].copy_(pos_ids)
        self.mask.copy_((self.range_buf < (self.start_pos + size)).long())

    def _ent_out(self, size: int) -> torch.Tensor:
        buf = self.step_entropy if size == 1 else self.verify_entropy.get(size)
        if buf is None:
            buf = torch.zeros(
                (1, size), dtype=torch.float32, device=self.device
            )
            if size == 1:
                self.step_entropy = buf
            else:
                self.verify_entropy[size] = buf
        return buf

    def _post_forward(self, size: int, logits: torch.Tensor) -> None:
        """图内：每行熵 + norm_logits(temp 0) + 历史缓冲写入（全部定长）。"""
        lg = logits
        if self.vocab_limit is not None and lg.shape[-1] != self.vocab_limit:
            lg = lg[..., : self.vocab_limit]
        idx = self.start_pos + self.range_buf[:size]  # (size,)
        # pad 行写到 current_seq_len 之外 —— 读端按逻辑长度切片，永不可见
        # （与 KV 槽位同一套回滚语义）。
        assert self.hist_logits_buf is not None and self.hist_prob_buf is not None
        self.hist_logits_buf.index_copy_(1, idx, lg)
        # 每行熵（0-dim 均值由调用方按真实 k 行计算，与 eager 语义一致）
        f = lg.float()
        lp = torch.log_softmax(f, dim=-1)
        self._ent_out(size).copy_((lp.exp() * lp).sum(dim=-1).neg())
        # norm_logits temp=0：argmax one-hot（与 src/utils.norm_logits 同式）
        am = lg.argmax(dim=-1, keepdim=True)
        onehot = torch.zeros_like(f).scatter_(-1, am, 1.0)
        self.hist_prob_buf.index_copy_(1, idx, onehot.to(self.hist_prob_buf.dtype))

    @property
    def last_entropy_rows(self) -> torch.Tensor | None:
        """最近一次 step/verify 的每行熵 (1, K)；未开启 post 捕获时为 None。"""
        return self._last_ent_rows

    # ────────────────────────────────────────── 内部
    def _forward_step(self):
        """一次单步前向；捕获与回放共用，保证两者走完全相同的算子。"""
        if self.capture_post:
            self._pre_forward_setup(1)
        outputs = self.model(
            self.step_ids,
            past_key_values=self.cache,
            cache_position=self.step_pos,
            attention_mask=self.mask,
            use_cache=True,
        )
        logits = outputs.logits
        if self.capture_hidden:
            hs = getattr(outputs, "hidden_states", None)
            if hs is not None and self.hidden_buf is not None:
                # 原地写入 ⇒ 图内合法，且地址不变
                self.hidden_buf.copy_(hs[-1][:, -1:, :].to(self.hidden_buf.dtype))
        if self.capture_post:
            self._post_forward(1, logits)
        return logits

    def _forward_verify(self, size: int):
        """一次 (1,size) 定长前向；捕获与回放共用。"""
        if self.capture_post:
            self._pre_forward_setup(size)
        outputs = self.model(
            self.verify_ids[size],
            past_key_values=self.cache,
            cache_position=self.verify_pos[size],
            attention_mask=self.mask,
            use_cache=True,
        )
        logits = outputs.logits  # (1, size, V)
        if self.capture_hidden:
            hs = getattr(outputs, "hidden_states", None)
            if hs is not None and self.verify_hidden[size] is not None:
                self.verify_hidden[size].copy_(
                    hs[-1].to(self.verify_hidden[size].dtype)
                )
        if self.capture_post:
            self._post_forward(size, logits)
        return logits

    def _set_step_inputs(self, token: torch.Tensor | None, pos: int) -> None:
        if token is not None:
            self.step_ids.copy_(token)
        self.start_pos.fill_(pos)
        if not self.capture_post:
            self.step_pos.fill_(pos)
            self.mask.zero_()
            self.mask[:, : pos + 1] = 1

    def _set_verify_inputs(self, size: int, tokens: torch.Tensor | None, pos: int) -> None:
        """回放前填充 (1,size) 输入缓冲：真实 k 个 token + 尾部沿用旧 token。

        pad 行不需要清零：只要 id 合法（曾经被写入过合法 token 或初始零），
        其输出被丢弃、其 KV 槽随回滚作废 —— 与显式 zero_ 等价但省一个 op。
        """
        if tokens is not None:
            k = int(tokens.shape[1])
            self.verify_ids[size][:, :k].copy_(tokens)
        self.start_pos.fill_(pos)
        if not self.capture_post:
            self.verify_pos[size].copy_(
                torch.arange(pos, pos + size, device=self.device)
            )
            self.mask.zero_()
            self.mask[:, : pos + size] = 1

    def _pick_verify_size(self, k: int) -> int | None:
        for size in self.verify_sizes:
            if size >= k:
                return size
        return None

    # ────────────────────────────────────────── 对外
    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """变长 prefill（eager，不进图），随后**首次**捕获各定长图。

        图已存在（跨样本复用）时只做 `reset + eager prefill`，不重捕获 ——
        StaticLayer.reset() 是原地 zero_，张量地址不变，图仍有效。

        返回最后一个位置的**原始 logits**，形状 (1, vocab)。
        """
        seq = int(input_ids.shape[1])
        if seq >= self.max_len:
            raise RuntimeError(
                f"prompt 长度 {seq} 超过图模式 max_len {self.max_len}；"
                "调大 KVCacheModel 的 max_length 或关闭 CUDA Graph"
            )
        reuse = self.graph is not None
        if reuse:
            self.begin_new_sequence()

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
        if self.logits_buf is None:
            self.logits_buf = torch.empty(
                (1, logits.shape[-1]), dtype=logits.dtype, device=logits.device
            )
        hs = getattr(outputs, "hidden_states", None)
        if self.capture_hidden and hs is not None:
            if self.hidden_buf is None:
                self.hidden_buf = torch.empty(
                    (1, 1, hs[-1].shape[-1]), dtype=hs[-1].dtype, device=hs[-1].device
                )
            self.hidden_buf.copy_(hs[-1][:, -1:, :])

        self.nnz = seq
        if reuse:
            return logits[:, -1].clone()

        # ── 首次：捕获全部图（单步 + 各验证尺寸）─────────────────────────
        vocab_w = int(logits.shape[-1])
        for size in self.verify_sizes:
            self.verify_ids[size] = torch.zeros(
                (1, size), dtype=torch.long, device=self.device
            )
            self.verify_pos[size] = torch.zeros(
                (size,), dtype=torch.long, device=self.device
            )
            self.verify_logits[size] = torch.zeros(
                (1, size, vocab_w), dtype=logits.dtype, device=logits.device
            )
            if self.capture_hidden and hs is not None:
                self.verify_hidden[size] = torch.zeros(
                    (1, size, hs[-1].shape[-1]),
                    dtype=hs[-1].dtype,
                    device=hs[-1].device,
                )

        torch.cuda.synchronize()
        # 热身：把 lazy init / 分配挡在图外。热身写入的 KV 槽位在 seq 之后，
        # 逻辑上已死（回滚语义），首轮真实回放会覆盖它们。
        with torch.inference_mode():
            self._set_step_inputs(None, seq)
            for _ in range(3):
                self._forward_step()
            for size in self.verify_sizes:
                if seq + size > self.max_len:
                    continue  # 该档位放不下（max_len 太紧），跳过并靠回退
                self._set_verify_inputs(size, None, seq)
                for _ in range(2):
                    self._forward_verify(size)
        torch.cuda.synchronize()

        # 捕获单步图
        self.step_ids.fill_(0)
        self._set_step_inputs(None, seq)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            captured = self._forward_step()
            self.logits_buf.copy_(captured[:, -1])  # 图内写固定缓冲区
        self.capture_count += 1

        # 捕获各档位验证图
        for size in self.verify_sizes:
            if seq + size > self.max_len:
                continue
            self.verify_ids[size].zero_()
            self._set_verify_inputs(size, None, seq)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_v = self._forward_verify(size)
                self.verify_logits[size].copy_(captured_v)
            self.verify_graphs[size] = graph
            self.capture_count += 1
        torch.cuda.synchronize()

        return logits[:, -1].clone()

    @torch.inference_mode()
    def begin_new_sequence(self) -> None:
        """跨样本复用：清空 KV/mask/指针，**保留**已捕获的图。

        StaticLayer.reset() 是原地 zero_（地址不变）⇒ 图内冻结的 cache 张量
        地址仍然有效。相比每样本重建 + 重捕获（13B ~534ms），这里只付一次
        清零（~1ms 量级）。
        """
        self.cache.reset()
        self.mask.zero_()
        self.prefill_logits = None
        self.nnz = 0
        self._last_ent_rows = None

    @torch.inference_mode()
    def step(self, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """写入一个 token 并前进一步（单步图）。

        返回 (原始 logits (1,vocab), 最后一层 hidden (1,1,H) 或 None)。
        前后处理进图时：熵行在 step_entropy，历史缓冲已由图内写入。
        """
        if self.graph is None or self.logits_buf is None:
            raise RuntimeError("step() 前必须先调用 prefill()")
        pos = self.nnz
        if pos >= self.max_len:
            raise RuntimeError(f"超出图模式 max_len {self.max_len}")
        self._set_step_inputs(token, pos)
        self.graph.replay()
        self.nnz = pos + 1
        self._last_ent_rows = self.step_entropy if self.capture_post else None
        return self.logits_buf, self.hidden_buf

    @torch.inference_mode()
    def verify(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """一个多 token 前向（(1,K) padding 验证图）。

        tokens: (1, k)，1 ≤ k ≤ 最大档位。取 ≥k 的最小捕获档位 K，尾部 padding，
        回放后返回**前 k 行**：
            (原始 logits (1,k,vocab), 最后一层 hidden 的最后一个真实行 (1,1,H) 或 None)
        无可用档位（k 超过最大档位 / 超出 max_len）时返回 None，调用方回退 eager。
        nnz 只前进 k —— pad 行写入的 K-k 个 KV 槽位随回滚语义作废。
        前后处理进图时：每行熵在 verify_entropy[K]（前 k 行有效），logits/prob
        历史已由图内 index_copy_ 写好（pad 行落在逻辑长度之外）。
        """
        k = int(tokens.shape[1])
        if k <= 0:
            return None
        size = self._pick_verify_size(k)
        if size is None or size not in self.verify_graphs:
            return None
        pos = self.nnz
        if pos + size > self.max_len or pos + k > self.max_len:
            return None
        tokens_long = tokens if tokens.dtype == torch.long else tokens.long()
        self._set_verify_inputs(size, tokens_long.to(self.device), pos)
        self.verify_graphs[size].replay()
        self.nnz = pos + k
        self._last_ent_rows = (
            self.verify_entropy.get(size) if self.capture_post else None
        )
        logits = self.verify_logits[size][:, :k]
        hidden = None
        if self.verify_hidden.get(size) is not None:
            hidden = self.verify_hidden[size][:, k - 1 : k]
        return logits, hidden

    def rollback(self, end_pos: int) -> None:
        """回滚 = 只改指针；旧槽位会在后续写入时被覆盖。"""
        self.nnz = max(0, min(int(end_pos), self.nnz))
