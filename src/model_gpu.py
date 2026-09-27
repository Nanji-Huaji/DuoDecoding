import torch
from transformers.cache_utils import DynamicCache

from .proposal_utils import (
    build_topk_proposal_history_step,
    concat_topk_proposal_history,
)
from .utils import (
    log_prob_tensor_if_invalid,
    norm_logits,
    numeric_debug_checks_enabled,
    rebuild_topk_uniform_probs,
    sample,
)

from collections.abc import Sequence
from typing import Any, Protocol, Iterator, TypeAlias, TypeGuard, cast, Optional


KVPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]
LegacyPastKeyValues: TypeAlias = tuple[KVPair, ...]


class EmbeddingLike(Protocol):
    weight: torch.Tensor


class CacheLike(Protocol):
    def crop(self, end_pos: int) -> None: ...
    def get_seq_length(self) -> int: ...
    def __iter__(self) -> Iterator[KVPair]: ...


PastKeyValues: TypeAlias = CacheLike | LegacyPastKeyValues | None


class ModelOutputLike(Protocol):
    logits: torch.Tensor | None
    past_key_values: PastKeyValues
    hidden_states: Sequence[torch.Tensor] | None


def _is_cache_like(value: PastKeyValues) -> TypeGuard[CacheLike]:
    return value is not None and hasattr(value, "get_seq_length")


def _is_legacy_past(value: PastKeyValues) -> TypeGuard[LegacyPastKeyValues]:
    return isinstance(value, tuple)


class CausalModel(Protocol):
    config: Any

    def __call__(self, *args: Any, **kwargs: Any) -> ModelOutputLike: ...
    def get_input_embeddings(self) -> torch.nn.Module: ...
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]: ...


class KVCacheModel:
    def __init__(
        self,
        model: CausalModel,
        temperature: float = 1,
        top_k: int = 0,
        top_p: float = 0,
        return_hidden_states: bool = False,
        max_length: int | None = None,
        use_cuda_graph: bool = False,
        verify_graph_sizes: Sequence[int] = (),
        graph_len_budget: int = 512,
    ) -> None:
        self._model: CausalModel = model
        # CUDA Graph 模式：把定长 decode 前向捕获成图回放，绕开 per-op 启动开销
        # （实测单步 68M 12.85×、1.1B 4.13×）。默认关闭以保持结果与历史实验
        # 逐位可复现；开启后仍与 eager 路径逐 token 一致
        # （见 scripts/test_graph_decode.py / test_graph_verify.py 的等价性验证）。
        #
        # verify_graph_sizes：多 token 前向（resync+草稿 的合成验证前向）用
        # (1,K) 定长 padding 图接管 —— 拆单步是负优化（每步整读一遍权重），
        # 这里按档位 padding，回放后取前 k 行（见 graph_decode.py 的论证）。
        self._use_cuda_graph = bool(use_cuda_graph)
        self._verify_graph_sizes = tuple(
            sorted({int(s) for s in verify_graph_sizes if int(s) >= 2})
        )
        self._graph_len_budget = int(graph_len_budget)
        self._graph_runner = None
        if self._use_cuda_graph:
            # 护栏：图只对 gamma>1 的草稿缓存有意义。如果在日志里看不到这一行，
            # 说明这条 eval_mode 的缓存构造路径没接线 —— 静默漏接会让"开图"
            # 看起来毫无效果（我们就在 adaptive_tridecoding 上踩过一次）。
            sizes = (
                f"，验证图档位 {list(self._verify_graph_sizes)}"
                if self._verify_graph_sizes
                else ""
            )
            print(
                f"[cuda-graph] 已启用图回放: {type(model).__name__} "
                f"vocab={getattr(model.config, 'vocab_size', '?')}{sizes}",
                flush=True,
            )
        self._past_key_values: PastKeyValues = None

        self._temperature: float = temperature
        self._top_k: int = top_k
        self._top_p: float = top_p
        self._has_explicit_max_length = max_length is not None
        self.max_length: int = max_length if max_length is not None else 16384

        self.hidden_states: Sequence[torch.Tensor] | None = None
        embeddings = cast(EmbeddingLike, model.get_input_embeddings())
        self.embedding_vocab_size = int(embeddings.weight.shape[0])

        if hasattr(model.config, "vocab_size"):
            self.vocab_size = int(model.config.vocab_size)
        elif hasattr(model.config, "text_config") and hasattr(
            model.config.text_config, "vocab_size"
        ):
            self.vocab_size = int(model.config.text_config.vocab_size)
        else:
            raise AttributeError("Vocab size not found in model config")

        # Pre-allocate buffers to eliminate O(N^2) memory allocations via torch.cat
        self._prob_buffer: torch.Tensor | None = None
        # 最近一次前向在**原始 logits（温度 1）**下的平均熵。
        #
        # 为什么需要它：RL 控制器把 entropy 当作状态特征，但原先是在 baselines.py
        # 里对 `_forward_with_kvcache` 的返回值再 softmax 一次——那个返回值已经是
        # `norm_logits` 归一化过的**概率**，再 softmax 就近似均匀分布，熵恒等于
        # ln(vocab)≈10.3735，归一化(min(entropy/10,1))后饱和成常数 1.0，特征完全
        # 失效（实测轨迹 300 步只有一个取值）。而且 --temp 0.0 时 norm_logits 直接
        # 返回 one-hot，熵恒为 0 —— 所以必须从**归一化之前的 logits** 算。
        #
        # 存 GPU 张量、property 里惰性 `.item()`：`.item()` 是一次 host 同步，
        # 放在每次前向里会把 CPU/GPU 流水线打断（图回放后这就是新的瓶颈）。
        # RL 每轮读一次 = 每轮 1 次同步，语义仍是"最后一次前向的熵"。
        self._last_entropy_t: torch.Tensor | None = None
        self._logits_buffer: torch.Tensor | None = None
        self._current_seq_len: int = 0

    def _new_dynamic_cache(self) -> DynamicCache:
        return DynamicCache(config=self._model.config)

    def _build_model_inputs(self, input_ids: torch.Tensor, *, use_cache: bool) -> dict:
        model_inputs: dict[str, object] = {
            "input_ids": input_ids,
            "use_cache": use_cache,
        }

        seq_len = input_ids.shape[1]
        device = input_ids.device
        past_seen_tokens = self.current_length if self._past_key_values is not None else 0
        attention_len = past_seen_tokens + seq_len if use_cache else seq_len
        attention_mask = torch.ones(
            (input_ids.shape[0], attention_len), dtype=torch.long, device=device
        )
        model_inputs["attention_mask"] = attention_mask

        return model_inputs

    def _prepare_generation_inputs(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: PastKeyValues,
    ) -> dict[str, object]:
        batch_size, seq_len = input_ids.shape
        cache_start = 0
        if past_key_values is not None and _is_cache_like(past_key_values):
            cache_start = past_key_values.get_seq_length()

        # 图模式下缓存是预分配的 StaticCache，它的内部计数**不会随我们的回滚后退**
        # （回滚只改 runner.nnz，旧槽位等后续写入覆盖）。如果这里仍以它为权威，
        # 回滚之后的 eager 回退（new_len > 1 的短后缀）就会把 KV 写到错位的偏移上，
        # 表现为 prob_history 越界（曾观测到 n1=358 而 len=341，差 17 = gamma2+1）。
        # 因此图模式下以 Python 侧的 _current_seq_len 为准。
        if self._graph_runner is not None:
            cache_start = self._current_seq_len

        attention_mask = torch.ones(
            (batch_size, cache_start + seq_len),
            dtype=torch.long,
            device=input_ids.device,
        )
        cache_position = torch.arange(
            cache_start,
            cache_start + seq_len,
            dtype=torch.long,
            device=input_ids.device,
        )

        if hasattr(self._model, "prepare_inputs_for_generation"):
            prepared_inputs = self._model.prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=True,
            )
            prepared_inputs["use_cache"] = True
            return cast(dict[str, object], prepared_inputs)

        model_inputs = self._build_model_inputs(input_ids, use_cache=True)
        model_inputs["past_key_values"] = past_key_values
        model_inputs["cache_position"] = cache_position
        return model_inputs

    @property
    def last_entropy(self) -> float | None:
        """最近一次前向在原始 logits（温度 1）下的平均熵，供 RL 控制器使用。

        惰性同步：这里才做 `.item()`（host 同步），前向路径只写 GPU 张量。
        """
        if self._last_entropy_t is None:
            return None
        return float(self._last_entropy_t.item())

    def _compute_entropy_tensor(self, sliced_logits: torch.Tensor) -> torch.Tensor:
        """原始 logits（温度 1）下的平均熵，保持为 GPU 张量（不同步）。"""
        _lg = sliced_logits.float()
        _lp = torch.log_softmax(_lg, dim=-1)
        return -(_lp.exp() * _lp).sum(dim=-1).mean()

    @property
    def _prob_history(self) -> torch.Tensor | None:
        if self._prob_buffer is None:
            return None
        return self._prob_buffer[:, : self._current_seq_len, :]

    @_prob_history.setter
    def _prob_history(self, value):
        pass

    @property
    def logits_history(self) -> torch.Tensor | None:
        if self._logits_buffer is None:
            return None
        return self._logits_buffer[:, : self._current_seq_len, :]

    @logits_history.setter
    def logits_history(self, value):
        pass

    @property
    def prob_history(self) -> torch.Tensor:
        if self._prob_history is None:
            raise ValueError("Probability history buffer is not initialized")
        return self._prob_history

    def _ensure_buffer_size(
        self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ):
        # Dynamically resize buffers to prevent OOM on large context while maintaining contiguous memory access
        if self._prob_buffer is None:
            if not self._has_explicit_max_length:
                self.max_length = max(2048, seq_len + 1024)
            elif seq_len > self.max_length:
                self.max_length = max(self.max_length * 2, seq_len + 1024)
            self._prob_buffer = torch.empty(
                (batch_size, self.max_length, self.vocab_size),
                device=device,
                dtype=dtype,
            )
            self._logits_buffer = torch.empty(
                (batch_size, self.max_length, self.vocab_size),
                device=device,
                dtype=dtype,
            )
            return

        if seq_len > self.max_length:
            old_len = self.max_length
            self.max_length = max(self.max_length * 2, seq_len + 1024)

            runner = getattr(self, "_graph_runner", None)
            if (
                runner is not None
                and runner.graph is not None
                and getattr(runner, "capture_post", False)
            ):
                # 图内 index_copy_ 冻结了旧缓冲地址；扩容换址 = 图回放写野地址。
                # 图模式桶长公式保证 end_pos < 缓冲长度，走到这里说明不变量被破坏。
                raise RuntimeError(
                    "前后处理进图模式下历史缓冲不允许扩容（图冻结了旧地址）；"
                    f"seq_len={seq_len} > max_length={old_len}，"
                    "请增大图缓存桶长（graph_len_budget）"
                )

            new_prob = torch.empty(
                (batch_size, self.max_length, self.vocab_size),
                device=device,
                dtype=dtype,
            )
            new_prob[:, :old_len, :] = self._prob_buffer
            self._prob_buffer = new_prob

            new_logits = torch.empty(
                (batch_size, self.max_length, self.vocab_size),
                device=device,
                dtype=dtype,
            )
            if self._logits_buffer is not None:
                new_logits[:, :old_len, :] = self._logits_buffer
            self._logits_buffer = new_logits

    def _raise_if_invalid_probs(self, probs: torch.Tensor, label: str) -> None:
        if not numeric_debug_checks_enabled():
            return
        if not log_prob_tensor_if_invalid(probs, label):
            return

        model_name = getattr(
            getattr(self._model, "config", None), "_name_or_path", "unknown"
        )
        probs_float = probs.detach().float()
        row_sums = probs_float.sum(dim=-1)
        raise ValueError(
            f"Invalid probability tensor before sampling for {model_name} at {label}: "
            f"shape={tuple(probs.shape)}, row_sum_min={float(row_sums.min().item())}, "
            f"row_sum_max={float(row_sums.max().item())}"
        )

    @torch.inference_mode()
    def _prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input: (batch_size, seq_len)
        output: (batch_size, vocab_size) - probabilities for the next token after the entire input sequence
        """
        self._validate_input_ids(input_ids)
        if self._use_cuda_graph:
            return self._prefill_graph(input_ids)
        seq_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        self._past_key_values = self._new_dynamic_cache()
        outputs = self._model(
            **self._prepare_generation_inputs(
                input_ids,
                past_key_values=self._past_key_values,
            )
        )
        logits = outputs.logits
        if logits is None:
            raise RuntimeError("Model returned logits=None in prefill")

        self._ensure_buffer_size(batch_size, seq_length, logits.device, logits.dtype)
        sliced_logits = logits[..., : self.vocab_size]
        assert self._logits_buffer is not None, (
            "Logits buffer should not be None after ensuring buffer size"
        )
        self._logits_buffer[:, :seq_length, :] = sliced_logits

        # 原始 logits 下的熵（温度 1，供 RL 控制器作状态特征；GPU 张量不同步）
        with torch.no_grad():
            self._last_entropy_t = self._compute_entropy_tensor(sliced_logits)
        probs = norm_logits(sliced_logits, self._temperature, self._top_k, self._top_p)
        log_prob_tensor_if_invalid(
            probs[:, -1, :],
            "KVCacheModel._forward_with_kvcache.initial_probs",
        )

        assert self._prob_buffer is not None, (
            "Probability buffer should not be None after ensuring buffer size"
        )
        self._prob_buffer[:, :seq_length, :] = probs
        self._current_seq_len = seq_length
        self._past_key_values = outputs.past_key_values
        self.hidden_states = outputs.hidden_states

        return probs[:, -1, :]

    def _graph_bucket_len(self, seq_length: int) -> int:
        """图缓存桶长：prompt + 生成预算 + 余量，按 256 取整、512 下限。

        为什么桶宽重要（实测）：StaticCache+SDPA 的 attention 按桶宽计算
        （mask 填充到 max_cache_len），与当前 pos 无关 —— 桶 512/1024/2048 的
        裸回放实测 6.42/7.24/8.81 ms（1.1B, k=17, pos=96），桶每多 1 个
        key 位置 ≈ +1.2µs。256 粒度 + 512 下限让桶紧贴 prompt+生成预算。
        太松的旧参数（1024 粒度 + 1536 下限）在 GSM8K 典型 prompt 下白付
        ~0.8ms/前向 × 每轮 4-5 次前向。

        初始建桶与超长重建共用同一公式，保证重建后的桶不会再被相近长度的
        prompt 立刻打穿。
        """
        need = seq_length + self._graph_len_budget
        return max(512, ((need + 255) // 256) * 256)

    @torch.inference_mode()
    def _prefill_graph(self, input_ids: torch.Tensor) -> torch.Tensor:
        """图模式下的 prefill：用 StaticCache 走完整段 prompt 并捕获后续定长图。

        为什么不能只把 decode 换掉：DynamicCache 会随解码增长，张量形状每步都变，
        而图冻结的是形状与地址。所以图模式必须**整体**换成按 max_length 预分配的
        StaticCache —— 代价是显存按 max_length 预留，收益是 decode 走图回放。
        prefill 本身是变长的、不进图（每样本一次，摊薄后影响很小）。

        跨样本复用：runner 已存在时不重建 —— StaticLayer.reset() 是原地 zero_，
        地址不变 ⇒ 图仍然有效，只付一次清零（对比每样本重捕获 13B ~534ms）。
        prompt 超出当前 max_len 时丢弃 runner 重建（罕见，打印一次性提示）。
        """
        from .graph_decode import GraphDecodeRunner

        batch_size, seq_length = input_ids.shape

        if (
            self._graph_runner is not None
            and seq_length >= self._graph_runner.max_len
        ):
            print(
                f"[cuda-graph] prompt {seq_length} 超出图缓存 "
                f"{self._graph_runner.max_len}，重建并重捕获",
                flush=True,
            )
            self._graph_runner = None
            self._past_key_values = None
            self.max_length = max(self.max_length, self._graph_bucket_len(seq_length))
            # 旧的历史缓冲按旧 max_length 分配；重建分支抬高了 max_length，
            # _ensure_buffer_size 会认为"还没超"而跳过扩容（新增长度夹在
            # 旧缓冲与新 max_length 之间时写入越界 —— 测试抓到过）。新样本
            # 反正从零开始，直接丢弃按新桶重分配。
            self._prob_buffer = None
            self._logits_buffer = None

        if self._graph_runner is None:
            if not self._has_explicit_max_length:
                # StaticCache 按 max_len 预分配（attention 也按桶宽计算，见
                # _graph_bucket_len 的实测注记）：按「prompt + 生成预算 + 余量」
                # 定桶，256 取整 + 512 下限。数据集内 prompt 长度有分布（GSM8K
                # 3-shot 约数百 token 差异），首样本定桶太小会让后续长 prompt
                # 频繁触发重建+重捕获（13B ~0.5s/次）—— graph_len_budget 已含
                # max_tokens+256 的余量兜底。
                self.max_length = self._graph_bucket_len(seq_length)
            device = self.device
            dtype = next(iter(self._model.parameters())).dtype
            runner = GraphDecodeRunner(
                self._model,
                max_len=self.max_length,
                device=device,
                dtype=dtype,
                verify_sizes=self._verify_graph_sizes,
            )
            self._graph_runner = runner

        assert self._graph_runner is not None
        runner = self._graph_runner

        # 历史缓冲必须在捕获**之前**就位：图内的 index_copy_ 会冻结它们的地址。
        # 模型 logits dtype == 参数 dtype（bf16），与 eager 路径创建时机等价。
        model_dtype = next(iter(self._model.parameters())).dtype
        self._ensure_buffer_size(batch_size, seq_length, self.device, model_dtype)
        assert self._prob_buffer is not None and self._logits_buffer is not None
        if self._prob_buffer.shape[1] < runner.max_len:
            raise RuntimeError(
                f"历史缓冲 {self._prob_buffer.shape[1]} < 图缓存 {runner.max_len}；"
                "图内 index_copy_ 会越界（桶长公式被破坏，请检查 max_length 配置）"
            )
        runner.attach_history_buffers(
            self._logits_buffer,
            self._prob_buffer,
            vocab_size=self.vocab_size,
            # 前后处理进图仅覆盖 temp==0（argmax one-hot 定长可捕获）；
            # temp>0 的 top-k/top-p 分支保守留在图外（回退旧行为）。
            capture_post=(self._temperature == 0),
        )
        raw_logits = runner.prefill(input_ids)

        full_logits = self._graph_runner.prefill_logits
        assert full_logits is not None
        # 历史缓冲已在捕获前创建（见上）；prefill 的整段 logits 走 eager 写入
        sliced_logits = full_logits[..., : self.vocab_size]
        assert self._logits_buffer is not None
        self._logits_buffer[:, :seq_length, :] = sliced_logits

        with torch.no_grad():
            self._last_entropy_t = self._compute_entropy_tensor(sliced_logits)
        probs = norm_logits(sliced_logits, self._temperature, self._top_k, self._top_p)
        log_prob_tensor_if_invalid(
            probs[:, -1, :], "KVCacheModel._prefill_graph.initial_probs"
        )
        assert self._prob_buffer is not None
        self._prob_buffer[:, :seq_length, :] = probs

        self._current_seq_len = seq_length
        self._past_key_values = runner.cache
        self.hidden_states = (
            (runner.hidden_buf,) if runner.hidden_buf is not None else None
        )
        return probs[:, -1, :]

    # @torch.compile()
    @torch.inference_mode()
    def _decode_step(self, last_input_id: torch.Tensor) -> torch.Tensor:
        """
        Decode one cached step (or a short cached suffix) after prefill.
        """
        if last_input_id.dtype != torch.long:
            last_input_id = last_input_id.to(torch.long)

        if last_input_id.shape[1] == 0:
            if self._current_seq_len <= 0:
                raise RuntimeError(
                    "No new input provided for decode step and no cache available"
                )
            return self.prob_history[:, self._current_seq_len - 1, :]

        past_key_values = self._past_key_values
        if past_key_values is None:
            raise RuntimeError("Decode step called before cache initialization")

        batch_size = last_input_id.shape[0]
        new_len = last_input_id.shape[1]

        # 图模式路由：
        #   new_len == 1 → 单步图（(1,1)）
        #   new_len > 1  → (1,K) 定长 padding 验证图（resync+草稿的合成前向）；
        #                 k 超出最大档位时 runner.verify 返回 None → eager 回退。
        #   batch > 1    → 一律 eager（管线是 bs=1，仅防御）。
        graph_used = False
        hidden_last: torch.Tensor | None = None
        if (
            self._graph_runner is not None
            and self._graph_runner.graph is not None
            and batch_size == 1
        ):
            runner = self._graph_runner
            if new_len == 1:
                # runner.step 返回的已经是最后一个位置的 (1, V)；下游统一按
                # (batch, new_len, V) 处理，这里补回序列维（view，不复制）
                logits, hidden_last = runner.step(last_input_id)
                logits = logits.unsqueeze(1)
                graph_used = True
            else:
                vout = runner.verify(last_input_id)
                if vout is not None:
                    # (1, k, V) 的前 k 行真实 logits；hidden 取最后一个真实行
                    logits, hidden_last = vout
                    graph_used = True
        if graph_used:
            # 图内缓存是 prefill 用的那个 StaticCache，地址不变，无需替换
            new_past_key_values = None
            self.hidden_states = (hidden_last,) if hidden_last is not None else None
            if runner.capture_post:
                # 前后处理已编进图：logits/prob 历史与每行熵都由图内写入
                # （pad 行落在 current_seq_len 之外，读端不可见）。这里只剩
                # 切视图 + 按真实 k 行求熵均值 —— 零 kernel 派发。
                end_pos = self._current_seq_len + new_len
                ent_rows = runner.last_entropy_rows
                if ent_rows is not None:
                    self._last_entropy_t = ent_rows[:, :new_len].mean()
                self._current_seq_len = end_pos
                q = self.prob_history[:, end_pos - 1, :]
                return q
        else:
            outputs = self._model(
                **self._prepare_generation_inputs(
                    last_input_id,
                    past_key_values=past_key_values,
                )
            )
            logits = outputs.logits

            if logits is None:
                raise RuntimeError("Model returned logits=None in decode step")

            new_past_key_values = outputs.past_key_values
            if new_past_key_values is None:
                raise RuntimeError("Model returned past_key_values=None in decode step")

        end_pos = self._current_seq_len + new_len

        if (
            not graph_used
            and self._graph_runner is not None
            and self._graph_runner.graph is not None
            and end_pos > self._graph_runner.max_len
        ):
            # eager 回退与图共用同一个定长 StaticCache：写入超出桶容量是
            # 静默显存越界（CUDA assert 或更糟）。管线里桶 = prompt +
            # max_tokens + 余量，正常不可达；测试/特殊配置需要显式失败。
            raise RuntimeError(
                f"图模式 eager 回退超出 StaticCache 容量：end_pos={end_pos} > "
                f"max_len={self._graph_runner.max_len}。增大 max_length 或 "
                "graph_len_budget（= max_tokens + 余量）"
            )

        self._ensure_buffer_size(batch_size, end_pos, logits.device, logits.dtype)

        sliced_logits = logits[..., : self.vocab_size]
        if self._logits_buffer is not None:
            self._logits_buffer[:, self._current_seq_len : end_pos, :] = sliced_logits

        # 原始 logits 下的熵（温度 1，供 RL 控制器作状态特征；GPU 张量不同步）
        with torch.no_grad():
            self._last_entropy_t = self._compute_entropy_tensor(sliced_logits)
        probs = norm_logits(sliced_logits, self._temperature, self._top_k, self._top_p)
        log_prob_tensor_if_invalid(
            probs,
            "KVCacheModel._forward_with_kvcache.cached_probs",
        )
        if self._prob_buffer is not None:
            self._prob_buffer[:, self._current_seq_len : end_pos, :] = probs

        self._current_seq_len = end_pos
        if not graph_used:
            self._past_key_values = new_past_key_values
            self.hidden_states = outputs.hidden_states
            if self._graph_runner is not None:
                # 不变量：runner.nnz 必须恒等于 _current_seq_len。
                # 图内回放由 runner.step() 自己推进，但这条 eager 回退（new_len > 1 的
                # 短后缀）不会动 nnz —— 如果不对齐，下一轮 current_length 偏小，
                # _generate_with_kvcache 会把已经在缓存里的 token 重新喂一遍，
                # 表现为 prob_history 越界（曾观测 n1=354 而 len=341，差 gamma2+1）。
                self._graph_runner.nnz = end_pos

        return probs[:, -1, :]

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        if input_ids.numel() == 0:
            return

        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        if min_id < 0 or max_id >= self.embedding_vocab_size:
            model_name = getattr(
                getattr(self._model, "config", None), "_name_or_path", "unknown"
            )
            raise ValueError(
                "Input token id out of embedding range for model "
                f"{model_name}: min={min_id}, max={max_id}, "
                f"embedding_vocab_size={self.embedding_vocab_size}, "
                f"configured_vocab_size={self.vocab_size}"
            )

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(torch.long)

        if self._past_key_values is None:
            return self._prefill(input_ids)

        cached_len = self.current_length
        last_input_id = input_ids[:, cached_len:]

        if last_input_id.numel() == 0:
            if self._current_seq_len <= 0:
                raise RuntimeError(
                    "No new input provided for decode step and no cache available"
                )
            return self.prob_history[:, self._current_seq_len - 1, :]

        return self._decode_step(last_input_id)

    @property
    def last_hidden_state(self) -> torch.Tensor:
        if self.hidden_states is None:
            raise ValueError("hidden_states is None")
        return self.hidden_states[-1]

    def _generate_with_kvcache(self, prefix: torch.Tensor, gamma: int) -> torch.Tensor:
        x = prefix
        if x.dtype != torch.long:
            x = x.to(torch.long)

        if gamma == 0:
            return x

        # First step: use _forward_with_kvcache to handle prefill or cache extension
        q = self._forward_with_kvcache(x)
        self._raise_if_invalid_probs(q, "KVCacheModel._generate_with_kvcache.q")
        next_tok = sample(q)
        if next_tok.dtype != torch.long:
            next_tok = next_tok.to(torch.long)
        new_tokens: list[torch.Tensor] = [next_tok]

        # Subsequent steps: use _decode_step with only the new token to avoid
        # growing x with torch.cat on each iteration
        for _ in range(gamma - 1):
            q = self._decode_step(new_tokens[-1])
            self._raise_if_invalid_probs(q, "KVCacheModel._generate_with_kvcache.q")
            next_tok = sample(q)
            if next_tok.dtype != torch.long:
                next_tok = next_tok.to(torch.long)
            new_tokens.append(next_tok)

        return torch.cat([x] + new_tokens, dim=1)

    def generate_with_rebuilt_topk(
        self,
        input: torch.Tensor,
        gamma: int,
        proposal_top_k: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = input
        if x.dtype != torch.long:
            x = x.to(torch.long)

        if gamma == 0:
            return x, None

        rebuilt_rows: list[torch.Tensor] = []
        new_tokens: list[torch.Tensor] = []

        # First step: use _forward_with_kvcache to handle prefill or cache extension
        q = self._forward_with_kvcache(x)
        self._raise_if_invalid_probs(q, "KVCacheModel.generate_with_rebuilt_topk.q")
        rebuilt_q = rebuild_topk_uniform_probs(q, proposal_top_k)
        self._raise_if_invalid_probs(
            rebuilt_q, "KVCacheModel.generate_with_rebuilt_topk.rebuilt_q"
        )
        rebuilt_rows.append(rebuilt_q.unsqueeze(1))
        next_tok = sample(rebuilt_q)
        if next_tok.dtype != torch.long:
            next_tok = next_tok.to(torch.long)
        new_tokens.append(next_tok)

        # Subsequent steps: use _decode_step with only the new token to avoid
        # growing x with torch.cat on each iteration
        for _ in range(gamma - 1):
            q = self._decode_step(new_tokens[-1])
            self._raise_if_invalid_probs(
                q, "KVCacheModel.generate_with_rebuilt_topk.q"
            )
            rebuilt_q = rebuild_topk_uniform_probs(q, proposal_top_k)
            self._raise_if_invalid_probs(
                rebuilt_q, "KVCacheModel.generate_with_rebuilt_topk.rebuilt_q"
            )
            rebuilt_rows.append(rebuilt_q.unsqueeze(1))
            next_tok = sample(rebuilt_q)
            if next_tok.dtype != torch.long:
                next_tok = next_tok.to(torch.long)
            new_tokens.append(next_tok)

        rebuilt_history = torch.cat(rebuilt_rows, dim=1)
        return torch.cat([x] + new_tokens, dim=1), rebuilt_history

    def generate_with_rebuilt_topk_metadata(
        self,
        input: torch.Tensor,
        gamma: int,
        proposal_top_k: Optional[int],
    ):
        x = input
        if x.dtype != torch.long:
            x = x.to(torch.long)

        if gamma == 0:
            return x, None, None

        rebuilt_rows: list[torch.Tensor] = []
        proposal_steps = []
        new_tokens: list[torch.Tensor] = []

        q = self._forward_with_kvcache(x)
        self._raise_if_invalid_probs(
            q, "KVCacheModel.generate_with_rebuilt_topk_metadata.q"
        )
        rebuilt_q = rebuild_topk_uniform_probs(q, proposal_top_k)
        self._raise_if_invalid_probs(
            rebuilt_q,
            "KVCacheModel.generate_with_rebuilt_topk_metadata.rebuilt_q",
        )
        rebuilt_rows.append(rebuilt_q.unsqueeze(1))
        proposal_meta = build_topk_proposal_history_step(q, proposal_top_k)
        if proposal_meta is not None:
            proposal_steps.append(proposal_meta)
        next_tok = sample(rebuilt_q)
        if next_tok.dtype != torch.long:
            next_tok = next_tok.to(torch.long)
        new_tokens.append(next_tok)

        for _ in range(gamma - 1):
            q = self._decode_step(new_tokens[-1])
            self._raise_if_invalid_probs(
                q, "KVCacheModel.generate_with_rebuilt_topk_metadata.q"
            )
            rebuilt_q = rebuild_topk_uniform_probs(q, proposal_top_k)
            self._raise_if_invalid_probs(
                rebuilt_q,
                "KVCacheModel.generate_with_rebuilt_topk_metadata.rebuilt_q",
            )
            rebuilt_rows.append(rebuilt_q.unsqueeze(1))
            proposal_meta = build_topk_proposal_history_step(q, proposal_top_k)
            if proposal_meta is not None:
                proposal_steps.append(proposal_meta)
            next_tok = sample(rebuilt_q)
            if next_tok.dtype != torch.long:
                next_tok = next_tok.to(torch.long)
            new_tokens.append(next_tok)

        rebuilt_history = torch.cat(rebuilt_rows, dim=1)
        rebuilt_history_meta = concat_topk_proposal_history(proposal_steps)
        return torch.cat([x] + new_tokens, dim=1), rebuilt_history, rebuilt_history_meta

    def _sample_from_topk_proposal(
        self,
        probs: torch.Tensor,
        proposal_top_k: Optional[int],
    ) -> torch.Tensor:
        proposal_meta = build_topk_proposal_history_step(probs, proposal_top_k)
        if proposal_meta is None:
            token = sample(probs)
            return token.to(torch.long) if token.dtype != torch.long else token

        topk_indices = proposal_meta.topk_indices[:, 0, :]
        topk_probs = proposal_meta.topk_probs[:, 0, :]
        tail_uniform_prob = proposal_meta.tail_uniform_prob[:, 0, :]
        topk_mass = topk_probs.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        tail_mass = (1.0 - topk_mass).clamp_min(0.0)
        region_probs = torch.cat((topk_mass, tail_mass), dim=-1)
        region_choice = torch.multinomial(region_probs, num_samples=1)

        token = torch.empty(
            (probs.shape[0], 1),
            dtype=torch.long,
            device=probs.device,
        )

        topk_rows = region_choice.squeeze(-1) == 0
        if topk_rows.any():
            normalized_topk = topk_probs[topk_rows] / topk_mass[topk_rows].clamp_min(1e-12)
            topk_pick = torch.multinomial(normalized_topk, num_samples=1)
            token[topk_rows] = torch.gather(topk_indices[topk_rows], 1, topk_pick)

        tail_rows = ~topk_rows
        if tail_rows.any():
            tail_topk_indices = topk_indices[tail_rows]
            batch_size, _, = tail_topk_indices.shape
            vocab_size = probs.shape[-1]
            candidate_mask = torch.ones(
                (batch_size, vocab_size),
                dtype=torch.bool,
                device=probs.device,
            )
            candidate_mask.scatter_(1, tail_topk_indices, False)
            candidate_weights = candidate_mask.to(probs.dtype)
            tail_pick = torch.multinomial(candidate_weights, num_samples=1)
            token[tail_rows] = tail_pick

        return token

    def generate_with_topk_metadata_only(
        self,
        input: torch.Tensor,
        gamma: int,
        proposal_top_k: Optional[int],
    ):
        x = input
        if x.dtype != torch.long:
            x = x.to(torch.long)

        if gamma == 0:
            return x, None

        proposal_steps = []
        new_tokens: list[torch.Tensor] = []

        q = self._forward_with_kvcache(x)
        self._raise_if_invalid_probs(
            q, "KVCacheModel.generate_with_topk_metadata_only.q"
        )
        proposal_meta = build_topk_proposal_history_step(q, proposal_top_k)
        if proposal_meta is not None:
            proposal_steps.append(proposal_meta)
        next_tok = self._sample_from_topk_proposal(q, proposal_top_k)
        if next_tok.dtype != torch.long:
            next_tok = next_tok.to(torch.long)
        new_tokens.append(next_tok)

        for _ in range(gamma - 1):
            q = self._decode_step(new_tokens[-1])
            self._raise_if_invalid_probs(
                q, "KVCacheModel.generate_with_topk_metadata_only.q"
            )
            proposal_meta = build_topk_proposal_history_step(q, proposal_top_k)
            if proposal_meta is not None:
                proposal_steps.append(proposal_meta)
            next_tok = self._sample_from_topk_proposal(q, proposal_top_k)
            if next_tok.dtype != torch.long:
                next_tok = next_tok.to(torch.long)
            new_tokens.append(next_tok)

        rebuilt_history_meta = concat_topk_proposal_history(proposal_steps)
        return torch.cat([x] + new_tokens, dim=1), rebuilt_history_meta

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        return self._generate_with_kvcache(input, gamma)

    def reset_for_new_sample(self) -> None:
        """跨样本复用：清空序列状态，保留已捕获的图与预分配缓冲。

        prob/logits 历史缓冲按 `_current_seq_len` 切片暴露，旧样本的残留行
        在新样本写满前不可见；graph runner 里 StaticCache 原地清零（地址不变，
        图仍有效）。eager 路径下 `_past_key_values=None` 等价于重建。
        """
        self._current_seq_len = 0
        self._past_key_values = None
        self.hidden_states = None
        self._last_entropy_t = None
        if self._graph_runner is not None:
            self._graph_runner.begin_new_sequence()

    @torch.no_grad()
    def rollback(self, end_pos: int):
        if self._past_key_values is None:
            return

        # 图模式：缓存是按 max_length 预分配的 StaticCache，没有 crop 的概念
        # （StaticCache 上的 crop 会转发到不存在的 StaticLayer.crop）。回滚只需把
        # 写入指针退回去，旧槽位会在后续写入时被覆盖 —— 这正是论文里"回滚=指针移动"
        # 的实现，也是图模式省掉一次 KV 内存搬运的额外收益。
        if self._graph_runner is not None:
            self._graph_runner.rollback(end_pos)
            self._current_seq_len = min(end_pos, self.current_length)
            return

        if _is_cache_like(self._past_key_values) and hasattr(
            self._past_key_values, "crop"
        ):
            self._past_key_values.crop(end_pos)
        else:
            assert _is_legacy_past(self._past_key_values)
            past_key_values_trimmed: list[KVPair] = []
            for kv in self._past_key_values:
                k, v = kv
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                past_key_values_trimmed.append((k, v))
            self._past_key_values = tuple(past_key_values_trimmed)

        # Keep history length aligned with the real cache length. Rolling back beyond
        # the cache should not expose uninitialized rows from the preallocated buffers.
        self._current_seq_len = min(end_pos, self.current_length)

    @property
    def device(self) -> torch.device:
        return next(iter(self._model.parameters())).device

    @property
    def current_length(self) -> int:
        if self._graph_runner is not None and self._graph_runner.graph is not None:
            return self._graph_runner.nnz
        if self._past_key_values is None:
            return 0
        if _is_cache_like(self._past_key_values):
            return self._past_key_values.get_seq_length()
        assert _is_legacy_past(self._past_key_values)
        return self._past_key_values[0][0].shape[2]

    def debug_state(self) -> dict[str, int]:
        prob_history_len = 0 if self._prob_buffer is None else self._current_seq_len
        logits_history_len = 0 if self._logits_buffer is None else self._current_seq_len
        return {
            "current_length": self.current_length,
            "tracked_seq_len": self._current_seq_len,
            "prob_history_len": prob_history_len,
            "logits_history_len": logits_history_len,
            "max_length": self.max_length,
        }

    def debug_row_sums(self, start: int, end: int) -> list[float]:
        if self._prob_buffer is None or self._current_seq_len <= 0:
            return []

        start = max(0, min(start, self._current_seq_len))
        end = max(start, min(end, self._current_seq_len))
        if end <= start:
            return []

        row_sums = self._prob_buffer[0, start:end, :].detach().float().sum(dim=-1)
        return [float(v.item()) for v in row_sums]

    def __len__(self) -> int:
        return self.current_length
