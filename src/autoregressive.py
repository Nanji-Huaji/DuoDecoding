"""Autoregressive decoding for the small and large baselines, and lookahead forward."""

from typing import List, Optional, Tuple

import torch

from .graph_decode import graph_mode_cache_kwargs
from .metrics import DecodingMetrics, get_empty_metrics
from .register import Register


@Register.register_decoding("large")
@Register.register_decoding("small")
@torch.inference_mode()
def autoregressive_sampling(
    self,
    prefix,
    use_early_stopping: bool = False,
    stop_sequences: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, DecodingMetrics]:
    if self.args.eval_mode == "small":
        model = self.draft_model
        # B39：small 模式实际跑的是 draft 模型，前向次数应记入
        # draft_forward_times；此前统一写 target_forward_times，模型身份错标。
        forward_times_key = "draft_forward_times"
        # CUDA Graph：AR 循环是定长 (1,1) 单步前向（无多 token 验证
        # 前向，cap 按 γ=1 + 4 与其它方法的 γ+4 写法对齐）—— 正是单
        # 步图的收益点；且该循环逐样本调用，不复用就会每样本重捕获一次
        # （~534ms，见 docs/graph_integration_status.md），收益全被吃
        # 掉。跨样本复用同一缓存对象的理由写在
        # _acquire_autoregressive_cache 里。
        graph_kw = graph_mode_cache_kwargs(self.args, cap=1 + 4)
    elif self.args.eval_mode == "large":
        model = self.target_model
        forward_times_key = "target_forward_times"
        # target 保持 eager（与 speculative_decoding 的 draft/target 取舍一致）：
        # graph_kw 为空 ⇒ 构造 KVCacheModel 的调用与接线前逐字节相同。
        graph_kw = {}
    else:
        raise RuntimeError(
            "Auto-Regressive Decoding can be used only in small / large eval mode!"
        )
    self.validate_input_ids(prefix, "autoregressive_sampling.prefix")
    prefix = prefix.to(model.device)
    model = self._acquire_autoregressive_cache(
        "_autoregressive_sampling_caches", graph_kw, model
    )
    model.vocab_size = self.vocab_size

    prefix_len = prefix.shape[1]
    max_tokens = prefix_len + self.args.max_tokens

    x = prefix

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    forward_times = 0

    start_event.record(stream=torch.cuda.current_stream())
    queuing_time = 0
    batch_delay = self.comm_env.batch_delay
    while x.shape[1] < max_tokens:
        queuing_time += batch_delay
        x = model.generate(x, 1)
        forward_times += 1

        if use_early_stopping and self._check_stopping_criteria(
            x, stop_sequences, prompt_len=prefix_len
        ):
            break

    end_event.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    elapsed_time = (
        start_event.elapsed_time(end_event) / 1000.0
    )  # Convert to seconds
    generated_tokens = x.shape[1] - prefix_len

    metrics = get_empty_metrics()
    metrics[forward_times_key] = forward_times
    metrics["generated_tokens"] = generated_tokens
    metrics["queuing_time"] = queuing_time
    metrics["wall_time"] = elapsed_time + queuing_time
    metrics["throughput"] = (
        generated_tokens / metrics["wall_time"] if metrics["wall_time"] > 0 else 0
    )

    return x, metrics


@torch.inference_mode()
def lookahead_forward(self, prefix, **kwargs):
    input_ids = prefix.cuda()

    output_ids, idx, accept_length_list = self.target_model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=False,
        temperature=0.0,
        max_new_tokens=self.args.max_tokens,
        num_assistant_tokens_schedule="constant",
    )
    new_token = len(output_ids[0][len(input_ids[0]) :])  # noqa: F841 (legacy lookahead)
    return output_ids
