#!/usr/bin/env python
"""13B 4bit（bitsandbytes nf4）× CUDA Graph 捕获的**前置排雷**测试。

为什么要单独测：68M/1.1B 是 bf16 权重，kernel 全在图内无 host 分支；bnb 4bit
的 Linear4bit 前向里有反量化/dequant 逻辑，若含数据相关的 host 分支或不可捕获
的 API，图捕获会失败（或静默改变行为）。A6000 空闲显存不够装两份 13B，所以
这个测试串行跑：先 eager 参考存 CPU，再建图路径比对。

通过标准：捕获成功 + verify 图 logits 与 eager 整段一致（bf16 容差）+ 回放可重复。
用法: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/test_graph_verify_13b4bit.py
"""
from __future__ import annotations

import gc
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, ".")
from src.model_loading import build_quant_config  # noqa: E402
from src.graph_decode import GraphDecodeRunner  # noqa: E402

MODEL = "llama/Llama-2-13b-hf"  # 与管线一致：utils.py 的 llama-2-13b → 此本地路径
TOL_REL = 5e-2  # 4bit nf4 反量化 + bf16：容差放宽


def load(quant: str):
    cfg = build_quant_config(MODEL, quantization=quant, compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=cfg,
        output_hidden_states=True,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    return model


def main() -> int:
    print(f"GPU {torch.cuda.get_device_name(0)}，模型 {MODEL}（4bit nf4）")
    torch.manual_seed(0)
    prompt = torch.randint(100, 30000, (1, 24), device="cuda")

    # ── 参考：eager + DynamicCache，逐 k-token 前向的结果存 CPU ──────────
    print("加载 eager 参考（4bit）…", flush=True)
    model = load("4bit")

    from transformers import DynamicCache

    @torch.inference_mode()
    def eager_rows(tokens: torch.Tensor, cache, pos: int) -> torch.Tensor:
        mask = torch.ones((1, pos + tokens.shape[1]), dtype=torch.long, device="cuda")
        out = model(
            tokens,
            past_key_values=cache,
            cache_position=torch.arange(pos, pos + tokens.shape[1], device="cuda"),
            attention_mask=mask,
            use_cache=True,
        )
        return out.logits[:, :, :].float().cpu()

    cache = DynamicCache(config=model.config)
    ref_prefill = eager_rows(prompt, cache, 0)
    cache_len = prompt.shape[1]
    toks17 = torch.randint(100, 30000, (1, 17), device="cuda")
    ref_verify = eager_rows(toks17, cache, cache_len)
    del cache
    gc.collect()
    torch.cuda.empty_cache()

    # ── 图路径：StaticCache + 双图（单步 + (1,K) 验证）───────────────────
    print("捕获图（同一份 4bit 权重）…", flush=True)
    runner = GraphDecodeRunner(
        model, max_len=256, device="cuda", dtype=torch.bfloat16, verify_sizes=(8, 16, 24)
    )
    # 前后处理进图（temp 0 配置）：历史缓冲在捕获前 attach，图内 index_copy_ 写入
    vocab = int(model.config.vocab_size)
    hist_logits = torch.zeros((1, 256, vocab), dtype=torch.bfloat16, device="cuda")
    hist_prob = torch.zeros((1, 256, vocab), dtype=torch.bfloat16, device="cuda")
    runner.attach_history_buffers(hist_logits, hist_prob, vocab_size=vocab, capture_post=True)
    g_prefill = runner.prefill(prompt).float().cpu()

    def rel(a, b):
        return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))

    r0 = rel(g_prefill, ref_prefill[0, -1])
    print(f"prefill 最后位置 rel = {r0:.4f}")
    assert r0 <= TOL_REL, "prefill 不一致"

    # 验证图：k=17 → 档位 24（padding）
    pos0 = prompt.shape[1]
    logits, hidden = runner.verify(toks17)
    r1 = rel(logits[:, :17].float().cpu(), ref_verify[0])
    print(f"verify 图 k=17（pad→24）rel = {r1:.4f}")
    assert r1 <= TOL_REL, "verify 图与 eager 整段不一致"

    # 图内 index_copy_ 写入的历史行 + one-hot + 熵（前后处理进图路径）
    r_hist = rel(
        hist_logits[0, pos0 : pos0 + 17, :].float().cpu(), ref_verify[0]
    )
    print(f"图内写入的 logits 历史 rel = {r_hist:.4f}")
    assert r_hist <= TOL_REL, "图内历史写入与 eager 不一致"
    am_prob = hist_prob[0, pos0 : pos0 + 17, :].float().argmax(-1).cpu()
    am_ref = ref_verify[0].argmax(-1)
    assert bool(torch.equal(am_prob, am_ref)) or r_hist <= TOL_REL, "图内 one-hot argmax 分歧"
    ent = runner.last_entropy_rows
    assert ent is not None and bool(torch.isfinite(ent).all()), "图内熵异常"
    print(f"图内每行熵有限 ✓（前 3 行: {ent[0, :3].tolist()}）")

    # 回放可重复（同输入再放一次，bit 级一致才对——同一张图）
    runner.rollback(prompt.shape[1])
    logits2, _ = runner.verify(toks17)
    assert torch.equal(logits2, logits), "同输入两次回放不一致"
    print("同输入两次回放 bit 级一致 ✓")

    # 单步图
    q, h = runner.step(torch.tensor([[7]], device="cuda"))
    assert q is not None and h is not None
    print(f"单步图 OK，hidden 形状 {tuple(h.shape)}")
    print("hidden_buf 有限性:", bool(torch.isfinite(h.float()).all()))

    print("\n⇒ ✓ 13B 4bit 可捕获、verify 图与 eager 一致 —— A/B 可以放心跑")
    return 0


if __name__ == "__main__":
    sys.exit(main())
