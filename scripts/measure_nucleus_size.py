#!/usr/bin/env python
"""top-p 核大小测量: 草稿分布上 |nucleus(p)| 与 top-k 覆盖质量的对照。

回答: R2 弱链路域该用 top-k(预算旋钮)还是 top-p(质量旋钮)做上行稀疏化。
方法: tiny-llama-1.1b 贪心解码 GSM8K 20 题 × 40 token, 逐位置统计
  - |nucleus(p)| @ p ∈ {0.8, 0.9, 0.95, 0.99}  (达到累计质量所需最小 token 数)
  - top-k 覆盖质量 @ k ∈ {64, 256, 300, 1024}   (前 k 个 token 的概率质量和)
输出: exp_logs/nucleus_size_report.txt (表格) + .json (原始分布)
用法: .venv/bin/python scripts/measure_nucleus_size.py [--device cpu]
"""
import argparse, json, os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1", help="cuda:1 / cpu")
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--model", default="llama/tiny-llama-1.1b")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map=args.device
    )
    model.eval()

    # data/test.jsonl = MT-Bench 格式 (question_id/category/turns): 开放生成域,
    # 熵更高, 正是 top-p 稀疏化最有趣的场景
    data = [json.loads(l) for l in open("data/test.jsonl") if l.strip()]
    prompts = []
    for d in data:
        if isinstance(d.get("turns"), list) and d["turns"]:
            prompts.append(d["turns"][0])
        elif isinstance(d.get("instruction"), str):
            prompts.append(d["instruction"])
        if len(prompts) >= args.n_prompts:
            break

    P_TARGETS = [0.8, 0.9, 0.95, 0.99]
    K_TARGETS = [64, 256, 300, 1024]
    nucleus = {p: [] for p in P_TARGETS}
    coverage = {k: [] for k in K_TARGETS}

    with torch.no_grad():
        for pi, prompt in enumerate(prompts):
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(args.device)
            out = model.generate(
                **ids, max_new_tokens=args.max_new, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
            )
            for step_logits in out.scores:
                probs = torch.softmax(step_logits[0].float(), dim=-1)
                sorted_p, _ = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_p, dim=0)
                for p in P_TARGETS:
                    n = int(torch.searchsorted(cum, torch.tensor(p, device=cum.device)).item()) + 1
                    nucleus[p].append(min(n, len(sorted_p)))
                for k in K_TARGETS:
                    coverage[k].append(float(sorted_p[:k].sum()))

    def stats(v):
        v = sorted(v)
        n = len(v)
        return {"mean": sum(v) / n, "p50": v[n // 2], "p90": v[9 * n // 10], "max": v[-1]}

    report = {
        "model": args.model,
        "n_positions": len(nucleus[0.9]),
        "nucleus_size": {str(p): stats(v) for p, v in nucleus.items()},
        "topk_coverage": {str(k): stats(v) for k, v in coverage.items()},
        "bytes_per_token": {
            "topk_300": 300 * 8,
            **{
                f"topp_{p}": int(round(stats(nucleus[p])["mean"])) * 8
                for p in P_TARGETS
            },
        },
    }
    os.makedirs("exp_logs", exist_ok=True)
    json.dump(report, open("exp_logs/nucleus_size_report.json", "w"), indent=2)

    lines = [f"top-p 核大小测量 ({report['n_positions']} 位置, {args.model})"]
    lines.append(f"{'p':>5} | {'均值':>6} {'p50':>6} {'p90':>7} {'max':>7} | {'期望字节/位':>10}")
    for p in P_TARGETS:
        s = report["nucleus_size"][str(p)]
        lines.append(f"{p:>5} | {s['mean']:>6.1f} {s['p50']:>6d} {s['p90']:>7d} {s['max']:>7d} | {int(round(s['mean']))*8:>10d}")
    lines.append(f"\n{'k':>5} | {'覆盖质量均值':>10} {'p50':>7} {'p10(最差10%)':>12}")
    for k in K_TARGETS:
        s = report["topk_coverage"][str(k)]
        lines.append(f"{k:>5} | {s['mean']:>10.3f} {s['p50']:>7.3f} {s['p90']:>12.3f}")
    lines.append(f"\n对照: top-k300 固定 {300*8}B/位 vs top-p0.9 期望 {report['bytes_per_token']['topp_0.9']}B/位")
    txt = "\n".join(lines)
    open("exp_logs/nucleus_size_report.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
