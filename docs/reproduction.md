# 复现验收口径

同一台机器上的回归用 `test/golden/compare.py`：除计时字段外全部 LOCKED，token 逐位一致。换 GPU/驱动后 token 不再逐位一致，按下面的分级验收。

## 分级

| 级别 | 内容 | 适用范围 |
|---|---|---|
| **R0 逻辑** | `pytest` 全绿；逻辑指纹一致 | 所有机器 |
| **R1 资产/配置** | `git_sha`、checkpoint sha、model revision、协议字段、seed/shots/带宽/γ/top-k 一致 | 所有机器 |
| **R2 行为** | accuracy/correctness 一致；计数类指标（tokens、forward、通信字节）在 ±2% 内 | 跨 GPU/驱动 |
| **D 分布** | 生成长度分布、token unigram 分布、accuracy 一致 | R2 的补充，跨机器 |
| **R3 逐 token** | token 序列逐位一致 | 同 GPU + 同驱动 + 同 torch/CUDA/transformers |
| **P 性能** | throughput/wall/comm 只在同硬件内比较 | 论文表 |

## D：分布一致性

```bash
python tools/compare_distributions.py baseline.jsonl new.jsonl
```

判定阈值（默认）：

| 指标 | 阈值 |
|---|---|
| 生成长度 two-sample KS | D ≤ 0.2 |
| token unigram cosine | ≥ 0.99 |
| token unigram total variation | ≤ 0.05 |
| accuracy 差 | ≤ 0.05 |
| token 一致率 | 只报告，不判失败 |

2026-10-31 实测（q15，本地 A6000 golden vs enine Blackwell）：

```
samples: [20, 20]
tokens:  [2533, 2523]
length:  KS D=0.100
unigram: cosine=0.99880, TV=0.0290
accuracy: [0.25, 0.25] diff=0.0
token match: 13/20 (diagnostic)
RESULT PASS
```

## 论文表

- 不同 GPU 的绝对值不合并；脚注写清 GPU 型号、驱动、torch/CUDA；
- Table VIII 在 Blackwell 上只作为同硬件相对表（5/10/46 Mbps 趋势），不改论文 A6000 数字。

## 已知缺口

metrics 还没有 `runtime` 字段（GPU 型号、torch/CUDA、transformers、attention backend）。跨机器 run 目前要从日志/环境记录反推，后续应补进 `get_save_dict`。
