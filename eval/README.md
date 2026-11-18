# 运行 evaluation

原来的 `python eval/eval_*.py ...` 单任务命令仍可使用。需要连续跑不同配置时，用 `exp.py` 的矩阵入口：它把同一组兼容模型的配置排在一起，分配一次 GPU，在同一个子进程中串行执行；不同组仍可并行。分别启动多条 shell 命令不会跨进程共享显存。

在 Python 中也可以显式复用权重。下面的 `args_a`、`args_b` 是两份已解析的评测配置：

```python
from eval.decoder import ModelStore
from eval.eval_gsm8k import EvalGSM8K
from eval.eval_mt_bench_noeval import EvalMTBench

models = ModelStore()
try:
    math_result = EvalGSM8K(args_a, models=models).eval()
    chat_result = EvalMTBench(args_b, models=models).eval()
finally:
    models.close()
```

- **权重按加载条件复用**：任务、带宽、采样参数不同不要求重载。实际使用的模型、精度、量化或放置方式不兼容时需要重载。缓存不会保存历次实验的所有模型。
- **运行状态不复用**：每次 `.eval()` 都使用独立的配置副本、解码器、KV/graph 缓存、策略适配器和指标。用新配置创建策略不改变原有显式 checkpoint 的读写语义。
- **任务与输出**：任务定义数据、prompt、多轮或多次采样、评分；运行器负责计时、非预热样本的指标累计和结果写入。单任务 JSONL 为兼容既有工具仍是追加写；`*_metrics.json` 和 `.eval()` 的返回值只描述本次运行，不从历史行重算。
- **加载错误**：模型、数据集或本地 JSONL 加载失败时，本次运行失败，不写零样本成功汇总。混合任务需要的各个数据集也必须能正常加载。
