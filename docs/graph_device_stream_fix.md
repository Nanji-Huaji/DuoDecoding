# 跨设备 CUDA Graph 空图修复与数值边界

日期：2026-11-13。基线：`8854ebf5788d0ca953b8b5a53900cd77749f8589`。

## 结论

- **已修复真实的跨设备 Graph 正确性缺陷**：第二个设备可能捕获空图，replay 不更新缓冲区，最终产生错误 token。
- 修复后，8 项真实双 GPU 定向回归与 336 项原生 CEE 矩阵全部通过；没有使用进程内 stream 修正开关替代生产补丁。
- **Qwen BF16 的严格数值等价仍未通过**。差异不仅在历史 prompt 行，也在生成概率行。此次 eager/Graph 的 28 组配对输出全部一致，但这不能证明采样分布完全相同或差异永远无害。本次不为消除这些差异扩大生产改动，也不放宽阈值。

## 根因与最小修复

PyTorch 2.6 的 `torch.cuda.graph.default_capture_stream` 是类级共享状态，不是每个设备各有一个默认 capture stream。第一个 runner 初始化后，另一个设备上的 runner 若继续省略 `stream=`，可能在错误设备的 stream 上捕获；模型运算实际上在其张量所在设备执行，图内没有相应运算。

新回归在原始代码上出现 PyTorch 的明确警告：`The CUDA Graph is empty ... wrong device or stream`。将输出缓冲区填成 NaN 后 replay，NaN 仍保留，7/7 logits 对比失败。这不是普通 BF16 舍入误差。[原生红灯日志](<.refactor/graph-device-fix-20261113/remote-evidence/native-red-v1/execution.log>)。

生产修改仅在 [graph_decode.py](<src/graph_decode.py#L353-L364>)：

1. 构造时将未指定索引的 `cuda` 固定到当时设备，创建 runner 自有的 device-local capture stream。
2. [prefill](<src/graph_decode.py#L573-L587>) 在 runner 的设备上下文执行，退出后恢复调用者设备。
3. warmup 使用同一 capture stream；step 图与各 verify 图都显式传入该 stream；三处已有同步显式指定 runner 设备。
4. 修正 capture count 注释：它是单步图加 verify 图的数量，跨样本复用时不增长。

没有修改 CEE wire probability law、采样、验证、回滚、私有 RNG、精度策略或 replay 热路径；没有新增逐 token 的全设备同步。独立只读复核未发现可操作缺陷。一个 runner 跨多个 host stream 并发使用仍需调用者自行建立顺序，本次不扩展该契约。

## 回归证据

环境：Python 3.10.18、Torch 2.6.0+cu124、Transformers 4.57.6、pytest 9.1.1；water4090 两张 RTX 4090，物理 GPU2/3。

| 检查 | 结果 |
| --- | --- |
| CPU 接线回归，修复前 → 修复后 | 4 failed → 4 passed；8 项 native 默认跳过 |
| 原生双 GPU，未修复基线 | 指定反例失败，空图警告，NaN 未被 replay 覆盖 |
| 修复后定向测试 | 12 passed：4 CPU 契约 + 8 native CUDA |
| 本地 CPU 门禁，default / importlib | 每种 1989 passed + 163 subtests，15 skipped |
| 远程 CPU 门禁，default / importlib | 每种 1989 passed + 163 subtests，15 skipped |
| 新命名空间 CUDA smoke | passed |
| 原生 tiny CEE 矩阵 | 336/336 passed：eager 168，Graph 168 |
| 额外逐前向诊断自检 | 14/14 passed |

15 项 CPU skip 包含原有 7 个缺失历史实验目录的集成检查，以及新增 8 个显式 opt-in CUDA 检查；6 个 required golden checkpoint 的严格资产门禁通过，没有将 CUDA skip 算作 native 通过。

[新增测试](<test/test_graph_device_streams.py>)覆盖：设备顺序正反、显式/未索引设备、capture_post 开关、非默认调用者 stream、改变输入并毒化输出、step、带 padding 的 verify、rollback、跨样本复用、图对象和捕获数量稳定、默认 CUDA RNG 不变。

336 项原生矩阵包括 FP32/BF16、T=0/0.7、CEE-DSD/DSSD、`(0,0,0)` / `(0,0,1)` / `(0,1,1)` 三种逻辑布局，以及预算边界、新样本、EOS 裁剪。共生成 1488 tokens；DSD/DSSD 的 168 组配对和 eager/Graph 的 168 组配对均一致。记录到实际 Graph step 800 次、verify 428 次、begin_new_sequence 288 次。新鲜计算与缓存概率的最大差异为 `2.3841858e-7`。

[原生矩阵结果](<.refactor/graph-device-fix-20261113/remote-evidence/native-cee-all-v2/result.json>)、[native 绿灯日志](<.refactor/graph-device-fix-20261113/remote-evidence/native-green-v1/execution.log>)、[本地 CPU manifest](<.refactor/graph-device-fix-20261113/cpu-full/manifest.json>)、[远程 CPU manifest](<.refactor/graph-device-fix-20261113/remote-evidence/cpu-v1/manifest.json>)。

## BF16：没有掩盖的剩余差异

使用离线、未量化、SDPA 的 Qwen1.5-0.5B/1.8B/7B Chat，布局 `(0,0,1)`，修复后的真实 eager 和 Graph 后端。两个 prompt，两个温度，两种 CEE 协议，共 56 次调用、248 个生成 tokens。

- 38 次通过严格检查；18 次为 numerical mismatch（eager 6，Graph 12）。整体状态仍为 `failed-numerical-equivalence`，进程返回 2。
- DSD/DSSD 的 28 组配对、eager/Graph 的 28 组配对，输出 tokens 全部一致。
- 额外记录 424 次实际 cache forward，检查 2452 个生成区域概率行（含重复检查与后来被 rollback/EOS 丢弃的候选）。这些行均无 argmax 变化，但 **152 次 forward** 至少有一行超出原有数值阈值。
- T=0 对比是归一化后的 greedy one-hot 检查，不据此推断原始 logits 位级一致。T=0.7 使用原有 BF16 `atol=0.008, rtol=0.02`，没有调大。

| 生成区域，与同模型 fresh eager 对比的最大单项概率差 | eager | Graph |
| --- | ---: | ---: |
| little | 0.0458984375 | 0.0458984375 |
| draft | 0.0625 | 0.060546875 |
| target | 0.044921875 | 0.046875 |

最终保留历史中，prompt-only 区域的最大差异为 0.14453125，生成区域为 0.0625。prompt-only 行界限是 `< prompt_length - 1`；最后一个 prompt 位置已经能预测第一个生成 token，不能排除。逐前向记录补充了仅检查最终保留历史的盲区。

已有独立 [无 cache / 无 CEE / 无 Graph 对照](<.refactor/water4090-20261113/remote-evidence/qwen-shape-control-v1/shape-control.json>) 表明：0.5B 同一 causal prompt 改变总计算长度 31→40，BF16 logits 差异 0.28125、概率差异 0.03125；FP32 分别降到约 `2.67e-5` / `2.94e-6`。同输入重复相等；固定总长度、只改变未来 token 时 prompt 输出差异为 0。这独立复现了 shape-dependent 低精度数值差异，不能将其都归为 cache/CEE 错误；但也不是对每个 1.8B/7B 差异的逐 kernel 归因证明。

因此，本次接受的是**空图/陈旧回放缺陷的修复与限定测试结果**，不是“Qwen 严格分布等价已经通过”。目前没有观察到这些数值差异改变本次输出，不值得为此贸然改动采样语义或全局精度设置。若后续实验要求不同 batch/cache/Graph 路径严格使用相同概率分布，需要单独制定计算形状、精度和参考实现契约。

[Qwen 完整结果与逐前向记录](<.refactor/graph-device-fix-20261113/remote-evidence/qwen-forward-v1/qwen/result.json>)；[机器可读汇总](<.refactor/graph-device-fix-20261113/summary.json>)。

## 安全与复现边界

- 新远程命名空间 `graph-device-fix-20261113`；原基线验证目录没有重置或覆盖。依赖环境和模型资产只读复用，无安装、无权重覆盖、无凭证拷贝。
- 每次 GPU guard 要求 GPU2/3 两次间隔 31 秒的空闲观察，启动前复查，运行中每 10 秒监测，仅允许自己的进程组。未选择或终止 GPU0/1 的任务。
- 一次紧接 smoke 的矩阵启动因 GPU2 残留 1% utilization 被 guard 拒绝，**没有启动 GPU 计算**；之后以新证据名称重新完成空闲检查。没有削弱 guard。
- 最终校验 352 个候选源码文件与本地完全一致，原基线源码未变；144 个包版本未变；25 个模型资产文件重新校验 SHA256。GPU2/3 最终没有计算进程。见 [最终绑定](<.refactor/graph-device-fix-20261113/remote-evidence/final-binding.json>)。
- 不声称验证了 13B、三物理 GPU、RPC、模型质量、速度收益或通用采样分布等价。

本地无 GPU 接线回归：

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -B -m pytest -q test/test_graph_device_streams.py
```

完整 CPU 门禁（输出目录需为新目录，需已具备 golden 资产）：

```bash
.venv/bin/python -I -B tools/run_cpu_gate.py \
  --output-dir /path/to/new-evidence --require-golden-assets
```

原生回归需要操作员先确认两张授权空闲卡，并采用与此次相同的 idle/foreign-process guard；仅设置可见设备变量不是资源预留。

```bash
CUDA_VISIBLE_DEVICES=<authorized-uuid-1>,<authorized-uuid-2> \
DUODECODING_RUN_CUDA_TESTS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts= \
  test/test_graph_device_streams.py
```
