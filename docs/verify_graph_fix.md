# 调度开销修复：定长 padding 验证图（2026-09-16）

本文记录对「CPU 侧逐算子调度占每轮 64%」的**已实施修复**。前置调查见
`docs/perf_investigation_log.md`（阶段四）与 `docs/graph_integration_status.md`。

## 先纠正旧结论：方案②「拆单步」对验证前向是负优化 ✗

阶段四写的修复方向 2（把多 token 前向拆成单步，让图路径只吃 `new_len==1`）
**只对草稿单步成立**。真实管线里每个模型每轮的主体是「上轮接受 token 的
resync + 本轮草稿」合成的**一个多 token 验证前向**：

| 前向 | 位置（baselines.py, adaptive_tridecoding） | k |
|---|---|---|
| 1.1B 验证 68M 提案 | `draft_model_cache.generate(x, 1)` | ≈ 接受+γ2 |
| 13B 验证 1.1B 提案 | `target_model_cache.generate(x, 1)` | ≈ 接受+γ1 |
| 各模型轮初 resync | `_generate_with_kvcache` 首步 | 1~2 |

拆成单步 = 每步整读一遍权重（内存墙地板）：

```
13B 验证 ~21 token：bf16 21×34ms = 714 ms（现整段 101 ms）✗✗ 7× 恶化
                    4bit 21×8.5ms = 178 ms（现 101 ms）✗ 仍恶化
1.1B 需要每个图档位显存 ~MB 级、单步 ~3ms × 17 = 51 ms ✗（现 ~35 ms）
```

## 正确修法：三件事（全部已实施）

### ① (1,K) 定长 padding 验证图（主体）

为每档尺寸 K 捕获一张 `(1,K)` 图：回放前把真实 k 个 token 拷进固定输入缓冲、
尾部 padding，回放后取**前 k 行** logits。安全性四条因果（详见
`src/graph_decode.py` 模块 docstring）：

1. causal 注意力 ⇒ 前 k 行与真实 k-token 前向一致（bf16 数值差 ~1%，与单步图同级）；
2. pad 行输出是垃圾但被切掉；
3. pad 行写入的 KV 槽位随回滚作废（rollback 本来就是"只移 nnz 指针"）；
4. StaticCache 零初始化 ⇒ 垃圾 KV 是有限值，不会 NaN 传染（且 pad 行不被读）。

档位自动推导：`{4,8,16,24,32,40,48,64} ∩ ≤ (γ1+γ2+4)`，`--graph_verify_sizes` 可覆盖。
回放时选 ≥k 的最小档（vLLM 式 padding-to-captured-size）。

### ② 图与 KV 缓存跨样本复用

原实现每样本重建 KVCacheModel ⇒ 图每样本重捕获（13B 实测 ~534 ms/样本，把
收益吃光）。`StaticLayer.reset()` 是**原地 zero_**（transformers 4.57 源码确认），
张量地址不变 ⇒ 图仍有效。现在三个缓存挂在 Baselines 实例上（`_adaptive_tridecoding_caches`），
样本间只 `reset_for_new_sample()`（清零 ~ms 级）。prompt 长度桶按 1024 取整、
1536 下限，超桶罕见触发重建+重捕获（有一次性打印）。

### ③ 去掉每前向的 `.item()` 同步

`_last_entropy` 原来每次前向 `float(...item())`（host 同步，打断 CPU/GPU 流水；
图回放后这就是新瓶颈）。改为存 GPU 张量，`last_entropy` property 惰性同步 ——
RL 每轮读一次 = 每轮 1 次同步，语义不变（仍是"最后一次前向的熵"）。

## 实施明细

| 文件 | 改动 |
|---|---|
| `src/graph_decode.py` | `GraphDecodeRunner`：verify 图（多档位）+ `begin_new_sequence()` 跨样本复用 + `verify()` 回放 |
| `src/model_gpu.py` | `_decode_step` 路由（k=1 单步图 / k>1 验证图 / 超档 eager 回退）；`_prefill_graph` 复用+重建；熵惰性同步；`reset_for_new_sample()` |
| `src/baselines.py` | `adaptive_tridecoding` 三缓存接线（含 target —— 验证图正是它的负载）+ 跨样本 stash |
| `src/utils.py` | `--graph_verify_sizes` 参数；`--use_cuda_graph` 帮助更新 |
| `scripts/test_graph_verify.py` | 等价性测试（新） |
| `scripts/bench_verify_graph.py` | 微基准（新） |
| `scripts/run_graph_ab_paired.sh` | `TAG_PREFIX`/`GATE_SKIP` 参数化 |
| `scripts/summarize_graph_ab.py` | 兼容 `vg_` 前缀 |
| `scripts/run_verify_graph_validation.sh` | 一站式验证：守门→冒烟→配对 A/B→微基准 |

`--use_cuda_graph` 仍默认关闭 ⇒ **eager 行为与历史逐位一致**，历史结果可复现 ✓。

## 等价性验证（已通过 ✓，`scripts/test_graph_verify.py`）

68M 与 1.1B（bf16，sdpa，temp 0）：

| 检查 | 68M | 1.1B |
|---|---|---|
| prefill | rel 0.00% | rel 0.00% |
| 图内档位 k∈{1..16}（含 padding、回滚 resync）| max rel 1.52% | 1.61% |
| 超档位 eager 回退（k=21）| 1.42% | 1.23% |
| 跨样本复用（无重捕获，capture_count 不变）| 1.24% | 0.95% |
| 超长 prompt 重建+重捕获（prompt≥max_len）| 1.31% | 1.45% |

（重建用例抓到并修复了一个真 bug：重建分支抬高 `max_length` 后，旧 prob/logits
缓冲因"未超新上限"被 `_ensure_buffer_size` 跳过扩容 ⇒ 写入越界。修法：重建时
丢弃历史缓冲按新桶重分配，且重建与初始建桶共用同一桶长公式。）

相对差异 ~1% = bf16 图回放 vs eager 的已知量级（历史实测 0.938%）。**不逐 token
恒等**（与旧 A/B 结论一致），准确率不受影响的验证靠配对 A/B。

**13B 4bit（bitsandbytes nf4）前置排雷 ✓**（`scripts/test_graph_verify_13b4bit.py`）：
bnb 的 Linear4bit 反量化可进图捕获；verify 图 k=17（pad→24）与 eager 整段
rel 0.82%，同输入两次回放 bit 级一致，hidden 有限。这是 A/B 前唯一未验证的
组件（bnb 核内有 host 分支的话捕获会直接失败）。

**长序列压力测试 ✓**（`scripts/stress_graph_verify.py`，300 次随机 k 前向 +
随机深度回滚，teacher-forced）：

| | 图 vs StaticCache-eager（主断言）| StaticCache vs Dynamic（信息）|
|---|---|---|
| 68M | max rel 2.13% | 2.33% |
| 1.1B | max rel 1.84% | 3.90% |

诊断结论（重要）：长上下文上图路径相对 **DynamicCache** eager 的漂移（实测可到
~4%）**全部来自 StaticCache 与 DynamicCache 的 kernel 数值路径差**（对照实验：
同一失败点上图=3.92% vs StaticCache-eager=3.90%，二者相同），图回放在
StaticCache 之上零额外误差。该模型级数值差不改准确率（历史 A/B 0.25=0.25）。
nnz 不变量 300 次全程保持、零重捕获。

## 预期与上限（诚实口径）

- 图直接攻击的是 232 ms/轮的调度开销里**最大头**（13B 验证 101 ms、1.1B 153 ms），
  理论上超过旧估计的 +25%（那个估计只修草稿单步回退）；
- 通信+排队 97 ms/轮（27%）**完全不受影响** ⇒ 接受长度仍是天花板（与阶段二/三
  结论一致）；
- 附带红利：进图后 13B 的地板从 34 ms（bf16）变为 8.5 ms（4bit）——量化在
  eager 下不省时间（launch-bound），在图回放下才兑现；
- 已知噪声源：bf16 数值漂移会偶发改变接受模式（前向次数/生成 token 数在两臂间
  可能不同 ≠ bug，`summarize_graph_ab.py` 的确定性告警会如实报出）。

## 端到端验证（已完成 ✓，2026-09-24 18:12-18:32 干净窗口）

守门 3.5h 后窗口放行（GPU1，第三方退出），自动链路跑完：冒烟 N=2（护栏 3/3）→
配对 A/B N=40 × 2 轮（`exp/vg_*`，`--use_cuda_graph` 唯一变量）→ 微基准。

### A/B 结果（GSM8K，temp 0，γ1=γ2=16，13B-4bit，MAXTOK=128）

| 指标 | eager | cuda graph | 判定 |
|---|---|---|---|
| **准确率** | 0.25 (10/40)，两轮一致 | **0.25 (10/40)，两轮一致** | ✓ 相等 |
| **吞吐（跨轮中位）** | 11.47 tok/s | **13.40 tok/s** | ✓ **1.17×**（达标 ≥1.15×）|
| 68M 计算时间 | 10.2 / 10.0 s | 5.9 / 6.2 s | 1.65× |
| 1.1B 计算时间 | 98.0 / 97.5 s | **45.9 / 46.1 s** | **2.13×** ✓✓ |
| 13B 计算时间 | 124.7 s | 121.4 / 121.5 s | 1.03× ✗（见下）|
| 确定性指标 | — | 生成 token +0.55%、前向次数 −1~4% | bf16 数值漂移的已知性质（同旧 A/B），非接线 bug |

### 归因（干净窗口微基准，`scripts/bench_verify_graph.py`）

| 模型 | 模拟轮 eager | 模拟轮 graph | 加速 | 结论 |
|---|---|---|---|---|
| 68M | 16.75 ms | 2.04 ms | **8.23×** | 纯 launch-bound，图全收 ✓ |
| 1.1B | 55.26 ms | 23.33 ms | **2.37×** | 调度占主体，图大头收走 ✓ |
| 13B 4bit | 235.78 ms | 207.89 ms | 1.13× | **bnb 4bit kernel 效率瓶颈，非调度** ✗ |
| 13B bf16 | 138.32 ms | 134.35 ms | 1.03× | 权重带宽瓶颈（26GB/768GB/s≈34ms 地板）✗ |

### 对原调查结论的重要修正 ✗→✓

「13B 验证 92% 是调度开销」**不成立**：13B 的验证前向是 **kernel/带宽瓶颈**
（bf16 贴 34ms 权重读取地板；4bit 的 bnb 反量化 kernel 低效到 8-10× 于其
6.5GB/8.5ms 地板），launch 间隙占比小 ⇒ 图回放救不了 13B。阶段三的线索
（4bit 与 bf16 eager 耗时几乎相同）正是这个 kernel 低效的表象。
**端到端 1.17× 的收益几乎全部来自 1.1B（2.13×）与 68M（1.65×）**。

### 后续路线（13B 的正确药方，超出本目标范围）

13B 验证要从 ~100 ms/轮 降下来，方向不是调度而是 kernel：
1. **换 4bit kernel**：AWQ/GPTQ + Marlin（vLLM/torchao）替代 bnb —— 若贴 8.5ms
   地板，13B 验证可到 ~15ms/轮（现 100ms）；
2. 或 **bf16 + 图**（26GB 显存放得下 A6000）：单次前向 ~40ms 级（权重地板），
   仍优于现 4bit-100ms —— 但被 1 限幅，两者上限一致；
3. 通信+排队 97 ms/轮 仍不受任何计算优化影响 ⇒ 接受长度仍是天花板（不变 ✓）。

---

## 第二轮优化（vg2）：前后处理进图 + 桶收紧（2026-09-25）

图回放收走模型内部调度后，小模型每次前向仍有 ~15-18 个回放前后的 Python/torch
op。本轮把**定长前后处理也编进图**，并顺带修掉采样热路径的 host 同步。**但隔离
测量推翻了动机的一半 —— 诚实记录如下。**

### 实施内容

1. **前后处理进图**（`graph_decode.py`）：回放前 host 只剩 2 个 op（拷 k 个
   token、填 `start_pos` 标量）；`cache_position = start_pos + range_buf[:K]`、
   `mask = range_buf < start_pos + K`、每行熵、temp-0 的 argmax one-hot、
   logits/prob 历史写入（`index_copy_` 按标量行索引）全部在图内。
   pad 行写到 `current_seq_len` 之外，读端不可见（与 KV 槽同一套回滚语义）。
   开启条件 temp==0（temp>0 的 top-k/top-p 分支保守留在图外）。
2. **`sample()` 去同步**（`utils.py`）：`invalid_rows.any()`（每次采样一次 host
   同步，打断 CPU/GPU 流水线）→ `torch.where` 无分支化，语义/RNG 逐位一致。
3. **桶收紧**：`_graph_bucket_len` 从 1024 粒度 + 1536 下限 → **256 粒度 + 512
   下限**（动机见下）。
4. **新护栏**：图模式 eager 回退超出 StaticCache 容量时显式 RuntimeError（原来
   是静默显存越界，被旧的 1536 下限掩盖；收紧后由压测炸出）。

### 隔离测量的三个事实（修正先前判断）

```
1.1B k=17 @pos96（图回放态）:
  裸 replay            8.01 ms   ← 全部时间在图内 GPU 执行
  runner.verify 全流程 8.05 ms   ← 输入拷贝+熵均值 wrapper ≈ 0.04ms
  _decode_step 图路径   8.08 ms   ← 全路径 wrapper 总共 ≈ 0.07ms
⇒ 事实①：CPU 派发早已与 GPU 异步重叠，「前后处理进图」单独计收益 ≈ 0
  （此前预估 +6~10% 是错的：0.68ms/前向 里 0.5ms 是图内小 kernel 的
   GPU 执行地板 ~3-6µs/个，不是 host 派发）。

桶宽 vs 裸 replay（StaticCache+SDPA 的 attention 按桶宽计算，与 pos 无关）:
  bucket 512 → 6.42 ms；1024 → 7.24 ms；2048 → 8.81 ms（≈ +1.2µs/key）
⇒ 事实②：桶宽是真杠杆 —— 这才是本轮收益来源。
```

### 微基准（干净窗口，模拟轮 = verify k=17 + resync + step）

| 模型 | eager | 图（vg，旧桶）| 图（vg2，紧桶）| 加速 |
|---|---|---|---|---|
| 68M | 16.9 ms | 2.04 ms | **1.64 ms** | 10.35× |
| 1.1B | 56.7 ms | 23.33 ms | **18.38 ms** | 3.08× |

（前后处理进图贡献 ~1-3%；主要来自桶收紧。）

### 端到端验证（`exp/vg2_*`，配对 A/B N=40×2，守门放行后跑）

| 指标 | eager | cuda graph（vg2）| 判定 |
|---|---|---|---|
| **准确率** | 0.25（两轮一致）| **0.25（两轮一致）** | ✓ 相等 |
| **吞吐（跨轮中位）** | 11.41 tok/s | **13.81 tok/s** | ✓ **1.21×**（vg 一轮为 1.17×）|
| 1.1B 列 | 99.2 / 100.1 s | **30.7 / 30.8 s** | 3.2× ✓（紧桶 + 图）|
| 68M 列 | 10.2 / 10.4 s | 5.0 / 5.4 s | 2× ✓ |
| 13B 列 | 84.4 / 85.6 s | 7.5 s | **计时归属位移，非真实 16×**（见下）|

**重要解读（诚实口径）**：`sample()` 去同步后前向异步化，「各模型秒数」列
不再等价于 GPU 占用 —— 13B 列从 84→7.5s（eager 臂也 124.7→84.4s）主要是
逐采样 host 同步消除后的**计时归属变化**（前向调用在 GPU 完成前返回，GPU 工作
与通信记账/其它 CPU 工作重叠）。**墙钟吞吐才是真话：11.41→13.81 = 1.21×**，
其中对 vg 一轮（1.17×）的增量 +3.5% 来自 紧桶 + sample 去同步 + 前后处理进图
（微基准里 1.1B 模拟轮 23.33→18.38ms 的主要成分是紧桶）。
确定性列仍显示漂移（bf16 已知性质），准确率不受影响 ✓。

### 修掉的一个基础设施 bug

`run_verify_graph_validation.sh` 守门调用 `wait_gpu_idle.py --timeout 45`，但该
脚本默认需 4 次连续达标、采样间隔 ~15s+查询耗时（实测节奏 24s/次）⇒ 数学上
不可能在 45s 内放行（昨天 18:12 的放行属侥幸节奏）。改为
`--samples 2 --interval 10`：对已空闲的卡 ~30s 确认即放行。
