# 验证闸门：重构前后到底能断言什么

配套计划：`docs/refactor_baselines_plan.md`。本文只回答一个问题——
**改完之后，凭什么说行为没变。**

结论先写：

| 变换 | 能断言什么 | 手段 |
|---|---|---|
| 抽取（代码搬家，原地调用） | 逻辑足迹不变、AST 可回读 | `tools/logic_fingerprint.py` |
| 合并（11 份变 1 张参数表） | **不能证明**；只能枚举差异并逐条裁决 | 足迹 + 人工矩阵 |
| 数值（同等输入下结果） | 聚合指标逐位、token 序列逐位 | metrics golden + tokens golden |

`golden` 只对「历史跑过的那 80 条输入」负责。不同的 token 序列可能撞出
同一组聚合指标，所以本轮把观测粒度从 metrics 下沉到 `gen_ids`。

---

## 1. 逻辑足迹（`tools/logic_fingerprint.py`）

把方法体按源码顺序展开成事件序列：

```
叶子调用 · RNG 抽取 · comm 计费操作 · args 口径读取
self 赋值 · 循环/异常结构 · return/break/continue/raise
```

**局部 helper 会被原地展开**（不含 `return` 的纯语句块），所以：

```python
# before
run():  comm_simulator.transfer(1); sample(2)
# after
run():  self._step()
_step(): comm_simulator.transfer(1); sample(2)
```

两者足迹完全一致。而少一次计费、换一次 RNG、挪一个 `return`，都会立刻分叉。

```bash
# PR 前后对比（旧版本从 git 读）
python tools/logic_fingerprint.py diff src/baselines.py --base main

# 跨文件抽取：把新模块一起纳入展开索引
python tools/logic_fingerprint.py diff src/baselines.py --base main \
    --include src/decoding/context.py
```

冻结快照在 `test/golden/logic/*.json`，由
`test/test_logic_fingerprint.py` 在每次 pytest 里执行。
**任何足迹变化都必须同 PR 更新快照并说明理由**——这是评审动作，不是
自动放行。

边界（重要）：

- 只看「有没有/顺序」，**不看条件表达式内部与参数取值**。把
  `if self.args.post_round:` 改成 `if self.args.post_round and ...` 不一定
  触发；把 `transfer(a)` 改成 `transfer(b)` 不触发。
- 含 `return` 的 helper 不展开。把带 `return` 的语句块抽成函数会改变
  「return 退出谁」，属于语义变更，足迹会从 `return` 变成 `call` 被抓住。
- 跨文件同名定义视为歧义，不展开——宁可比出差异，也不要静默洞穿。

因此它是**必要条件检测器**，不是等价性证明。

## 2. token 级 golden（`test/golden/gammar/*/tokens.json`）

原始 run 的逐样本 `gen_ids` 直接从主树 `exp/` 收割，不需要 GPU：

```bash
python tools/harvest_tokens.py --exp /path/to/main/exp        # 收割
python tools/harvest_tokens.py --exp /path/to/main/exp --check # 只核对
```

重跑 golden 配置后逐样本对比：

```bash
python test/golden/compare.py --tokens llama_free_g8 \
    exp/gammar/<new_run>/adaptive_tridecoding_gsm8k.jsonl
```

第一条漂移会报成 `样本 i: token[pos]`，比“总耗时变了”可诊断得多。

边界：

- 身份是 run 内的**记录顺序**。`sample_idx` 在每个 task 内恒为 0，不能当键。
- 只覆盖 sample_seed=1234、gsm8k、temp=0 的 7 个 gammar 配置。
- 不覆盖 MT-Bench 等 temp>0 路径；那里只能统计等价。

## 3. 聚合 metrics（已有）

`test/golden/gammar/*/metrics.json` + `test/golden/compare.py`：
除 `TIMING_FIELDS`（墙钟测量）外全部字段精确相等，含
`comm_trace_edge_cloud` 的逐轮字节。校准双跑（2026-10-11）已裁决
LOCKED 字段逐位稳定。

```bash
python test/golden/compare.py llama_free_g8 <new_metrics.json>
```

边界：它锁的是**聚合口径**。token 序列变了但总数/接受数/字节恰好不变
的情形，只有 tokens golden 能抓住。

## 4. PR 纪律

1. 抽取类 PR：`logic_fingerprint diff --base <PR 基点>` 必须 PASS。
2. 口径变更类 PR：同一个 PR 里更新 `test/golden/logic/*.json` 或
   `tokens.json`，并在描述里写「为什么这个数字应该变」。
3. golden 重跑需要 GPU 空闲窗口；CPU 侧闸门（pytest 子集，约 1 秒）每次
   都跑，GPU 侧闸门按 Phase 节奏跑。
4. 解释不了的漂移 = bug，不是“噪声”。

## 5. 已知缺口

- 13 个解码方法里，只有 `adaptive_tridecoding` 有端到端 golden；其余
  12 个靠单测 + 逻辑足迹。补法：每个方法收一份小模型 golden
  （llama-68m / tiny-llama-1.1b，80 样本，temp=0）。
- 没有 round 级固件（计划 Phase 0.3）：`verify_and_commit_round` 动刀前
  必须补。
- 逻辑足迹看不见数据流（见 §1 边界），跨文件移动必须配合 `--include`。
- `src/model/lade/`、`src/model/rest/` 是第三方，不纳入任何冻结快照。

## 6. 两条命令记住

```bash
python tools/logic_fingerprint.py diff src/baselines.py --base main   # 逻辑
python test/golden/compare.py --tokens <tag> <new.jsonl>              # 决策
```
