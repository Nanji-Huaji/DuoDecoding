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
# S2 后：当前实现已包化，main 上仍是旧文件；映射只作用于旧侧入口
.venv/bin/python tools/logic_fingerprint.py diff src/baselines/core.py \
    --base main --base-path src/baselines.py \
    --include src/baselines_helpers.py

# 若比较基点也已包化，省略 --base-path；后续跨文件抽取追加相应 --include。
```

冻结快照在 `test/golden/logic/*.json`，由
`test/test_logic_fingerprint.py` 在每次 pytest 里执行。
**任何足迹变化都必须同 PR 更新快照并说明理由**——这是评审动作，不是
自动放行。

2026-11-02 的 `adaptive_tridecoding` 诊断抽取按同一规则处理，**不新增
helper 独立冻结条目**：两个新 helper 以「原地展开」进入调用方
`Baselines.adaptive_tridecoding` 的既有冻结条目，
`test/golden/logic/baselines.json` 的条目集合不变（当前 15 条）。
`diff` 输出里形如 `+ 新增入口 Baselines._trace_*` 的行只是信息性的展开
索引，不单独冻结；只有调用方的事件序列真的变了，才需要评审快照。

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
- **Phase 0.3 的 round 级固件仍未建立**：`verify_and_commit_round` 这个
  函数目前还不存在（计划里 0.3 留到 Phase 3 随函数成形再建）。2026-11-02 的
  WIP 加的是**行为级轮门禁**，不是那个固件：
  - 位置：`test/test_adaptive_tridecoding_rounds.py`（工作树新文件，未提交）；
  - 接缝：只走公开入口 `Baselines.adaptive_tridecoding`，使用真实
    `resolve_stage_verification`、真实 `CommunicationSimulator`（测试子类只在
    `simulate_transfer` 记录请求账、`_charge_transfer` 记录结算账，实际计费仍
    由父类完成）；KV 缓存是脚本化
    fake，但保留 `prob_history` 的行语义与 rollback 截断；
  - 覆盖：`per_transfer` + `charge_residual_payload=False`，以及默认
    `per_round` + `charge_residual_payload=True`，两种口径都覆盖三轮全接受
    和两阶段都拒绝；断言 token、指标、rollback、有序请求/结算账及 flush。
    默认口径额外核对残差消息字节、确定性的通信时间，以及历史上
    `wall_time` 在末轮 `flush_round` 前读取的顺序（本批不修正该口径）；
    诊断 JSONL 用例覆盖
    两轮非对称行、true logits、rebuilt 降级和不可写路径；
  - 边界：它锁的是单方法在脚本化输入下的 token/metrics/逐轮计费，既不能
    代替 Phase 0.3 的「固定张量 → 精确输出」固件，也不构成对
    `verify_and_commit_round` 任何拆分的背书；没有远端 GPU 验收。
- 逻辑足迹看不见数据流（见 §1 边界），跨文件移动必须配合 `--include`。
- `src/model/lade/`、`src/model/rest/` 是第三方，不纳入任何冻结快照。

## 6. 两条命令记住

```bash
.venv/bin/python tools/logic_fingerprint.py diff src/baselines/core.py \
    --base main --base-path src/baselines.py \
    --include src/baselines_helpers.py                              # 逻辑
python test/golden/compare.py --tokens <tag> <new.jsonl>              # 决策
```

## 7. 2026-11-02 WIP 快照：`adaptive_tridecoding` 诊断抽取能断言什么

工作树（基于 `main@50c43ba`，未提交）把该方法里的三段纯诊断代码原位抽成
`_trace_adaptive_cache_preverify` / `_trace_adaptive_calibration`
（AST：方法跨度 1,062 → 944 行，helper 38 + 114 行）。三个开关
`CACHE_CHECK_TRACE` / `ARP_CALIB_TRACE` / `TOPK_TRACE` 的读写点与门控
原样保留；helper 不读 `self.args`，也不调用 `transfer` /
`simulate_transfer` / `sample`，所以内省、计费、采样三条口径都没有变化。

| 变换 | 当前能断言 | 依据 |
|---|---|---|
| 诊断段原位抽取 | 调用方足迹一致、AST 可回读 | 指纹 `diff --base main` 判定 PASS（helper 展开进调用方条目） |
| 内省/计费/采样不变 | 结构上未触碰 | helper 体内无 args / 计费 / RNG 调用 |
| 轮级行为 | 仅脚本化场景：三轮全接受、两阶段拒绝 | `test_..._rounds.py`（新文件，尚未提交） |
| 诊断 JSONL | 两轮非对称行、true/rebuilt、追加与路径失败 | `test/test_adaptive_tridecoding.py`；交换缓存、错行、覆盖写四种内存变异均被断言捕获 |

仍**不能**断言：真实模型/真实分布下的轮级行为、
`verify_and_commit_round`（尚不存在）的等价性、gammar golden 的端到端数值。
本节保留诊断抽取阶段快照；续做的策略抽取与门禁结果见 §8，没有 GPU 侧结论。

## 8. 2026-11-02 续做：主级策略连续块与扩展门禁

本节保留主级策略抽取阶段的快照，最新门禁见 §9。

在 §7 的诊断抽取之上，原位提取 `_select_adaptive_main_config`（46 行），
`adaptive_tridecoding` 944 → 921 行。只返回应用 cap 后的 `transfer_top_k`，
保留 `select_config → cap → ARP threshold → _next_gamma1` 顺序；外层条件、
计时采样点、reward/step、两级验证与 EOS/flush 不移动。详见
`docs/baselines_split_map.md` §11。

| 门禁 | 本次结果与范围 |
|---|---|
| 全量 CPU pytest | `564 passed, 7 skipped, 88 subtests passed`（2 个既有 warning） |
| main 原实现对照 | 内存编译 `main:src/baselines.py` 的原方法，去注册装饰器后临时替换入口；同一组 adaptive/round/policy 测试 `27 passed, 3 subtests passed`，不改磁盘源码 |
| importlib 收集 | 全量以 `--import-mode=importlib` 运行，同样 `564 passed, 7 skipped, 88 subtests passed`；`test/conftest.py` 显式加入测试目录以支持共享 fake 的兄弟模块导入 |
| 逻辑指纹 | `diff src/baselines.py --base main` PASS，调用方 308 事件一致；三个 helper 原地展开，冻结 fixture 未改 |
| 严格 AST 回读 | 三个 helper 去 docstring/尾返回，并按实参重命名后展开；整个方法 body（条件、实参、表达式）与 main 完全相同，不只比较调用名 |
| 协议内省 | 四个通信开关及 `gamma1/gamma2` 保留，注册别名/no-ARP/opportunistic wrapper 的消费集不变 |

新增/扩展的行为门禁只走公开入口：

- `test/test_adaptive_tridecoding_rounds.py`：10 个用例，覆盖三轮全接受、
  两级拒绝、`per_round+residual`、`per_transfer+residual`、两层 EOS 各自的
  结算位置，以及“前一轮拒绝 → 下一轮第一层 EOS”的跨轮结算和缓存截断效果。
  全部使用真实验证与通信计费，KV 概率行脚本化。
- `test/test_adaptive_tridecoding_policies.py`：7 个用例，锁 little top-k
  不泄漏；主 top-k 使用不同动作序列（初始 2 → 第一轮选 3 → 第二轮选 6
  后 cap 到 4），区分旧 proposal 与本轮新结算值；gamma 下一轮才生效，
  `gamma_dim=1` 不覆盖；ARP 多步 predict 使用旧阈值；`disable_rl_update`
  不抑制 select/reward、只抑制 step/save；机会门控同轮复用 RL 新阈值，
  skip/transfer 都仍走真实第二层验证。完整记录并断言 RL 状态六字段：
  bandwidth、latency、ARP acc_probs、entropy、task、training；使用不同链路
  带宽/NTT、非零不同熵、非空不同 ARP 概率与 task 哨兵，防止参数错位。
- `test/test_adaptive_tridecoding.py` 共 10 个用例，其中 4 个诊断场景覆盖
  JSONL 两轮追加、非对称行、true/rebuilt、不可写路径，以及三个开关分别单开
  （3 个 subtests）。单开时核对两轮内容、另两个文件不存在、外部环境恢复，
  明确锁定 TOPK_TRACE 不依赖 ARP_CALIB_TRACE，其余为既有行为/wrapper 测试。

子代理分别对残差/rollback/EOS/set_round，以及 top-k 泄漏、gamma 覆盖、
机会阈值复用、更新开关与本轮结算做临时变异检查；这些定向变异均被测试捕获。
另补查新 helper 返回值丢失/未 cap、RL 特征错位与 TOPK/ARP 门控错误耦合，
相应定向变异也被公开入口断言捕获。这不是系统化 mutation coverage，也不是
GPU/gammar golden 的替代品。

独立审阅使用 `byr-llm-gw / deepseek-v4.1-flash`，分别覆盖生产缺陷、测试
可靠性、方案与文档。生产与方案维度无 findings；测试维度指出的状态特征
转发、诊断单开组合、importlib 收集三项已补强，再跑上述全量门禁；原审阅者
增量复审确认三项全部关闭，无新增 finding。

本次保留的边界：真实模型/真实 ARP/RL 权重、训练数值轨迹、CUDA Graph
跨样本复用、EOS 在块中间及其与 max_tokens 截断的组合尚未新增验证。
策略替身只记录动作/调用顺序，不验证真实 reward 数值或团队奖励。
函数调用开销仍进入原有 wall-mode `compute_s` 测量区间；不宣称时间数值逐位相同。
Phase 0.3 的 `verify_and_commit_round` 固件仍不存在；不做 S2/S3，未提交。

## 9. 2026-11-02 第三批：little 策略与拒绝概率载荷共同段

本节是包化前快照；S2 最新布局和门禁见 §10。

本批只提取 `_select_adaptive_little_config` 与 `_adaptive_rejection_prob_bytes`，
方法 921 → 888 行；不移动真实验证、概率行选择、残差消息、计时、reward、
EOS、`simulate_transfer` 或 flush 点位。不修改任何逻辑指纹 fixture。

| 门禁 | 结果 |
|---|---|
| 默认 pytest / 全量 importlib pytest | 两者均 `569 passed, 7 skipped, 90 subtests passed`，2 个既有 warning |
| main 原方法对照 | 同组三个 adaptive 文件 `32 passed, 5 subtests passed` |
| 足迹 / 完整 AST 回读 | 单文件调用方 308 事件一致，diff PASS；五 helper、六调用点按真实实参/尾返回展开后方法 body 与 main 完全相同 |
| 协议内省 | 原方法与现方法的四个通信键、`gamma1/gamma2` code-object 消费集一致；五个入口/别名查询一致 |

AST 比对固定 Python 3.10.18（`.venv/bin/python`），归一化规则见拆分地图
§11；本批 SHA256 仍是 `9632a2c687de9d92aaa22940d4ee2a7638c9db350e1bccaa02c4081a60189932`。
不同 Python 版本的 AST 序列化字节串可能不同，不直接比较跨版本哈希。

### 新门禁能断言什么

- policies 7 → **10** 个用例：little `gamma_dim>1` 时首轮用 args.gamma2、
  次轮用上一轮动作、结束状态留下新动作；little top-k 仍不泄漏到提案与计费；
  static 即使挂了策略/ARP 也不调用；`last_gamma=None/0` 回落到本轮深度。
  已有的六字段状态转发、阈值与训练开关断言继续生效。
- rounds 10 → **12** 个用例：新增真实量化/真实验证/真实计费的 bits=8 两层
  拒绝账（k=2 的合并拒绝各 20 B），以及 top-k=None/0 的整行计费（各 22 B）。
  同一脚本切到 bits=16，第二层从拒绝变接受，证明不只改了计费数字。
  测试运行器每次重建惰性的 CommConfig，确保同实例位宽对照确实读取新配置。
- 负向检查：忽略上一轮 gamma、让新 gamma 同轮生效、漏掉 None/0 fallback，
  以及漏 bits 覆盖、强制全宽、只计 bits 不量化等定向变异均被公开入口断言捕获。

生产与测试由 `byr-llm-gw / deepseek-v4.1-flash` 两名独立审阅者复核，
均为 No findings；分别重做了实际实参驱动的 AST 回读、量化非原地检查、
live-out 核对及公开入口定向变异，未修改生产/测试代码。

### 不扩大的结论

新增位宽测试是 `per_transfer`、不补残差消息的口径；不声称覆盖低位宽与
`per_round/charge_residual/EOS` 的组合，或 bits=4 与 `topk>=vocab` 的完整入口矩阵。
真实 ARP/RL 权重、真实 reward 数值/团队奖励、CUDA Graph 复用、训练数值轨迹、
GPU/gammar golden 仍未验收；一次函数调用开销仍进入原 wall-time 奖励区间。
该批结束时完整 layer/`verify_and_commit_round` 未建立，未做 S2/S3、未提交。

## 10. 2026-11-02 S2：最小兼容包化的验收范围

比较对象为包化前的 **5,825 行 WIP** 与 `src/baselines/core.py`，不是把原先
adaptive 抽取的改动混算为 S2。core 仍 5,825 行，adaptive 仍 888 行：
只有 20 个 relative import 各加一级，逐字归一化和完整模块 AST 对照相同。
`src/baselines/__init__.py` 是 eager 导入兼容层；S3/算法/方法体不动。

| 最终验证 | 结果 |
|---|---|
| 默认 / importlib 全量 pytest | 均 `587 passed, 7 skipped, 90 subtests passed`，2 个既有 warning |
| 改名路径指纹比较 | `core.py --base main --base-path src/baselines.py --include src/baselines_helpers.py` PASS（68 个入口） |
| 冻结逻辑 fixture | 仅 path 变化；原 15 条 entries、include、rev 均不变 |
| S2 原样迁移 | 5,825 → 5,825 行；20 个 relative import +1，其余文本与完整模块 AST 相同 |

本批新增 9 个包契约用例、8 个指纹 CLI 用例和 1 个未暂存迁移扫描用例。

### 兼容面与持久门禁

- `test/golden/baselines_exports.json` 从**包化前快照**生成，冻结 105 个可读名字
  与 79 个星号名字；`test_baselines_package` 按此清单核对 `getattr`、`dir`、
  显式/下划线导入、星号导出及对象身份，不使用当前 core 的名字集合自证。
- 冻结 24 个注册名及方法映射，三个独立冷启动顺序验证注册完整性与别名同一对象；
  reload 入口不重复注册，旧全局 pickle 名与新 pickle 往返均验证。
- patch 只承诺真实定义模块 `src.baselines.core.X`，不承诺旧 package globals
  写透。已有依赖 patch 迁至 core，公开导入仍走兼容入口；真实调用测试验证
  经旧入口导入的 helper 的 `__globals__` 属于 core，patch 实际生效。
- 四个源码门禁改扫 core，逻辑 fixture **只改 path**；原 15 条 entries、
  include 和 rev 未变化，未重新生成冻结值来放行。
- 全仓 CLI call-site 扫描同时包含已跟踪与未忽略新 .py/.sh，并过滤工作区
  已删除路径；真实临时 git 仓库用例锁住“未暂存旧文件删除/新 core 新增”。
  不以暂存文件规避失败，不扩大到 .venv/vendored/被忽略产物。
- `logic_fingerprint diff --base-path` 只替换旧侧入口路径，include 仍分别从
  git base/工作区读取。CLI 测试覆盖默认同路径、改名、行为漂移、入口与修订
  错误、无同名入口拒绝，以及**只改 include 文件**必须 FAIL，防止旧侧误读工作区。

### 预期变化，不冒充逐字节不变

`Baselines.__module__` / 新 pickle 定义路径是 `src.baselines.core`，包 `__file__`
指向 `__init__.py`；历史显式导入仍支持，但不保证包的 `vars()` / `__dict__`
列出虚拟导出名。`COMM_TRACE.caller` 的 basename 从 **baselines.py → core.py**，
行号及调用栈深度不变；trace 分析器按 caller 分组，跨版本比较应归一化该标签。
不改 `communication.py` 伪造旧文件名，也不宣称 trace 原文全等。

生产/导入面与测试/工具分别由 `byr-llm-gw / deepseek-v4.1-flash` 独立审阅。
初审提出的旧侧 include 来源缺口、过期门禁命令、trace basename 说明三项 P3
均已修复；原审阅者增量复核确认全部关闭，无新增 finding。

本批未跑 GPU/gammar，不新增真实权重/reward/CUDA Graph 的结论；算法等价性
依赖归一化文本/AST、既有行为门禁及导入/注册/patch 契约，不以“包化成功”代替
真实模型端到端数值验收。S3 未开始；未暂存、未提交。
