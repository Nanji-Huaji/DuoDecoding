# CU-HLM 首个小批次：迁出 `uncertainty_decoding` 与三项定向修正

本页记录阶段 B 第一个小批次的实施边界、先红后绿证据和未解决项。它不替代
`docs/refactor_stage_a_status.md` 的现行总状态，也不把本批结果说成整个 CU-HLM
协议已经验收。

## 1. 授权与写入边界

- 分支：`refactor/cuhlm-contracts-20261111`；基点：`58034cd`（阶段 A PR #31
  已合并）。
- 本批按 CPU 代码变更验收；远端 GPU 实验固定在阶段 A 合并版本上，单独记录。实现期间未安装依赖或下载模型。
- 实际写入：`src/baselines/cuhlm.py`（新增）、`src/baselines/core.py`（仅装配）、
  `src/baselines/__init__.py`（模块清单说明）、`src/communication.py`、`test/`
  中与 CUHLM 直接相关的测试和 `test/golden/logic/baselines.json`、本页。
- 未改指纹解析器、`sampling.py`、`verification.py`、核心数学工具、GPU 验收
  工具和历史 HTML。
- 原工作区两个未跟踪 HTML 报告保持原样。

## 2. 迁移：完整循环进入独立 mixin

`uncertainty_decoding` 的完整循环（含注册装饰器、CUDA Graph 接线、EOS 检查、
通信计费、拒绝重采样与 metrics 收尾）已从 `core.py` 实际搬到
[`src/baselines/cuhlm.py`](<../src/baselines/cuhlm.py>) 的 `CuhlmMixin`；
`core.py` 只保留导入和 MRO 装配：

```python
from .cuhlm import CuhlmMixin as _CuhlmMixin

class Baselines(_RuntimeResources, _AdaptiveDecodingMixin, _CuhlmMixin, ...):
```

- 注册名 `uncertainty_decoding`、别名 `cuhlm` 不变；`src.baselines` 的包级
  兼容导入和核心星号导出面不变，因此 `test/golden/baselines_exports.json`
  无需重冻。
- 循环仍是该算法自己的 `while`，没有并入 SD/DSD/DSSD 或 CEE-SD 的验证骨架，
  也没有新建转发 helper 或通用 flags round。
- 未迁移 `cee_cuhlm`、`_select_cuhlm_stage_config`、
  `_finalize_cuhlm_verification`；它们仍属于 CEE 组合，留待后续批次。
- 迁移后方法体除 docstring 分行外，只差三项定向修改（下述）；逻辑指纹仍为
  **91 events、`052491e9665476aa`**，但条目键从
  `Baselines.uncertainty_decoding` 改为 `CuhlmMixin.uncertainty_decoding`，
  fixture 的 `include` 新增 `src/baselines/cuhlm.py`。其余 14 个条目逐项计算后
  未变。指纹只是结构证据，不是数学或数据流等价证明。

## 3. 三项定向修正

### 3.1 构造 CUHLM 时透传真实词表大小

`uncertainty_decoding` 的 plain `CUHLM` 与 `PreciseCUHLM` 构造现在都显式传
`vocab_size=self.vocab_size`。此前默认 32000：Qwen 的 151936 词表会让
`_calculate_compressed_vocab_size` 进入“长度不匹配”保底分支并返回
`min(300, vocab_size // 100)`，即 k*=300，和论文式 (26) 无关。

定向测试用记录构造参数的 fake 断言两条分支都收到 151936；真实 k* 仍由
分布决定，不在本批重抽整个 CUHLM 数学库。

本批**只改 `uncertainty_decoding` 的两个构造点**。`cee_cuhlm` 的两个构造点
仍使用 32000 默认值；这是后续批次的问题，不能据此声称整个 CEE 已修复。

### 3.2 式 (26) 的 `x_d` 使用实际 draft token

论文式 (26) 的 `x_d` 是**本轮实际采样的 draft token** 在当前概率分布上的概率，
不是 top-1 概率的快捷替代。修改：

- `CUHLM.determine_transfer_strategy` 新增 `draft_token: Optional[int] = None`，
  并把它显式传给 `_calculate_compressed_vocab_size`；
- `uncertainty_decoding` 每轮先算
  `draft_token = int(x[0, -1].item())`，同一个 token 同时用于
  `calculate_uncertainty` 和 `determine_transfer_strategy`；
- 未提供 `draft_token` 的旧调用者保留 top-1 兼容 fallback。`terminal_prob`、
  `cee_cuhlm` 等并未在本批改为真实 token，不能声称全 CEE 路径已修复。

反例 V3（纯函数，`theta=0.25`）：

```text
probs = [0.5, 0.3, 0.2], u = 0.5, d = 1
x_d = 0.3  -> k* = 2
top-1=0.5  -> k* = 1
```

另有公开接口测试（阈值 0.4）使用默认 `theta=0.1` 的反例：
`probs=[0.85,0.05,0.10]`、`u=0.5` 时 top-1 给 k=1，显式
`draft_token=1` 给 k=2。真实 `uncertainty_decoding` 调用则用 fake 断言策略
收到了本轮采样 token（fake draft 返回 0），同时锁住不确定度与 k* 使用同一
token。

### 3.3 式 (10) 的触发边界改为严格大于

论文式 (10) 是 `u > threshold` 才传输。实现从 `>=` 改为 `>`：等号走机会跳过。
M=20 的分歧计数让 `16/20 = 0.8` 恰好可达，因此这个边界不是纯理论问题。定向
测试断言 `u == 0.8` 时 `determine_transfer_strategy` 返回 `(False, 0)`，
`u=0.81` 才触发。

该比较位于共享的 `CUHLM` 类，所以 `cee_cuhlm` 也受影响：等于阈值的轮次从
“触发”变为“跳过”。已有 CEE 测试没有在等号处下断言（使用确定性 0.0/0.55/0.6
等值或 fake），97 项 CEE 相关测试全部通过，因此没有为通过而删除或改写断言。
这是有意的论文语义对齐，本批仍不宣称 CEE 的其余协议已修。

## 4. 本批明确保留的行为

- 采样与拒绝路径不变：实际 proposal `q` 用于接受判定，拒绝残差仍来自压缩重构
  `x_hat`；机会跳过仍然存在。因此这条路径**不是**可自动声称 target 精确的路径，
  本批没有把它套进 SD/DSD/DSSD 的严格验证器。
- 显式 legacy 验证桥继续保留：`uncertainty_decoding` 仍调用
  `_prepare_legacy_verification_inputs`。本批只把
  `test_legacy_verification_bridge.py` 的白名单路径从
  `src/baselines/core.py::Baselines.uncertainty_decoding` 更新为
  `src/baselines/cuhlm.py::CuhlmMixin.uncertainty_decoding`，调用次数不变。
- 通信模式不变：首轮上下文上行、跳过零通信、触发一次压缩上行加一次 0 字节
  下行、排队只在触发时计、拒绝在服务器端基于 `x_hat` 重采样、`flush_round`
  结算顺序均原样保留。
- CUDA Graph 缓存键、EOS 早停、`max_tokens` 预算和 metrics 字段不变。
- 本页不重复“bonus 或截断 proposal 后整体仍无偏”的断言。该说法未在本批得到
  独立证明：CUHLM 用实际 `q` 接受、用压缩 `q_hat` 残差拒绝，且可跳过 target，
  这些条件不能由普通 SD 的无偏性论证直接推出。

## 5. 先红后绿记录

红例（实现前）：

- `test_comm_simulation.py::TestCuhlmCompressedVocab`：2 failed（等号仍触发；
  `determine_transfer_strategy` 不接受 `draft_token`），6 passed。
- `test_uncertainty_decoding.py`：3 failed（两个构造点没有 `vocab_size`，
  真实循环没有把采样 token 传给策略）。
- `test_baselines_package.py`：1 collection error（`src.baselines.cuhlm` 尚不存在）。
- `test_legacy_verification_bridge.py` + `test_legacy_graph_cache_identity.py`：
  1 failed、2 passed、42 errors（旧路径/旧模块目标）。

绿例（实现后，命令均为
`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider ...`）：

| 范围 | 结果 |
|---|---|
| `test_comm_simulation.py::TestCuhlmCompressedVocab` + `test_uncertainty_decoding.py` | 11 passed |
| `test_legacy_graph_cache_identity.py` | 42 passed |
| `test_baselines_package.py` + `test_legacy_verification_bridge.py` + `test_comm_config_reads.py` | 29 passed |
| `test_logic_fingerprint.py` + `test_logic_fingerprint_corpus.py` | 14 passed |
| `test_cee_refactor.py` + `test_adaptive_tridecoding_policies.py` + `test_cee_sd_contracts.py` + `test_cee_sd_orchestration.py` | 97 passed |
| `test_cuda_graph_wiring.py` + `test_cache_context_contracts.py` + `test_decoding_avg_topk_metrics.py` | 45 passed |
| `test_comm_simulation.py` + `test_unified_comm_accounting.py` + `test_protocol_spec.py` + `test_prob_bits_wiring.py` | 118 passed、70 subtests |
| `test_tk_slt.py` + `test_mode_features.py` + `test_mode_features_registry.py` + `test_legacy_modes.py` | 39 passed、15 subtests |
| `test_cli_surface.py` + `test_mode_features.py` + `test_mode_features_registry.py` | 19 passed |

以上是实现阶段的定向 CPU 测试。随后由 Lead 运行双模式完整 CPU 门禁：

| 模式 | 结果 | 耗时 |
|---|---|---|
| 默认导入 | 1385 passed、7 skipped、163 subtests | 97.81s |
| importlib | 1385 passed、7 skipped、163 subtests | 92.76s |

门禁记录位于 `.refactor/cuhlm-20261111/final-cpu/`，两次返回码均为 0，
`source_unchanged=true`。环境为 Python 3.10.18、PyTorch 2.6.0、
Transformers 4.57.6、pytest 9.1.1。GLM-5.3 完成一次只读批次终审，未发现新增缺陷。
这仍是 CPU 验收，不包含本批 CUHLM 的真实模型或 GPU 结果。

`src/baselines/cuhlm.py` 的 scoped Ruff 通过；`core.py`、`communication.py` 和
若干既有测试文件仍有本批之前就存在的 Ruff 告警，本批未做无关格式化。

## 6. 尚未解决的协议/数学问题（后续批次）

- `cee_cuhlm` 的两个 CUHLM 构造点仍未透传 `vocab_size`，且每轮
  `determine_transfer_strategy` 仍走 top-1 fallback；需要单独的 CEE 批次。
- 不采样前向、bonus/预算/EOS、严格窗口和近似分布协议未处理。当前循环的
  `n = prefix_len + 1 - 1` 也保持原样。
- `_calculate_compressed_vocab_size` 在长度不匹配时仍返回
  `min(300, vocab_size // 100)`；`denominator <= 0` 时仍返回硬编码 30。
  这些是保底路径，不是本批已证明的数学结论。
- 阈值等号语义现在对共享 `CUHLM` 的所有调用者生效；若后续需要按方法区分，
  应在调用方显式表达，而不是恢复 `>=`。
- 本批的指纹和 golden 更新只覆盖实际变动的入口/方法；没有整体重冻，也没有
  把结构指纹当作正确性证明。
