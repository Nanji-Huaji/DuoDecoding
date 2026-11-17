# CEE-CUHLM 阶段拆分与 A6000 验证（2026-11-15）

> 历史记录：本文验证的是保留旧协议的结构重构。2026-11-16 已另行按独立 CUHLM 修改外层协议；当前行为和验证见 [协议对齐说明](<cee_cuhlm_protocol_alignment_20261116.md>)。本文的旧输出对比不验证新协议。

基点：`850d75e125dadd730a0242d49ef96f665c54ec41`。生产改动限于 [cee_cuhlm.py](<../src/baselines/cee_cuhlm.py>)，继续保留 [上一轮说明](<cee_cuhlm_refactor_20261114.md>)中的旧近似协议、NTT 传参与最终 EOS/缓存端点修复。

## 拆分结果

公开方法 `cee_cuhlm` 从 **548 → 285 行**（AST `def..end`，含注释、不含装饰器），文件从 640 → 760 行。签名、注册、模块级 patch 目标不变。

| 私有阶段 | 职责 |
|---|---|
| `_build_cuhlm_comm_simulator` | precise/plain 构造、显式 NTT 参数与 plain 阈值 |
| `_acquire_cuhlm_caches` | 三缓存获取、Graph 身份和词表配置 |
| `_run_little_to_draft_stage` | Little 提案、上行计费、原始 Draft 概率上的内层拒绝采样 |
| `_scan_cuhlm_uncertainty` | 外层 skip/trigger 扫描与首个触发的上行计费 |
| `_finalize_cuhlm_metrics` | 计数、通信账和既有 checkpoint 守卫 |

13 个累计值改用局部 `stats` 字典，避免大量参数往返；没有新公共接口或运行上下文框架。外层 trigger/bonus、rollback、下行、预算与 EOS 收尾保留在入口，不强行抽出高耦合的小片段。仅删除两处未读的 `None` 初始化。

旧协议仍是：Little→Draft 按 raw Draft 验证；carried/fresh 行来源不同；外层低不确定度跳过、首个触发强制残差，并不执行独立 CUHLM 的 target 接受判定。历史默认设备 RNG 和丢弃样本也保留。本次不是概率协议升级。

## CPU 与只读审查

- 实现与实验由 `byr-llm-gw/glm-5.3` 完成。[公开入口 CPU 契约](<../test/test_cee_cuhlm_contracts.py>)在改前/改后均 **30 passed**；相关守卫初跑 81 passed / 70 subtests，另有一个 legacy bridge 作用域坐标失败。
- `deepseek-v4.1-flash` 独立只读审查未发现运行行为回归；确认上述守卫需要随抽取更新。
- Lead 已将 [legacy bridge 台账](<../test/test_legacy_verification_bridge.py>)从入口两处改为入口一处、内层 helper 一处；总调用数仍为二，不放宽允许消费者。
- [完整语料逻辑指纹](<../test/golden/logic/baselines.json>)仍为 **139 events / `9d843c8a8576d6c1`**，五个尾返回 helper 会原地展开。指纹只是辅证，不代替行为测试。`gamma1/gamma2` 读取留在入口，保留协议内省。

## 真实 CUDA 前后对比

机器：`4xA6000-tiantianyi`（`10.160.8.194`），物理 GPU 2，单卡。实验前四卡仅有轻量 Xorg 占用。复用现有 Python 3.10.12 / torch 2.6.0+cu124 / transformers 4.57.6；没有安装依赖或修改驱动、旧 checkout、权重。

远端目录为 `~/DuoDecoding-refactor-20261115-cee-cuhlm`。BEFORE 来自基点归档；AFTER 仅替换本模块，完整 SHA256 为 `e7a12dcae5feea909feacd75675bce860d6a7fcf215370549d781d487ebd85b3`，与 Lead 审查的最终源码一致。

**41 个独立配置，72 组前后配对**：tiny 31 个配置先单跑一次，再在含 smoke 的 41 配置运行中复跑。因此不能把 72 解释为 72 个不同场景。

- tiny Llama FP32/BF16 × eager/CUDA Graph；skip、trigger、混合阈值及多轮运行。
- 预算 0/1/3/6/12、温度 0/0.7、precise/plain、NTT 40/7ms、零深度、top-k 4/300/None。
- 首 token/中段最终 EOS、prompt EOS、disable_eos_stop、同 owner 三样本 Graph 复用。
- 10 个真实权重 smoke 配置：缓存的 JackFram/llama-68m，FP32/BF16、eager/Graph；其中一例用 luvres/Llama-3-187M-Instruct 作 target，检查不同词表大小下的调用链。不是质量评测。

Lead 重新读取四份 JSON 并逐例比较：**72/72 无非计时差异**。输出 tokens、计数、通信指标（字节/能耗/连接数/模拟时延）、CPU/CUDA RNG 摘要一致。

证据边界：

- 生成脚本先排除总 wall/computation time、throughput、checkpoint overhead 等壁钟字段；比较时另排除 little/draft/target 的六个 wall/computation 字段。不是“仅六个字段未比较”，也不由此宣称性能无变化。
- 缓存用弱引用记录：25 条记录含调用后仍存活的缓存端点/概率与 logits 摘要，这些摘要一致；已销毁的 eager 缓存记录为空，不能声称每个 GPU 用例都检查了三缓存端点。CPU 合同另有真实缓存端点断言。
- 单卡实验未覆盖多 GPU 拓扑、RL 更新/checkpoint 路径或大模型质量基准。

原始证据（本地忽略目录，不随 git 提交）：[实验脚本](<../.refactor/cee-cuhlm-20261115/gpu-validation/gpu_cee_cuhlm_compare.py>)、[tiny 前](<../.refactor/cee-cuhlm-20261115/gpu-validation/before-tiny.json>) / [后](<../.refactor/cee-cuhlm-20261115/gpu-validation/after-tiny.json>)、[含 smoke 前](<../.refactor/cee-cuhlm-20261115/gpu-validation/before-smoke.json>) / [后](<../.refactor/cee-cuhlm-20261115/gpu-validation/after-smoke.json>)。实验代理记录峰值显存约 2.9 GiB，结束后卡 2 恢复 10 MiB / 0%。

最终集成测试见 [baseline 汇总](<baselines_refactor_20261115.md>)。
