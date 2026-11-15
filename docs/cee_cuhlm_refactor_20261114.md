# CEE-CUHLM 独立迁移与边界修复

## 范围

基线为 `8854ebf5788d0ca953b8b5a53900cd77749f8589` 加上此前已验证的 Graph stream 修复。此次不改 Graph 实现、不调整 BF16 容差，也不改评测层或 KV cache 内部。

- 注册名 `cee_cuhlm`、调用签名和 `Baselines.cee_cuhlm` 保持可用；真实定义移至 [cee_cuhlm.py](../src/baselines/cee_cuhlm.py)。
- [core.py](../src/baselines/core.py) 保留组合入口和 `_finalize_cuhlm_verification` 兼容导出；共享的阶段配置选择仍供 CEE-SD 使用。
- 先做逐字方法体迁移，后单独修复 precise NTT 接线和最终 EOS／预算裁剪。没有引入统一解码循环。

## 保留的旧概率契约，不代表正确性认证

这一模式既不是“两层独立 CUHLM”，也没有采用 [wire-draft-v1](cee_cascade_20261111.md)。

1. 三个模型缓存仍使用 `top_k=top_p=0`；传输 top-k 只影响后续均匀尾重构，不能用通用缓存构造器替换这些有效配置。
2. Little 候选按旧 top-k 均匀尾重构分布生成，第一层按 raw Draft 概率接受／纠正；Little uncertainty threshold 在此不作门控。
3. 外层 carried token 使用 raw Draft 历史行，fresh token 使用重构行，两者来源不同。不得将 carried token 自动改标为 `Q_D`。
4. 外层先跳过低不确定性候选；第一次触发即强制采样 `(P-Q)+`，没有 Target 接受判定，也没有把选定的 k* 编码实际用于残差。`P=Q` 时旧残差零质量回退到 token 0 仍存在。这是待协商的算法问题，不是本次“修好了”的内容。
5. 真实缓存历史比已采样序列少一行，因此 all-skip 路径未门控最后一个候选，而是丢弃后从同一行再采“bonus”。两层验证 `generate(..., 1)` 也仍各有丢弃的抽样。`gamma1=gamma2=0` 仍保留旧的无门控 Draft continuation。
6. 候选、接受均匀数、残差、bonus 和 uncertainty probes 仍消耗默认设备 RNG。未改为私有 RNG，也未用无采样前向替换验证调用。

以上行为由公开注册入口上的真实 CPU Llama／KV cache 测试记录，不是目标模型分布等价证明。下一批若要对齐独立 CUHLM，必须先明确 trigger 接受规则、原始 q 与重构 qhat 的用途、all-skip 尾部及 RNG 策略。

## 两项独立行为修正

### Precise NTT

旧入口遗漏 `ntt_ms_edge_cloud` 和 `ntt_ms_edge_end`，导致 precise 模式无论默认或显式参数都使用模拟器自身的 200/20ms 默认值。现在把入口参数明确传入。

公开计费回归覆盖 plain／precise：一个全跳过轮有一次 cloud prompt 和四次 edge-end 报文（含两个 0B 响应）；cloud NTT 加 13ms、edge-end 加 7ms，通信时间应增加 **41ms**。修复前 precise 增量为 0，plain 对照正常。

Precise 与 stochastic 同时指定时仍保留 precise 优先的旧行为；此项没有暗改。

### 最终 EOS 与缓存端点

最后一个块达到预算时不会再进入下一轮，旧代码因而可能返回确认 EOS 之后的 token。最终输出需要同时满足预算和生成段首个 EOS（包含 EOS），并裁掉所有缓存中超出最终上下文的推测后缀；最后一个输出 token 保持未前向状态，不延长落后的缓存。

只检查确认后的输出：prompt 中的 EOS、被 Draft 拒绝的 Little EOS 不应触发停止。`disable_eos_stop` 继续生效。

仍按整块计算／抽样后裁剪；本次没有改变短预算的候选数量或尾部概率律。生成计数与返回输出对齐，accepted／forward／通信计数仍描述实际完成的工作，可能包含最终裁剪掉的推测工作。

## 兼容边界

- `args.max_tokens` 仍为新增 token 数；显式 `max_tokens` 仍为绝对序列端点。尚未统一，也未新增输入／预算合法性策略。
- 零预算仍构造缓存并收费一次 edge-end prompt；Graph 缓存复用／reset 与有效参数身份保留。
- 通信仍为 per-transfer、paper payload 成本假设；0B 报文仍收 NTT，round ID 仍为空。所谓 8-bit 概率计费不是已实现的 8-bit 概率序列化。
- all-skip 不调用 Target；之后的 trigger 仍用完整 proposal 追赶 Target。
- `disable_rl_update` 守卫不变；未宣称无 checkpoint 副作用。

## 验证记录

[公开入口契约测试](../test/test_cee_cuhlm_contracts.py) 使用真实 tiny CPU Llama 和 `KVCacheModel`。只替换 CUDA 时钟，并在需确定分支时脚本化 uncertainty gate；另有真实 uncertainty probe 的 RNG 观测。

- 迁移前 characterization：**8 passed**。
- 纯结构迁移后：相关注册、兼容、旧适配白名单、通信扫描、缓存身份、逻辑指纹和解码测试 **110 passed**。
- 两个方法 AST 完全一致；纯迁移逻辑指纹仍为 **136 / `2e729db6cd34e5c1`**。
- NTT 红绿：precise 失败、plain 对照通过；修复后公开入口测试 **10 passed**。
- EOS／缓存裁剪红绿：修正空缓存观测方式后 **10 failed + 4 passed**；修复后连同旧回归／缓存身份测试 **80 passed**。首次测试脚本错误的日志也保留，不计作生产缺陷。
- 最终公开入口契约 **30 项**，加逻辑指纹测试共 **42 passed**；新增文件 Ruff 检查通过。将独立探针中的 trigger→skip 落后 Target、显式绝对预算与禁用 EOS 四个场景固化为回归测试；一次脚本误把单个 trigger 标志设为持续触发，修正为显式 trigger→skip 后通过，没有额外生产修改。除两项行为修正外，仅做格式整理及保留计时调用的未使用局部变量重命名。
- 只更新该模式的指纹为 **139 / `9d843c8a8576d6c1`**：增加一次最终 EOS 检查、一次有界 rollback 调用及相应循环；所有其他模式指纹不变。
- 独立只读 reviewer：**No findings**。额外 10 个真实 CPU 场景覆盖 trigger 后 EOS、先 trigger 后 skip 的落后 Target、显式绝对预算、禁用 EOS 等；plain／precise 多轮 NTT 差分均为独立推导的 114ms，输出、消息与 RNG 不变。
- 冻结后的全仓 CPU 门禁：本地与 water4090 独立副本各跑 default／importlib，四次均为 **2079 passed、15 skipped、163 subtests passed**，均启用 `--require-golden-assets`，源码前后不变。15 个跳过是 7 个缺失的历史实验目录和 8 个显式 opt-in CUDA 回归，不计作通过；远端另有 15 条可选 acc-head 注册表缺失／默认路径回退警告，不代表完整生产资源加载已验证。
- 两地绑定同一批 **359 个源码文件散列、HEAD、dirty/untracked 清单和子模块提交**。子模块显示注释本地为 `heads/main`、远端为 `d340008`，所以不声称两地原始 snapshot JSON 完全相同；每个门禁各自严格绑定其所在环境。
- [正式原生工具](<native_validation.md>) 的公开 CLI 契约为 **59 passed × 2 模式**。本轮实机入口因 GPU2/3 非空闲返回 `resource-refused`／exit3：**0/336 项执行，未启动 worker，非原生验收通过**。它的矩阵针对 CEE-DSD/DSSD，不覆盖本模式的原生执行。
- 汇总及来源见本地 [整体验证记录](<../.refactor/native-validation-20261114/integration-summary.json>)。旧 Graph 代码／测试／文档及用户 HTML 散列保持不变；两个旧远端 checkout 的源码快照和 144 个包版本保持不变。没有安装、升级、提交或推送。

逐步日志、迁移前源码及 AST 比对保存在本地 `.refactor/cee-cuhlm-20261114/`，不覆盖此前 Graph 原生验收证据。CPU 测试不是 CUDA Graph 或真实 Qwen 数值验收。
