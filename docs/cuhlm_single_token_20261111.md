# CUHLM 单 token 生命周期（2026-11-11 第二批）

基点 `41b591a`（PR #33 合并版本），分支
`refactor/cuhlm-single-token-20261111`。本批修正 plain `uncertainty_decoding` /
`cuhlm`，接续[首批迁移](<cuhlm_refactor_20261111.md>)，不代表 CUHLM 整体验收完成。

## 协议与实现

[论文 v2](https://arxiv.org/html/2505.11788v2#S4.SS2) 的 II-A、IV-B 和
Algorithm 1 每轮只定稿一个 token。接受时保留 draft，拒绝时采一个残差 token，
没有 full-accept 后的 target bonus。原实现却先 `target.generate(x, 1)` 采一个
丢弃的样本，接受后再采 bonus，导致预算 1 可能生成 2 个 token。

[CUHLM 实现](<../src/baselines/cuhlm.py>) 现在：

- 触发时调用 `target.forward_context(prefix)`，只处理旧 prefix，不 forward
  本轮候选或抽取丢弃样本；验证概率取 `prefix_len-1` 行。
- 接受只保留 draft；拒绝只追加一个残差 token。删除 bonus 及未来端点 rollback。
- 每轮结束，draft cache 比输出少一个尚未 forward 的 token。触发轮 target
  也停在旧 prefix；skip 轮 target 不动，可能落后更多，下一次触发时一次追赶。
- 草稿 EOS 被拒绝不终止；定稿 EOS 在下一次 draft 前停止；prompt EOS 不误停。
  零预算不 forward、不采样。
- skip 直接定稿的 draft 也计入 `draft_accepted_tokens`。该字段表示 draft
  定稿数，不是目标准确率；`generated_tokens = draft_accepted_tokens + 拒绝轮数`。
- 保留首轮 context transfer、一次压缩上行及 index-only 下行、pending resync
  与轮末 flush；skip 仍无逐轮通信，排队只在触发时计。

接受仍使用历史原 `q(d)`，拒绝仍使用重建 `qhat`。这种区分来自论文，
不能直接替换成普通精确 SD 验证器；其低精度与退化分布问题尚未在本批解决。

## 测试与收尾

[公开入口回归](<../test/test_uncertainty_decoding.py>) 先得到 **8 failed、6 passed**，
覆盖多余 target 采样、单 token 预算、skip 计数和缓存端点。实现后该文件
**14 passed**，并保留词表大小、实际 draft token 与通信顺序回归。

[真实 CPU 缓存测试](<../test/test_cuhlm_cpu_contracts.py>) 使用 tiny Llama 和生产
`KVCacheModel`，仅替换 CUDA 计时及触发策略。核心证据：

| 场景 | 结果 |
|---|---|
| prompt 长度 2、预算 1，accept/reject/skip | 均只生成一个 token |
| 触发接受 | target 输入长度 `[2]`，两缓存端点 2，输出长度 3 |
| 连续接受三轮 | 恰好三次 multinomial，无 bonus 或丢弃样本 |
| skip 两轮后首次触发 | target 一次 prefill 输入长度 `[4]` |
| trigger/skip/skip/trigger | target 输入长度 `[2,3]`，末轮端点 5 |
| 拒绝 | 只抽 draft 和 residual 两次，target 端点仍为旧 prefix |
| target.generate 失败替身 | 混合 trigger/skip 正常通过 |
| EOS 与零预算 | 拒绝草稿 EOS 不误停；定稿 EOS 不多 forward；prompt EOS 不误停 |

GLM-5.3 对实现者提交的五文件改动做了一次只读复查，未报告发现，复跑了
29 项入口/真实缓存测试、12 项指纹测试和 3 项 legacy bridge 测试。

Lead 随后补了异常适配器的防御性检查：如果 target 返回的历史缺少验证行，
legacy bridge 会裁成空窗口，移除 bonus 后将无进展地重复同一 prefix。
有界反例先复现此行为；现在要求 `actual_gamma == 1`，不满足就明确报错。
这不表示真实 eager 后端已经出现该故障，也不扩展本批数值契约。
收尾后的入口、真实缓存与 legacy bridge 三文件合计 **33 passed**。

[逻辑指纹](<../test/golden/logic/baselines.json>) 只更新 CUHLM 一项：
`91 / 052491e9665476aa` → 实现者版本 `88 / c9dafa505ebe1c6f` →
加入缺行检查后的 `90 / eaf44fd1b5f6baf4`。变化包括移除两个 rollback 调用及
accept 采样、增加 forward_context、generate 次数 2→1，以及缺行 ValueError。
另外 14 项和解析器保持不变；历史 golden 数值未修改。

完整严格 CPU 门禁已完成，启用 `--require-golden-assets`：

| 导入方式 | 结果 | 耗时 |
|---|---|---|
| default | 1427 passed、7 skipped、163 subtests | 87.23s |
| importlib | 1427 passed、7 skipped、163 subtests | 106.47s |

14 项历史权重指纹实际通过；7 项 skip 均为缺失原始 run。源码端点一致，
只剩原有两类 warning。日志位于本地忽略目录
`.refactor/cuhlm-single-token-20261111/final-cpu-strict/`。
另一次 Ruff 调用被本地 uv 缓存的只读权限拦截，未执行，不计作 lint 通过。

## 后续数值工作

本批未改共享模型、通信、采样/验证模块或 CEE-CUHLM 循环。仍需处理：

- FP32 实际采样律与历史原概率的差异。Lead 已复现：BF16 `[.1,.3,.6]`
  与其 FP32 采样权重作为 P 时，实际采样权重相同，但旧 raw-q 验证仍可误拒绝。
- 压缩残差退化：`q=[.95,.04,.01]`、`k=1`、`P=qhat=[.95,.025,.025]`，
  候选 `d=1` 可以被拒绝，正残差却为零。现有 `max_fn` 会回退到 argmax
  one-hot（平局 token 0），不是论文归一化残差；本批未改变该行为。
- k 搜索不含 V，未满足条件或长度不匹配时的固定回退不是已证明的保守策略。
  BF16 `[.1,.2,.7]`、`theta=0` 已实测得到 `k=0`。
- CEE-CUHLM 的 `vocab_size` / `draft_token` 透传及独立组合验证。

本批没有 CUDA graph 实跑结果。阶段 A 的独立远端队列首轮等待一小时未获得
空闲 GPU，运行 0 个实验，详见[远端记录](<stage_a_remote_20261111.md>)。
