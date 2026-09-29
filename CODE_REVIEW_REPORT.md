# DuoDecoding 业务代码审查报告

**审查日期**：2026-03 · **审查对象**：`/home/tiantianyi/code/DuoDecoding`
**范围**（经确认）：`src/` 顶层业务模块、训练管理脚本（`auto_train_manager*.py` 等）、评测流水线（`eval/` + `exp.py` + 根目录小脚本）。
**排除**：vendored 第三方实现（`src/model/`、`src/SpecDec_pp`、`src/quantize`）、`test/`、`scripts/`、notebook、数据与 checkpoint。
**方法**：5 个分区审查（解码运行时 / 通信仿真 / 模型加载 / RL 训练 / 评测流水线）+ 主审对全部 critical/high 发现逐一读码核验，交叉指控双侧取证。约 1.9 万行业务代码。

---

## 0. 执行摘要

单看代码质量，`graph_decode.py`、`decoding_ops.py`、`rl_reward.py`、`model_gpu.py` 属于优秀水平（单位换算、CUDA Graph 安全论证、残差采样数学均核验无误）。**但仓库的整体健康度被一个结构性模式拖垮：同一逻辑被复制 4~11 份后各自演化**。11 份解码循环、9 份 model_id 判定链、4 份传输计费公式、2 份训练编排器、2 份批量实验器——本报告绝大多数 high 级 bug（口径分叉、漏接开关、越界不截断）都是拷贝分叉的直接产物。第二个系统性模式是**静默降级**：`except Exception → print → 从零继续` 的写法贯穿 checkpoint 加载、trace 读取、状态文件解析，让"坏了"和"正常"不可区分。

**最需要立即处理的 10 项**：

| # | 问题 | 影响 |
|---|------|------|
| 1 | `auto_train_manager_adaptive.py` 构造即 `TypeError`（`get_rl_agent_spec` 位置传参），默认启动脚本也不存在 | adaptive 编排器整体不可用 |
| 2 | `dist_spec`/`uncertainty_decoding` 每轮把**整条序列**上行计费（O(L²) 字节膨胀），prompt 首轮还被计费两次 | 两个被比较基线的通信时延口径失真 |
| 3 | `uncertainty_decoding` 把**原始 logits** 喂给按"概率分布"设计的 CUHLM 压缩词表公式 | CUHLM 基线核心超参 k\* 无意义 |
| 4 | CUHLM 压缩词表搜索是 O(V²) Python 双循环 + 逐元素 `.item()` 同步 | 32k/151k 词表下单次调用分钟~小时级 |
| 5 | `eval_specbench.py` 不解包 `(prefix, metrics)` 元组 | SpecBench 入口首个样本即崩 |
| 6 | `speculative_decoding_with_bandwidth` 签名缺 `**kwargs` | 经评测 harness 调用必 `TypeError` |
| 7 | `cee_cuhlm` 把 max_tokens 放大 γ1+γ2+1 且不截断；`cee_dssd/cee_dsd/ceesd_without_arp` 越界不截断 | 4 个方法的吞吐被系统性高估 |
| 8 | mt_bench/noeval/cnndm/xsum 的 `seed_set` 永不入集 | `-n K` 时 K 份样本逐字相同，均值被污染 |
| 9 | RL checkpoint 加载吞掉一切异常后"Starting fresh" + 非原子写入 + SIGKILL | 无人值守训练成果可被静默丢弃 |
| 10 | fresh clone 时 `acc_head_registry.json`（在子模块内）缺失 → `parse_arguments` 在 argparse 默认值阶段裸崩 | README 的克隆步骤跑不通 |

---

## 一、Bug（按严重度）

### 1.1 Critical

**B1. adaptive 训练编排器构造即崩溃** 【🗑 已删除 2026-03】：文件与配套测试整体删除（零生产引用，功能由 exp.py adaptive 路径承担）
`auto_train_manager_adaptive.py:74-81` 以两个位置参数调用 `get_rl_agent_spec(ADAPTIVE_METHOD, ROLE_MAIN, ...)`，而 `src/rl_agent_registry.py:143-150` 的签名在 `role` 之后全部 keyword-only（有 `*` 分隔），`_validate_role` 也只认 `main/little` → 实例化即 `TypeError`。其默认 `start_script="cmds/train_rl_mixed_adaptive.sh"`（第 40 行）全仓不存在。测试文件 `test/test_auto_train_manager_adaptive.py:21` 虽实例化该类，但显然未在当前代码状态下运行过。
**修改**：改为 `get_rl_agent_spec(ROLE_MAIN, little_model=None, draft_model=..., target_model=...)` 并补脚本；或确认无人使用后整文件删除（与 `exp.py` 的 adaptive 路径重复）。

### 1.2 High：功能直接崩溃

**B2. `eval_specbench.py` 不解包解码返回的二元组**（`eval_specbench.py:245-255`） 【✅ 已修复 2026-03】：两处调用点（warmup+主循环）补元组解包，并新增 `_specbench_metrics.json` 落盘
`generate_ids = decoding(input_ids)` 之后直接 `generate_ids.shape[1]`。engine 所有已注册解码方法都返回 `(prefix, metrics)`（如 `engine.py:1021`）。对照正确写法 `eval_humaneval.py:249-250` `if isinstance(generate_ids, tuple): generate_ids, _ = generate_ids`。整个 SpecBench 入口是坏的，且该脚本不写任何 `_metrics.json`。
**修改**：补元组解包 + metrics 落盘。

**B3. `speculative_decoding_with_bandwidth` 签名缺 `**kwargs`**（`engine.py:1024-1029`） 【✅ 已修复 2026-03】
评测 harness 统一以 `partial(decoding, ..., use_stochastic_comm=..., ntt_ms_edge_cloud=..., ntt_ms_edge_end=..., use_early_stopping=...)` 装配（`eval/eval_xsum.py:140-148` 等），该函数不接受这些参数 → 首个样本即 `TypeError`。其余 11 个方法全部有 `**kwargs` 兜底。该模式同时出现在 `engine.load_model` 分支列表（`engine.py:436-437`），说明它被认为是可用模式。
**修改**：补齐与 `sd` 一致的参数或统一 `**kwargs`。

**B4. fresh clone 上 `parse_arguments` 在 argparse 默认值阶段崩溃**（`utils.py:562-578` + `acc_head_registry.py:82-84`） 【✅ 已修复 2026-03】：注册表加存在性检查降级空表，三个 default 改 None 走后置解析
三个参数默认值在**每次**解析时求值：`default=resolve_acc_head_path("tiny-llama-1.1b", "llama-2-13b")` → `load_acc_head_registry()` 无任何存在性检查地 `open(_REGISTRY_PATH)`。该 JSON 位于 `src/SpecDec_pp` **子模块**内（`.gitmodules:1-3`），README 的克隆步骤（`git clone` + `uv sync`）不含 `git submodule update --init`，也不含 `scripts/setup/download_assets.py` → 任何 eval 入口在参数解析前就 `FileNotFoundError`，栈指向 argparse 内部，极难定位。另外 `utils.py:1200-1201` 在非显式传参时总会覆盖该默认值——默认值里的这次注册表读取纯属多余。
**修改**：`load_acc_head_registry` 开头 `if not _REGISTRY_PATH.exists(): return {}`（降级为纯默认路径）+ warning；三个 default 改为 `None`，统一在 parse 后解析（1185-1201 已有该逻辑）；README 补子模块初始化步骤。

### 1.3 High：科学计量错误（研究代码的最高优先级）

**B5. 两个基线把整条序列重复上行计费**（`baselines.py:1324`、`baselines.py:1626`） 【✅ 用户已修（commit 9020c22）】
`comm_simulator.transfer(x, None, "edge_cloud")`——而 `KVCacheModel.generate` 返回**完整序列**（`model_gpu.py:683` `return torch.cat([x] + new_tokens, dim=1)`），每轮按 `(prompt+已生成+γ)×8B` 重复计费整条前缀；`dist_spec` 第 1 轮还先发一次 prompt（1273-1274），prompt 被计费两次。对照 dssd 已改为增量上行（`baselines.py:991-996` `_collect_dssd_uplink_payload` → `collect_verification_payload` 取 `x[:, prefix_len:...]`）。受影响的 `dist_spec` 与 `uncertainty_decoding`(cuhlm) 恰好都是**被比较的对照基线**，O(L²) 膨胀会系统性压低其 TPS。
**修改**：与 dssd 统一只传 `x[:, prefix_len:]`；若全量重发是刻意的协议假设，加注释并在论文口径中声明。

**B6. CUHLM 压缩词表公式吃原始 logits**（`baselines.py:1627-1641` 误用方；`communication.py:935-1009` 被误用方）【✅ 已修复 2026-03】
`_calculate_compressed_vocab_size` 的注释与公式（`x_d = sorted_probs[0]`、`residual_mass = 1 - Σ`、softplus 分母）只对归一化概率有意义，但 `uncertainty_decoding` 传入的是 `logits_history`（原始 logits，`model_gpu.py:349`）。logits 之和量级任意，`x_d` 可 >1 使分母为负直接 `return 30`（:980-981）。该基线的核心自适应超参 k\* 实际不可信。
**✅ 已对照原论文复核（arXiv:2505.11788v2，Oh et al., IEEE TC）**：公式实现本身忠实于论文——代码的分子 Σ|x_i−x̂_i| 即论文式 (23)/(26) 的分子，分母 `(1−x_d)·softplus(−1) + x_d·softplus(−β_d)` 与式 (26)（Proposition 2）逐项一致，β_d 的线性拟合 a=0.815、b=−0.066 对应论文 Section III 的 uncertainty→rejection 模型。**但论文全部公式都定义在词表概率分布 x(t) 上**：第 II-A 节明确 x(t) 由 logit 经 softmax 归一化而来，式 (16) 的 top-k+均匀尾重构、式 (23)/(26) 的残差质量与 x_d 均为概率域量。论文没有任何一步把 logits 代入这些公式。旁证：`CUHLM.terminal_prob`（`communication.py:1011+`）内部调用同一 `determine_transfer_strategy(uncertainty, current_probs)` 时传的就是**概率**张量——只有 `uncertainty_decoding` 这个调用点传 logits。结论：bug 在调用方而非公式实现。另两处顺带核实：① docstring 自称"公式(24)"，实际实现的是**在线**变体（式 (26) 约束），(24) 是离线时间平均版——引用编号笔误；② O(V²) 暴力搜索（B7）与论文无关，argmin 可由前缀和 O(V) 求解，纯实现问题。
**修改**：调用方改传 `prob_history` 行；函数开头校验 `abs(sum-1) < eps` 防御。

**B7. CUHLM 压缩词表搜索 O(V²)**（`communication.py:984-999`）【✅ 已修复 2026-03：cumsum+searchsorted 向量化，151936 词表单次 26ms，与暴力循环等价性有测试覆盖】
`for k in range(1, vocab)` 内层 `for i in range(k, V)` 且逐元素 `.item()`（每次一次 GPU→CPU 同步）。V=32000 时约 5×10⁸ 次同步，单次调用分钟~小时级，且在 per-token 主循环上（`baselines.py:1638`）。即使修好 B6，此循环也让基线跑不完。
**修改**：预计算 `cum = cumsum(sorted_probs)`，分子 Σ|x_i−u| 可由前缀和闭合表达，整体 O(V) 向量化后二分搜索最小 k。

**B8. 四个方法越过 max_tokens 不截断 / 直接放大预算**（`baselines.py:4179-4180`、2636-2696、4963-5022、5356-5415）【✅ 已修复 2026-03】
`cee_cuhlm`：`max_tokens = prefix.shape[1] + self.args.max_tokens + gamma1 + gamma2 + 1` 且末尾无截断；`cee_dssd/cee_dsd/ceesd_without_arp` 循环内一次最多追加 γ2+γ1+2 个 token 也无截断。对照 `dist_spec` 的注释（1499-1504）明确写着必须截断"否则会多拿 token，使配对质量比较与时延/吞吐统计都不公平"（F33 契约）。
**修改**：统一在返回前 `prefix = prefix[:, :prompt_len + self.args.max_tokens]`；`cee_cuhlm` 去掉放大。

**B9. `dssd` 的 metrics 在 EOS 截断之前结算**（`baselines.py:1114-1148` vs `1169`） 【✅ 已修复 2026-03】：截断上移到 metrics 结算前（wall_time 不回调——计算是真实花费）
`_stop_at_eos` 在 metrics 全部算完后才执行，被截掉的 token 已计入 `generated_tokens`/吞吐，wall_time 也未随之下调。对照 `adaptive_tridecoding` 是循环内截断后 break 再统计（3728-3731 → 4064）。
**修改**：截断移到 metrics 结算前，或截断后重算 `generated_tokens`。

**B10. EOS 中途截断只接入 4/11 个方法**（`baselines.py:2698-2712` 注释自认缺陷；调用点仅 1169/2808/3114/3153/3729/4038） 【✅ 已修复 2026-03】：7 个方法循环体顶部统一接入 `_stop_at_eos`（含 `_tri_prompt_len` 捕获）；engine `_check_stopping_criteria` 增加 prompt_len 参数扫描整个生成段，修掉 chunk 中部 EOS 漏检
`dist_spec`、`uncertainty_decoding`、`tridecoding`、`ceesd_without_arp`、`cee_cuhlm`、`cee_dssd`、`cee_dsd` 及 engine.py 全部方法均未接 `_stop_at_eos`；engine 的 `_check_stopping_criteria`（328-347）只查最后一个位置，chunk 中部 EOS 漏检后继续解码到 max_tokens。同一评测里不同方法 EOS 行为不同 → 生成长度、质量分、时延都不可比。
**修改**：在所有方法返回前统一接 `_stop_at_eos`（建议引擎层做一次），且在 metrics 结算前完成。

**B11. 多样本评测的 seed 永不变化**（`eval_mt_bench.py:305-308`、noeval/cnndm/xsum 同构） 【✅ 已修复 2026-03】：4 个 eval 文件补 `seed_set.add`，与 humaneval/specbench 对齐
`while self.seed in self.seed_set` 恒为假（`seed_set` 为空集且**从不 add**；全仓库仅 humaneval:232、specbench:214 有 add）。`-n K` 时 K 次采样逐字相同——MT-bench 对同一答案打 K 次分。当前 exp.py 传 n=1 未暴露；gsm8k 则用第三套方案（`seed + rep*7919`）。
**修改**：抽公共 `resolve_sample_seed()`，内部维护 seed_set。

**B12. `finalize_verification` 用请求 γ 而非 actual γ 构建回滚计划**（`decoding_ops.py:604`；对照正确写法 :556-560） 【✅ 已修复 2026-03】：与 resolve_stage_verification 同公式计算 actual_gamma
`build_rollback_plan(prefix_len, gamma, n)` 用函数参数（请求值），而 `resolve_stage_verification` 用 `verification_inputs.actual_gamma`。当 `actual_gamma < gamma`（prob_history 被截短，`prepare_verification_inputs:129-134`）且草稿全被接受时，`all_accepted` 误判为 False → bonus token 从 max(p−q) 残差分布而非目标分布 p 采样，序列尾轮输出分布偏离目标。调用方 `engine.py:961、1197-1213` 传的都是请求值。
**修改**：`finalize_verification` 增加 `actual_gamma` 参数，与 `resolve_stage_verification` 同一公式。

**B13. 评估时 RL 探索未冻结：部分调用点漏传 `training=False`**（`rl_adapter.py:691` 默认 True；`baselines.py:2511-2513`、`2970-2972` 未传 vs `3500`、`3800` 正确传） 【✅ 已修复 2026-03】：3 处补 `training=not disable_rl_update`（与既有 2 处正确写法一致）
checkpoint 的 ε≈0.01 → 评估时约 1% 决策是纯随机动作，且 ceesd 与 ceesd_without_arp 的对照被不同的探索噪声污染。
**修改**："评估冻结"下沉到 adapter（构造时读 `disable_rl_update` 或提供 `eval()` 开关），不依赖 5 个调用点各自记参数。

### 1.4 Medium（bug）

| # | 问题 | 位置 → 修改 |
|---|------|------------|
| B14 | 【✅ 已修复：删条件重复调用（_trace_comm 内部已有门控）】`transfer()` 对每笔传输写**两条**相同 COMM_TRACE 记录（重构残留：658 无条件调 + 667 条件再调，`_trace_comm` 内部已有门控） | `communication.py:658-676` → 删除 667-676 |
| B15 | 【✅ 已修复：原子 helper + 全阶段接线 + 模式门控三管齐下】`_uplink_prob_payload()` 把"量化张量"与"计费位宽"绑定返回，调用点无法只取其一（B16 讨论中确认的信任缺口由此闭环）：①tri 两方法的 8 个上行点全部接线——stage-1 初次上行（原静默忽略）、两级拒绝上行（原忽略或只计费不量化——后者即"收 bits 钱传全宽信息"的活跃实例）、stage-2 样板收敛；②tridecoding stage-2 的 verify override 原先重建**未量化**副本（注释承诺"验证看到同一个 q̂"从未兑现），现与载荷同源；③ModeSpec 增 `supports_prob_bits` 能力位（精确集 {tridecoding, adaptive_tridecoding}），argparse 对不支持模式 bits<16 显式 parser.error（此前静默忽略）。**bits<16 为 opt-in：默认 16 时全路径逐字节不变（测试锁死）；位宽生效后 stage-1/拒绝上行的计费与接受判据会变（协议语义本该如此），涉及 run 需重跑** | 原子 helper + 能力位 + 校验 |
| B16 | 【✅ 已定性修正+关缝：**潜伏 API 语义缝隙，非活跃计费 bug**——经复核，全部调用点（量化 2 处、传参 2 处、手工计费 2 处）均以严格 `< 16` 门控，不一致窗口（fp32 + bits∈[16,32)）当前不可达；原描述"收 16bit 钱传 32bit 信息"写成了现在时，属过度陈述。已结构性关死：穿透阈值提为 `PROB_QUANT_PASS_THROUGH_BITS=16` 常量单源，计费条件镜像量化门（`min(16, 8*elem)`），全部行为零变化（含 argparse 默认值 16 恰在窗口左端点的贴边行为，3 项边界测试锁死）；不变量不再依赖散落调用点各自记得门控】量化早退 `bits>=16` 与计费 `bits < 8*element_size` 不一致 | 常量单源 + 计费条件镜像量化门 |
| B17 | 下行回传计费口径分裂：旧方法 token 与位置索引分两次传输付 **2 次 NTT**（1107-1108、2019-2020、2482、4824），新方法合并付 1 次（4042-4050） | 抽 `send_downlink_token()` helper 全仓统一 |
| B18 | `avg_top_k` 语义分叉：无 transfer_top_k 时 dist_spec 记 `args.top_k`（采样参数），dssd/tridecoding 记 0 | 统一为"只统计传输压缩 top-k，未压缩记 0" |
| B19 | `--use_cuda_graph` 三档接线：tridecoding 完全漏接（`build_adaptive_tridecoding_caches` 无图参数），cee_\* 裸接（无验证图档位、无跨样本复用 → 每样本重捕获 ~534ms） | 全部改走 `_graph_mode_cache_kwargs` + 复用属性 |
| B20 | RL adapter 直接改写全局 `self.args.gamma1/gamma2`（2386、2514），跨样本/跨任务残留；对照 adaptive_tridecoding 已用实例属性 | 统一为实例级 `_next_gammaN` |
| B21 | 【✅ 已修复：死块连同只守卫它的 assert 一并删除】`ceesd_without_arp` 每轮对整条 logits 历史做 `norm_logits` 后**从未使用**（O(L×V) softmax + 跨设备拷贝） | 删除 |
| B22 | 【✅ 已修复：vocab 查表移回映射前用原始别名命中；重复键去重】`model_zoo` 的 `vocab_size` 查表在 zoo 映射**之后**（`utils.py:297-306`），键是别名、值是路径 → 字典对最常用别名全部 miss，每次 parse 都读 config.json 或走 `AutoConfig.from_pretrained(trust_remote_code=True)` 网络回退；dict 还有重复键 `"llama-2-70b"`（234/263） | 查表移到映射前，或删字典（engine 反正会重算） |
| B23 | 【✅ 已修复：未部署集合 + 解析阶段显式 ValueError】zoo 把未部署模型映射为 `"deepseek-1.3b还没部署"` 等占位符（268-279），垃圾路径静默传播 | 删键或映射时 raise |
| B24 | 【✅ 已修复：default=None + parse 后置块开头 parser.error（先于 acc-head/RL 解析）】默认 `--draft_model codellama-7b` / `--target_model codellama-70b`（319-320）不可解析：zoo 无此键、本地无此目录、HF 无此仓库名 | 改为真实存在的对或 `default=None` 必填 |
| B25 | 【✅ 已修复：显式 (draft, target, little) 固定优先级】legacy RL 回退路径用 **set 迭代**生成顺序（`rl_agent_registry.py:119` `for model_name in {little, draft, target}`），跨进程 hash 随机 → 多个 legacy checkpoint 并存时加载哪个纯凭运气，可能迁错模型对的 agent | 改显式列表按固定优先级排序 |
| B26 | 【✅ 已修复：分歧别名（qwen-3-*/llama-2-chat-7b）并入 CANONICAL_MODEL_ALIASES 统一归一】zoo 别名与 `CANONICAL_MODEL_ALIASES` 分歧：zoo 用 `qwen-3-0.6b`，registry 只认 `qwen3-0.6b`；`llama-2-chat-7b` vs `llama-2-7b-chat` 同病 → 合法别名静默错过已注册对，落到不存在的默认路径（实测复现） | zoo 映射后统一过 canonicalize，两表合并为单一事实源 |
| B27 | `norm_numpy_logits` 的 top-k/top-p 过滤被注释禁用（`utils.py:1313`），接口名不副实；唯一调用方是死代码 model_cpu.py | 删函数或恢复过滤 |
| B28 | 【✅ 已修复：benchmark=False，复现优先】`seed_everything` 同时设 `cudnn.deterministic=True` 与 `benchmark=True`——后者选最快但不一定确定的算法，伪复现保证 | 二选一 |
| B29 | warmup 计数一族 off-by-one 且各脚本不一致：mt_bench/noeval n=10 实际 9 次（break 在生成前），cnndm/xsum n=5 → 4 次，humaneval/specbench 恰好 10 次，gsm8k 0 次 | 抽公共 `warmup(n)`，统一"先检查后生成" |
| B30 | 【✅ 已修复：accumulate_metrics() 单点化，6 个 eval 脚本接入；死键/幽灵键删除、KeyError 路径关闭；gsm8k/humaneval 的 connect_times 由恒空改为累加（分叉统一的预期变化）】metrics 合并循环 5 种分叉实现，排除列表互相矛盾且含死键（`little_acceptance_rate/draft_acceptance_rate` 全 src 不存在）；humaneval/mt_bench 两处在键缺失时潜伏 KeyError | `src/metrics.py` 提供单一 `merge_metrics()` |
| B31 | cnndm/xsum 解码异常路径：吞异常 + 用 EOS 占位后 `num_tokens = 1 - prompt_len < 0` 混入均值，污染 tokens/s 与 ROUGE | 异常时 continue 或记 status 剔除，`max(0, ...)` 钳制 |
| B32 | `eval_cnndm.py:166` 硬编码 `use_early_stopping=True` 覆盖 exp.py 传入值；`eval_mt_bench_noeval.py:141-148` partial 丢 `use_stochastic_comm` | 参数统一由 partial 装配函数生成 |
| B33 | 【✅ 已修复：不支持的任务显式告警并返回空串】MT-bench few-shot 是静默 no-op：`get_few_shot_prompt("mt_bench", ...)` 无对应分支返回空串（`few_shot_examples.py:83-108`），`--num_shots 3` 无效无告警 | 未知 task raise，或删调用点 |
| B34 | chat 模板后未关 `add_special_tokens`：`eval_mixed.py:300-302`、`eval_humaneval.py:119-125`（只豁免 Llama-3.1）→ Llama-3/3.2 系潜在双 BOS | 所有 chat 模板路径统一 `add_special_tokens=False` |
| B35 | mt_bench/noeval 以 append 写 jsonl 却用全文件重算速度（185/497），重跑时速度混入旧数据、accuracy 只算本次 | 记录文件偏移或只统计本次写入行 |
| B36 | 【✅ 已修复：告警 + 回退 -1.0】`best.pth` 缺 `best_tps` 键时回退 `+inf`（`rl_adapter.py:315-318`）→ `new_best` 恒 False，"best" 永不更新且无警告 | 告警并回退 -1.0，或迁移脚本补键 |
| B37 | 【✅ 已修复：模式由 self.models 生成 + `--*_model <name>` 完整参数匹配 + list exec】pkill 清理模式与 MODEL_SERIES 大小写不一致，llama 系列 target 残留进程杀不掉；裸子串误伤面大 | patterns 由 `self.models` 生成 + 精确匹配 |
| B38 | 【✅ 已修复：改 adaptive_tridecoding.*】`adaptive_tridecoding` 里的校验标签写成 `cee_cuhlm.*`（3723-3727，复制粘贴错标签） | 改 label |
| B39 | `engine.py` `small` 模式把 draft 前向计数进 `metrics["target_forward_times"]`（777-798，模型身份错标） | 改键 |
| B40 | `_apply_top_k_compression` 用 `len(probs)` 判 top-k 上界（`communication.py:496、919` 两份拷贝同病），对 (B,V) 取 B 维；`compressed_probs[top_k_indices]` 对 2D 不安全。当前调用方恰好都传 1D，属潜伏 【✅ 已修复：`shape[-1]` + `scatter_(-1)`，附 2D 回归测试】 | 改 `probs.shape[-1]` |
| B41 | 【✅ 已修复：clamp_min(0) + 显式非 top-k 尾掩码（两份拷贝+batch 版）】`rebuild_full_probs`/`compress_rebuild_probs` 的 `residual_mass` 无 clamp（:529/:580），top_k_sum>1 时产出负"概率"；`zero_mask==0` 会把本就为 0 的 top-k 项也换 uniform | `(1-sum).clamp_min(0)` + 显式非-top-k 掩码（对齐 `utils.rebuild_topk_probs` 已有的保护） |
| B42 | 【✅ 已修复：浮点走 float 路径】`AdaptiveDecodingDebugger.tensor()` 把浮点张量 `.to(torch.long)`（`adaptive_debug.py:36-42`）→ 概率/熵记录全 0 | 浮点走 float 路径 |
| B43 | `verify_draft_sequence` 指标用请求 γ 而非 actual γ（`decoding_ops.py:498`）；serial 模式 token 只取 batch 0（:492） | 改 actual_gamma；声明 bs=1 约束 |
| B44 | 【✅ 已修复：钳制 + 负值 ValueError，4 项回归测试】`build_draft_probs_override` 对 `stage_start_len=0` 静默切成 `[:, :-1]`，潜伏 | `max(stage_start_len-1, 0)` + 断言 |

---

## 二、隐患（数据完整性 / 环境脆弱 / 竞态 / 安全）

**R1. RL 训练成果可被静默丢弃（三条链叠加）**【✅ 已修复 2026-03（含一处勘误）】
① 保存非原子：`torch.save` 直接覆写目标文件、replay buffer 走裸 `pickle.dump`（`rl_adapter.py:270-288`），且**每个样本**调用一次（`baselines.py:593-599`）；② manager 收敛时 `os.killpg(SIGKILL)` 10 秒强杀（`auto_train_manager.py:312-319`），检测到新 TPS 又立即拷贝可能正在写的文件（470-478）；③ 加载侧 `except Exception: print("Starting fresh.")`（`rl_adapter.py:326-327`）把文件损坏/shape 不匹配（换 action_space 必触发）与文件不存在一律静默重训，随后 save 覆盖旧成果。load 还有**半加载**问题：policy/target 网络已被覆盖后才失败，"fresh" 实为半新半旧。
**修改**：tmp + `os.replace` 原子写；非 FileNotFoundError 一律 re-raise 并带 traceback；load 先全量校验（含维度匹配）再一次性 commit；SIGTERM 优先并给 flush 时间。

**R2. GPU 选择竞态（三处独立实现同病）**
`auto_train_manager.py:245-275`（nvidia-smi 瞬时快照排序取前 N）、`exp.py:344-369`（启动时一次性 NVML 快照 + 仅进程内 `threading.Lock`，acquire 后**永不复查**）、`exp_threshold_vs_quality.py:141-147`（`itertools.cycle` 轮询固定 GPU 列表，完全不看空闲）。两个进程并行启动时互相不可见（大模型加载耗时数分钟）→ 同卡双跑 OOM。`exp.py:393` 还用 `"70b" in str(config)` 字符串嗅探触发多卡分配。
**修改**：跨进程文件锁（如 `exp/.gpu_lock/<gpu>.lock` + 心跳续期），选定即落盘，Popen 前二次复查。

**R3. 并行实验固定端口 29051**（`exp.py:96-98`）
`cmd_temp` 写死 `--main_process_port 29051`，`max_workers=4` 时多个 `accelerate launch` 并发共用同一 rendezvous 端口；`engine.py:94-100` 在 `RANK` 存在时真的 `dist.init_process_group(env://)` 绑端口。≥2 卡空闲即撞车（单卡机器 max_workers 收敛到 1 被掩盖）。
**修改**：端口按实验分配或传 0 让 accelerate 自选。

**R4. summary JSON 非原子写 + 孤儿子进程**（`exp.py:493-495`、`exp_threshold_vs_quality.py:153-154`） 【✅ 已修复：tmp+os.replace（两文件）；exp.py 改 Popen(start_new_session) + 异常 killpg 收整棵子树】
每实验完成后原地重写整个 JSON，中断即半个文件（下游 `calculate_consistency.py:90-91` 直接 `json.load` 失败）；exp.py 被杀时 accelerate 子进程树不受管，孤儿进程继续占卡，与 R2 叠加。
**修改**：`.tmp` + `os.replace`；子进程 `start_new_session=True` 并在异常时 killpg。

**R5. `torch.load` / `pickle.load` 无 `weights_only`**（`rl_adapter.py:293、320-322`、`baselines.py:100`） 【✅ 已修复：checkpoint/acc-head 加 weights_only=True；replay buffer 白名单 unpickler（numpy/torch/builtins）】
checkpoint 路径来自 CLI/registry 可指向任意文件，反序列化即任意代码执行面；torch≥2.6 默认翻转后行为突变。replay buffer 的裸 pickle 无等价开关。
**修改**：`weights_only=True`（当前 payload 均基础类型可过）；buffer 改 torch.save/safetensors 或白名单 unpickler。

**R6. 无输出看门狗**（`auto_train_manager.py:543-610`）
主循环仅 `poll()`，子进程活着但日志无进展（GPU hang/加载卡死/脚本等输入）时无人值守运行**无限期**挂起，retry_dssd/scheduled_train 都是一次性的不会接手。
**修改**：last_progress_time 超时（如 30 分钟）告警 + 有限重启。

**R7. 模块级全局可变状态**（`communication.py:71-88、1070-1074、1169-1173`）
`_NTT_TRACE_STATE` 被所有模拟器实例共享（多实例交错计费时 RTT trace 游标与实例不对应）；`_has_logged` 类属性吞掉第二个实例的初始化日志，且 Precise 系两个类一个用 logging 一个用 print、三段带宽打印顺序不一致（至少一边标错）。
**修改**：trace 状态收进实例构造参数；日志去重交给 logging。

**R8. 通信统计无界增长**（`communication.py:136-168、460-484`）
`stats` 与 bandwidth/ntt/topk/draft_len 四个 history 随传输次数线性增长，dump 时再 `.copy()` 翻倍；RL 环境下每次决策一条记录。
**修改**：`snapshot_and_reset()` 或增量聚合。

**R9. 多 GPU 加载的三处不对称**（`model_loading.py:67-148`）
① `_single_gpu_max_memory` 只读 device 0 显存却把同一额度套给所有卡（异构 24G+48G 时小卡 OOM）；② reserve 固定挂最后一张卡，与 `select_dual_model_devices` 把 draft 放 cuda:0 的布局脱节（target 分片会与 draft 抢显存）；③ `num_gpus≥3` 非 60B 场景只产出 cuda:0/cuda:1，第三张卡永远空闲。
**修改**：逐卡计算 max_memory；reserve 卡号由实际布局传入；tri 分支分散 little/draft。

**R10. `import` 即全局生效的副作用**（`engine.py:10-11`、`profile_cee_dsd.py:18`）
`transformers.set_verbosity(40)` + `warnings.filterwarnings("ignore")` 在 import 时压制一切告警（掩盖真实问题）；`ProfiledBaselines.cee_dsd` 与生产实现**同名注册**到类级共享 dict——任何人 import 它都会静默替换生产解码路径（当前两文件均无引用者，属地雷）。
**修改**：告警配置移到 main 入口；profile 改为上下文包裹式插桩。

**R11. 环境与路径脆弱性**
`exp.py:96` 硬编码 `/home/tiantianyi/.../accelerate` + `shell=True` 拼命令（路径含空格即错乱）；registry 默认根与 zoo 本地路径全是 **cwd 相对**（从子目录启动全部指向幻影位置）；`scheduled_train.py:48` 相对路径依赖启动 cwd；`cmd_temp` 把模型名内插进 shell 字符串。
**修改**：`shutil.which("accelerate")`；根路径用 `Path(__file__)` 推导；`subprocess.run([list])`。

**R12. `argparse` 的 `type=bool` 陷阱**（`auto_train_manager.py:629-634`）
`--adaptive_decoding` 用 `type=bool`：任何非空字符串（含 `"False"`）都为 True，且解析结果从未使用。
**修改**：删参数或 `BooleanOptionalAction`。

---

## 三、设计缺陷（结构性根因）

**D1. 复制粘贴演化是本报告多数 high bug 的共同根因**
- `Baselines` 类 10 个解码方法各 300~930 行（`adaptive_tridecoding` ≈925 行），同一骨架（构造 simulator → 构造 caches → while 循环 → verify/rollback → 40 行 metrics 结算）重复 10 份，simulator 构造 10 处近似拷贝、metrics 结算 7 份近同——B5/B8/B9/B10/B15/B17/B18/B19 全部分叉于此。engine.py 内 `speculative_decoding` 与 `..._with_bandwidth` 亦为拷贝对（metrics 已分叉：前者有 `loop_times`，后者无）。
- eval/ 7 个数据集入口：model_id 判定 if-链 **9 份分叉且互相矛盾**——同一组合 `(llama-68m, Llama-2-13b)` 在 gsm8k 映射 `"vicuna"`、xsum 映射 `"llama-2-chat"`、mixed 映射 `"base"`（跨数据集质量口径不可比）；未知组合一半 raise 一半静默默认 vicuna。
- 传输字节计费 **≥4 处各算各的**（`transfer()` / `__call__` / `_simulate_topk_prob_transfer` / 手搓公式），`__call__` 还是零调用者的死代码。
- 两个 TrainingManager ~80% 复制且已漂移（同一收敛 bug 两份、状态文件失败语义相反、双卡阈值不同）；`adaptiveexp.py` 是 `exp.py` 的陈旧分叉（缺 summary 落盘、Literal GPU id 与实际不符）。
- `collect_confidence.py` / `profile_cee_dsd.py` 各 fork 300+ 行解码循环，主流程修复（如 L1 计费口径收敛）不会同步到副本——用它们校准的阈值数据对应旧动力学。

**重构方向**（按收益排序）：
1. 抽 `DecodingContext`（simulator + caches + metrics 收集器）与 `verify_and_commit_round()`，metrics 结算收成一个函数（7 处手写"先算 throughput 再加 queuing 再重算"的重复即消失）；
2. `resolve_model_id(draft, target) -> ModelId` 单一函数 + 显式表，未知组合 fail-fast；
3. 载荷计费收敛为 `CommunicationSimulator.charge_payload(link, n_tokens, n_probs, prob_bits, topk, ...)` 唯一入口；
4. `merge_metrics()` / `warmup()` / `resolve_sample_seed()` / `send_downlink_token()` 五个公共函数消灭 eval 族分叉；
5. 删除：`model_cpu.py`、`model_gpu_new.py`（2026-04 一次提交后从未更新的旧快照，"_new"命名与演化方向相反，零引用）、`src/tp.py`（import 不存在的 `gpt_fast_model`，根本不可导入）、`eval/eval.py`（Eval 类无人引用且 dispatch 兜底自相矛盾）、`adaptiveexp.py`、`profile_cee_dsd.py`（或改造）、engine.py `_prepare_stop_tokens`/`_should_stop`（~95 行死代码）、`communication.__call__`。

**D2. eval_mode 知识散布 ≥5 处（Shotgun Surgery）**
新增一个模式要同步改：装饰器注册（baselines.py:807-808 等）、`engine.load_model` 两处分支列表（428-438、559-572）、baselines `uses_main_rl`/`uses_little_rl` 两集合（446-465）、`load_acc_head` 第三集合（663-670）、utils.py 特判（1219）——漏改即静默走错加载/RL 路径。`register.py:32-33` 的 `hasattr` 兜底还能把 `--eval_mode __init__` 之类的任意属性名当解码方法返回；同名注册静默覆盖。
**修改**：集中为 `MODE_FEATURES: dict[str, ModeSpec]`（needs_little/uses_main_rl/uses_acc_head/…）单点查询；注册时检测重名 raise；删 hasattr 兜底。

**D3. `parse_arguments` 巨型函数 + 解析期副作用**
单函数 930 行、约 90 个参数，通信/RL/解码/acc-head/curriculum/controlled 实验全部混在一个 argparse；解析中夹带 `os.makedirs(args.exp_name)`（1235-1236）；acc-head 默认值在定义期读注册表（B4）；`utils.py` 同时承载参数解析/模型注册/采样/trace IO/vocab 探测五种职责（Divergent Change）；`read_trace_file` 与 `return_closest_mean_index` 有 ~30 行逐字重复的 block 解析且都是裸 `except:`，硬编码 5.0 带宽下限与 `--min_bandwidth_mbps` 参数口径重复。RL 路径默认解析逻辑在 utils/exp/baselines **三处复制**（utils.py:1203-1233、exp.py:565-621、baselines.py:486-548，最后一处的 `or` 右支实际不可达）。
**修改**：拆 `args/`（按域分组）、`model_zoo.py`、`sampling.py`、`trace_io.py`；目录创建移到 engine/exp 层；`resolve_default_rl_paths()` 下沉 registry。

**D4. 训练/推理循环耦合**（`rl_adapter.py:712-724、979-982`）
`select_config` 内部做 `store_transition` + `update()`（推理热路径里跑优化器，靠 `torch.enable_grad()` 补救），状态靠三段式 `last_state/last_action/last_reward` 握手，任何调用点漏 `step(reward)` 即静默不学习；`ceesd_without_arp` 还把 top-k 动作维度重载为草稿长度（`self.args.gamma2 = next_k`），同一动作语义随模式漂移，而正式的 `topk_thr_gamma` 动作空间已存在。吞吐指标（best.pth 判据）混入优化器耗时。DDQN 的 `self.gamma`（折扣因子）与 draft γ 撞名。
**修改**：决策/学习分离（select 只读 + 显式 learn）；返回 Action dataclass；折扣因子改名 discount。

**D5. 收敛判据与课程式训练冲突（潜伏）**
manager 对 TPS 序列做 0.5% 窗口停滞检验即杀训练（`auto_train_manager.py:485-498`），而课程训练每步重采链路条件（TPS 随课程变难系统性下行）。当前 `train_rl_mixed.sh` 未传 curriculum 参数（start==end）故未触发；课程化后的相邻窗口差可能 <0.5% → 课程中段被误判收敛。
**修改**：manager 识别 curriculum 参数，课程结束后再判收敛。

---

## 四、代码风格建议与修改示例

**S1. 命名**
- `model_gpu_new.py` 名不副实（实为旧快照）→ 删除而非改名。
- DDQN `self.gamma` 折扣因子与 draft γ 撞名 → `discount`。
- `rebuild_topk_uniform_probs` 是 `rebuild_topk_probs` 的纯转发 Middle Man → 删别名。
- `_reward_by_gamma` 是全期累计均值却按周期打印 → deque 窗口 + 改名。
- 诊断标签错位：`model_gpu._prefill` 里写 `"KVCacheModel._forward_with_kvcache.initial_probs"`（355-357，从旧函数拷来）。

**S2. 魔法数与常量**
- 6 字节 reject/accept 消息在三处硬编码（`communication.py:696/701`、`baselines.py:3710`）→ `REJECT_MSG_BYTES = 6` 模块常量 + 构成注释。
- `read_trace_file` 的 `5.0` 带宽下限与 `min_bandwidth_mbps` 重复口径 → 统一引用。
- `INT_SIZE = 4` 在 metrics.py:4 定义、engine.py:51 import 后又在 :78 重复定义 → 只留一处。
- `prob_size = vocab_size * 4` 硬编码（`baselines.py:4476`）→ `element_size()`。

**S3. 异常处理**
- 全仓裸 `except:` / `except Exception: pass` 约 10 处（`utils.py:1455/1497/1560`、`baselines.py:435`、`auto_train_manager.py` 读状态、rl_adapter buffer 写入等）→ 收窄异常类型，至少 `logger.warning` 一次。NTT trace 加载的三连 `except Exception: continue` 最后抛 `FileNotFoundError`，把权限错/格式错都伪装成"文件不存在"。
- "Starting fresh" / "加载训练状态失败" 类静默降级 → 至少升为 `logger.error` + traceback，并区分"不存在"与"不兼容"。

**S4. 死代码清理清单**（全部零引用，grep+git 证据）
`src/model_cpu.py`（267 行，内部还藏雷：未定义的 `self.vocab_size`、`_prob_history` 立即被覆盖）、`src/model_gpu_new.py`（187 行）、`src/tp.py`（162 行，不可导入）、`eval/eval.py`（Eval 类）、`adaptiveexp.py`、`engine.py:_prepare_stop_tokens/_should_stop`（~95 行）与 `prob_with_flag`、`compute_stage_reward`（仅测试引用，生产三处内联同公式）、`uncertainty_decoding` 的 `compressed_prob` 死赋值与 `n = prefix_len + 1 - 1`、`adapter.py:52 cum_acc_prob`、CUHLM 逐字复制父类的两个 override、两份拷贝里未使用的 `nonzero_mask`、humaneval 的 `para_sd_hybrid` 双胞胎分支与 `t = 0` 恒零计时。

**S5. 其他风格**
- help 文本中英混杂（同一 argparse 里 `--use_cuda_graph` 是中文长文、`--temp` 是英文）；建议统一语言并在 CLI 里指向文档。
- `@torch.inference_mode()` 与 `@torch.no_grad()` 混用（eval_mt_bench vs 其他）→ 统一前者。
- `pyproject.toml` 依赖混入垃圾包：`dataset==1.6.2`（≠`datasets`，是 SQL 工具包，连带 alembic/sqlalchemy）、`google==3.0.0`（命名空间占位包）、`logger==1.4`（py2 时代包）、`ipdb`/`ipykernel` 应移 dev 依赖 → 清理后 `uv export` 重新生成 requirements.txt。
- `metrics_dumper.py:37-49` 用 importlib 按路径 exec `eval/utils.py` → 改正常 import。
- `hf_download.py` 全文 Tab 缩进与仓库 4 空格不符；`upload.py`（硬编码个人 repo）、`main.py`（hello world）、`test.py`（手工 32B 草稿）应移出仓库根或删除。
- `exp.py` 配置构建在模块层无 `if __name__` 保护，`scan_threshold.py` `from exp import *` 连带执行整个 sweep；阈值扫描参数不进 exp_name，事后只能翻 summary 对目录。
- `graph_decode.py` 返回静态缓冲视图（step/verify），当前消费方安全但 docstring 未警示"下次回放失效"→ 补一句或加 `.clone()` 调试开关。
- 注释残留：`collect_confidence.py:415-424` 的 AI 自问自答注释（"Ah, EvalMTBench inherits..."）直接暴露复制粘贴未消化。

---

## 五、修复优先级路线图

**第一批（科学计量正确性，改完需重跑受影响基线）**：B5、B6+B7、B8、B9+B10、B12、B13、B17、B18。
**第二批（功能可用性）**：B1、B2、B3、B4、B11、B15、B19、R3。
**第三批（无人值守鲁棒性）**：R1、R2、R4、R6、R7、B36、B37。
**第四批（结构收敛，防止回归）**：D1 的五个公共抽取、D2 模式注册表、D3 参数拆分、S4 死代码清理、pyproject 清理。

---

## 修复记录（2026-03）

**CUHLM 一组（经论文 arXiv:2505.11788v2 复核后修复）：**
1. `baselines.py uncertainty_decoding`：`determine_transfer_strategy` 改传 `_get_current_probs(prob_history)`（概率分布），不再传原始 logits；删除死赋值 `compressed_prob` 与重复的 `_get_current_probs`（B6）。
2. `communication.py _calculate_compressed_vocab_size`：O(V²) 双循环 → cumsum+searchsorted O(V) 向量化（V=151936 单次 26ms）；docstring 引用勘误（式 (24) 离线 → 实为式 (26) 在线规则）；新增输入总质量防御告警（|Σ−1|>0.05 时警告"疑似 logits"）（B7）。
3. `_apply_top_k_compression`（两份拷贝）：`len(probs)` → `probs.shape[-1]`；花式索引 → `scatter_(-1, ...)`，对 (..., V) 多维正确（B40）。
4. `--uncertainty_threshold` 接线补全：`cee_cuhlm` 硬编码 0.8 → `getattr(args, ...)`；两处 `PreciseCUHLM` 构造新增该参数（`uncertainty_decoding` 原本已接）。
5. 新增回归测试 `test/test_comm_simulation.py::TestCuhlmCompressedVocab`（5 项：向量化 vs 暴力等价、均匀分布 k\*=1、logits 输入告警、阈值边界语义、2D 压缩），全套 16 passed。

**注意**：k\* 修复后 `uncertainty_decoding`（cuhlm 基线）的压缩字节数计费口径会变化（此前 k\* 是噪声）——与修复前跑出的 CUHLM 通信字节数不可比，涉及该基线的实验需重跑。

**B8 + R1 修复（2026-03 第二批）：**
1. B8：五个方法（`ceesd_without_arp`、`adaptive_tridecoding`、`cee_cuhlm`、`cee_dssd`、`cee_dsd`）在 `generated_tokens` 结算前统一插入 `if prefix.shape[1] > max_tokens: prefix = prefix[:, :max_tokens]` 截断（F33 契约）。比报告多修一处：`adaptive_tridecoding` 属"截断存在但排在 metrics 之后"的变体，吞吐同样高估；`cee_cuhlm` 的 γ1+γ2+1 预算放大已移除（`cee_dssd` 的 `buffer_size = max_tokens + γ1 + γ2 + 1` 是 KV-cache 容量分配，语义正确，未动）。**这五个方法的吞吐/质量数字修复前后不可比，需重跑。**
2. R1：① `DDQNAgent.save` / replay buffer / `save_training_status` 全部改 tmp + `os.replace` 原子写（杀进程窗口变无害）；② `load` 重写：文件不存在→返回 False（正常 fresh 路径），损坏/不兼容→抛 RuntimeError 并附原因与抢救提示；新增 `_assert_state_dict_compatible` 先纯校验键集与形状再应用，杜绝半加载；③ `RLNetworkAdapter` 候选链改 `_try_load`：损坏候选高声 CRITICAL、改名 `.corrupt` 保留证据（不会被下次 save 覆盖）、继续尝试下一候选；buffer 损坏降级为告警跳过。
3. R1 勘误：复核发现 `stop_training`（`auto_train_manager.py:464-476`）**原本就是 SIGTERM→等10s→SIGKILL 升级式**，报告原表述"直接 SIGKILL"不准确；但因训练子进程未装 SIGTERM handler，默认处置仍是立即终止——真正的数据丢失窗口由①的原子写关闭。
4. 新增回归测试 `test/test_rl_checkpoint_durability.py`（8 项：原子写无 .tmp 残留、损坏抛错不静默、shape 不兼容抛错且网络零污染、正常往返、buffer 损坏降级）。全套件对照：HEAD 25 failed/89 passed → 修复后 25 failed/97 passed（25 个失败为 temperature_sampling 等既有问题，与改动无关；`test_eval_mixed_adaptive.py` 在 HEAD 上即因 `eval/eval_mixed_adaptive.py` 缺失而无法收集）。

**B/R 第三批修复（2026-03，崩溃级 + 高性价比 medium + R4/R5 + B10/B11/B12/B13）：**
1. B1：`auto_train_manager_adaptive.py` 与配套测试**整体删除**（经确认零生产引用、构造即 TypeError、与 exp.py adaptive 路径重复）。
2. B2/B3/B4：SpecBench 两处元组解包 + metrics 落盘；`speculative_decoding_with_bandwidth` 补 `**kwargs`；acc-head 注册表存在性检查 + argparse 默认值后置解析（fresh clone 不再裸崩）。
3. B9/B10：dssd 截断上移 metrics 前；**7 个方法**（dist_spec/uncertainty_decoding/tridecoding/ceesd_without_arp/cee_cuhlm/cee_dssd/cee_dsd）循环顶统一接入 `_stop_at_eos`；engine `_check_stopping_criteria` 新增 `prompt_len` 参数扫描整个生成段，修掉 chunk 中部 EOS 漏检。**B10 使全部方法在 EOS 后停止计算与通信——生成长度/时延/质量口径全部改变，涉及实验需重跑。**
4. B11/B12/B13：4 个 eval 文件补 `seed_set.add`；`finalize_verification` 改用 actual_gamma（尾轮 bonus 从目标分布采样）；3 处 RL `select_config` 补 `training=not disable_rl_update`（评估冻结探索）。
5. B14/B25/B36/B38/B41/B42：trace 双写删除；legacy 路径固定优先级；best_tps 回退 -1.0；校验标签勘误；rebuild 三处 clamp+显式尾掩码；调试器浮点路径。
6. R4/R5：summary JSON 原子写（exp.py + exp_threshold）；exp.py 子进程改 `start_new_session` + 异常 killpg；`torch.load` 加 `weights_only=True`（checkpoint/acc-head）；replay buffer 白名单 unpickler。
7. R10 查证结论：`profile_cee_dsd.py` 的 `ProfiledBaselines` 与生产 `cee_dsd` 同名注册到类级共享 dict，但**全仓无任何 import 链**（纯休眠地雷）；建议后续把插桩改为上下文包裹式或改名注册，暂未动。
8. 全量测试对照：HEAD 25 failed/89 passed → 本批后 **21 failed/97 passed**（少的 4 个 = B1 删除的 adaptive 测试文件；21 个失败全部为 temperature_sampling 等既有问题，`comm` 对照**零新增失败**）。

**结构性设计缺陷修复（2026-03，D1–D5 专批，按 D5→D2→D4→D3→D1 顺序独立提交，与 bug 修复严格分批）：**
1. **D5（a561913）**：auto_train_manager 的 TPS 停滞收敛判定与课程学习（TPS 系统性下降）冲突——检测启动脚本中的 `--curriculum_{bw,ntt}_{start,end}` 激活课程、`--curriculum_total_steps` 定契约，课程未结束时 `check_convergence` 恒 False（一次性告警）；子进程 Step 日志实时跟踪进度，无契约时保守不干预。
2. **D2（041960f）**：eval_mode 知识单点化——新增 `src/mode_features.py`（`ModeSpec(models/uses_main_rl/uses_little_rl/acc_head)` 全模式表），engine.load_model 五分支、baselines 的 RL 集合/acc_head 档位、utils 的 little-RL 路径门全部改查表；`test_mode_features.py` 10 项含四组字段与迁移前硬编码集合的**逐元素相等断言**（迁移等价性证明）。register.py 加固：异函数同名注册 raise（同函数多别名合法）、名字校验前移、删除 `hasattr` 反射兜底（`--eval_mode __init__` 类任意属性名不再被当解码方法）、`speculative_decoding_with_bandwidth` 补显式注册。
3. **D4（1e8df4b）**：RL 决策/学习分离——学习步从 `select_config` 热路径抽出为 `_flush_transition`（与 `save` 的终止转移共用，buffer 序列逐条等价）；DDQN 折扣因子 `self.gamma`→`self.discount`（与 draft γ 撞名）；**吸收 B20**：ceesd_without_arp 的 RL 草稿长度改方法局部变量，不再残留全局 Namespace（test_cee_refactor 增污染回归断言）。act() 纯决策 + Action dataclass 需重设计三段握手并重跑 RL 验证，如实缓议。
4. **D3（204f65e）**：utils 1615→1190 行、五种职责拆三种——`src/sampling.py`（8 个分布运算）、`src/trace_io.py`（轨迹 IO 去重：~30 行重复解析块下沉 `_parse_runs`，硬编码 5.0 提为 `BANDWIDTH_FLOOR_MBPS`，裸 except 收窄）、`src/model_zoo.py`；utils 再导出保兼容（既有 import 零改动）；RL 默认路径解析四份复制下沉 `resolve_rl_agent_paths()`。有意保留：parse_arguments 的 `os.makedirs`（迁移需逐一核点全部入口）；baselines 构造期 `getattr(...) or spec.latest_path` 防御（测试可达）。过程事故一次（脚本拼接截断 exp.py，py_compile 掩盖）——git 恢复 + 加完整性护栏重做，教训入档。
5. **D1（5ad02b2，可安全范围）**：`eval/model_ids.py` 单表——9 处内联 model_id if-链（各 ~40 行）迁入规则表，**各数据集规则原样保留**（xsum 的 Llama-2 矛盾映射、mixed 的 base/chat 区分、eval/mt_bench* 的硬失败兜底 vs 其它静默 vicuna——分叉如实入表并注释，test_model_ids.py 8 项锁住差异）；communication.py 死码 `__call__`（零调用者、与 simulate_transfer 口径已漂移）删除。
6. **D1 明确缓议**（结构统一会无声改变测量数字，不属可安全重构）：①10 方法解码骨架抽取（时序插桩点存在真实测量语义差异）；②simulator 构造工厂（10 处在 8 个维度漂移，工厂退化为 kwargs 转发）；③指标结算块统一（"重复"实为吞吐分母/token 计账的语义漂移，部分即已知未修 bug 领域）；④副本脚本改 import 主实现（标定数据对应旧动力学，需用户决策重标定）。
7. 全量测试：D 系列五批各自独立验证 127→137→145 passed / 0 failed（累计新增 18 项回归测试）+ opportunistic 10 passed。

**第五批修复（2026-03，测试套件复活：21 个既有失败 triage 全清）：**
1. **B45（新发现，真生产 bug）**：`cee_cuhlm` 的精确仿真分支（baselines.py:4302）缺 `channel_gain`/`noise_power_watt` 必填参数——自 c3b51c6 引入参数起 `use_precise_comm_sim=True` 即 TypeError。因默认 False 而潜伏；6 个测试红了一路无人看。已对齐 1544 处完整调用修复。
2. B19 部分：6 处 `args.use_cuda_graph` 直取改 `getattr(..., False)`（最小 Namespace/编程构造的 args 不再崩）。
3. 测试腐化清理（API 演进后替身/断言未跟）：
   - `VerificationInputs` 构造补 `selected_draft_p`（4 处，值=gather(draft_probs,2,indices).squeeze(-1)，形状 (b,γ)）
   - `.n` 已从 AcceptanceResult 移除 → 断言改走 `materialize_acceptance` 派生（2 处）；fake 构造 `accepted_count` 张量化（3 处）
   - comm 替身补 `set_round`/`flush_round` 协议方法（两个文件的基类，覆盖 3 个子类）
   - `_FakeCache` 补 `generate_with_rebuilt_topk_metadata`（返回三元组、meta=None 透传）+ `**kwargs` 构造
   - 10 个 fake verify 签名补 `draft_topk_history=None`
   - `avg_effective_proposal_len` 键断言删除（生产已移除该键）
4. 旧设计期望更新（语义确认后改写）：cee_cuhlm 全接受时 target 真跳过（CUHLM 机会跳过设计）→ target_forward_times 断言 1→0；little/draft accepted 断言 0→>0（写于对应阶段接入前）；DecodingAdapter 期望补 `stop_mode='cumulative'`。
5. 全量测试：**21 failed/106 passed → 127 passed / 0 failed（首次全绿）**；test_opportunistic_rl_training 10 passed。

**第四批修复（2026-03，模型解析链 B22/B23/B24/B26/B33）：**
1. B24：`--draft_model/--target_model` 默认 codellama-7b/70b（不可解析）移除，改 default=None；必填检查放在 parse 后置块开头（parser.error，先于 acc-head/RL 解析——否则 None 会让 canonicalize 以 AttributeError 崩，实测发现的次生问题），model_zoo 内保留同款检查兜底直接调用方。
2. B23：未部署模型（deepseek-1.3b/6.7b、vicuna-7b-v1.5/v1.3）从 zoo 占位符改为解析阶段显式 ValueError；zoo 表同步删除占位条目。
3. B22：vocab_size 查表移回 zoo 映射之前用原始别名命中（此前映射后查表，别名全 miss，每次 parse 读 config.json 或走网络）；重复键 llama-2-70b 去重。
4. B26：zoo 与 CANONICAL_MODEL_ALIASES 的分歧别名（qwen-3-0.6b/1.7b/14b、llama-2-chat-7b）并入 canonical 表统一归一——registry 在 zoo 映射前拿到原始别名（RL 解析先于 model_zoo），两种拼法此前指向不同 series、静默错过已注册对；canonicalize 另加空名防御。
5. B33：few_shot 对不支持任务（mt_bench 等对话式基准）显式 UserWarning 并返回空串，--num_shots 失效不再静默。
6. 测试：新增 test/test_model_resolution.py 9 项（canonical 归一/必填/未部署拒绝/vocab 别名命中+路径映射/few-shot 告警）；test_dsd_target_placement 与 test_opportunistic_rl_training 的 argv 按 B24 必填语义适配。全量对照：21 failed/106 passed，零回归。



为公平起见，以下高风险区域经逐项推导/换算核验**未发现问题**：
- `rl_reward.py` 全部模式量纲一致（λ EMA tokens/s × s = tokens；byte 计价 MB 换算正确；slo 的 ms→s 正确；`t_total` 有下界）。
- RL 网络状态单位修复到位：所有调用点统一走 `bandwidth_*_mbps`/`ntt_*_ms` 显式属性；`_check_state_units` 有防御。
- `FactoredRecurrentQNetwork` 的 Q 布局与 `select_config` 解码严格一致（C-order 核验）。
- 通信单位换算族：RTT trace ms→s 及 history ×1000 回毫秒、Mbps/MBps/bps/Bps→B/s、拥塞 NTT 公式、trace 按仿真时间推进、能耗只按 tx_time、Precise 香农容量——全部正确。
- `graph_decode.py` CUDA Graph 实现严谨：结果图内拷出、mask/pos 由标量图内重算、pad 行 KV 作废论证成立、扩容断言兜底、capture_post 仅 temp==0（与 argmax one-hot 语义一致）。
- `decoding_ops.py` 接受/回滚核心（cummin 连续接受、rejection_offset、top-k+均匀尾两域混合残差采样）逐步推导无误；课程采样边界（progress 两端、low≤high、几何插值单调）正确。
- `nvml.py` init/shutdown 有 try/finally，句柄不泄漏；debug 探针全部环境变量门控，生产路径零开销。
- README 声明的 best→latest→legacy→scratch 四步回退在 main agent 上与实现一致（little agent 的 opportunistic 例外是一个文档级偏差）。
