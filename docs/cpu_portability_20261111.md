# CPU 测试可移植性修复（2026-11-11）

干净 clone（`58034cd`，子模块代码已初始化、无本地 `models/`、无 RL checkpoint）全量两种收集模式均 `18 failed / 1361 passed / 7 skipped`。本批只改测试、测试 helper 与 CPU 门禁入口，不改 `src/`、golden 数据/manifest、依赖清单或 GPU 工具。

## 失败根因与改动

| 文件 | 根因 | 改动 |
|---|---|---|
| [test_checkpoints_cli.py](<../test/test_checkpoints_cli.py>) | `status`/`pull` 直接读生产 manifest 与生产文件 | `status` 用 `tmp_path` 最小 manifest+文件并 patch `ROOT`/`cm.MANIFEST`，保留 0/1 返回码；`pull` 用临时 manifest 真跑 `fetch_checkpoints --dry-run`，验证 0（文件就位）与 1（缺失）两种输入 |
| [test_fixed_prefix_distribution_probe.py](<../test/test_fixed_prefix_distribution_probe.py>) | 写死仓库 `.venv/bin/python` | 子进程改用 `sys.executable` |
| [test_model_resolution.py](<../test/test_model_resolution.py>) | `auto` 测试假定生产 `models/` 已存在 | 临时 repo root + `configs/model_repos.json`，自给覆盖“目录存在→local”与“不存在→hub” |
| [test_golden_baselines.py](<../test/test_golden_baselines.py>)、[test_golden_gammar.py](<../test/test_golden_gammar.py>) | 14 项资产检查直接 `read_bytes()`，缺文件即 `FileNotFoundError` | 共用 [golden_assets.py](<../test/golden_assets.py>)（测试 helper）：先核对全部已存在文件的 `sha256_16`，再处理缺失；路径统一从仓库根解析 |
| [test/conftest.py](<../test/conftest.py>) | 无严格模式开关 | 注册 `--require-golden-assets` 与 `require_golden_assets` fixture |
| [tools/run_cpu_gate.py](<../tools/run_cpu_gate.py>) | 无资产策略 | 两模式透传 `--require-golden-assets`，固定加 `-rs`，manifest 记录 `golden_assets.required/flag`；source 前后快照逻辑未改 |

资产策略（[test_golden_asset_policy.py](<../test/test_golden_asset_policy.py>) 用小字节文件覆盖匹配/漂移/默认缺失/严格缺失/混合缺失+漂移）：

- 默认：缺失 → `pytest.skip` 并写明“未验收，非通过”（**不是 pass**）；
- 严格：`--require-golden-assets` → 缺失 `pytest.fail`；
- 任意已存在文件指纹漂移 → 两种策略下都失败；先检查全部存在文件，缺失 role 不会掩盖另一 role 的漂移；
- golden 数据、`manifest.json`、`sha256_16` 均未修改。

## 证据（本机 clean-sim，`git archive HEAD` + 子模块代码符号链接，无 `.venv`/`models`/RL checkpoint）

```text
red   （HEAD 测试，5 个文件）        : 18 failed, 68 passed, 7 skipped
green （修复后，7 个文件, default）  : 93 passed, 21 skipped   （14 资产 skip + 7 原始 run skip）
green （修复后, importlib）          : 93 passed, 21 skipped
strict（--require-golden-assets 两模式）: 14 failed, 45 passed, 7 skipped
```

本机完整资产 + strict 定向运行：`107 passed, 7 skipped`，原 14 项 `sha256_16` 断言全部真实执行通过。原始日志在 `.refactor/cpu-portability-20261111/clean-sim-*.log`（忽略目录）。

## 完整 CPU 门禁

本机使用 `--require-golden-assets` 跑完整门禁，默认导入与 importlib 均为
**1400 passed、7 skipped、163 subtests**（91.71s / 86.07s）。两次均执行了
14 项历史权重指纹检查，运行前后源码一致。记录位于
`.refactor/cpu-portability-20261111/final-cpu-strict/`。

Lead 另补充两种策略、两种 role 顺序的交叉测试，并修正普通模式下混合
“缺失+漂移”失败消息误称启用严格模式的问题：先复现 2 failed，修正后
资产策略与门禁接线共 22 passed。

## 干净远端与隔离安装

远端另建源码副本，在 `b0d1cfbf544211d4779a5d86beb9a2604e2513f2`
运行门禁；没有补入生产 models、RL checkpoint 或 acc-head 注册表。

| 模式 | 结果 | 耗时 |
|---|---|---|
| default | 1386 passed、21 skipped、163 subtests | 188.05s |
| importlib | 1386 passed、21 skipped、163 subtests | 144.66s |

21 项 skip 包括 14 项缺失权重与 7 项缺失历史 run，日志逐项列明原因。
两次返回码均为 0、源码端点一致，文件哈希和子模块提交与本机验收版本相同。
远端再单独开启严格模式，两种导入方式均得到预期的 **14 项缺资产失败**，
没有 error 或 skip；验证脚本同时核对了失败原因。

解释器来自新建的 `.venv-cpu-isolated`，不是原有共享环境：
Python 3.10.12、torch 2.6.0、transformers 4.57.6、pytest 9.1.1、ruff 0.15.4。
先安装开发工具，再以 `uv pip install --require-hashes -r requirements.txt`
安装锁定生产依赖；`uv pip check` 检查 145 个包通过，Baselines 与
FastChat 评测模板入口导入成功。离线缓存最初缺 ruff，随后从 PyPI 完成安装；
没有下载模型或改动原有环境、权重。远端 acc-head 注册表缺失的 warning 保留。

完整远端 CPU 日志已收集到 `.refactor/cpu-portability-20261111/isolated-clean-cpu/`。
本批 Ruff 检查只报告原有 [文档示例超长行](<../test/test_golden_baselines.py#L23>)，
未为消除它修改无关内容。

## 使用边界

- 需要完整权重指纹验收时，显式使用 `--require-golden-assets`，避免缺失被跳过。
- 干净 clone 的 14 项 skip 不算权重验收；本机严格门禁已实际通过这 14 项。
- 本批不含 GPU 结果，也不改变生产算法或历史 golden 数字。
