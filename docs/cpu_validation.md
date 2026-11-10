# CPU 验证与依赖来源

实施状态见[阶段 A 状态](<refactor_stage_a_status.md>)，约定见[路线图](<refactor_roadmap_20261110.md>)。

## 1. 唯一生产依赖真源

[项目声明](<../pyproject.toml>)与[生产锁](<../uv.lock>)是唯一真源。[pip 清单](<../requirements.txt>)只是同锁导出，不能独立手改包、版本、marker 或 hash。本次未修改两份生产输入；离线导出的 146 条记录包括 `torch==2.6.0`、`transformers==4.57.6`、`fastchat==0.1.0`，不再维护 `fschat` 分叉。

[导出工具](<../tools/export_locked_requirements.py>)只调用 `uv export --frozen --offline --no-dev --no-emit-project --format requirements-txt`，不解析新依赖、不安装包、不下载 Python。缓存使用工作区忽略目录。默认检查；只有显式 `--write` 才同时再生成清单与[导出身份记录](<../test/golden/dependency_export.json>)：

```bash
.venv/bin/python -B tools/export_locked_requirements.py
.venv/bin/python -B tools/export_locked_requirements.py --write
```

记录包含生产输入 SHA-256、完整导出字节 SHA-256、生成命令与 uv 版本。本次使用 uv 0.8.15。保留全部产物 hash；pip 会因此启用 hash 校验，安装产物必须符合生产锁中的允许值。不要为了让检查变绿而手改身份记录；更新依赖时必须重新离线导出并评审清单差异。

[依赖契约测试](<../test/test_dependency_contract.py>)不要求测试机安装 uv，不执行安装或网络操作。它将完整清单与绑定生产输入的已验证导出记录比较，因此删除传递包、修改平台条件或修改 hash 都会失败，而不是只检查版本属于锁。另有关键版本/名称、直接依赖约束及开发工具 pin 检查；使用现有 `packaging` 实现 PEP 440。Python 3.10 缺少必需 TOML 解析器会失败，不能 skip 后假绿。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider test/test_dependency_contract.py
```

## 2. CPU 验收入口

先停止所有源码写入，再使用[CPU 门禁](<../tools/run_cpu_gate.py>)：

```bash
.venv/bin/python -B tools/run_cpu_gate.py \
  --output-dir .refactor/stage-a-20261110/cpu-gate-$(date +%s)
```

- 使用调用它的 Python；不安装任何包。输出目录必须不存在，不复用旧证据。
- 依次执行完整测试目录的默认收集与 `--import-mode=importlib`，记录日志、JUnit、解释器及关键包版本。
- 设置 GPU 隐藏、Hugging Face 离线、禁自动 pytest 插件及禁 pyc 变量；移除 `PYTEST_ADDOPTS`/`PYTEST_PLUGINS`，并用 `-o addopts=` 防止宿主配置偷偷只收集而不执行。
- 对主要源码/测试/工具目录、主仓库 Python 文件（包括测试实际导入的[根实验入口](<../exp.py>)）、依赖清单及 pytest 配置记录前后快照，包含未跟踪源码；任一测试非零退出或端点快照不相等都会失败。
- **端点快照相等不证明中间从未改动后恢复。** 子模块只记录 Git 修订/状态，不是所有嵌套字节的哈希；模型权重也不在该源码快照内。最终冻结须由执行者确认。
- 环境变量不是通用网络/设备安全沙箱，也不能阻止任意配置 hook 或恶意 Python 路径注入。门禁不是生产依赖安装验证器。

[入口单元测试](<../test/test_cpu_gate_runner.py>)包括真实 pytest 子进程反例：即使环境要求加载外来插件、配置写了 `--collect-only`，故意失败的测试仍会真实执行，两模式都必须返回失败。

## 3. 开发工具与隔离安装限制

[开发工具清单](<../requirements-dev.txt>)固定 `pytest==9.1.1` 与 `ruff==0.15.4`。这只是直接 pin，不是完整 dev 传递锁；生产锁不包含这两个工具。不能把当前共享环境的测试通过，称为清洁环境安装已验证。

本次创建了工作区内隔离工具环境，但空工作区缓存中没有 pytest；离线安装明确失败，**清洁环境安装尚未验证**。未修改共享环境、未联网补包。scoped Ruff 来自宿主已有的 0.15.4，不代表开发清单已安装成功。允许包下载后，才可在独立环境补做完整生产锁加开发工具安装及门禁；不得用 `--system-site-packages` 冒充清洁安装。

复现离线工具安装尝试（不要把工具环境当成已装完整生产依赖的测试环境）：

```bash
UV_PYTHON_DOWNLOADS=never uv venv \
  .refactor/stage-a-20261110/dependency/tool-venv-new --python 3.10
UV_CACHE_DIR="$PWD/.refactor/stage-a-20261110/dependency/uv-cache" \
  uv pip install --offline \
  --python .refactor/stage-a-20261110/dependency/tool-venv-new/bin/python \
  -r requirements-dev.txt
```

## 4. 证据边界

CPU 测试可以核对数学 toy law、接口、fake/scripted 状态机、真实微型随机初始化后端、注册/导入及模拟通信账；不能证明真实权重的跨设备执行、CUDA graph capture/replay、性能或数值质量。

GPU/真实权重验收仍须单独授权，记录设备 UUID、只读权重身份、有效配置、预算和原始结果。历史工件、CPU fake cache 或 graph 构造测试不能替代当前版本的真实执行。最终报告应分别列出 CPU 门禁、离线导出、清洁安装与 GPU 验收的状态，不合并成一个“全部通过”。
