# 环境配置与迁移指南

本仓库的代码可以 `git clone`，但**跑不起来所需的东西大多不在 git 里**：模型权重、接受率预测头
（acc_head）、HF 数据集缓存。这份文档说明如何在新机器上把环境一次配好。

## 快速开始

```bash
git clone git@github.com:Nanji-Huaji/DuoDecoding.git && cd DuoDecoding
bash scripts/setup/setup_env.sh                 # 论文最小集，约 52GB
```

跑完会打印一份自检报告；**全部 ✓ 即可开工**。脚本是幂等的，任何一步失败都可以直接重跑续传。

```bash
# 可选：需要 70B 扩展实验（额外 ~386GB）
bash scripts/setup/setup_env.sh --only paper-70b
# 可选：只补资产、不动 Python 环境
bash scripts/setup/setup_env.sh --skip-deps
# 可选：只做一次自检
bash scripts/setup/setup_env.sh --check-only
```

## 机器要求

| 项 | 要求 | 说明 |
|---|---|---|
| GPU | **≥2 张 48GB**（本仓库在 2×RTX A6000 上开发） | 三档模型需同时驻留：68M + 1.1B + 13B(4bit) |
| 显存 | 13B 以 4bit 加载，论文默认 `--target_quantization 4bit` | 实测两卡各占约 20-26GB |
| 磁盘 | `paper` 集 ~60GB，`all` 需 ~900GB | 模型目录 `llama/` 被 gitignore，不占仓库 |
| Python | 3.10（`.python-version`） | 由 `uv` 自动装 |
| 网络 | 可访问 HuggingFace（或镜像） | gated 模型只能走官方站点 |

## 目录与资产来源速查

| 资产 | 位置 | 在 git 里？ | 获取方式 |
|---|---|---|---|
| 源码 / 配置 / 成本模型 | `src/` `eval/` `configs/` | ✓ | `git clone` |
| 任务数据 jsonl | `data/*.jsonl` | ✓ | `git clone` |
| **模型权重** | `llama/<name>/` | ✗ gitignore | `download_assets.py` |
| **acc_head 预测头** | `src/SpecDec_pp/checkpoints/acc_head/` | ✗ **不在 git** | `download_assets.py`（HF: `ArcticHuaji/specdecpp-acc-heads`） |
| HF 数据集缓存 | `~/.cache/huggingface/` | ✗ | `download_assets.py` 预热 |

> ⚠️ **acc_head 是最容易漏的一项**：它是方法的核心组件，缺失时**代码能 import、能跑完，
> 但结果全错**。`acc_head_registry.json` 里有 12 对，论文必需的是
> `llama-68m--to--tiny-llama-1.1b` 与 `tiny-llama-1.1b--to--llama-2-13b`。

## 脚本说明

```
scripts/setup/
├── setup_env.sh        主入口：预检 → uv → 依赖 → 目录 → 资产 → 自检
├── assets.json         清单：模型集、模型来源、数据集、必需 acc_head
├── download_assets.py  按清单下载（幂等、续传、支持镜像/代理）
└── check_env.py        自检：依赖 / GPU / 资产 / **代理是否真的在转发**
```

### 常用参数

```bash
# 下载器
python scripts/setup/download_assets.py --list              # 只看清单
python scripts/setup/download_assets.py --only paper         # 论文集
python scripts/setup/download_assets.py --only llama-68m,llama-160m
python scripts/setup/download_assets.py --hf-endpoint https://hf-mirror.com
python scripts/setup/download_assets.py --all-acc-heads      # 全部 12 对（~600MB）

# 自检
python scripts/setup/check_env.py --smoke                    # 含一次 GPU 前向
python scripts/setup/check_env.py --require-proxy             # 代理不可用即判错
```

## gated 模型（`meta-llama/*`）

Llama-2 系列需要授权，**镜像站不支持**：

```bash
export HF_TOKEN=hf_xxxxxxxx        # Settings → Access Tokens
# 并先在 https://huggingface.co/meta-llama/Llama-2-13b-hf 点接受许可
```

未设置 `HF_TOKEN` 时脚本会**明确跳过并说明原因**，不会静默失败。

## 网络：一个容易误判的坑

代理软件（clash/mihomo）对 `CONNECT` 请求**一律返回 `200 Connection established`——
即使上游节点已经失效**。因此"curl 首行是 200"**不能**作为代理可用的证据：

```bash
# ✗ 错误判据：只看到 CONNECT 应答
curl -sI -x http://127.0.0.1:7890 https://github.com | head -1
#    → HTTP/1.1 200 Connection established   ← 陷阱！上游其实已死

# ✓ 正确判据：看真实 HTTP 状态码（000 = 失败）
curl -s -o /dev/null -w "HTTP=%{http_code}\n" -x http://127.0.0.1:7890 https://github.com
```

`check_env.py` 用的是**真正完成一次 TLS 握手**的判据，所以能立刻暴露这类"假通"。

### 节点失效后必须重载配置

订阅更新**不会**自动生效：clash 只在启动时读一次 `config.yaml`。

```bash
# 1) 重新下载订阅（机场会轮换节点地址，旧配置里的地址会全部失效）
curl -sL -A "clash-verge/v1.5.0" -o /tmp/new_config.yaml "<你的订阅链接>"
# 2) 备份后替换
cp -a ~/clash/config.yaml ~/clash/config.yaml.bak-$(date +%Y%m%d_%H%M%S)
install -m 664 /tmp/new_config.yaml ~/clash/config.yaml
# 3) 通过 API 重载（无需重启进程，不影响其它会话）
curl -s -X PUT "http://127.0.0.1:9090/configs?force=true" \
     -H "Content-Type: application/json" \
     -d '{"path":"'"$HOME"'/clash/config.yaml"}' -w "reload HTTP=%{http_code}\n"
```

诊断技巧：用 clash 的测速接口看**是否有节点活着**（全部报错 = 订阅整体失效）：

```bash
curl -s "http://127.0.0.1:9090/proxies/<节点名>/delay?timeout=3000&url=http://www.gstatic.com/generate_204"
```

### HuggingFace 镜像

```bash
bash scripts/setup/setup_env.sh --hf-endpoint https://hf-mirror.com
# 等价于 export HF_ENDPOINT=https://hf-mirror.com
```

镜像适合下载**公开**模型（`JackFram/*`、`TinyLlama/*`）；**gated 模型仍须走官方站点**。

## git 推送（HTTPS 失败时）

如果 `git push` 报 `gnutls_handshake() failed` / `SSLEOFError`，先按上面的方法确认代理**真的在转发**。
若 HTTPS 长期不通（例如校园网封 443），可改走 SSH——**22 端口往往仍然可达**：

```bash
ssh -T git@github.com          # Permission denied (publickey) = 网络通、只差密钥
git remote set-url origin git@github.com:Nanji-Huaji/DuoDecoding.git
```

私钥有 passphrase 时需要 agent 解锁（密码不必外泄）：

```bash
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519
```

> 另一个常见故障：`GIT_ASKPASS` 指向 VSCode 卸载后遗留的脚本，导致
> `fatal: cannot run .../askpass.sh`。`unset GIT_ASKPASS` 即可。

## 故障排查

| 症状 | 原因 | 修复 |
|---|---|---|
| `gnutls_handshake() failed` / `SSLEOFError` | 代理节点失效（CONNECT 仍返回 200） | 重下订阅 + API 重载；用 `check_env.py` 确认 |
| 所有节点 `An error occurred in the delay test` | 订阅整体失效 | 更新订阅并重载；确认 `expire` 未过期 |
| `cannot run .../askpass.sh` | `GIT_ASKPASS` 指向已删除文件 | `unset GIT_ASKPASS` |
| `Permission denied (publickey)` 但服务器说 `Server accepts key` | 私钥有 passphrase 且无 agent | `ssh-add ~/.ssh/id_ed25519` |
| 模型加载报缺文件 | 只下了部分分片 | 重跑 `download_assets.py`（幂等续传） |
| 评测卡在 HF 重试 | 数据集缓存缺失 | `download_assets.py --skip-models --skip-accheads` |
| 结果对但指标异常（准确率极低） | **acc_head 缺失或未加载** | 检查 `src/SpecDec_pp/checkpoints/acc_head/` |
| `CUDA out of memory` | 13B 未用 4bit，或其它进程占卡 | 加 `--target_quantization 4bit`；`nvidia-smi` 查占用 |
| 吞吐数字波动大 | 机器上有其它租户/任务在跑 | 用确定性指标（tok/round、云端调用次数）而非 wall-clock |

## 迁移检查清单

```bash
bash scripts/setup/setup_env.sh --smoke        # 预检 + 依赖 + 资产 + 自检 + GPU 前向
```

1. `git clone` 后确认 `data/` 下有 8 个 `*.jsonl`（外加 1 个 5G 数据集目录）
2. `scripts/setup/setup_env.sh` 输出 **全部 ✓**
3. `llama/` 下三个论文模型齐全（68m / 1.1b / 13b）
4. `src/SpecDec_pp/checkpoints/acc_head/` 下两对检查点就位（**不在 git 里**）
5. `check_env.py` 里「本地代理实际转发」为 ✓（若需联网）
6. 冒烟：`bash scripts/run_baseline_table.sh 0`
