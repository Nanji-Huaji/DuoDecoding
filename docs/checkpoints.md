# Checkpoint 管理（2026-10-21）

checkpoint 和口径/参数一样是实验资产：没有清单就无法回答"这张表用的是哪份权重"。
本方案把所有权重收进单一根目录、用 manifest 索引，并通过 Hugging Face Hub 分发。

## 1. 目标布局

```
checkpoints/                    # 主物理根（可用 DDQ_CHECKPOINT_ROOT 覆盖）
  manifest.json                 # 唯一权威清单（入 git，覆盖两个根）
  README.md
  rl/{main,little}/<pair>/      # RL agents（原 checkpoints/rl_agents）
  <series>/ ...                 # qwen15/qwen/llama/vicuna/best（历史系列目录）
  rl_* / *_backup_* / legacy... # 历史实验，manifest 标 status=legacy

src/SpecDec_pp/checkpoints/acc_head/   # 第二物理根（SpecDec_pp 子模块）
  <pair>/<run>/...              # 验收头；由同一份 manifest 索引
```

为什么不把 acc heads 也搬进主 `checkpoints/`：它们在 `SpecDec_pp` **子模块**里，
移出会让子模块变脏（或需要改子模块仓库）。因此采用"一个 manifest、两个物理根"：

- manifest 的 `remote_path` 是仓库相对路径（`checkpoints/...` 或
  `src/SpecDec_pp/checkpoints/...`），Hugging Face 仓库保持同样布局；
- 下载后文件落在原位置，**现有代码、golden manifest、脚本都不需要改路径**；
- `checkpoints/rl_agents -> rl` 符号链接保留旧路径兼容。

## 2. manifest.json

```json
{
  "version": 1,
  "root": "checkpoints",
  "hub": {"repo_id": "Nanji-Huaji/DuoDecoding-checkpoints", "repo_type": "model", "revision": "main"},
  "entries": {
    "rl/main/<pair>/latest.pth": {
      "path": "rl/main/<pair>/latest.pth",
      "sha256": "719cac...", "size": 4986658,
      "role": "rl_main", "pair": "<pair>", "status": "production",
      "legacy_paths": ["checkpoints/rl_agents/main/<pair>/latest.pth"]
    }
  }
}
```

- `status`: `production`（评测/golden 用）/ `legacy`（历史实验）；
- `role`: `rl_main` / `rl_little` / `acc_head` / `series_legacy` / `experiment_legacy`；
- `legacy_paths`: 下载后自动建的兼容链接；
- 文件级 sha256 是唯一内容标识，**golden/结果工件应引用 manifest 条目而不是裸路径**。

## 3. 工作流

1. **训练**：写 `checkpoints/staging/<run-id>/`，不碰生产路径；
2. **晋升**：人工确认 eval/golden 证据后，把产物移入 `rl/...` 或 `acc_head/...`，跑
   `python tools/checkpoint_manifest.py --write` 更新清单；
3. **评测**：只读。`--disable_rl_update` 已保证不写 checkpoint（PR #13）；
4. **分发**：`tools/upload_checkpoints.py` 上传，`tools/fetch_checkpoints.py` 下载并校验。

## 4. 统一下载（RL agents + acc heads）

`tools/fetch_checkpoints.py` 是唯一的 checkpoint 下载入口，按 manifest 逐条取源：

| 组 | 来源 | 说明 |
|---|---|---|
| `rl` | `<user>/DuoDecoding-checkpoints`（本仓库镜像） | RL agents；需先上传 |
| `acc_head` | `ArcticHuaji/specdecpp-acc-heads`（已有公共仓库） | 由 `acc_head_registry.json` 的 `hf_repo`/`hf_subpath` 决定，逐文件下载 |
| `legacy` | 同上（manifest 里已有源的条目） | 历史实验，默认不下载 |

```bash
# 只下论文必需的 acc heads（RL 镜像就绪前）
python tools/fetch_checkpoints.py --groups acc_head

# 全部实验 checkpoint（RL 镜像就绪后）
python tools/fetch_checkpoints.py --groups rl,acc_head

# 只校验本地完整性 / 预览下载计划
python tools/fetch_checkpoints.py --verify-only
python tools/fetch_checkpoints.py --dry-run

# 指定 RL 镜像仓库（覆盖 manifest.hub.repo_id）
python tools/fetch_checkpoints.py --rl-repo <user>/DuoDecoding-checkpoints
```

基础模型（Llama/Qwen 权重、数据集）不在本脚本范围，仍用现有入口：
`scripts/download_models.py`（模型 + acc heads）与
`scripts/setup/download_assets.py`（论文最小集 / 全量资产）。

## 5. Hugging Face Hub（推荐）

仓库建议命名：`<user>/DuoDecoding-checkpoints`（model 类型，私有或公开均可）。

```bash
# 一次性：登录 + 创建 repo
hf auth login
python tools/upload_checkpoints.py \
  --repo-id <user>/DuoDecoding-checkpoints --create --only production

# 新机器：下载 + 校验 + 建兼容链接
python tools/fetch_checkpoints.py --only production
```

- 上传内容 = manifest 里 `status=production` 的文件（RL agents + acc heads，约 700 MB）；
- 历史实验（`legacy`）默认不传；需要时 `--only all`；
- 下载后会逐文件校验 sha256，不匹配立即失败；
- 网络受限的机器可先在有网环境下载后拷贝整个 `checkpoints/`。

### GitHub LFS 备选

GitHub 可达但 LFS 免费额度通常只有 1 GB 存储/带宽，本仓库生产权重约 700 MB、
历史实验另有 ~900 MB，容易超额；且 LFS 指针与现有 100 个已跟踪二进制叠加会继续
膨胀历史。**优先 HF**，GitHub 只放 manifest + 脚本。

## 6. 迁移记录

- 2026-10-21：`checkpoints/rl_agents` 改名为 `rl` 并保留兼容链接；
  acc heads 留在 SpecDec_pp 子模块，纳入同一份 manifest；
  停止用 git 跟踪 `checkpoints/**` 二进制（历史仍保留）；
- 同期停止用 git 跟踪 `checkpoints/**` 二进制（历史仍保留），只跟踪
  `manifest.json` 与 README。

## 7. 待办

- [ ] 训练脚本改为写 `staging/` + 显式 promote；
- [ ] `rl_agent_registry` / `acc_head_registry` 改为经 manifest 解析并校验 sha；
- [ ] 评测结果记录 `checkpoint_ids`（与 `git_sha` 并列）；
- [ ] golden manifest 绑定 checkpoint id（替代裸路径 + sha）。
