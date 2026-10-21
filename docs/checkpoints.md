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

## 3. 训练与晋升（staging → promote）

训练默认仍写生产路径；要启用 staging：

```bash
python eval/eval_mixed.py ... --use_rl_adapter --disable_rl_update False   --rl_output_dir checkpoints/staging/<run-id> --rl_init_strategy fresh
```

- latest/best/.buffer 写到 `checkpoints/staging/<run-id>/<role>/<pair>/`；
- 训练过程不碰 `checkpoints/rl/**`（评测侧另有 `disable_rl_update` 只读守卫）；
- `resume` 时若 staging 已有 checkpoint 就续训；没有则回退读生产/legacy 并迁入 staging。

验证后显式晋升：

```bash
# 预览
python tools/promote_checkpoint.py   --staging checkpoints/staging/<run-id> --dry-run
# 执行（自动备份旧文件 + 更新 manifest provenance）
python tools/promote_checkpoint.py   --staging checkpoints/staging/<run-id> --evidence "q15 golden PASS"
```

晋升行为：

- 校验 payload（policy/target/optimizer + 动作格）；
- 旧生产文件移到 `checkpoints/archive/<timestamp>/<role>/<pair>/`；
- 复制到 `checkpoints/rl/<role>/<pair>/`；
- 重算 sha256/size，写回 `manifest.json`，并记录
  `provenance: {promoted_at, staging, evidence}`；
- 之后按需重新收割 golden、上传 HF。

## 4. 评测与分发

1. **评测**：只读。`--disable_rl_update` 已保证不写 checkpoint（PR #13）；
2. **分发**：`tools/fetch_checkpoints.py` 下载并校验，
   `tools/upload_checkpoints.py` 上传 RL 镜像。

## 5. 统一下载（RL agents + acc heads）

`tools/fetch_checkpoints.py` 是唯一的 checkpoint 下载入口，按 manifest 逐条取源：

| 组 | 来源 | 说明 |
|---|---|---|
| `rl` | `<user>/DuoDecoding-checkpoints`（本仓库镜像） | RL agents；需先上传 |
| `acc_head` | `ArcticHuaji/specdecpp-acc-heads`（已有公共仓库） | 由 `acc_head_registry.json` 决定；默认补 `scripts/setup/assets.json` 里的 **6 个论文必需对**，`--all-acc-heads` 补全 12 对 |
| `legacy` | 同上（manifest 里已有源的条目） | 历史实验，默认不下载 |

```bash
# 只下论文必需的 acc heads（RL 镜像就绪前；默认 6 个必需对）
python tools/fetch_checkpoints.py --groups acc_head

# 补全全部 12 对 / 指定 pair
python tools/fetch_checkpoints.py --groups acc_head --all-acc-heads
python tools/fetch_checkpoints.py --groups acc_head --acc-head-pairs qwen3-0.6b--to--qwen3-1.7b

# 全部实验 checkpoint（RL 镜像就绪后）
python tools/fetch_checkpoints.py --groups rl,acc_head

# 只校验本地完整性 / 预览下载计划
python tools/fetch_checkpoints.py --verify-only
python tools/fetch_checkpoints.py --dry-run

# 指定 RL 镜像仓库（覆盖 manifest.hub.repo_id）
python tools/fetch_checkpoints.py --rl-repo <user>/DuoDecoding-checkpoints
```

acc heads 的"是否齐全"按 `model.safetensors` / `pytorch_model.bin` 判断，已有目录会跳过；
manifest 只索引本机已有的文件，因此新 clone 上 acc heads 由 registry 补齐后再跑
`checkpoint_manifest.py --write` 即可纳入 sha 校验。

基础模型（Llama/Qwen 权重、数据集）不在本脚本范围，仍用现有入口：
`scripts/download_models.py`（模型 + acc heads）与
`scripts/setup/download_assets.py`（论文最小集 / 全量资产）。

## 6. Hugging Face Hub（推荐）

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

## 7. 迁移记录

- 2026-10-21：`checkpoints/rl_agents` 改名为 `rl` 并保留兼容链接；
  acc heads 留在 SpecDec_pp 子模块，纳入同一份 manifest；
  停止用 git 跟踪 `checkpoints/**` 二进制（历史仍保留）；
- 同期停止用 git 跟踪 `checkpoints/**` 二进制（历史仍保留），只跟踪
  `manifest.json` 与 README。

## 8. 待办

- [x] 训练可写 `staging/`（`--rl_output_dir`）+ 显式 `promote`（含备份/provenance）；
- [ ] `rl_agent_registry` / `acc_head_registry` 改为经 manifest 解析并校验 sha；
- [ ] 评测结果记录 `checkpoint_ids`（与 `git_sha` 并列）；
- [ ] golden manifest 绑定 checkpoint id（替代裸路径 + sha）。
