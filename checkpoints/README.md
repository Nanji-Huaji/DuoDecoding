# Checkpoints

本目录是唯一的 checkpoint 根目录；`manifest.json` 是权威索引（路径、sha256、角色、
状态、旧路径映射）。

- 下载 / 校验 / 恢复旧路径：`python tools/fetch_checkpoints.py`
- 重新生成清单：`python tools/checkpoint_manifest.py --write`
- 上传到 Hugging Face：`python tools/upload_checkpoints.py --repo-id <user>/DuoDecoding-checkpoints`

二进制不进入 git（历史提交仍保留）。`checkpoints/rl_agents` 是指向 `rl` 的
符号链接；acc heads 仍在 `src/SpecDec_pp/checkpoints/acc_head`（子模块），
由同一份 manifest 索引。下载脚本按 `remote_path` 把文件放回仓库相对位置，
代码路径无需改动。

模型卡（RL adapter 的架构与用途）见 `MODEL_CARD.md`。
