#!/bin/bash
# v2 接力：等 v1 链（训练+探针）完全结束 → 论文域钉参重训 v2 → v2 探针 → 追加汇总。
# v1 教训：eval_mixed 的 curriculum 默认(20,50/0,5)会劫持命令行 46/50；
# v2 用 curriculum start=end=46,46/50,50 钉死论文域（已验证采样恒定），
# trace 带宽随机性 + L1 拥塞 NTT 动态在正确基值上叠加。
set -u
cd "$(dirname "$0")/.."

V1_WATCHER=${V1_WATCHER:-2712084}

echo "[chain-v2] $(date '+%F %T') 等 v1 链(watcher $V1_WATCHER)结束..."
while kill -0 "$V1_WATCHER" 2>/dev/null; do sleep 30; done
echo "[chain-v2] $(date '+%F %T') v1 链结束，开始论文域钉参重训"

OUT=checkpoints/rl_inregime_pin EXP=rl_inregime_pin PORT=29962 \
  bash scripts/run_rl_inregime_training.sh 1 500 \
  > exp_logs/nohup_train_inregime_pin.log 2>&1
echo "[chain-v2] $(date '+%F %T') v2 训练 exit=$?"

NEWCKPT=checkpoints/rl_inregime_pin PORT=29795 \
TAGA=t5a_ceesd_inregime2 TAGB=t5a_ceesd_inregime2_sn \
PROBE_TAGS="t5a_ceesd_inregime2 t5a_ceesd_inregime2_sn" \
SUMMARY=exp_logs/t5a_inregime2_probe_summary.txt \
  bash scripts/chain_probe_after_inregime_training.sh \
  > exp_logs/nohup_chain_probe_v2.log 2>&1

echo "[chain-v2] $(date '+%F %T') 全部完成"
