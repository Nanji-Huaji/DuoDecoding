#!/bin/bash
# v3 接力：reward 修复版训练（λ自适应 + 排队计价 + graph 标定）→ 探针 → 汇总。
set -u
cd "$(dirname "$0")/.."

echo "[chain-v3] $(date '+%F %T') v3 训练启动（reward 修复: λ自适应/排队计价/graph成本）"
OUT=checkpoints/rl_inregime_v3 EXP=rl_inregime_v3 PORT=29973 \
LAMBDA=0 RL_EXTRA="--rl_charge_queue" \
  bash scripts/run_rl_inregime_training.sh 1 500 \
  > exp_logs/nohup_train_inregime_v3.log 2>&1
echo "[chain-v3] $(date '+%F %T') v3 训练 exit=$?"

NEWCKPT=checkpoints/rl_inregime_v3 PORT=29799 \
TAGA=t5a_ceesd_inregime3 TAGB=t5a_ceesd_inregime3_sn \
PROBE_TAGS="t5a_ceesd_inregime3 t5a_ceesd_inregime3_sn" \
SUMMARY=exp_logs/t5a_inregime3_probe_summary.txt \
  bash scripts/chain_probe_after_inregime_training.sh \
  > exp_logs/nohup_chain_probe_v3.log 2>&1

echo "[chain-v3] $(date '+%F %T') 全部完成"
