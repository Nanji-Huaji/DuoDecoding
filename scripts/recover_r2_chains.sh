#!/bin/bash
# 恢复链: 清掉单位错误污染的 r215 数据 → 重跑 NTT15+trace 全套(跳过训练,
# checkpoints 未受污染) → 补跑 NTT10 域 DRA 探针(首链死于"编辑运行中脚本") → 双汇总
set -u
cd "$(dirname "$0")/.."
PING=data/sigcomm-5gmemu-5g-mmWave-uplink-data/ping/static/5g/away_p1.list

echo "[recover] 清除污染的 r215 数据 $(date +%H:%M:%S)"
rm -rf exp/r215_*

echo "[recover] NTT15+ping-trace 阶梯+DRA探针 (TRAIN=0) $(date +%H:%M:%S)"
TRAIN=0 NTT_MS=15 TAGPFX=r215 NTT_TRACE_FILE=$PING NTT_TRACE_SCALE=1.0 \
PORT=29830 SUMMARY=exp_logs/r215_ladder_summary.txt \
  bash scripts/r2_weaklink_ladder.sh >> exp_logs/nohup_recover.log 2>&1

echo "[recover] NTT10 域 DRA 探针补跑 (静态阶梯已在, 幂等跳过) $(date +%H:%M:%S)"
TRAIN=0 NTT_MS=10 TAGPFX=r2 \
PORT=29831 SUMMARY=exp_logs/r2_ladder_summary.txt \
  bash scripts/r2_weaklink_ladder.sh >> exp_logs/nohup_recover.log 2>&1

echo "[recover] 双汇总 $(date +%F\ %H:%M:%S)"
echo "===== NTT15 + 真实RTT trace ====="; cat exp_logs/r215_ladder_summary.txt
echo; echo "===== NTT10 (合成NTT, 对照) ====="; cat exp_logs/r2_ladder_summary.txt
echo "=== 恢复链完成 ==="
