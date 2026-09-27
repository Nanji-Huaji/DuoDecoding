#!/bin/bash
# NTT15 重锚链: 等当前 R2@NTT10 链结束 → 用真实 ping trace 回放重跑阶梯+DRA探针
# 判据口径不变(每验证块 tok/fwd, 诚实计费); R2@NTT10 结果保留作敏感性对照。
set -u
cd "$(dirname "$0")/.."
CHAIN_PID=${CHAIN_PID:-2769810}
echo "[reanchor] 等待 R2@NTT10 链 (PID=$CHAIN_PID) 结束... $(date +%H:%M:%S)"
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 60; done
echo "[reanchor] 前链结束 $(date +%H:%M:%S)，启动 NTT15+ping-trace 重锚"

PING=data/sigcomm-5gmemu-5g-mmWave-uplink-data/ping/static/5g/away_p1.list
NTT_MS=15 TAGPFX=r215 NTT_TRACE_FILE=$PING NTT_TRACE_SCALE=1.0 \
PORT=29820 SUMMARY=exp_logs/r215_ladder_summary.txt \
  bash scripts/r2_weaklink_ladder.sh
echo "[reanchor] 全部完成 $(date +%F\ %H:%M:%S)"
