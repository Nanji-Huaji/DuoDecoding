#!/bin/bash
#
# CEE-SD RL Agent 重训脚本
# 串行训练多个模型系列，每次启动前轮询GPU空闲状态
#
# 用法:
#   bash cmds/retrain_ceesd_rl.sh [model_series...]
#
# 示例:
#   bash cmds/retrain_ceesd_rl.sh llama qwen qwen15   # 指定系列
#   bash cmds/retrain_ceesd_rl.sh                       # 默认全部3个系列

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

# ============================================================
# 配置
# ============================================================
DEFAULT_SERIES=("llama" "qwen" "qwen15")
SERIES=("${@:-${DEFAULT_SERIES[@]}}")

# GPU 轮询参数
GPU_ID=${GPU_ID:-0}
MIN_FREE_MB=${MIN_FREE_MB:-30000}   # GPU 空闲显存阈值 (MB)
POLL_INTERVAL=${POLL_INTERVAL:-60}   # 轮询间隔 (秒)

LOG_DIR="logs/retrain"
mkdir -p "$LOG_DIR"

# ============================================================
# 函数
# ============================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_gpu_free() {
    local gpu_id=$1
    local min_free_mb=$2

    local free_mb
    free_mb=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v gpu="$gpu_id" '$1 == gpu {print $2}')

    if [[ -z "$free_mb" ]]; then
        log "WARNING: 无法查询 GPU $gpu_id 状态，跳过检查"
        return 0
    fi

    if [[ "$free_mb" -ge "$min_free_mb" ]]; then
        return 0  # 空闲
    else
        log "GPU $gpu_id 显存不足: ${free_mb}MB 空闲 / 需要 >= ${min_free_mb}MB"
        return 1  # 繁忙
    fi
}

wait_for_gpu() {
    local gpu_id=$1
    local min_free_mb=$2

    log "等待 GPU $gpu_id 空闲 (需要 >= ${min_free_mb}MB)..."
    while ! check_gpu_free "$gpu_id" "$min_free_mb"; do
        log "  GPU $gpu_id 仍繁忙, ${POLL_INTERVAL}s 后重试..."
        sleep "$POLL_INTERVAL"
    done
    log "GPU $gpu_id 已空闲，开始训练"
}

train_series() {
    local series=$1
    local log_file="$LOG_DIR/retrain_${series}_$(date '+%Y%m%d_%H%M%S').log"

    log "============================================================"
    log "开始训练系列: $series"
    log "日志文件: $log_file"
    log "============================================================"

    # 等待 GPU 空闲
    wait_for_gpu "$GPU_ID" "$MIN_FREE_MB"

    # 启动训练
    uv run python auto_train_manager.py --model "$series" 2>&1 | tee "$log_file"
    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log "系列 $series 训练正常结束"
    else
        log "系列 $series 训练异常退出 (exit code: $exit_code)"
    fi

    return $exit_code
}

# ============================================================
# 主流程
# ============================================================

log "================================================================"
log "CEE-SD RL Agent 重训启动"
log "系列列表: ${SERIES[*]}"
log "目标 GPU: $GPU_ID | 最小空闲: ${MIN_FREE_MB}MB | 轮询间隔: ${POLL_INTERVAL}s"
log "================================================================"

START_TS=$(date +%s)
FAILED_SERIES=()

for series in "${SERIES[@]}"; do
    if train_series "$series"; then
        log "✓ $series 完成"
    else
        log "✗ $series 失败"
        FAILED_SERIES+=("$series")
    fi
    # 系列之间短暂冷却，释放 GPU 资源
    log "冷却 10s..."
    sleep 10
done

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))

log "================================================================"
log "全部训练结束"
log "总耗时: ${DURATION}s ($(($DURATION / 60))m $(($DURATION % 60))s)"
if [[ ${#FAILED_SERIES[@]} -gt 0 ]]; then
    log "失败系列: ${FAILED_SERIES[*]}"
else
    log "所有系列训练成功 ✓"
fi
log "================================================================"
