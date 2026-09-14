#!/usr/bin/env bash
# DuoDecoding 一键环境配置（迁移到新机器时跑这个）
#
# 覆盖：系统预检 → uv → Python 依赖 → 目录骨架 → 模型/检查点/数据集 → 自检
# 特性：幂等（可重复跑，已完成的步骤自动跳过）、可断点续传、不硬编码任何密钥
#
# 用法:
#   bash scripts/setup/setup_env.sh                      # 默认：论文最小集（~52GB）
#   bash scripts/setup/setup_env.sh --only all           # 全部模型（~770GB）
#   bash scripts/setup/setup_env.sh --skip-deps          # 只补资产，不动 Python 环境
#   bash scripts/setup/setup_env.sh --hf-endpoint https://hf-mirror.com
#   bash scripts/setup/setup_env.sh --check-only         # 只做自检
#
# gated 模型（meta-llama/*）需要先:
#   export HF_TOKEN=hf_xxx    # 并在 HuggingFace 页面点接受 Llama-2 许可
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=$(pwd)
# 只在交互式终端上色；重定向到日志/CI 时输出纯文本，避免转义码污染
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; DIM=$'\033[2m'; RST=$'\033[0m'
else
  RED=''; GRN=''; YEL=''; DIM=''; RST=''
fi
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s✓%s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '%s!%s %s\n' "$YEL" "$RST" "$*"; }
die()  { printf '%s✗%s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }
step() { printf '\n%s── %s %s\n' "$DIM" "$*" "$RST"; }

ONLY=paper
PROXY=auto
HF_ENDPOINT_ARG=()
SKIP_DEPS=0
SKIP_ASSETS=0
CHECK_ONLY=0
SMOKE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --only)         ONLY="$2"; shift 2 ;;
    --proxy)        PROXY="$2"; shift 2 ;;
    --hf-endpoint)  HF_ENDPOINT_ARG=(--hf-endpoint "$2"); shift 2 ;;
    --skip-deps)    SKIP_DEPS=1; shift ;;
    --skip-assets)  SKIP_ASSETS=1; shift ;;
    --check-only)   CHECK_ONLY=1; shift ;;
    --smoke)        SMOKE=1; shift ;;
    -h|--help)      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              die "未知参数: $1（用 --help 查看）" ;;
  esac
done

say "════════════════════════════════════════════════════════════════"
say " DuoDecoding 环境配置   仓库: $ROOT"
say "════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────── 1. 系统预检
step "1/6 系统预检"

[ -f pyproject.toml ] && [ -f uv.lock ] || die "不在仓库根目录（缺 pyproject.toml / uv.lock）"
ok "仓库结构正常"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)
  ok "GPU: $GPU_LINE"
  GPU_N=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  [ "$GPU_N" -ge 2 ] && ok "检测到 $GPU_N 张卡（本仓库的并行 campaign 假设 ≥2）" \
                     || warn "只检测到 $GPU_N 张卡；scripts/camp_*.sh 默认用 GPU0/GPU1"
else
  warn "未检测到 nvidia-smi —— 没有 GPU 只能跑 CPU 冒烟测试"
fi

FREE_GB=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
case "$ONLY" in
  paper)     NEED_GB=60  ;;
  paper-70b) NEED_GB=450 ;;
  all)       NEED_GB=900 ;;
  *)         NEED_GB=60  ;;
esac
if [ "${FREE_GB:-0}" -ge "$NEED_GB" ]; then
  ok "磁盘可用 ${FREE_GB}GB（需要约 ${NEED_GB}GB）"
else
  warn "磁盘可用 ${FREE_GB}GB < 建议 ${NEED_GB}GB —— 可能装不下，考虑 --only paper"
fi

# ─────────────────────────────────────────── 2. uv
step "2/6 Python 工具链 (uv)"

if command -v uv >/dev/null 2>&1; then
  ok "uv 已安装: $(uv --version)"
else
  warn "未找到 uv，正在安装到 ~/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv 安装失败（检查网络/代理）"
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv 安装后仍不可用，请手动加入 PATH"
  ok "uv 安装完成: $(uv --version)"
fi

# ─────────────────────────────────────────── 3. 依赖
step "3/6 Python 依赖"

if [ "$CHECK_ONLY" -eq 1 ] || [ "$SKIP_DEPS" -eq 1 ]; then
  warn "跳过依赖安装"
else
  if [ -d .venv ]; then
    ok "已存在 .venv（Python $(.venv/bin/python -V 2>/dev/null | awk '{print $2}')），执行增量同步"
  fi
  # uv.lock 是权威来源；--frozen 保证在不同机器上装出完全一致的版本
  if uv sync --frozen 2>/dev/null; then
    ok "uv sync --frozen 完成（与 uv.lock 完全一致）"
  else
    warn "--frozen 失败（可能网络中断或 lock 与 pyproject 不一致），回退到普通 sync"
    uv sync || die "依赖安装失败：检查网络/代理，或看 docs/setup.md 的镜像配置"
    ok "uv sync 完成"
  fi
  [ -x .venv/bin/python ] || die ".venv/bin/python 不存在，venv 创建失败"
  ok "虚拟环境就绪: $(.venv/bin/python -V)"
fi

# ─────────────────────────────────────────── 4. 目录骨架
step "4/6 目录骨架"

for d in llama exp exp_logs checkpoints src/SpecDec_pp/checkpoints/acc_head; do
  if [ -d "$d" ]; then
    ok "$d/ 已存在"
  else
    mkdir -p "$d" && ok "$d/ 已创建"
  fi
done
[ -d data ] && ok "data/ 已存在（$(ls data/*.jsonl 2>/dev/null | wc -l) 个 jsonl 随 git 带过来）" \
            || warn "data/ 缺失 —— 通常在 git 里，检查是否完整 clone"

# ─────────────────────────────────────────── 5. 外部资产
step "5/6 模型 / acc_head / 数据集"

if [ "$CHECK_ONLY" -eq 1 ] || [ "$SKIP_ASSETS" -eq 1 ]; then
  warn "跳过资产下载"
else
  if [ -z "${HF_TOKEN:-}" ]; then
    warn "未设置 HF_TOKEN —— gated 模型（meta-llama/*）会被跳过"
    warn "  需要时: export HF_TOKEN=hf_xxx  （并在 HF 页面接受 Llama-2 许可）"
  else
    ok "HF_TOKEN 已设置"
  fi
  set +e
  .venv/bin/python scripts/setup/download_assets.py \
      --only "$ONLY" --proxy "$PROXY" "${HF_ENDPOINT_ARG[@]+"${HF_ENDPOINT_ARG[@]}"}"
  ASSET_RC=$?
  set -e
  [ "$ASSET_RC" -eq 0 ] && ok "资产下载全部完成" \
                        || warn "部分资产未完成（rc=$ASSET_RC）—— 修好网络后重跑本脚本即可续传"
fi

# ─────────────────────────────────────────── 6. 自检
step "6/6 环境自检"

SMOKE_ARG=(); [ "$SMOKE" -eq 1 ] && SMOKE_ARG=(--smoke)
set +e
.venv/bin/python scripts/setup/check_env.py --model-set "$ONLY" "${SMOKE_ARG[@]+"${SMOKE_ARG[@]}"}"
CHECK_RC=$?
set -e

say ""
say "════════════════════════════════════════════════════════════════"
if [ "$CHECK_RC" -eq 0 ]; then
  ok "环境配置完成，可以开始跑实验"
  say ""
  say "  论文主表:      bash scripts/run_baseline_table.sh 0"
  say "  环境复检:      .venv/bin/python scripts/setup/check_env.py --smoke"
  say "  补齐资产:      .venv/bin/python scripts/setup/download_assets.py --only paper"
else
  warn "自检有未通过项（见上面 ✗）—— 把输出贴出来即可定位"
  say ""
  say "  常见修复见 docs/setup.md；重新执行本脚本是安全的（幂等）"
fi
say "════════════════════════════════════════════════════════════════"
exit "$CHECK_RC"
