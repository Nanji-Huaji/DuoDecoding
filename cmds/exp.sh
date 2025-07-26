# # 自回归 Baseline
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --num_processes 1 \
    --main_process_port 29051 \
    eval/eval_mt_bench.py \
    --eval_mode large \
    --gamma 5 \
    -n 1 \
    -e vicuna \
    --draft_model vicuna-68m \
    --target_model tiny-vicuna-1b \
    --max_tokens 128 \
    --temp 0 \
    --use-gpt-fast-model true \
    --exp_name auto-regressive-baseline_$(date +%Y%m%d_%H%M%S) 