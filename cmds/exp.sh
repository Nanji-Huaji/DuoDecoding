# # 自回归 Baseline
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes 1 \
    --main_process_port 29051 \
    eval/eval_mt_bench.py \
    --eval_mode tridecoding \
    --gamma 5 \
    -n 1 \
    -e vicuna \
    --use-gpt_fast_model false \
    --draft_model vicuna-68m \
    --target_model tiny-vicuna-1b \
    --max_tokens 128 \
    --temp 0 \
    --exp_name tridecoding_expirement_$(date +%Y%m%d_%H%M%S) \


# CUDA_VISIBLE_DEVICES=1 accelerate launch \
#     --num_processes 1 \
#     --main_process_port 29051 \
#     eval/eval_mt_bench.py \
#     --eval_mode large \
#     --gamma 5 \
#     -n 1 \
#     -e vicuna \
#     --draft_model vicuna-68m \
#     --target_model vicuna-13b-v1.5 \
#     --max_tokens 128 \
#     --temp 0 \
#     --exp_name auto-regressive-baseline-13b_$(date +%Y%m%d_%H%M%S) \

# 猜测解码
# CUDA_VISIBLE_DEVICES=0 accelerate launch \
#     --num_processes 1 \
#     --main_process_port 29051 \
#     eval/eval_mt_bench.py \
#     --eval_mode sd \
#     --gamma 5 \
#     -n 1 \
#     -e vicuna \
#     --draft_model vicuna-68m \
#     --target_model vicuna-13b-v1.5 \
#     --max_tokens 128 \
#     --temp 0 \
#     --exp_name speculative_decoding_target_13b_draft_68m_$(date +%Y%m%d_%H%M%S) \

# CUDA_VISIBLE_DEVICES=0 accelerate launch \
#     --num_processes 1 \
#     --main_process_port 29051 \
#     eval/eval_mt_bench.py \
#     --eval_mode sd \
#     --gamma 5 \
#     -n 1 \
#     -e vicuna \
#     --draft_model vicuna-68m \
#     --target_model tiny-vicuna-1b \
#     --max_tokens 128 \
#     --temp 0 \
#     --exp_name speculative_decoding_target_1b_draft_68m_$(date +%Y%m%d_%H%M%S) \

# CUDA_VISIBLE_DEVICES=1 accelerate launch \
#     --num_processes 1 \
#     --main_process_port 29051 \
#     eval/eval_mt_bench.py \
#     --eval_mode sd \
#     --gamma 5 \
#     -n 1 \
#     -e vicuna \
#     --draft_model tiny-vicuna-1b \
#     --target_model vicuna-13b-v1.5 \
#     --max_tokens 128 \
#     --temp 0 \
#     --exp_name speculative_decoding_target_13b_draft_1b_$(date +%Y%m%d_%H%M%S) \
