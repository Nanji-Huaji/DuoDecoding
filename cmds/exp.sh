# 自回归 Baseline
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
#     --exp_name speculative_decoding-1b_$(date +%Y%m%d_%H%M%S) \

    # CUDA_VISIBLE_DEVICES=0 accelerate launch \
    # --num_processes 1 \
    # --main_process_port 29051 \
    # eval/eval_mt_bench.py \
    # --eval_mode tridecoding \
    # --gamma 5 \
    # --gamma1 6 \
    # --gamma2 6 \
    # -n 1 \
    # -e vicuna \
    # --draft_model vicuna-68m \
    # --target_model tiny-vicuna-1b \
    # --max_tokens 128 \
    # --temp 0 \
    # --exp_name tri_quality_examination-1b__$(date +%Y%m%d_%H%M%S) 

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

# CUDA_VISIBLE_DEVICES=0 accelerate launch \
#     --num_processes 1 \
#     --main_process_port 29051 \
#     eval/eval_mt_bench.py \
#     --eval_mode tridecoding_with_bandwidth \
#     --gamma 5 \
#     -n 1 \
#     -e vicuna \
#     --use-gpt_fast_model false \
#     --draft_model vicuna-68m \
#     --target_model tiny-vicuna-1b \
#     --max_tokens 128 \
#     --temp 0 \
#     --edge_cloud_bandwidth 22 \
#     --edge_end_bandwidth 4 \
#     --cloud_end_bandwidth 4 \
#     --exp_name tridecoding_width_bandwidth_$(date +%Y%m%d_%H%M%S) \

# CUDA_VISIBLE_DEVICES=0 accelerate launch \
#     --num_processes 1 \
#     --main_process_port 29051 \
#     eval/eval_mt_bench.py \
#     --eval_mode uncertainty_decoding \
#     --gamma 5 \
#     -n 1 \
#     -e vicuna \
#     --use-gpt_fast_model false \
#     --draft_model vicuna-68m \
#     --target_model tiny-vicuna-1b \
#     --max_tokens 128 \
#     --temp 0 \
#     --edge_cloud_bandwidth 22 \
#     --edge_end_bandwidth 4 \
#     --cloud_end_bandwidth 4 \
#     --exp_name uncertainty_decoding_$(date +%Y%m%d_%H%M%S) \
#     --uncertainty_threshold 0.8 
