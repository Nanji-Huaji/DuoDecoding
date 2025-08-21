    CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --num_processes 1 \
    --main_process_port 29051 \
    eval/eval_mt_bench.py \
    --eval_mode tridecoding \
    --gamma 5 \
    --gamma1 6 \
    --gamma2 6 \
    -n 1 \
    -e vicuna \
    --draft_model vicuna-68m \
    --target_model tiny-vicuna-1b \
    --max_tokens 128 \
    --temp 0 \
    --exp_name tri_quality_examination-1b__$(date +%Y%m%d_%H%M%S) 
