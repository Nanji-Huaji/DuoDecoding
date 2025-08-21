CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes 1 \
    --main_process_port 29051 \
    eval/eval_mt_bench.py \
    --eval_mode sd \
    --gamma 5 \
    -n 1 \
    -e vicuna \
    --draft_model vicuna-68m \
    --target_model tiny-vicuna-1b \
    --max_tokens 128 \
    --temp 0 \
    --exp_name sd_qual_exmaination-1b_$(date +%Y%m%d_%H%M%S) 