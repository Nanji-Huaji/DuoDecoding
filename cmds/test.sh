CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes 1 \
    --main_process_port 29051 \
    eval/eval_mt_bench_noeval.py \
    --eval_mode large \
    --gamma 5 \
    -n 1 \
    -e vicuna \
    --little_model llama-68m \
    --draft_model tiny-llama-1.1b \
    --target_model tiny-llama-1.1b \
    --max_tokens 128 \
    --temp 0 \
    --exp_name test_expirement_$(date +%Y%m%d_%H%M%S) 