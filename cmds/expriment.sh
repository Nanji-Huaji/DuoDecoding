# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e llama --draft_model llama-68m --target_model tiny-vicuna-1b --max_tokens 128 --temp 0 # 68m 1b

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e llama --draft_model tiny-vicuna-1b --target_model llama-2-13b --max_tokens 128 --temp 0 # 1b 13b

# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e llama --draft_model llama-68m --target_model llama-2-13b --max_tokens 128 --temp 0 # 68m 1b

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e llama --draft_model llama-68m --target_model tiny-vicuna-1b --max_tokens 128 --temp 0 # 68m 1b

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode para_sd --gamma 5 -n 1  -e llama --draft_model tiny-vicuna-1b --target_model llama-2-13b --max_tokens 128 --temp 0 # 1b 13b 