# ===========
# Speculative Decoding Evaluation Commands
# ===========

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e vicuna --draft_model vicuna-68m --target_model tiny-vicuna-1b --max_tokens 128 --temp 0 # 68m 1b

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e vicuna --draft_model tiny-vicuna-1b --target_model vicuna-13b-v1.5 --max_tokens 128 --temp 0 # 1b 13b

# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e vicuna --draft_model vicuna-68m --target_model vicuna-13b-v1.5 --max_tokens 128 --temp 0 # 68m 1b

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e llama --draft_model llama-68m --target_model tiny-vicuna-1b --max_tokens 128 --temp 0 # 68m 1b

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e llama --draft_model llama-68m --target_model tiny-vicuna-1b --max_tokens 128 --temp 0 # 68m 1b

# ===========
# Auto-Regressive Evaluation Commands
# ===========

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode large --target_model tiny-vicuna-1b --draft_model vicuna-68m --gamma 5 -n 1 -e vicuna --max_tokens 128 --temp 0

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode large --target_model vicuna-13b-v1.5 --draft_model vicuna-68m --gamma 5 -n 1 -e vicuna --max_tokens 128 --temp 0

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e llama --draft_model llama-68m --target_model tiny-vicuna-1b --max_tokens 128 --temp 0 # 68m 1b

# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e llama --draft_model tiny-vicuna-1b --target_model llama-2-13b --max_tokens 128 --temp 0 # 1b 13b

# CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode sd --gamma 5 -n 1  -e llama --draft_model llama-68m --target_model llama-2-13b --max_tokens 128 --temp 0 # 68m 1b

# ===========
# Tridecoding Evaluation Commands
# ===========
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29051 eval/eval_mt_bench.py --eval_mode tridec --gamma 5 -n 1  -e vicuna --smallest_model vicuna-68m --draft_model tiny-vicuna-1b --target_model vicuna-13b-v1.5 --max_tokens 128 --temp 0.1 