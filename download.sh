huggingface-cli download meta-llama/Llama-3.2-1B --local-dir llama/llama-3.2-1b 

# 下载 Vicuna-13B-v1.5
huggingface-cli download lmsys/vicuna-13b-v1.5 \
    --local-dir ./vicuna/vicuna-13b-v1.5 \
    --local-dir-use-symlinks False \
    --resume-download

# 下载llama-68m
huggingface-cli download JackFram/llama-68m \
    --local-dir ./llama/llama-68m \
    --local-dir-use-symlinks False \
    --resume-download

# 下载llama-2-13b-hf
huggingface-cli download meta-llama/Llama-2-13b-hf \
    --local-dir ./llama/llama-2-13b-hf \
    --local-dir-use-symlinks False \
    --resume-download