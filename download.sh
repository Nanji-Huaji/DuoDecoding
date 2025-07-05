huggingface-cli download meta-llama/Llama-3.2-1B --local-dir llama/llama-3.2-1b 

# 下载 Vicuna-13B-v1.5
huggingface-cli download lmsys/vicuna-13b-v1.5 \
    --local-dir ./llama/vicuna-13b-v1.5 \
    --local-dir-use-symlinks False \
    --resume-download

