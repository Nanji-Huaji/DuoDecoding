#!/bin/bash

# 实验脚本 - 带宽测试配置

# 配置5: 563 Mbps (end-edge) + 34.6 Mbps (end/edge-cloud)
echo "Running experiment: bandwidth_test_563_34_6"
CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --num_processes 1 \
    --main_process_port 29051 \
    eval/eval_mt_bench.py \
    --eval_mode tridecoding_with_bandwidth \
    -e llama \
    --draft_model tiny-llama-1.1b \
    --target_model Llama-2-13b \
    --little_model llama-68m \
    --max_tokens 128 \
    --temp 0.0 \
    --gamma1 4 \
    --gamma2 24 \
    --edge_end_bandwidth 563.0 \
    --edge_cloud_bandwidth 34.6 \
    --cloud_end_bandwidth 34.6 \
    --exp_name bandwidth_test_563_34_6_$(date +%Y%m%d_%H%M%S)

# 配置4: 350 Mbps (end-edge) + 25.0 Mbps (end/edge-cloud)
echo "Running experiment: bandwidth_test_350_25_0"
CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --num_processes 1 \
    --main_process_port 29052 \
    eval/eval_mt_bench.py \
    --eval_mode tridecoding_with_bandwidth \
    -e llama \
    --draft_model tiny-llama-1.1b \
    --target_model Llama-2-13b \
    --little_model llama-68m \
    --max_tokens 128 \
    --temp 0.0 \
    --gamma1 4 \
    --gamma2 24 \
    --edge_end_bandwidth 350.0 \
    --edge_cloud_bandwidth 25.0 \
    --cloud_end_bandwidth 25.0 \
    --exp_name bandwidth_test_350_25_0_$(date +%Y%m%d_%H%M%S)

# 配置3: 200 Mbps (end-edge) + 15.0 Mbps (end/edge-cloud)
echo "Running experiment: bandwidth_test_200_15_0"
CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --num_processes 1 \
    --main_process_port 29053 \
    eval/eval_mt_bench.py \
    --eval_mode tridecoding_with_bandwidth \
    -e llama \
    --draft_model tiny-llama-1.1b \
    --target_model Llama-2-13b \
    --little_model llama-68m \
    --max_tokens 128 \
    --temp 0.0 \
    --gamma1 4 \
    --gamma2 24 \
    --edge_end_bandwidth 200.0 \
    --edge_cloud_bandwidth 15.0 \
    --cloud_end_bandwidth 15.0 \
    --exp_name bandwidth_test_200_15_0_$(date +%Y%m%d_%H%M%S)

# 配置2: 100 Mbps (end-edge) + 5.0 Mbps (end/edge-cloud)
echo "Running experiment: bandwidth_test_100_5_0"
CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --num_processes 1 \
    --main_process_port 29054 \
    eval/eval_mt_bench.py \
    --eval_mode tridecoding_with_bandwidth \
    -e llama \
    --draft_model tiny-llama-1.1b \
    --target_model Llama-2-13b \
    --little_model llama-68m \
    --max_tokens 128 \
    --temp 0.0 \
    --gamma1 4 \
    --gamma2 24 \
    --edge_end_bandwidth 100.0 \
    --edge_cloud_bandwidth 5.0 \
    --cloud_end_bandwidth 5.0 \
    --exp_name bandwidth_test_100_5_0_$(date +%Y%m%d_%H%M%S)

# 配置1: 33.2 Mbps (end-edge) + 0.14 Mbps (end/edge-cloud)
echo "Running experiment: bandwidth_test_33_2_0_14"
CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --num_processes 1 \
    --main_process_port 29055 \
    eval/eval_mt_bench.py \
    --eval_mode tridecoding_with_bandwidth \
    -e llama \
    --draft_model tiny-llama-1.1b \
    --target_model Llama-2-13b \
    --little_model llama-68m \
    --max_tokens 128 \
    --temp 0.0 \
    --gamma1 4 \
    --gamma2 24 \
    --edge_end_bandwidth 33.2 \
    --edge_cloud_bandwidth 0.14 \
    --cloud_end_bandwidth 0.14 \
    --exp_name bandwidth_test_33_2_0_14_$(date +%Y%m%d_%H%M%S)

echo "所有实验配置已完成"
