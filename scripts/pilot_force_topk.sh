#!/bin/bash
# pilot: 验证 --rl_force_topk 后字节随 k 拉开 (2跑 ~8min)
set -u
cd "$(dirname "$0")/.."
GPU=1; PORT=29840
ACC=src/SpecDec_pp/checkpoints/acc_head
COMMON="--data_path data --num_shots 3 --max_tokens 128 --eval_data_num 80
  --temp 0.0 --sample_seed 1234 --random_sample --num_samples_per_task 1
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
  --edge_end_bandwidth 563 --cloud_end_bandwidth 46
  --ntt_ms_edge_cloud 15 --ntt_ms_edge_end 0.317 --batch_delay 0.05
  --ntt_trace_file data/sigcomm-5gmemu-5g-mmWave-uplink-data/ping/static/5g/away_p1.list
  --comm_round_trip_mode per_round --no-charge_residual_payload --transfer_top_k_cap 0
  --small_draft_threshold 0.6 --draft_target_threshold 0.7
  --uncertainty_threshold 0.8 --use_stochastic_comm --stochastic_ntt
  --use_rl_adapter --disable_rl_update --use_cuda_graph
  --gamma1 16 --gamma2 16 --gamma 5 --rl_force_threshold 0.4
  --small_draft_acc_head_path $ACC/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3
  --draft_target_acc_head_path $ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3
  --main_rl_path checkpoints/rl_r2_weaklink/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth
  --little_rl_path checkpoints/rl_r2_weaklink/little/llama-68m--to--tiny-llama-1.1b/latest.pth
  --main_rl_best_path checkpoints/rl_r2_weaklink/main/tiny-llama-1.1b--to--llama-2-13b/best.pth
  --little_rl_best_path checkpoints/rl_r2_weaklink/little/llama-68m--to--tiny-llama-1.1b/best.pth"

for k in 64 1024; do
  tag="r215p_bw0.5_k${k}"
  rm -rf exp/$tag; mkdir -p exp/$tag
  echo "### $tag ($(date +%H:%M:%S))"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode adaptive_tridecoding -e $tag \
    --edge_cloud_bandwidth 0.5 --transfer_top_k $k --rl_force_topk $k $COMMON \
    > exp_logs/${tag}.log 2>&1
  echo "  exit=$?"
done
.venv/bin/python - <<'PYEOF'
import json, glob
for k in [64, 1024]:
    m = json.load(open(glob.glob(f"exp/r215p_bw0.5_k{k}/*metrics.json")[0]))
    b = m["edge_cloud_data_bytes"]; f = m["target_forward_times"]
    print(f"k={k:>4}: WAN字节={b/1e6:.2f}MB ({b/f/1e3:.1f}KB/轮) Thr={m['throughput']:.2f} "
          f"tokfwd={m['generated_tokens']/f:.2f} comm={m['communication_time']:.1f}s")
PYEOF
echo "PILOT_DONE"
