
# RL-based Top-K Selector for Distributed Speculative Decoding

This repository contains the trained model checkpoints (`.pth`) and replay buffers for a Reinforcement Learning (RL) adapter designed to dynamically optimize the **Top-K sparsity parameter** in distributed speculative decoding scenarios.

## Overview

In distributed speculative decoding, the optimal sparsity level ($k$) for transmission depends heavily on real-time network conditions and model confidence. This DDQN (Deep Double Q-Network) agent learns to select the best $k$ to balance communication overhead (bandwidth/latency) with draft acceptance rates.

### Model Architecture
*   **Type:** Deep Double Q-Network (DDQN)
*   **Input (State Space):**
    *   Network Bandwidth (Normalized)
    *   Network Latency (Normalized)
    *   Draft Acceptance Probability
    *   Model Entropy
    *   Task ID (One-hot encoded: `mt_bench`, `gsm8k`, `cnndm`, `xsum`, `humaneval`, or `unknown`)
*   **Output (Action Space):** Selection of $k$ from candidates: `[0, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 32000]`.

## Repository Contents

*   `rl_adapter.pth`: The PyTorch state dictionary containing weights for the Policy Net, Target Net, and Optimizer.
*   `rl_adapter.pth.buffer`: (Optional) The serialized replay buffer, allowing for continued training from the current state.

## Usage

To use these checkpoints, ensure you have the `RLNetworkAdapter` class definition in your codebase.

```python
import torch
from your_module import RLNetworkAdapter  # Assuming the provided class is saved here

# 1. Initialize the adapter
# This automatically looks for 'checkpoints/rl_adapter.pth' by default
adapter = RLNetworkAdapter(args=None, device="cuda")

# 2. Get the optimal K based on current metrics
current_k = adapter.select_k(
    bandwidth_mbps=500.0,
    latency_ms=20.0,
    draft_acc_prob=0.85,
    entropy=2.5,
    task_name="gsm8k",
    training=False  # Set to False for inference/evaluation
)

print(f"Selected Top-K: {current_k}")
```

## Training Details

The model continuously updates its policy based on the reward signal (system throughput or latency reduction) received after selecting a specific $k$.

*   **Algorithm:** DDQN
*   **Epsilon Decay:** Enabled (starts at 1.0, decays to 0.01)
*   **Target Update Frequency:** Every 10 steps

## License

