---
library_name: pytorch
license: mit
tags:
  - speculative-decoding
  - text-generation
  - auxiliary-model
  - pytorch
---

# SpecDec++ Acceptance Prediction Heads

This repository hosts acceptance prediction heads used by `SpecDec++` and related
tri-decoding experiments.

These checkpoints are not standalone language models. Each checkpoint is a small
auxiliary head trained for a directed model pair:

- `small_draft_acc_head_path` corresponds to `little_model -> draft_model`
- `draft_target_acc_head_path` corresponds to `draft_model -> target_model`

The heads are intended to be used with the `AcceptancePredictionHead` class from:

- `src/SpecDec_pp/specdec_pp/wrap_model.py`

and the decoding / evaluation code in:

- `src/SpecDec_pp/specdec_pp/hf_generation.py`
- `src/SpecDec_pp/specdec_pp/evaluate.py`

## Layout

Checkpoints are organized by directed model pair and run name:

```text
<source_alias>--to--<target_alias>/<run_name>/
```

Examples:

```text
qwen3-1.7b--to--qwen3-14b/exp-weight6-layer3/
tiny-vicuna-1b--to--vicuna-13b-v1.5/exp-weight6-layer3/
```

This structure reflects the actual semantics of the prediction head: a head is
trained for a specific source-target edge, not for a single model in isolation.

## Available Heads

Current published directories include:

- `llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3`
- `tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3`
- `llama-2-7b-chat--to--llama-2-chat-70b/exp-weight6-layer3`
- `vicuna-68m--to--tiny-vicuna-1b/exp-weight6-layer3`
- `tiny-vicuna-1b--to--vicuna-13b-v1.5/exp-weight6-layer3`
- `qwen1.5-0.5b-chat--to--qwen1.5-1.8b-chat/exp-weight-layer3`
- `qwen1.5-1.8b-chat--to--qwen1.5-7b-chat/exp-weight6-layer3`
- `qwen3-0.6b--to--qwen3-1.7b/exp-weight6-layer3`
- `qwen3-1.7b--to--qwen3-14b/exp-weight6-layer3`
- `qwen3-14b--to--qwen3-32b/exp-weight6-layer3`

## Usage

### 1. Download a specific checkpoint

Because this repository stores multiple heads under subdirectories, the most
robust loading flow is:

1. use `snapshot_download()` to fetch the desired subdirectory
2. load the local folder with `AcceptancePredictionHead.from_pretrained()`

```python
from huggingface_hub import snapshot_download

repo_id = "ArcticHuaji/specdecpp-acc-heads"
pair = "qwen3-1.7b--to--qwen3-14b"
run_name = "exp-weight6-layer3"

local_root = snapshot_download(
    repo_id=repo_id,
    allow_patterns=[f"{pair}/{run_name}/*"],
)

local_ckpt = f"{local_root}/{pair}/{run_name}"
```

### 2. Load the head

```python
from wrap_model import AcceptancePredictionHead

acc_head = AcceptancePredictionHead.from_pretrained(local_ckpt)
```

If needed, move the head to the same device / dtype as the assistant-side model:

```python
import torch

acc_head = acc_head.to(torch.bfloat16).to("cuda:0")
acc_head.eval()
```

### 3. Example integration

```python
from huggingface_hub import snapshot_download
from wrap_model import AcceptancePredictionHead

repo_id = "ArcticHuaji/specdecpp-acc-heads"
pair = "qwen3-1.7b--to--qwen3-14b"
run_name = "exp-weight6-layer3"

local_root = snapshot_download(
    repo_id=repo_id,
    allow_patterns=[f"{pair}/{run_name}/*"],
)

assist_acc_head = AcceptancePredictionHead.from_pretrained(
    f"{local_root}/{pair}/{run_name}"
)
```

## Pair Semantics

For tri-decoding style pipelines, the two head paths should be interpreted as:

- `small_draft_acc_head_path`: `little_model -> draft_model`
- `draft_target_acc_head_path`: `draft_model -> target_model`

Examples:

- `vicuna-68m--to--tiny-vicuna-1b` is used for `little -> draft`
- `tiny-vicuna-1b--to--vicuna-13b-v1.5` is used for `draft -> target`
- `qwen3-0.6b--to--qwen3-1.7b` is used for `little -> draft`
- `qwen3-1.7b--to--qwen3-14b` is used for `draft -> target`

## Notes

- These checkpoints are intended for research and experimentation.
- Not every possible model pair is covered.
- Different directories may use slightly different run names such as
  `exp-weight6-layer3` or `exp-weight-layer3`, reflecting the original local
  training outputs.
- Some folders may include `trainer_state.json`; it is auxiliary metadata rather
  than a required file for inference.

## Related Code

Primary codebase:

- `https://github.com/Kaffaljidhmah2/SpecDec_pp`

Local integration used in the author's experiments:

- `DuoDecoding/src/SpecDec_pp/specdec_pp/wrap_model.py`
- `DuoDecoding/src/SpecDec_pp/specdec_pp/evaluate.py`

## Citation

If you use these checkpoints, please cite the corresponding SpecDec / SpecDec++
work and mention the specific model pair(s) used in your experiments.
