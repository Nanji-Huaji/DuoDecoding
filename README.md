# DuoDecoding
[![arXiv.2503.00784](https://img.shields.io/badge/arXiv-2503.00784-red)](https://arxiv.org/abs/2503.00784) [![Hugging Face Paper Page](https://img.shields.io/badge/🤗%20Paper%20Page-2503.00784-yellow)](https://huggingface.co/papers/2503.00784)

This repo contains the implementation for the paper [Hardware-aware Heterogeneous Speculative Decoding with Dynamic Multi-Sequence Drafting](https://arxiv.org/abs/2503.00784). We propose deploying the draft model on CPU, which shifts drafting computational overhead to CPU and enables parallel decoding.

## Setup

1. Create a conda environment with Python 3.10:

```sh
conda create -n duodec python=3.10
conda activate duodec
```

2. Install Python bindings for llama.cpp:
```sh
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```


3. Install other required packages:
```sh
git clone https://github.com/KaiLv69/DuoDecoding.git
cd DuoDecoding
pip install -r requirements.txt
```

4. Set model path in `src/utils.py`.

5. (Optional) Install draftretriever and create a datastore for REST:
```sh
bash src/model/rest/datastore/datastore.sh
pip install src/model/rest/DraftRetriever/wheels/draftretriever-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl
```

## Evaluation
We provide evaluation scripts for the experiments reported in our paper.
- To evaluate the baseline methods on Llama-2-7b:
```sh
bash cmds/baseline_llama.sh
```
- To evaluate DuoDecoding on Llama-2-7b:
```sh
bash cmds/duodec_llama.sh
```
- To evaluate baseline methods on Vicuna-7b-v1.5:
```sh
bash cmds/baseline_vicuna.sh
```
- To evaluate DuoDecoding on Vicuna-7b-v1.5:
```sh
bash cmds/duodec_vicuna.sh
```


## Bugs and Questions
If you have any questions related to the code or the paper, feel free to email Kai (klv23@m.fudan.edu.cn). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!


## Acknowledgments

This repo builds upon the following excellent repos: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [Spec-Bench](https://github.com/hemingkx/Spec-Bench), [parallelspeculativedecoding](https://github.com/smart-lty/parallelspeculativedecoding).

## Citation
Please cite our paper if you find the repo helpful:
```bibtex
@misc{lv2025duodecodinghardwareawareheterogeneousspeculative,
      title={DuoDecoding: Hardware-aware Heterogeneous Speculative Decoding with Dynamic Multi-Sequence Drafting}, 
      author={Kai Lv and Honglin Guo and Qipeng Guo and Xipeng Qiu},
      year={2025},
      eprint={2503.00784},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.00784}, 
}
```