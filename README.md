# DuoDecoding

This is an experiment framework based on Duodecoding.

## Setup


Firstly, install the requirements with:
```bash
pip install -r requirements.txt
```

Then, download models in the following:

Llama Series:
    - Llama-68M
    - TinyLlama-1.1B
    - Llama-2-13B

Vicuna Series:
    - Vicuna-68M
    - TinyVicuna-1B
    - Vicuna-13B-v1.5

And put them to:
```
./<llama/vicuna>/<your-model-dir>
```

## Run


Run experiments via `./exp.py`:
```bash
python exp.py
```

### Args

This script is used to run given experiments automatically. Basically, the following args are needed:

| Args  Name                      | Meaning                                                      | Choice                                                       |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| eval_mode                       | The decoding method selected for the experiment              | `dist_spec`, `dist_split_spec`, `tridecoding`, `uncertainty_decoding` |
| draft_model                     | The draft model for speculative decoding methods             | Currently, `tiny-llama-1.1b`, `tiny-vicuna-1b`, `llama-68m`,  and `vicuna-68m` are available |
| target_model                    | The target model for speculative decoding methods            | Currently, `tiny-llama-1.1b`, `tiny-vicuna-1b`, `Llama-2-13b,` and `vicuna-13b-v1.5` are available. |
| small_model                     | The smallest model for tridecoding methods                   | Currently, `llama-68m`,  and `vicuna-68m` are available      |
| gamma                           | The number of drafted tokens on speculative decoding. <br>Automatically ignored on autoregressive or tridecoding methods. | An int                                                       |
| gamma1                          | The number of drafted tokens in end-edge level of the tri-decoding method. <br>Automatically ignored on other methods. | An int                                                       |
| gamma2                          | The number of drafted tokens in the edge-cloud level of tri-decoding method. <br>Automatically ignored on other methods. | An int                                                       |
| edge_end_bandwidth              | The bandwidth between the edge and the end device. <br>Only available on the methods implemented for transmission simulation. | A float                                                      |
| edge_cloud_bandwidth            | The bandwidth between the edge and the cloud device. <br/>Ibid. | A float                                                      |
| cloud_end_bandwidth             | The bandwidth between the edge and the end device. <br>Ibid. | A float                                                      |
| max_tokens                      | Max tokens for each sample of the dataset.                   | An int                                                       |
| temp                            | Temperature for the target model. Default: 0.0               | A float                                                      |
| exp_name                        | The name of your experiment.                                 | A str                                                        |
| ntt_ms_edge_cloud [Depreciated] | The non-transmission time between the edge and the cloud device. | A float                                                      |
| ntt_ms_edge_end [Depreciated]   | Ibid.                                                        | A float                                                      |
| transfer_top_k                  | Args for top-k compression on the methods that were implemented for transmission simulation. | An int                                                       |


### Adding New Experiments

Experiments are defined based on the following Typeddict:
```python
class ExpConfig(TypedDict):
    CUDA_VISIBLE_DEVICES: Literal['0', '1'] # Sorry for only considered only for the machines having 1 or 2 gpus.
    eval_mode: str
    edge_end_bandwidth: int
    edge_cloud_bandwidth: int
    cloud_end_bandwidth: int
    transfer_top_k: int
    exp_name: str
    use_precise: bool
    ntt_ms_edge_cloud: int | float
    ntt_ms_edge_end: int | float
```

`ExpConfig` is not the same as the table above, for currently only do the experiments for llama series. You can add args on the table above to the typeddict and pass them to the scripts if you need.

After defined your experiment configs, you can append them to the `config_to_run` list. And the experiment configs are to be run if the script is executed.


### Reading the Results

`table_generator_ver2.ipynb` is used for generating the table of experiment results.

After you executed the experiment script, it automatically generates a json file named `experiment_summary_<experiment_date>.json`. You can change the `file_path` var to your json file and run the jupyter notebook:

```python
# ... exixting codes
def main():
    # 读取实验数据
    file_path = "experiment_summary_20250920_145859.json" # change here
# ... existing codes
```

