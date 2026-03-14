# 通信模拟文档

对通信的模拟由将传输时间累加实现。具体而言，每次传输时间 $t$ 满足 $t = \frac{\rm Size}{B_{\rm Channel}} + {\rm NTT} $。所以，所有传输的总时间满足

$$
t_{\rm total} = \sum_{c \in {\rm Channels}} \frac{{\rm Size}_c}{B_c} + {\rm NTT}_c
$$

约定传输速率 $B$ 的单位为 ${\rm Mbps}$，数据大小 ${\rm Size}$ 的单位为 ${\rm byte}$。

## 通信模拟器

### `CommunicationSimulator`

```python
    def __init__(
        self,
        bandwidth_edge_cloud,
        bandwidth_edge_end,
        bandwidth_cloud_end,
        protocol_overhead_bytes: int = 0,
        transfer_top_k: Optional[int] = None,
        dimension: Dimension = "Mbps",
        ntt_ms_edge_end: float = 20,
        ntt_ms_edge_cloud: float = 200
    )
```

参数

- bandwidth_edge_cloud: edge-cloud 之间的带宽
- bandwidth_edge_end: edge-end 之间的带宽
- bandwidth_cloud_end: cloud-end 之间的带宽


```python
    def simulate_transfer(
        self,
        data_size_bytes: int | float,
        link_type: Literal["edge_cloud", "edge_end", "cloud_end"],
        add_to_stats=True,
    ) -> float
```

参数：

- data_size_bytes: 传输的数据大小
- link_type: 传输链路
- add_to_stats: 是否纳入最后的统计

返回值:

- float，为传输所花费的时间

## 调试检查开关

仓库里有两类可选调试检查，默认关闭，避免对正常实验和跑分造成额外开销。

### `DUODEC_DEBUG_NUMERICS`

用途：

- 检查概率张量是否包含 `NaN`、`Inf`、负值或行和异常
- 检查 acceptance ratio 是否非法

主要代码位置：

- `src/utils.py` 中的 `log_prob_tensor_if_invalid`
- `src/utils.py` 中的 `log_ratio_if_invalid`
- 调用点包括 `src/model_gpu.py` 和 `src/engine.py`

说明：

- 该开关作用在生成内循环中
- 打开后会增加额外的 `detach().float()`、reduction 和 `.item()` 同步开销
- 只建议在排查数值稳定性问题时启用

### `DUODEC_DEBUG_TOKEN_CHECKS`

用途：

- 在 MT-Bench 评测末尾检查输出 token 是否落在 `len(tokenizer)` 范围内
- 用于定位 tokenizer / added special tokens 相关问题

主要代码位置：

- `eval/eval_mt_bench_noeval.py`

说明：

- 该检查只在单次生成完成后执行一次
- 相比数值检查，性能影响较小

示例：

```bash
DUODEC_DEBUG_NUMERICS=1 DUODEC_DEBUG_TOKEN_CHECKS=1 python exp.py
```
