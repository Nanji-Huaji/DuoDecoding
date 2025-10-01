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

