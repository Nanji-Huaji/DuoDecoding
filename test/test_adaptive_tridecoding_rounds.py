"""轮级 CPU 行为门禁：Baselines.adaptive_tridecoding。

Seam
----
只通过公开入口 Baselines.adaptive_tridecoding 观察两件事：

1. 返回的 (tokens, metrics)；
2. **逐轮有序通信账本**——CommunicationSimulator 子类在真正的计费入口
   simulate_transfer 记录 (round, kind, link, bytes, topk, draft_len)；
   per_round 口径下再在 _charge_transfer 记录每轮每链路的实际结算。
   transfer 仍走父类的真实实现（先算字节再转发给 simulate_transfer），
   所以账本里的字节就是生产代码这次实际要求计费的字节。

覆盖矩阵：per_transfer（逐消息结算）× {全接受多轮, 两级拒绝,
两级拒绝 + 拒绝残差载荷}；per_round + charge_residual_payload=True
（每轮合并结算 + 拒绝残差载荷）× {全接受多轮, 两级拒绝,
拒绝轮之后继续跑一轮并在轮 2 触发 EOS 停机}；EOS 门禁覆盖内层 EOS
被 target 验证接受、target bonus EOS，以及轮 2 的内层 EOS；所有 terminal
response/downlink 都在停机前计费，并覆盖 per_transfer/per_round+residual。
prob_payload_bits=8 的量化拒绝 × {per_transfer
两级拒绝（补充 bits=16 对照组）, transfer_top_k=None/0 整行拒绝载荷}。

量化拒绝用例不是"先拒绝再倒推字节"：stage-2 的 bonus 位置是否拒绝由
8-bit 对数域量化后的 proposal 概率决定（同一脚本 bits=16 接受、bits=8
拒绝），因此拒绝字节确实来自量化后的计费路径。

两条跨用例不变式也在测试里显式断言：
* 同一组拒绝分布下，per_transfer 与 per_round+residual 的**载荷字节**
  （data bytes、逐链路结算和）必须逐字节相同，差别只在往返次数与
  NTT，即 communication_time 的差 = RTT 数差；
* 拒绝轮的截断必须真的退回缓存长度（不只是记录了 rollback 调用），
  否则下一轮会从错误的位置继续取行 —— 用绝对行号脚本化分布来观测。

允许的轻量替身（用户已确认）
----------------------------
* torch.cuda.Event：CPU 上没有 CUDA；
* KVCacheModel：脚本化概率行，但保留真实的 prob_history 长度语义
  （行 i = 消费完位置 i 后对位置 i+1 的预测）与 rollback 截断语义；
* CommunicationSimulator：继承真实实现，只增加有序账本；计费/统计仍由
  _charge_transfer 完成。

**绝不**替换 CEE 的 proposal/verification stage 或共享采样/回滚实现：
所有测试都走真实的 strict acceptance、拒绝残差采样与 rollback。
拒绝测试只固定 acceptance 判据里
的均匀随机数 torch.rand（r=0.75），不触碰验证逻辑本身。
_ablation="static" 只关闭 RL/ARP 策略头（否则需要真实 GPU 适配器），
验证、KV 缓存推进/rollback 与逐轮计费路径仍是生产代码。

测试里的期望值全部是手工推导的独立字面值，不从生产实现里读取。
"""

import os
import unittest
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import patch

import torch

from src.baselines import Baselines
from src.communication import CommunicationSimulator
from src.proposal_utils import (
    build_topk_proposal_history_step,
    concat_topk_proposal_history,
)
from src.utils import rebuild_topk_uniform_probs, sample

VOCAB_SIZE = 4

#: 三个诊断 JSONL 门控。外部 shell 若设置它们，生产代码会在固定点位写文件
#: （CACHE_CHECK_TRACE / ARP_CALIB_TRACE / TOPK_TRACE）；测试里先清空再恢复，
#: 防止外部路径污染本用例，也不在仓库产生杂散文件。
DIAGNOSTIC_ENV_KEYS = ("CACHE_CHECK_TRACE", "ARP_CALIB_TRACE", "TOPK_TRACE")

#: 全接受三连轮的逐事件账本（per_transfer 口径）。
#: 轮增长 = gamma2 + 1 + gamma1 + 1 = 1 + 1 + 2 + 1 = 5 token。
ALL_ACCEPT_LEDGER = [
    # prompt 在 while 之前传输一次，此时尚未 set_round -> round=None
    (None, "transfer", "edge_end", 8, 0, 0),  # prompt 1 token x 8 B
    # round 1: stage-1 1 token；stage-2 4 token
    (1, "transfer", "edge_end", 12, 0, 0),  # 8 B token + 4 B probability
    (1, "simulate", "edge_end", 12, 0, 0),  # 4 B index + 8 B accepted token
    (1, "transfer", "edge_cloud", 24, 0, 3),  # up: 3 tokens
    (1, "transfer", "edge_cloud", 48, 0, 4),  # stage-2 payload: 4 tokens + 4 probs
    (1, "simulate", "edge_cloud", 12, 300, 4),
    (1, "simulate", "edge_end", 12, 0, 0),    # downlink bonus token
    # round 2
    (2, "transfer", "edge_end", 12, 0, 0),
    (2, "simulate", "edge_end", 12, 0, 0),
    (2, "transfer", "edge_cloud", 16, 0, 2),
    (2, "transfer", "edge_cloud", 48, 0, 4),
    (2, "simulate", "edge_cloud", 12, 300, 4),
    (2, "simulate", "edge_end", 12, 0, 0),
    # round 3
    (3, "transfer", "edge_end", 12, 0, 0),
    (3, "simulate", "edge_end", 12, 0, 0),
    (3, "transfer", "edge_cloud", 16, 0, 2),
    (3, "transfer", "edge_cloud", 48, 0, 4),
    (3, "simulate", "edge_cloud", 12, 300, 4),
    (3, "simulate", "edge_end", 12, 0, 0),
]

ALL_ACCEPT_METRICS = {
    "generated_tokens": 15,
    "little_generated_tokens": 3,
    "little_accepted_tokens": 3,
    "draft_generated_tokens": 12,
    "draft_accepted_tokens": 12,
    "little_forward_times": 3,
    "draft_forward_times": 9,
    "target_forward_times": 3,
    "queuing_time": 0.375,
    "edge_end_data_bytes": 116,
    "edge_cloud_data_bytes": 236,
    "connect_times": {"edge_end": 10, "cloud_end": 0, "edge_cloud": 9},
    "comm_trace_edge_cloud": [
        [24, 1], [48, 1], [12, 1],
        [16, 2], [48, 2], [12, 2],
        [16, 3], [48, 3], [12, 3],
    ],
    "edge_cloud_topk_history": [0, 0, 300, 0, 0, 300, 0, 0, 300],
    "edge_cloud_draft_len_history": [3, 4, 4, 2, 4, 4, 2, 4, 4],
}

#: 单轮“两级都拒绝”的逐事件账本：拒绝位置额外付 top-k 概率 + 6 B 拒绝开销。
REJECTION_LEDGER = [
    (None, "transfer", "edge_end", 8, 0, 0), # prompt（轮外）
    (1, "transfer", "edge_end", 12, 0, 0),   # stage-1 payload
    # 4 B idx + 8 B token + k=2 * 4 B probs + 6 B reject overhead = 26
    (1, "simulate", "edge_end", 26, 0, 0),
    (1, "transfer", "edge_cloud", 16, 0, 2), # up: prefix [0, 2] 2 tokens
    (1, "transfer", "edge_cloud", 24, 0, 2), # stage-2 payload: 2 tokens + 2 probs
    (1, "simulate", "edge_cloud", 26, 2, 2),
    (1, "simulate", "edge_end", 12, 0, 0),   # downlink stage-2 reject token
]

REJECTION_METRICS = {
    "generated_tokens": 1,
    "little_generated_tokens": 1,
    "little_accepted_tokens": 0,
    "draft_generated_tokens": 2,
    "draft_accepted_tokens": 0,
    "little_forward_times": 1,
    "draft_forward_times": 2,
    "target_forward_times": 1,
    "queuing_time": 0.25,
    "edge_end_data_bytes": 58,
    "edge_cloud_data_bytes": 66,
    "connect_times": {"edge_end": 4, "cloud_end": 0, "edge_cloud": 3},
    "comm_trace_edge_cloud": [[16, 1], [24, 1], [26, 1]],
    "edge_cloud_topk_history": [0, 0, 2],
    "edge_cloud_draft_len_history": [2, 2, 2],
}

REJECTION_ROLLBACKS = {
    "little": [1, 1],   # stage-1 proposer -> n1+1 ; 轮末 n2+1 再次截断
    "draft": [1, 1],    # stage-1 verifier -> n1+1 ; stage-2 proposer -> n2+1
    "target": [1],      # stage-2 verifier -> n2+1 (=1)
}

#: ---- prob_payload_bits=8：量化拒绝载荷门禁 -----------------------------
#: 手工推导的脚本分布（不是从生产实现反推）：
#:   · little 行 0 = one-hot token 1 -> stage-1 proposal 的接受比
#:     draft[1]/1 = 0.512 < r=0.75，第一层拒绝；残差只有 token 2。
#:   · draft 行 0 = (0, 0.512, 0.488, 0)：它同时是 stage-1 的 verifier 行和
#:     stage-2 bonus 位置的 proposal 行。8-bit 对数域量化把 0.512/0.488
#:     压到同一档，量化后为 (0, 0.5, 0.5, 0)。stage-2 的 bonus token 是 2，
#:     目标行给 0.37：bits=16 时 0.37/0.488=0.758 ≥ 0.75 接受；bits=8 时
#:     0.37/0.5=0.74 < 0.75 拒绝。所以第二层的拒绝是量化直接造成的。
#:   · 两层的拒绝 token 都由真实残差唯一确定（第一层 token 2；第二层
#:     max(target - proposal, 0) 唯一非零的 token 3），不依赖采样随机性。
#:   · draft/target 行 1 都是 one-hot token 3：bits=16 对照组里 stage-2
#:     第 1 个验证位置（草稿 token 3）的接受比为 1，从而该层全接受。
QUANT_PROB_BITS = 8
QUANT_DRAFT_ROW_0 = (0.0, 0.512, 0.488, 0.0)
QUANT_TARGET_ROW_0 = (0.0, 0.50, 0.37, 0.13)


def _quantized_rejection_scripts():
    """每次调用返回新 dict，避免两个用例共享可变脚本。"""
    return {
        "draft": {0: QUANT_DRAFT_ROW_0, 1: _one_hot(3)},
        "target": {0: QUANT_TARGET_ROW_0, 1: _one_hot(3)},
    }


#: bits=8 的逐消息账本（两级都拒绝）：
#:   prompt 8；
#:   stage-1 payload = 8 B token + 1 个 8-bit 概率 = 9；
#:   stage-1 拒绝合并 = 4 idx + 8 token + k=2 个 8-bit 概率 + 6 reject = 20；
#:   stage-2 上行 prefix [0,2] = 16；
#:   stage-2 payload = 2 token × 8 + 2 个 8-bit 概率 = 18；
#:   stage-2 拒绝合并（同上）= 20；下行 token 12。
QUANT_REJECTION_LEDGER = [
    (None, "transfer", "edge_end", 8, 0, 0),
    (1, "transfer", "edge_end", 9, 0, 0),
    (1, "simulate", "edge_end", 20, 0, 0),
    (1, "transfer", "edge_cloud", 16, 0, 2),
    (1, "transfer", "edge_cloud", 18, 0, 2),
    (1, "simulate", "edge_cloud", 20, 2, 2),
    (1, "simulate", "edge_end", 12, 0, 0),
]

#: 同一脚本、bits=16 的对照账本：第一层拒绝字节回到全宽（12 + 26），
#: 第二层在 bonus 位置接受，合并计费是不含拒绝载荷的 12 B，也没有第二层
#: 拒绝 token 的重采样；只有下行 token 一笔。
QUANT_CONTROL_LEDGER = [
    (None, "transfer", "edge_end", 8, 0, 0),
    (1, "transfer", "edge_end", 12, 0, 0),
    (1, "simulate", "edge_end", 26, 0, 0),
    (1, "transfer", "edge_cloud", 16, 0, 2),
    (1, "transfer", "edge_cloud", 24, 0, 2),
    (1, "simulate", "edge_cloud", 12, 2, 2),
    (1, "simulate", "edge_end", 12, 0, 0),
]

#: topk=None/0：没有压缩行，概率按整行 4 个元素计费。
#:   stage-1 payload 仍是 9（只传被选中的 1 个概率）；
#:   两级拒绝合并 = 4 idx + 8 token + 4 个 8-bit 概率 + 6 reject = 22；
#:   stage-2 payload = 16 token + 2 个 8-bit 概率 = 18。
QUANT_FULL_ROW_LEDGER = [
    (None, "transfer", "edge_end", 8, 0, 0),
    (1, "transfer", "edge_end", 9, 0, 0),
    (1, "simulate", "edge_end", 22, 0, 0),
    (1, "transfer", "edge_cloud", 16, 0, 2),
    (1, "transfer", "edge_cloud", 18, 0, 2),
    (1, "simulate", "edge_cloud", 22, 0, 2),
    (1, "simulate", "edge_end", 12, 0, 0),
]

#: 量化拒绝账本的链路字节和：8+9+20+12 与 16+18+20。
QUANT_REJECTION_DATA_BYTES = {"edge_end": 49, "edge_cloud": 54}
#: 整行拒绝：8+9+22+12 与 16+18+22。
QUANT_FULL_ROW_DATA_BYTES = {"edge_end": 51, "edge_cloud": 56}
#: bits=16 对照：8+12+26+12 与 16+24+12。
QUANT_CONTROL_DATA_BYTES = {"edge_end": 58, "edge_cloud": 52}

#: ---- per_round + charge_residual_payload=True 口径 ----------------------
#: per_round 只改变"谁在何时结算"：simulate_transfer 仍逐消息累积，真正的
#: _charge_transfer 发生在 set_round()/flush_round()（每条链路每轮一次）。
#: 因此下面同时记录 requested ledger（逐消息）与 charge ledger（逐轮结算）。
#: 残差开启后，每个拒绝位置额外补 k*4 + element 字节（k=2, fp32 -> 12 B）。

PER_ROUND_ACCEPT_LEDGER = [
    (None, "transfer", "edge_end", 8, 0, 0),  # prompt（轮外）
    (1, "transfer", "edge_end", 12, 0, 0),
    (1, "simulate", "edge_end", 12, 0, 0),
    (1, "transfer", "edge_cloud", 24, 0, 3),
    (1, "transfer", "edge_cloud", 48, 0, 4),
    (1, "simulate", "edge_cloud", 12, 2, 4),
    (1, "simulate", "edge_end", 12, 0, 0),
    (2, "transfer", "edge_end", 12, 0, 0),
    (2, "simulate", "edge_end", 12, 0, 0),
    (2, "transfer", "edge_cloud", 16, 0, 2),
    (2, "transfer", "edge_cloud", 48, 0, 4),
    (2, "simulate", "edge_cloud", 12, 2, 4),
    (2, "simulate", "edge_end", 12, 0, 0),
    (3, "transfer", "edge_end", 12, 0, 0),
    (3, "simulate", "edge_end", 12, 0, 0),
    (3, "transfer", "edge_cloud", 16, 0, 2),
    (3, "transfer", "edge_cloud", 48, 0, 4),
    (3, "simulate", "edge_cloud", 12, 2, 4),
    (3, "simulate", "edge_end", 12, 0, 0),
]

#: 轮末结算：edge_end 每轮 12+12+12=36；edge_cloud 每轮
#: 24/16(up) + 48(payload) + 12(downlink) = 84/76/76，
#: topk=max(0,0,2)=2，draft_len=up_token + 4 + 4。
PER_ROUND_ACCEPT_CHARGES = [
    (None, "flushed", "edge_end", 8, 0, 0),   # set_round(1) 结算 prompt
    (1, "flushed", "edge_end", 36, 0, 0),
    (1, "flushed", "edge_cloud", 84, 2, 11),
    (2, "flushed", "edge_end", 36, 0, 0),
    (2, "flushed", "edge_cloud", 76, 2, 10),
    (3, "flushed", "edge_end", 36, 0, 0),
    (3, "flushed", "edge_cloud", 76, 2, 10),
]

#: 10 Mbps = 1.25e6 B/s；NTT edge_cloud=10 ms、edge_end=1 ms。
#: edge_end: 4*0.001 + 116/1.25e6 = 0.0040928
#: edge_cloud: 3*0.01 + 236/1.25e6 = 0.0301888
PER_ROUND_ACCEPT_METRICS = {
    "generated_tokens": 15,
    "little_generated_tokens": 3,
    "little_accepted_tokens": 3,
    "draft_generated_tokens": 12,
    "draft_accepted_tokens": 12,
    "little_forward_times": 3,
    "draft_forward_times": 9,
    "target_forward_times": 3,
    "queuing_time": 0.375,
    "edge_end_data_bytes": 116,
    "edge_cloud_data_bytes": 236,
    "edge_end_comm_time": 0.0040928,
    "communication_time": 0.0342816,
    # All three rounds are settled before the final wall/throughput snapshot.
    "wall_time": 0.4092816,
    "connect_times": {"edge_end": 4, "cloud_end": 0, "edge_cloud": 3},
    "comm_trace_edge_cloud": [[84, 1], [76, 2], [76, 3]],
    "edge_cloud_topk_history": [2, 2, 2],
    "edge_cloud_draft_len_history": [11, 10, 10],
    "edge_cloud_bandwidth_history": [10.0, 10.0, 10.0],
}

PER_ROUND_REJECTION_LEDGER = [
    (None, "transfer", "edge_end", 8, 0, 0),
    (1, "transfer", "edge_end", 12, 0, 0),
    (1, "simulate", "edge_end", 12, 0, 0),   # residual payload: 2*4 + 4
    (1, "simulate", "edge_end", 26, 0, 0),   # 4 idx + 8 token + 2*4 probs + 6 reject
    (1, "transfer", "edge_cloud", 16, 0, 2),
    (1, "transfer", "edge_cloud", 24, 0, 2),
    (1, "simulate", "edge_cloud", 12, 0, 0), # residual payload
    (1, "simulate", "edge_cloud", 26, 2, 2),
    (1, "simulate", "edge_end", 12, 0, 0),
]

#: 单轮结算：edge_end = 12+12+26+12 = 62；edge_cloud = 16+24+12+26 = 78。
PER_ROUND_REJECTION_CHARGES = [
    (None, "flushed", "edge_end", 8, 0, 0),
    (1, "flushed", "edge_end", 62, 0, 0),
    (1, "flushed", "edge_cloud", 78, 2, 6),
]

#: edge_end: 2*0.001 + 70/1.25e6 = 0.002056
#: edge_cloud: 0.01 + 78/1.25e6 = 0.0100624
PER_ROUND_REJECTION_METRICS = {
    "generated_tokens": 1,
    "little_generated_tokens": 1,
    "little_accepted_tokens": 0,
    "draft_generated_tokens": 2,
    "draft_accepted_tokens": 0,
    "little_forward_times": 1,
    "draft_forward_times": 2,
    "target_forward_times": 1,
    "queuing_time": 0.25,
    "edge_end_data_bytes": 70,
    "edge_cloud_data_bytes": 78,
    "edge_end_comm_time": 0.002056,
    "communication_time": 0.0121184,
    # Includes final-round communication plus .25s queuing.
    "wall_time": 0.2621184,
    "connect_times": {"edge_end": 2, "cloud_end": 0, "edge_cloud": 1},
    "comm_trace_edge_cloud": [[78, 1]],
    "edge_cloud_topk_history": [2],
    "edge_cloud_draft_len_history": [6],
    "edge_cloud_bandwidth_history": [10.0],
}

#: ---- per_transfer + charge_residual_payload=True 口径 -------------------
#: 残差载荷与往返口径无关：per_transfer 下每个拒绝位置也立刻多一条
#: k*4 + element = 12 B 的消息，插在"验证完成"与"合并计费"之间。
#: 因此下面把 per_round 的拒绝相关账本按逐消息展开重算一遍。
PER_TRANSFER_RESIDUAL_REJECTION_LEDGER = [
    (None, "transfer", "edge_end", 8, 0, 0),  # prompt（轮外）
    (1, "transfer", "edge_end", 12, 0, 0),    # stage-1 payload
    (1, "simulate", "edge_end", 12, 0, 0),    # stage-1 残差载荷
    (1, "simulate", "edge_end", 26, 0, 0),    # 4 idx + 8 token + 2*4 probs + 6 reject
    (1, "transfer", "edge_cloud", 16, 0, 2),
    (1, "transfer", "edge_cloud", 24, 0, 2),
    (1, "simulate", "edge_cloud", 12, 0, 0),  # stage-2 残差载荷
    (1, "simulate", "edge_cloud", 26, 2, 2),
    (1, "simulate", "edge_end", 12, 0, 0),    # 下行 stage-2 拒绝 token
]

#: 第二列是 trace_kind：per_transfer 下父类 simulate_transfer 显式传 None
#: （COMM_TRACE 未开时没有 trace 标签），语义仍是逐消息即时结算。
PER_TRANSFER_RESIDUAL_REJECTION_CHARGES = [
    (None, None, "edge_end", 8, 0, 0),
    (1, None, "edge_end", 12, 0, 0),
    (1, None, "edge_end", 12, 0, 0),
    (1, None, "edge_end", 26, 0, 0),
    (1, None, "edge_cloud", 16, 0, 2),
    (1, None, "edge_cloud", 24, 0, 2),
    (1, None, "edge_cloud", 12, 0, 0),
    (1, None, "edge_cloud", 26, 2, 2),
    (1, None, "edge_end", 12, 0, 0),
]

#: edge_end: 5 次往返 × 1 ms + 70 B / 1.25e6 B/s = 0.005056
#: edge_cloud: 4 次往返 × 10 ms + 78 B / 1.25e6 B/s = 0.0400624
#: communication_time = 0.0451184；per_transfer 下全部在 wall_time 读取前入账，
#: 故 wall_time = 0.0451184 + 0.25 s 排队。
PER_TRANSFER_RESIDUAL_REJECTION_METRICS = {
    "generated_tokens": 1,
    "little_generated_tokens": 1,
    "little_accepted_tokens": 0,
    "draft_generated_tokens": 2,
    "draft_accepted_tokens": 0,
    "little_forward_times": 1,
    "draft_forward_times": 2,
    "target_forward_times": 1,
    "queuing_time": 0.25,
    "edge_end_data_bytes": 70,
    "edge_cloud_data_bytes": 78,
    "edge_end_comm_time": 0.005056,
    "communication_time": 0.0451184,
    "wall_time": 0.2951184,
    "connect_times": {"edge_end": 5, "cloud_end": 0, "edge_cloud": 4},
    "comm_trace_edge_cloud": [[16, 1], [24, 1], [12, 1], [26, 1]],
    "edge_cloud_topk_history": [0, 0, 0, 2],
    "edge_cloud_draft_len_history": [2, 2, 0, 2],
    "edge_cloud_bandwidth_history": [10.0, 10.0, 10.0, 10.0],
}

#: ---- Target-verified EOS gates ----------------------------------------
#: Inner EOS is only a proposal. These scripts give target positive support for
#: it, so terminal output is retained only AFTER target verification and reply.
EOS_STAGE1_TOKEN = 2
EOS_STAGE2_TOKEN = 3
EOS_STAGE1_LEDGER = [
    (None, "transfer", "edge_end", 8, 0, 0),
    (1, "transfer", "edge_end", 12, 0, 0),
    (1, "simulate", "edge_end", 12, 0, 0),
    (1, "transfer", "edge_cloud", 24, 0, 3),
    (1, "transfer", "edge_cloud", 48, 0, 4),
    (1, "simulate", "edge_cloud", 12, 300, 4),
    (1, "simulate", "edge_end", 12, 0, 0),
]
EOS_STAGE1_CHARGES = [
    (None, "edge_end", 8),
    (1, "edge_end", 12),
    (1, "edge_end", 12),
    (1, "edge_cloud", 24),
    (1, "edge_cloud", 48),
    (1, "edge_cloud", 12),
    (1, "edge_end", 12),
]
# 44B / 4 edge-end RTTs; 84B / 3 edge-cloud RTTs. The full block
# is verified/billed even though the committed output ends at inner EOS.
EOS_STAGE1_METRICS = {
    "generated_tokens": 2,
    "little_generated_tokens": 1,
    "little_accepted_tokens": 1,
    "draft_generated_tokens": 4,
    "draft_accepted_tokens": 4,
    "little_forward_times": 1,
    "draft_forward_times": 3,
    "target_forward_times": 1,
    "queuing_time": 0.125,
    "edge_end_data_bytes": 44,
    "edge_cloud_data_bytes": 84,
    "edge_end_comm_time": 0.0040352,
    "communication_time": 0.0341024,
    "wall_time": 0.1591024,
    "connect_times": {"edge_end": 4, "cloud_end": 0, "edge_cloud": 3},
    "comm_trace_edge_cloud": [[24, 1], [48, 1], [12, 1]],
    "edge_cloud_topk_history": [0, 0, 300],
    "edge_cloud_draft_len_history": [3, 4, 4],
    "edge_cloud_bandwidth_history": [10.0, 10.0, 10.0],
}
EOS_STAGE1_PER_ROUND_CHARGES = [
    (None, "flushed", "edge_end", 8, 0, 0),
    (1, "flushed", "edge_end", 36, 0, 0),
    (1, "flushed", "edge_cloud", 84, 300, 11),
]
# Coalesced: 2 end RTTs + 1 cloud RTT; final flush precedes derived time.
EOS_STAGE1_PER_ROUND_METRICS = {
    **EOS_STAGE1_METRICS,
    "edge_end_comm_time": 0.0020352,
    "communication_time": 0.0121024,
    "wall_time": 0.1371024,
    "connect_times": {"edge_end": 2, "cloud_end": 0, "edge_cloud": 1},
    "comm_trace_edge_cloud": [[84, 1]],
    "edge_cloud_topk_history": [300],
    "edge_cloud_draft_len_history": [11],
    "edge_cloud_bandwidth_history": [10.0],
}
# Outer bonus EOS has the same requests and billing, but commits all 5 tokens.
EOS_STAGE2_LEDGER = EOS_STAGE1_LEDGER
EOS_STAGE2_CHARGES = EOS_STAGE1_CHARGES
EOS_STAGE2_METRICS = {**EOS_STAGE1_METRICS, "generated_tokens": 5}
EOS_STAGE2_PER_ROUND_CHARGES = EOS_STAGE1_PER_ROUND_CHARGES
EOS_STAGE2_PER_ROUND_METRICS = {**EOS_STAGE1_PER_ROUND_METRICS, "generated_tokens": 5}

#: Round 1 rejects at both levels; round 2's inner EOS is target-accepted.
#: The target row at absolute position 2 supports that inner EOS explicitly.
MIXED_ROUND_EOS_TOKEN = 0
MIXED_ROUND_LEDGER = [
    (None, "transfer", "edge_end", 8, 0, 0),
    (1, "transfer", "edge_end", 12, 0, 0),
    (1, "simulate", "edge_end", 12, 0, 0),
    (1, "simulate", "edge_end", 26, 0, 0),
    (1, "transfer", "edge_cloud", 16, 0, 2),
    (1, "transfer", "edge_cloud", 24, 0, 2),
    (1, "simulate", "edge_cloud", 12, 0, 0),
    (1, "simulate", "edge_cloud", 26, 2, 2),
    (1, "simulate", "edge_end", 12, 0, 0),
    (2, "transfer", "edge_end", 12, 0, 0),
    (2, "simulate", "edge_end", 12, 0, 0),
    (2, "transfer", "edge_cloud", 16, 0, 2),
    (2, "transfer", "edge_cloud", 36, 0, 3),
    (2, "simulate", "edge_cloud", 12, 2, 3),
    (2, "simulate", "edge_end", 12, 0, 0),
]
MIXED_ROUND_CHARGES = [
    (None, "flushed", "edge_end", 8, 0, 0),
    (1, "flushed", "edge_end", 62, 0, 0),
    (1, "flushed", "edge_cloud", 78, 2, 6),
    (2, "flushed", "edge_end", 36, 0, 0),
    (2, "flushed", "edge_cloud", 64, 2, 8),
]
# End: 3*.001 +106/1.25e6; cloud:2*.01+142/1.25e6; queue=.25.
MIXED_ROUND_METRICS = {
    "generated_tokens": 3,
    "little_generated_tokens": 2,
    "little_accepted_tokens": 1,
    "draft_generated_tokens": 5,
    "draft_accepted_tokens": 3,
    "little_forward_times": 2,
    "draft_forward_times": 4,
    "target_forward_times": 2,
    "queuing_time": 0.25,
    "edge_end_data_bytes": 106,
    "edge_cloud_data_bytes": 142,
    "edge_end_comm_time": 0.0030848,
    "communication_time": 0.0231984,
    "wall_time": 0.2731984,
    "connect_times": {"edge_end": 3, "cloud_end": 0, "edge_cloud": 2},
    "comm_trace_edge_cloud": [[78, 1], [64, 2]],
    "edge_cloud_topk_history": [2, 2],
    "edge_cloud_draft_len_history": [6, 8],
    "edge_cloud_bandwidth_history": [10.0, 10.0],
}


class _TestBaselines(Baselines):
    def load_data(self):
        return None

    def preprocess(self, input_text):
        return input_text

    def postprocess(self, input_text, output_text):
        return output_text

    def eval(self):
        return None


class _FakeCudaEvent:
    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing

    def record(self, stream=None):
        return None

    def elapsed_time(self, other):
        return 0.0


def _one_hot(token: int):
    values = [0.0] * VOCAB_SIZE
    values[token] = 1.0
    return tuple(values)


class _RowScriptedCache:
    """KVCacheModel 的轻量替身，保留真实的 prob_history/rollback 语义。

    * 每次前向消费一个新位置，并在 prob_history 追加一行；
    * 行 i 表示“消费完位置 i 后对位置 i+1 的预测”；
    * current_length == len(prob_history)，rollback 按真实实现截断。

    scripts 按角色给出少量位置的显式概率行；未脚本化的位置回落到
    default_token 的 one-hot 行。这样全接受路径（两侧分布相同、比值 1）
    不依赖随机数；拒绝路径的比值由脚本行精确控制。
    """

    instances = []
    scripts = {}
    default_token = 1
    #: Auxiliary old-oracle cache API instrumentation; not a public billing contract.
    metadata_calls = 0

    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        self.model = model
        self.role = model.kind
        self.device = model.device
        self.vocab_size = VOCAB_SIZE
        self.constructor_kwargs = kwargs
        self._prob_rows = []
        self.rollback_calls = []
        self.hidden_states = None
        self.last_entropy = None
        type(self).instances.append(self)

    @property
    def prob_history(self):
        if not self._prob_rows:
            raise ValueError("Probability history buffer is not initialized")
        return torch.stack(self._prob_rows, dim=1)

    @property
    def logits_history(self):
        return None

    @property
    def current_length(self):
        return len(self._prob_rows)

    def _row(self, position):
        override = type(self).scripts.get(self.role, {}).get(position)
        values = (override if override is not None
                  else _one_hot(type(self).default_token))
        return torch.tensor([list(values)], dtype=torch.float)

    def _consume(self, input_ids):
        target_length = int(input_ids.shape[1])
        if target_length <= self.current_length:
            return self._prob_rows[-1]
        for position in range(self.current_length, target_length):
            self._prob_rows.append(self._row(position))
        return self._prob_rows[-1]

    def forward_context(self, prefix):
        assert prefix.shape[1] >= self.current_length
        return self._consume(prefix)

    def _extend_one(self):
        self._prob_rows.append(self._row(self.current_length))
        return self._prob_rows[-1]

    def generate(self, prefix, gamma):
        q = self._consume(prefix)
        if gamma <= 0:
            return prefix.clone()
        new_tokens = [sample(q)]
        for _ in range(gamma - 1):
            new_tokens.append(sample(self._extend_one()))
        return torch.cat([prefix] + new_tokens, dim=1)

    def generate_with_rebuilt_topk(self, prefix, gamma, proposal_top_k):
        q = self._consume(prefix)
        if gamma <= 0:
            return prefix.clone(), None
        rebuilt_rows = []
        new_tokens = []
        for step in range(gamma):
            rebuilt = rebuild_topk_uniform_probs(q, proposal_top_k)
            rebuilt_rows.append(rebuilt.unsqueeze(1))
            new_tokens.append(sample(rebuilt))
            if step < gamma - 1:
                q = self._extend_one()
        return (
            torch.cat([prefix] + new_tokens, dim=1),
            torch.cat(rebuilt_rows, dim=1),
        )

    def generate_with_rebuilt_topk_metadata(self, prefix, gamma, proposal_top_k):
        """与生产 KVCacheModel 同协议：额外返回 TopKProposalHistory。

        旧 oracle 的 residual 路径使用此接口；新 CEE 使用统一的 wire q，
        该 fixture API 不决定其采样法则或计费分支。fake 只用生产
        proposal_utils 的构造器把每一步 q 变成同一份元数据，不复制算法。
        """
        type(self).metadata_calls += 1
        q = self._consume(prefix)
        if gamma <= 0:
            return prefix.clone(), None, None
        rebuilt_rows = []
        new_tokens = []
        meta_steps = []
        for step in range(gamma):
            rebuilt = rebuild_topk_uniform_probs(q, proposal_top_k)
            rebuilt_rows.append(rebuilt.unsqueeze(1))
            step_meta = build_topk_proposal_history_step(q, proposal_top_k)
            if step_meta is not None:
                meta_steps.append(step_meta)
            new_tokens.append(sample(rebuilt))
            if step < gamma - 1:
                q = self._extend_one()
        meta = concat_topk_proposal_history(meta_steps)
        return (
            torch.cat([prefix] + new_tokens, dim=1),
            torch.cat(rebuilt_rows, dim=1),
            meta,
        )

    def rollback(self, end_pos):
        self.rollback_calls.append(end_pos)
        keep = min(int(end_pos), self.current_length)
        self._prob_rows = self._prob_rows[:keep]


class _RecordingCommunicationSimulator(CommunicationSimulator):
    """真实计费 + 有序账本。

    simulate_transfer 是唯一真正计费的入口（transfer 先编码成字节再调用它）。
    在调用父类之前记录：轮号、调用来源、链路、实际字节、top-k 与 draft_len。
    父类实现保持不变，因此 metrics 里的 data/connect/trace 仍是生产口径。
    """

    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger = []
        #: 真正进入 _charge_transfer 的结算账（per_round 下每轮每链路一笔）。
        self.charge_ledger = []
        self.set_round_calls = []
        self.flush_round_calls = []
        self._in_transfer = 0
        type(self).instances.append(self)

    def set_round(self, round_idx):
        self.set_round_calls.append(int(round_idx))
        super().set_round(round_idx)

    def flush_round(self):
        self.flush_round_calls.append(self._round_idx)
        super().flush_round()

    def transfer(self, tokens, prob, link_type, *args, **kwargs):
        self._in_transfer += 1
        try:
            return super().transfer(tokens, prob, link_type, *args, **kwargs)
        finally:
            self._in_transfer -= 1

    def simulate_transfer(self, data_size_bytes, link_type, *args, **kwargs):
        self.ledger.append(
            (
                self._round_idx,
                "transfer" if self._in_transfer else "simulate",
                link_type,
                data_size_bytes,
                int(kwargs.get("topk", 0) or 0),
                int(kwargs.get("draft_len", 0) or 0),
            )
        )
        return super().simulate_transfer(
            data_size_bytes, link_type, *args, **kwargs
        )

    def _charge_transfer(
        self,
        data_size_bytes,
        link_type,
        add_to_stats=True,
        topk=0,
        draft_len=0,
        trace_kind="direct",
    ):
        # 父类在 per_transfer 下逐消息调用（trace_kind=direct），在 per_round
        # 下由 flush_round 每轮每链路调用一次（trace_kind=flushed）。
        self.charge_ledger.append(
            (
                self._round_idx,
                trace_kind,
                link_type,
                data_size_bytes,
                int(topk or 0),
                int(draft_len or 0),
            )
        )
        return super()._charge_transfer(
            data_size_bytes,
            link_type,
            add_to_stats=add_to_stats,
            topk=topk,
            draft_len=draft_len,
            trace_kind=trace_kind,
        )


def _constant_rand(value):
    def _rand(size, **kwargs):
        return torch.full(
            size, float(value), dtype=torch.float, device=kwargs.get("device")
        )

    return _rand


class AdaptiveTriDecodingRoundGateTests(unittest.TestCase):
    def setUp(self):
        self._saved_diagnostic_env = {}
        for key in DIAGNOSTIC_ENV_KEYS:
            self._saved_diagnostic_env[key] = os.environ.get(key)
            os.environ.pop(key, None)
        self.addCleanup(self._restore_diagnostic_env)

        _RowScriptedCache.instances = []
        _RowScriptedCache.scripts = {}
        _RowScriptedCache.default_token = 1
        _RowScriptedCache.metadata_calls = 0
        _RecordingCommunicationSimulator.instances = []

        args = Namespace(
            eval_mode="adaptive_tridecoding",
            edge_cloud_bandwidth=10.0,
            edge_end_bandwidth=10.0,
            cloud_end_bandwidth=10.0,
            min_bandwidth_mbps=5.0,
            comm_bw_model="instant",
            stochastic_ntt=False,
            ntt_trace_file="",
            ntt_trace_scale=1.0,
            comm_trace_mode="static",
            comm_round_trip_mode="per_transfer",
            charge_residual_payload=False,
            transfer_top_k_cap=0,
            force_full_vocab_transfer=False,
            prob_payload_bits=16,
            max_tokens=1,
            temp=1.0,
            top_k=0,
            top_p=0.0,
            batch_delay=0.0,
            seed=0,
            exp_name="test",
            eval_dataset="unit",
            little_model="little",
            draft_model="draft",
            target_model="target",
            gamma=1,
            gamma1=2,
            gamma2=1,
            use_cuda_graph=False,
            use_early_stopping=False,
            disable_rl_update=True,
            disable_eos_stop=True,
            dump_network_stats=False,
        )
        self.instance = object.__new__(_TestBaselines)
        self.instance.args = args
        self.instance.accelerator = type("Accel", (), {"is_main_process": True})()
        self.instance.vocab_size = VOCAB_SIZE
        self.instance.num_acc_tokens = []
        self.instance.draft_forward_times = 0
        self.instance.target_forward_times = 0
        self.instance.little_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "little"}
        )()
        self.instance.draft_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "draft"}
        )()
        self.instance.target_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "target"}
        )()
        self.instance.small_draft_adapter = None
        self.instance.draft_target_adapter = None
        self.instance.rl_adapter = None
        self.instance.little_rl_adapter = None

    def tearDown(self):
        _RowScriptedCache.instances = []
        _RowScriptedCache.scripts = {}
        _RowScriptedCache.default_token = 1
        _RowScriptedCache.metadata_calls = 0
        _RecordingCommunicationSimulator.instances = []

    def _restore_diagnostic_env(self):
        for key, value in self._saved_diagnostic_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _run(
        self,
        *,
        max_tokens,
        batch_delay,
        transfer_top_k,
        gamma1=None,
        comm_round_trip_mode="per_transfer",
        charge_residual=False,
        rand_value=None,
        eos=None,
        prob_payload_bits=16,
    ):
        self.instance.args.max_tokens = max_tokens
        self.instance.args.batch_delay = batch_delay
        self.instance.args.comm_round_trip_mode = comm_round_trip_mode
        self.instance.args.charge_residual_payload = charge_residual
        self.instance.args.prob_payload_bits = prob_payload_bits
        # CommConfig 在实例上惰性缓存；同一用例里先跑 bits=8 再跑 bits=16
        # 对照组时必须让它按新 args 重建，否则第二次仍读到旧位宽。
        self.instance._comm_config = None
        if gamma1 is not None:
            self.instance.args.gamma1 = gamma1
        # eos=None 保持旧行为：args.disable_eos_stop 仍是 setUp 里的 True，
        # 且不注入 tokenizer（_eos_token_id 走 getattr 兜底返回 None）。
        # 给 eos 时才真正打开生产代码的 EOS 停机路径。
        if eos is not None:
            self.instance.args.disable_eos_stop = False
            self.instance.tokenizer = type(
                "_EosTokenizer", (), {"eos_token_id": eos}
            )()
        prefix = torch.tensor([[0]], dtype=torch.long)
        patches = [
            patch("src.baselines.runtime.KVCacheModel", _RowScriptedCache),
            patch(
                "src.baselines.simulators.CommunicationSimulator",
                _RecordingCommunicationSimulator,
            ),
            patch("src.baselines.core.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.core.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.core.torch.cuda.synchronize", return_value=None),
        ]
        if rand_value is not None:
            patches.append(patch("torch.rand", side_effect=_constant_rand(rand_value)))
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            output, metrics = self.instance.adaptive_tridecoding(
                prefix,
                transfer_top_k=transfer_top_k,
                _ablation="static",
            )
        simulator = _RecordingCommunicationSimulator.instances[-1]
        return output, metrics, simulator

    def test_all_accept_three_rounds_matches_tokens_metrics_and_round_ledger(self):
        output, metrics, simulator = self._run(
            max_tokens=15,
            batch_delay=0.125,
            transfer_top_k=300,
        )

        # 独立期望：起点 [0]，三轮每轮 +5 token，全部接受的都是 token 1。
        self.assertEqual(output.tolist(), [[0] + [1] * 15])
        for name, expected in ALL_ACCEPT_METRICS.items():
            self.assertEqual(metrics[name], expected, msg=name)
        self.assertEqual(simulator.ledger, ALL_ACCEPT_LEDGER)
        self.assertEqual(simulator.set_round_calls, [1, 2, 3])
        self.assertEqual(simulator.flush_round_calls, [3])

        little, draft, target = _RowScriptedCache.instances
        self.assertEqual(little.role, "little")
        self.assertEqual(draft.role, "draft")
        self.assertEqual(target.role, "target")
        # little: stage-1 proposer -> n1+1 = P+1。
        # draft verifier keeps P+1 forwarded positions; outer proposer requests P+4.
        # target verifier keeps P+4 forwarded positions, excluding sampled bonus.
        self.assertEqual(little.rollback_calls, [2, 7, 12])
        self.assertEqual(draft.rollback_calls, [2, 5, 7, 10, 12, 15])
        self.assertEqual(target.rollback_calls, [5, 10, 15])

    def test_rejection_round_charges_reject_payload_then_rolls_back_caches(self):
        _RowScriptedCache.scripts = {
            # stage-1：proposal 是 one-hot token 1（比例 1.0），draft verifier
            # 只给 token 1 概率 0.5 -> ratio=0.5 < r=0.75，拒绝。
            # 拒绝残差 max(verifier - proposal, 0) 唯一非零在 token 2。
            "draft": {0: (0.0, 0.5, 0.5, 0.0)},
            # stage-1 的拒绝残差采到 token 2，作为 stage-2 的 bonus token；
            # stage-2 proposal 行 0 仍是 (0,0.5,0.5,0)，target verifier 只给
            # token 2 概率 0.25 -> ratio=0.5，再次拒绝；残差
            # max(target - proposal, 0) 唯一非零在 token 3。
            "target": {0: (0.0, 0.5, 0.25, 0.25)},
        }
        output, metrics, simulator = self._run(
            max_tokens=1,
            batch_delay=0.25,
            transfer_top_k=2,
            gamma1=1,
            rand_value=0.75,
        )

        # stage-1 残差 -> 2（成为 stage-2 的 bonus token）；stage-2 残差 -> 3。
        self.assertEqual(output.tolist(), [[0, 3]])
        for name, expected in REJECTION_METRICS.items():
            self.assertEqual(metrics[name], expected, msg=name)
        self.assertEqual(simulator.ledger, REJECTION_LEDGER)
        self.assertEqual(simulator.set_round_calls, [1])
        self.assertEqual(simulator.flush_round_calls, [1])

        by_role = {cache.role: cache for cache in _RowScriptedCache.instances}
        for role, expected in REJECTION_ROLLBACKS.items():
            self.assertEqual(by_role[role].rollback_calls, expected, msg=role)

    def test_quantized_reject_payload_bytes_both_stages(self):
        _RowScriptedCache.scripts = _quantized_rejection_scripts()
        output, metrics, simulator = self._run(
            max_tokens=1,
            batch_delay=0.25,
            transfer_top_k=2,
            gamma1=1,
            rand_value=0.75,
            prob_payload_bits=QUANT_PROB_BITS,
        )

        self.assertEqual(output.tolist(), [[0, 3]])
        # 两级拒绝合并都按 k=2 个 8-bit 概率计费（20 B），而不是 2×fp32 的
        # 26 B；stage-2 payload 的 2 个概率也只算 2 B。
        self.assertEqual(simulator.ledger, QUANT_REJECTION_LEDGER)
        for link, expected in QUANT_REJECTION_DATA_BYTES.items():
            self.assertEqual(metrics[f"{link}_data_bytes"], expected, msg=link)

        by_role = {cache.role: cache for cache in _RowScriptedCache.instances}
        for role, expected in REJECTION_ROLLBACKS.items():
            self.assertEqual(by_role[role].rollback_calls, expected, msg=role)

        # 对照组：脚本分布完全相同，只把位宽恢复成 16。stage-2 bonus 位置的
        # 接受比 0.37/0.488=0.758 ≥ 0.75 -> 该层全接受，没有第二笔拒绝载荷。
        # 这证明上面的 stage-2 拒绝是 8-bit 量化（0.488 -> 0.5）造成的。
        _RowScriptedCache.instances = []
        _RecordingCommunicationSimulator.instances = []
        _RowScriptedCache.scripts = _quantized_rejection_scripts()
        control_output, control_metrics, control_simulator = self._run(
            max_tokens=1,
            batch_delay=0.25,
            transfer_top_k=2,
            gamma1=1,
            rand_value=0.75,
            prob_payload_bits=16,
        )

        self.assertEqual(control_output.tolist(), [[0, 2]])
        self.assertEqual(control_simulator.ledger, QUANT_CONTROL_LEDGER)
        for link, expected in QUANT_CONTROL_DATA_BYTES.items():
            self.assertEqual(
                control_metrics[f"{link}_data_bytes"], expected, msg=link
            )

    def test_quantized_reject_payload_without_topk_bills_full_row(self):
        # transfer_top_k=None 与 0 在 proposal 侧都退化为"不压缩"；拒绝位置的
        # 概率载荷因此按整行 4 个元素计费（4 × 8-bit = 4 B），而不是 2 个元素。
        for transfer_top_k in (None, 0):
            with self.subTest(transfer_top_k=transfer_top_k):
                _RowScriptedCache.instances = []
                _RecordingCommunicationSimulator.instances = []
                _RowScriptedCache.scripts = _quantized_rejection_scripts()
                output, metrics, simulator = self._run(
                    max_tokens=1,
                    batch_delay=0.25,
                    transfer_top_k=transfer_top_k,
                    gamma1=1,
                    rand_value=0.75,
                    prob_payload_bits=QUANT_PROB_BITS,
                )

                self.assertEqual(output.tolist(), [[0, 3]])
                self.assertEqual(simulator.ledger, QUANT_FULL_ROW_LEDGER)
                for link, expected in QUANT_FULL_ROW_DATA_BYTES.items():
                    self.assertEqual(metrics[f"{link}_data_bytes"], expected, msg=link)

                by_role = {
                    cache.role: cache for cache in _RowScriptedCache.instances
                }
                for role, expected in REJECTION_ROLLBACKS.items():
                    self.assertEqual(
                        by_role[role].rollback_calls, expected, msg=role
                    )

    def _assert_metrics(self, metrics, expected):
        for name, value in expected.items():
            if isinstance(value, float):
                self.assertAlmostEqual(metrics[name], value, places=12, msg=name)
            else:
                self.assertEqual(metrics[name], value, msg=name)

    def test_all_accept_per_round_residual_matches_flush_and_comm_time(self):
        output, metrics, simulator = self._run(
            max_tokens=15,
            batch_delay=0.125,
            transfer_top_k=2,
            comm_round_trip_mode="per_round",
            charge_residual=True,
        )

        self.assertEqual(output.tolist(), [[0] + [1] * 15])
        # All requested bytes must be present in settled charges and final metrics.
        self._assert_settled_accounting(metrics, simulator)
        self._assert_metrics(metrics, PER_ROUND_ACCEPT_METRICS)
        # 逐消息账本：消息顺序与每消息字节不变，只是 simulate 变成 pending。
        self.assertEqual(simulator.ledger, PER_ROUND_ACCEPT_LEDGER)
        # 真正结算的账本：每条链路每轮一笔，prompt 在 set_round(1) 单独结算。
        self.assertEqual(simulator.charge_ledger, PER_ROUND_ACCEPT_CHARGES)
        self.assertEqual(simulator.set_round_calls, [1, 2, 3])
        self.assertEqual(simulator.flush_round_calls, [None, 1, 2, 3])
        # communication_time 完全由 (bytes, NTT, 带宽) 决定；edge-cloud 分量
        # 不能从 metrics 单独读到，用同一模拟器属性交叉校验。
        self.assertAlmostEqual(
            simulator.edge_cloud_comm_time,
            PER_ROUND_ACCEPT_METRICS["communication_time"]
            - PER_ROUND_ACCEPT_METRICS["edge_end_comm_time"],
            places=12,
        )

        little, draft, target = _RowScriptedCache.instances
        self.assertEqual(little.rollback_calls, [2, 7, 12])
        self.assertEqual(draft.rollback_calls, [2, 5, 7, 10, 12, 15])
        self.assertEqual(target.rollback_calls, [5, 10, 15])

    def test_rejection_per_round_residual_payload_and_flush_order(self):
        _RowScriptedCache.scripts = {
            # 与 per_transfer 拒绝用例同一组分布：stage-1 ratio=0.5 -> 拒绝，
            # 残差采到 2；stage-2 ratio=0.5 -> 拒绝，残差采到 3。
            "draft": {0: (0.0, 0.5, 0.5, 0.0)},
            "target": {0: (0.0, 0.5, 0.25, 0.25)},
        }
        output, metrics, simulator = self._run(
            max_tokens=1,
            batch_delay=0.25,
            transfer_top_k=2,
            gamma1=1,
            comm_round_trip_mode="per_round",
            charge_residual=True,
            rand_value=0.75,
        )

        self.assertEqual(output.tolist(), [[0, 3]])
        # Residual billing must not leave uncharged or inconsistent ledger bytes.
        self._assert_settled_accounting(metrics, simulator)
        self._assert_metrics(metrics, PER_ROUND_REJECTION_METRICS)
        # 每个拒绝位置多一条 12 B 的残差消息（2*4 + 4）。
        self.assertEqual(simulator.ledger, PER_ROUND_REJECTION_LEDGER)
        # 结算顺序：prompt 先单独结算，然后本轮 edge_end、edge_cloud 各一笔。
        self.assertEqual(simulator.charge_ledger, PER_ROUND_REJECTION_CHARGES)
        self.assertEqual(simulator.set_round_calls, [1])
        self.assertEqual(simulator.flush_round_calls, [None, 1])
        self.assertAlmostEqual(simulator.edge_cloud_comm_time, 0.0100624, places=12)

        by_role = {cache.role: cache for cache in _RowScriptedCache.instances}
        for role, expected in REJECTION_ROLLBACKS.items():
            self.assertEqual(by_role[role].rollback_calls, expected, msg=role)

    def test_rejection_per_transfer_residual_charges_each_message(self):
        _RowScriptedCache.scripts = {
            # 与两个 per_round 拒绝用例同一组分布（stage-1 残差 -> 2，
            # stage-2 残差 -> 3），只有往返口径不同。
            "draft": {0: (0.0, 0.5, 0.5, 0.0)},
            "target": {0: (0.0, 0.5, 0.25, 0.25)},
        }
        output, metrics, simulator = self._run(
            max_tokens=1,
            batch_delay=0.25,
            transfer_top_k=2,
            gamma1=1,
            charge_residual=True,
            rand_value=0.75,
        )

        self.assertEqual(output.tolist(), [[0, 3]])
        self._assert_settled_accounting(metrics, simulator)
        self._assert_metrics(metrics, PER_TRANSFER_RESIDUAL_REJECTION_METRICS)
        # 残差消息插在"验证完成"与"合并计费"之间：先补 k*4+element=12 B，
        # 再结 26 B = 4 B 索引 + 8 B token + 2*4 B probs + 6 B reject。
        self.assertEqual(simulator.ledger, PER_TRANSFER_RESIDUAL_REJECTION_LEDGER)
        # per_transfer：9 条消息 = 9 次结算，没有轮末合并。
        self.assertEqual(
            simulator.charge_ledger, PER_TRANSFER_RESIDUAL_REJECTION_CHARGES
        )
        self.assertEqual(simulator.set_round_calls, [1])
        # 末端 flush_round 仍被调用一次，但 per_transfer 下没有 pending 可结算。
        self.assertEqual(simulator.flush_round_calls, [1])

        # 口径不变式：两种往返模式的**载荷字节**必须逐字节相同（同一组
        # 分布、同一组消息），差别只在往返次数——per_transfer 比 per_round
        # 多付 3 个 edge_end RTT 与 3 个 edge_cloud RTT。
        self.assertEqual(
            metrics["edge_end_data_bytes"],
            PER_ROUND_REJECTION_METRICS["edge_end_data_bytes"],
        )
        self.assertEqual(
            metrics["edge_cloud_data_bytes"],
            PER_ROUND_REJECTION_METRICS["edge_cloud_data_bytes"],
        )
        charged = {"edge_end": 0, "edge_cloud": 0, "cloud_end": 0}
        for entry in simulator.charge_ledger:
            charged[entry[2]] += entry[3]
        self.assertEqual(
            charged["edge_end"],
            PER_ROUND_REJECTION_METRICS["edge_end_data_bytes"],
        )
        self.assertEqual(
            charged["edge_cloud"],
            PER_ROUND_REJECTION_METRICS["edge_cloud_data_bytes"],
        )
        self.assertAlmostEqual(
            metrics["communication_time"],
            PER_ROUND_REJECTION_METRICS["communication_time"] + 3 * 0.001 + 3 * 0.01,
            places=12,
        )

        by_role = {cache.role: cache for cache in _RowScriptedCache.instances}
        for role, expected in REJECTION_ROLLBACKS.items():
            self.assertEqual(by_role[role].rollback_calls, expected, msg=role)

    def _assert_settled_accounting(self, metrics, simulator):
        # Dense/sparse cache generation APIs are private. The billing flag's
        # observable contract is complete, consistent requests/settlement/metrics.
        for link in ("edge_end", "edge_cloud", "cloud_end"):
            requested = sum(e[3] for e in simulator.ledger if e[2] == link)
            charged = sum(e[3] for e in simulator.charge_ledger if e[2] == link)
            self.assertEqual(metrics[f"{link}_data_bytes"], requested, msg=link)
            self.assertEqual(charged, requested, msg=link)
        self.assertEqual(simulator._pending_bytes, {})
        self.assertEqual(
            metrics["edge_cloud_data_bytes"],
            sum(e[0] for e in metrics["comm_trace_edge_cloud"]),
        )

    def test_eos_inner_is_target_verified_and_terminal_reply_is_paid(self):
        # Target supports inner EOS explicitly; no inner-layer early commit.
        _RowScriptedCache.scripts = {
            "draft": {1: _one_hot(EOS_STAGE1_TOKEN)},
            "target": {1: _one_hot(EOS_STAGE1_TOKEN)},
        }
        output, metrics, simulator = self._run(
            max_tokens=15,
            batch_delay=0.125,
            transfer_top_k=300,
            comm_round_trip_mode="per_transfer",
            charge_residual=False,
            eos=EOS_STAGE1_TOKEN,
        )
        self.assertEqual(output.tolist(), [[0, 1, EOS_STAGE1_TOKEN]])
        self._assert_metrics(metrics, EOS_STAGE1_METRICS)
        self.assertEqual(simulator.ledger, EOS_STAGE1_LEDGER)
        self.assertEqual(
            [(e[0], e[2], e[3]) for e in simulator.charge_ledger], EOS_STAGE1_CHARGES
        )
        self.assertEqual(simulator.set_round_calls, [1])
        self.assertEqual(simulator.flush_round_calls, [1])
        self._assert_settled_accounting(metrics, simulator)
        little, draft, target = _RowScriptedCache.instances
        self.assertEqual(little.rollback_calls, [2])
        self.assertEqual(draft.rollback_calls, [2, 5])
        self.assertEqual(target.rollback_calls, [5])
        self.assertEqual([c.current_length for c in (little, draft, target)], [1, 4, 5])

    def test_eos_inner_per_round_is_target_verified_and_terminal_reply_is_paid(self):
        # Target supports inner EOS explicitly; no inner-layer early commit.
        _RowScriptedCache.scripts = {
            "draft": {1: _one_hot(EOS_STAGE1_TOKEN)},
            "target": {1: _one_hot(EOS_STAGE1_TOKEN)},
        }
        output, metrics, simulator = self._run(
            max_tokens=15,
            batch_delay=0.125,
            transfer_top_k=300,
            comm_round_trip_mode="per_round",
            charge_residual=True,
            eos=EOS_STAGE1_TOKEN,
        )
        self.assertEqual(output.tolist(), [[0, 1, EOS_STAGE1_TOKEN]])
        self._assert_metrics(metrics, EOS_STAGE1_PER_ROUND_METRICS)
        self.assertEqual(simulator.ledger, EOS_STAGE1_LEDGER)
        self.assertEqual(simulator.charge_ledger, EOS_STAGE1_PER_ROUND_CHARGES)
        self.assertEqual(simulator.set_round_calls, [1])
        self.assertEqual(simulator.flush_round_calls, [None, 1])
        self._assert_settled_accounting(metrics, simulator)
        little, draft, target = _RowScriptedCache.instances
        self.assertEqual(little.rollback_calls, [2])
        self.assertEqual(draft.rollback_calls, [2, 5])
        self.assertEqual(target.rollback_calls, [5])
        self.assertEqual([c.current_length for c in (little, draft, target)], [1, 4, 5])

    def test_eos_target_bonus_is_target_verified_and_terminal_reply_is_paid(self):
        # Target supports inner EOS explicitly; no inner-layer early commit.
        _RowScriptedCache.scripts = {"target": {4: _one_hot(EOS_STAGE2_TOKEN)}}
        output, metrics, simulator = self._run(
            max_tokens=15,
            batch_delay=0.125,
            transfer_top_k=300,
            comm_round_trip_mode="per_transfer",
            charge_residual=False,
            eos=EOS_STAGE2_TOKEN,
        )
        self.assertEqual(output.tolist(), [[0, 1, 1, 1, 1, EOS_STAGE2_TOKEN]])
        self._assert_metrics(metrics, EOS_STAGE2_METRICS)
        self.assertEqual(simulator.ledger, EOS_STAGE2_LEDGER)
        self.assertEqual(
            [(e[0], e[2], e[3]) for e in simulator.charge_ledger], EOS_STAGE2_CHARGES
        )
        self.assertEqual(simulator.set_round_calls, [1])
        self.assertEqual(simulator.flush_round_calls, [1])
        self._assert_settled_accounting(metrics, simulator)
        little, draft, target = _RowScriptedCache.instances
        self.assertEqual(little.rollback_calls, [2])
        self.assertEqual(draft.rollback_calls, [2, 5])
        self.assertEqual(target.rollback_calls, [5])
        self.assertEqual([c.current_length for c in (little, draft, target)], [1, 4, 5])

    def test_eos_target_bonus_per_round_is_target_verified_and_terminal_reply_is_paid(
        self,
    ):
        # Target supports inner EOS explicitly; no inner-layer early commit.
        _RowScriptedCache.scripts = {"target": {4: _one_hot(EOS_STAGE2_TOKEN)}}
        output, metrics, simulator = self._run(
            max_tokens=15,
            batch_delay=0.125,
            transfer_top_k=300,
            comm_round_trip_mode="per_round",
            charge_residual=True,
            eos=EOS_STAGE2_TOKEN,
        )
        self.assertEqual(output.tolist(), [[0, 1, 1, 1, 1, EOS_STAGE2_TOKEN]])
        self._assert_metrics(metrics, EOS_STAGE2_PER_ROUND_METRICS)
        self.assertEqual(simulator.ledger, EOS_STAGE2_LEDGER)
        self.assertEqual(simulator.charge_ledger, EOS_STAGE2_PER_ROUND_CHARGES)
        self.assertEqual(simulator.set_round_calls, [1])
        self.assertEqual(simulator.flush_round_calls, [None, 1])
        self._assert_settled_accounting(metrics, simulator)
        little, draft, target = _RowScriptedCache.instances
        self.assertEqual(little.rollback_calls, [2])
        self.assertEqual(draft.rollback_calls, [2, 5])
        self.assertEqual(target.rollback_calls, [5])
        self.assertEqual([c.current_length for c in (little, draft, target)], [1, 4, 5])

    def test_eos_in_second_round_flushes_both_target_verified_rounds(self):
        # Round 1 rejects at both levels. Round 2's inner EOS is now supported
        # by target as well, so the carried proposal is accepted before stopping.
        _RowScriptedCache.scripts = {
            "draft": {0: (0.0, 0.5, 0.5, 0.0), 2: _one_hot(MIXED_ROUND_EOS_TOKEN)},
            "target": {0: (0.0, 0.5, 0.25, 0.25), 2: _one_hot(MIXED_ROUND_EOS_TOKEN)},
        }
        output, metrics, simulator = self._run(
            max_tokens=15,
            batch_delay=0.125,
            transfer_top_k=2,
            gamma1=1,
            comm_round_trip_mode="per_round",
            charge_residual=True,
            rand_value=0.75,
            eos=MIXED_ROUND_EOS_TOKEN,
        )
        self.assertEqual(output.tolist(), [[0, 3, 1, MIXED_ROUND_EOS_TOKEN]])
        self._assert_metrics(metrics, MIXED_ROUND_METRICS)
        self.assertEqual(simulator.ledger, MIXED_ROUND_LEDGER)
        self.assertEqual(simulator.charge_ledger, MIXED_ROUND_CHARGES)
        self.assertEqual(simulator.set_round_calls, [1, 2])
        self.assertEqual(simulator.flush_round_calls, [None, 1, 2])
        self.assertAlmostEqual(simulator.edge_cloud_comm_time, 0.0201136, places=12)
        self._assert_settled_accounting(metrics, simulator)
        little, draft, target = _RowScriptedCache.instances
        self.assertEqual(little.rollback_calls, [1, 1, 3])
        self.assertEqual(draft.rollback_calls, [1, 1, 3, 5])
        self.assertEqual(target.rollback_calls, [1, 5])
        self.assertEqual([c.current_length for c in (little, draft, target)], [2, 4, 5])


if __name__ == "__main__":
    unittest.main()
