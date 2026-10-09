"""解码模式能力表（D2：eval_mode 知识单点化）。

此前新增一个模式要同步改 ≥5 处——装饰器注册、engine.load_model 两处
分支列表、baselines 的 uses_main_rl/uses_little_rl 两集合、load_acc_head
第三集合、utils 的 RL 路径特判——漏改即静默走错加载/RL 路径。现在全部
消费点改为查询本表；新增模式只需在 MODE_FEATURES 加一行（可选：再挂
装饰器注册解码函数）。

字段语义：
- models: 模型加载拓扑
  * "small"  仅 draft（端侧小模型自回归）
  * "large"  仅 target（云端大模型自回归）
  * "dual"   draft + target 双模型
  * "tri"    little + draft + target 三模型
- uses_main_rl / uses_little_rl: 是否挂 RL adapter（决定 checkpoint 解析
  与 agent 构建）
- acc_head: 验收头档位
  * "none"         不加载
  * "draft_target" 单头（draft→target，adaptive_decoding）
  * "both"         双头（small_draft + draft_target，tri 族）；
                    ceesd_without_arp 按"无 ARP"设计用 RL 但不带头
- supports_prob_bits: 上行概率载荷是否参与 --prob_payload_bits 位宽
  压缩（B15）。只有 tri 系双投机阶段协议（tridecoding/adaptive_
  tridecoding）接线；其余模式显式拒绝 bits<16 而非静默忽略
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeSpec:
    models: str
    uses_main_rl: bool = False
    uses_little_rl: bool = False
    acc_head: str = "none"
    supports_prob_bits: bool = False


_TRI_RL_NO_HEAD = dict(models="tri", uses_main_rl=True, uses_little_rl=True)
_TRI_RL_BOTH_HEADS = dict(
    models="tri", uses_main_rl=True, uses_little_rl=True, acc_head="both"
)

MODE_FEATURES = {
    # 自回归基线
    "small": ModeSpec(models="small"),
    "large": ModeSpec(models="large"),
    # 双模型族（投机解码及其通信感知变体）
    **{
        name: ModeSpec(models="dual")
        for name in (
            "sd",
            "dsd",
            "dssd",
            "dist_spec",
            "dist_split_spec",
            "uncertainty_decoding",
            "cuhlm",
            # TK-SLT（WCSP'25, Zheng & Yang）：DSD + top-K 稀疏 logits 上行。
            # 论文用 68M 草稿 + 7B 验证（端侧小模型当草稿，同 dsd/cuhlm 拓扑）。
            "tk_slt",
            "tkslt",
            "speculative_decoding_with_bandwidth",
            # 已删除 "speculative_decoding_with_bandwidth_full_prob"（幽灵模式）：
            # 该名字从未有任何解码实现注册，能通过能力表校验却会在
            # get_decoding_method() 抛 NotImplementedError。它想表达的
            # "强制传完整词表概率" 是 CLI 开关 --force_full_vocab_transfer
            # 的职责，不该再复制成一个 eval_mode 名字；该开关的覆盖范围见
            # docs/protocol.md（当前只有部分方法消费它）。
        )
    },
    # 双模型 + 主 RL + 单验收头
    "adaptive_decoding": ModeSpec(
        models="dual", uses_main_rl=True, acc_head="draft_target"
    ),
    # 三模型族：无 RL / 无头
    "tridecoding": ModeSpec(models="tri", supports_prob_bits=True),
    "target_only": ModeSpec(models="tri"),
    # 三模型 + 双 RL + 双验收头（CEE 族）
    **{
        name: ModeSpec(**(_TRI_RL_BOTH_HEADS | {"supports_prob_bits": name == "adaptive_tridecoding"}))
        for name in (
            "adaptive_tridecoding",
            "cee_sd",
            "cee_sd_opportunistic",
            "cee_cuhlm",
            "cee_dsd",
            "cee_dssd",
        )
    },
    # "无 ARP"变体（两种拼写）：用 RL、不带头
    "ceesd_without_arp": ModeSpec(**_TRI_RL_NO_HEAD),
    "ceesd_w/o_arp": ModeSpec(**_TRI_RL_NO_HEAD),
}


def get_mode_spec(eval_mode: str) -> ModeSpec:
    """按 eval_mode 取能力描述；未知模式显式报错并列出可用名。"""
    spec = MODE_FEATURES.get(eval_mode)
    if spec is None:
        raise ValueError(
            f"未知 eval_mode {eval_mode!r}（mode_features 能力表无此条目；"
            f"可用: {sorted(MODE_FEATURES)}）"
        )
    return spec
