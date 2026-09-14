import json
import os
import pickle
import random
import warnings
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.rl_reward import RewardConfig, RewardLogger, RewardShaper

# 可选的动作追踪：设置环境变量 RL_ACTION_TRACE=/path/to/actions.jsonl 后，
# 每次 select_config 都会把 (网络条件, 状态, 选中动作) 追加写入该文件。
# 默认关闭，不影响任何行为，仅用于离线分析策略在不同网络条件下的选择。
ACTION_TRACE_PATH = os.environ.get("RL_ACTION_TRACE")

# 每次决策的奖励明细（含生效的 gamma/topk/阈值与当时的 bw/ntt）。
# 动机：只看窗口均值无法回答「奖励本身是否偏好更大的 gamma」，而这个问题的答案
# 决定了「策略不用大 gamma」到底是奖励的错还是学习的错 —— 实测是前者（F10）。
REWARD_TRACE_PATH = os.environ.get("RL_REWARD_TRACE")

# 定义候选的 K 值
TOPK_CANDIDATES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# 定义候选的阈值
THRESHOLD_CANDIDATES = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
# 候选的 draft 长度（每个 WAN 往返内验证的草稿 token 数）。
# 诊断（2026-09-11）：仿真通信时间的 ~97% 是往返时延（RTT），而往返次数 ≈ tokens/gamma，
# 但 gamma 原本是固定超参（默认 4），不在动作空间里 —— 于是策略只能去压"字节数"
# （只占通信时间的 3%），真正的 97% 拧不动。冻结策略下实测 gamma 4 -> 16：
# 往返次数 -17.7%、通信时间 -17.7%、计算时间 -11.9%（每轮只多几次小模型前向，
# 却省下一次 13B 验证前向）、吞吐 +22.1%，接受率不变。
GAMMA_CANDIDATES = [2, 4, 8, 16]

# 定义任务列表以进行 One-Hot 编码
KNOWN_TASKS = ["mt_bench", "gsm8k", "cnndm", "xsum", "humaneval"]
TASK_MAP = {name: i for i, name in enumerate(KNOWN_TASKS)}
UNKNOWN_TASK_ID = len(KNOWN_TASKS)


# ==========================================
# 核心修改 1: 具有时序表征能力的 Q-Network (DRQN 风格)
# ==========================================
class RecurrentQNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=128, num_layers=2):
        super(RecurrentQNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 特征提取层
        self.fc_embed = nn.Linear(feature_dim, hidden_dim)

        # 核心：LSTM 层，用于捕捉网络状态的时序特征
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Dueling Network 架构
        self.val_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.adv_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        x = torch.relu(self.fc_embed(x))
        lstm_out, _ = self.lstm(x)
        last_timestep_out = lstm_out[:, -1, :]

        val = self.val_fc(last_timestep_out)
        adv = self.adv_fc(last_timestep_out)

        return val + adv - adv.mean(1, keepdim=True)


class FactoredRecurrentQNetwork(nn.Module):
    """Branching (factored) dueling Q-network for combinatorial action spaces.

    动作空间是 (gamma, top-k, threshold) 的笛卡尔积（4x11x8 = 352）。扁平输出层
    要为每个组合单独学一个 Q 值，而实测 9669 次决策摊下来每个组合只被访问 ~27 次，
    gamma 的边缘效应完全淹没在组合噪声里 —— 策略因此收敛到小 gamma，尽管奖励
    明确显示 gamma=16 每次决策高 +0.83（reward 排序已用探针数据验证过）。

    这里按 Tavakoli et al. (2018) 的 branching dueling 结构把 Q 分解为各维度之和：

        Q(s, a) = V(s) + sum_h [ A_h(s, a_h) - mean_{a'_h} A_h(s, a'_h) ]

    每个维度各自做 dueling 中心化。由于 Q 对 a 可加，argmax 可以**精确地**逐维度
    分解（不是近似），而且每次决策都会给所有维度的 head 提供梯度：每个 gamma 取值
    的有效样本量从"该组合被访问的次数"提升到"全部决策次数"（约 |A| 倍）。

    为了完全不改动训练/推理循环，forward() 仍返回扁平的 [B, action_dim]，
    布局与 select_config 的解码一致（gamma 最外层、top-k 居中、threshold 最快）。
    """

    def __init__(self, feature_dim, head_dims, hidden_dim=128, num_layers=2):
        super().__init__()
        if len(head_dims) != 3:
            raise ValueError("FactoredRecurrentQNetwork expects exactly 3 head dims")
        self.head_dims = [int(d) for d in head_dims]
        self.action_dim = int(np.prod(self.head_dims))
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_embed = nn.Linear(feature_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

        self.val_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.adv_fc = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim),
                )
                for dim in self.head_dims
            ]
        )

    def forward(self, x):
        x = torch.relu(self.fc_embed(x))
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]

        val = self.val_fc(h).view(-1, 1, 1, 1)
        q = val
        for i, head in enumerate(self.adv_fc):
            adv = head(h)
            adv = adv - adv.mean(1, keepdim=True)  # 每个维度内部中心化
            shape = [1, 1, 1, 1]
            shape[i + 1] = self.head_dims[i]
            q = q + adv.view(adv.shape[0], *shape[1:])
        return q.reshape(x.shape[0], self.action_dim)


class DDQNAgent:
    def __init__(
        self,
        feature_dim,
        action_dim,
        seq_len=8,
        hidden_dim=128,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
        buffer_size=5000,
        batch_size=32,
        reward_scale=0.01,
        target_update_freq=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        name="RL-Agent",
        init_seed: int | None = None,
        head_dims: list | None = None,
    ):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.target_update_freq = target_update_freq
        self.device = device
        self.update_count = 0
        self.name = name
        self.init_seed = init_seed
        self._random = random.Random(init_seed)
        self.best_tps = -1.0
        self.reward_history = deque(maxlen=100)

        with torch.random.fork_rng(devices=[]):
            if init_seed is not None:
                torch.manual_seed(init_seed)
            if head_dims:
                if int(np.prod(head_dims)) != action_dim:
                    raise ValueError(
                        f"head_dims {list(head_dims)} do not multiply to action_dim {action_dim}"
                    )
                self.head_dims = [int(d) for d in head_dims]
                make_net = lambda: FactoredRecurrentQNetwork(  # noqa: E731
                    feature_dim, self.head_dims, hidden_dim
                )
            else:
                self.head_dims = None
                make_net = lambda: RecurrentQNetwork(  # noqa: E731
                    feature_dim, action_dim, hidden_dim
                )
            self.policy_net = make_net().to(self.device)
            self.target_net = make_net().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = deque(maxlen=buffer_size)

    def select_action(self, state_seq, training=True):
        if training and self._random.random() < self.epsilon:
            return self._random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.reward_history.append(reward)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self._random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 核心修复: 确保在 enable_grad 环境下运行，因为调用者(推理循环)通常在 no_grad 下
        with torch.enable_grad():
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            rewards = rewards * self.reward_scale

            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            current_q_values = self.policy_net(states).gather(1, actions)
            loss = self.loss_fn(current_q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.update_count % 10 == 0:
                avg_reward = (
                    np.mean(self.reward_history) if self.reward_history else 0.0
                )
                print(
                    f"[{self.name}] Step: {self.update_count}, Loss: {loss.item():.4f}, Reward: {avg_reward:.4f}, Epsilon: {self.epsilon:.4f}"
                )

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        # 尝试获取模型系列名称，用于校验
        model_series = os.environ.get("MODEL_SERIES_NAME", "unknown")

        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_count": self.update_count,
                "model_series": model_series,
                "init_seed": self.init_seed,
                "best_tps": self.best_tps,
            },
            path,
        )
        buffer_path = path + ".buffer"
        try:
            with open(buffer_path, "wb") as f:
                pickle.dump(list(self.memory)[-2000:], f)
        except Exception:
            pass

    def load(self, path):
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)

                # 严格校验模型系列
                current_series = os.environ.get("MODEL_SERIES_NAME")
                saved_series = checkpoint.get("model_series")
                if (
                    current_series
                    and saved_series
                    and saved_series != "unknown"
                    and saved_series != current_series
                ):
                    print(
                        f"CRITICAL WARNING: Checkpoint at {path} belongs to model series '{saved_series}', but current environment is '{current_series}'!"
                    )
                    # 为了向后兼容和避免不必要的崩溃，我们在这里默认只打印警告。
                    # 如果需要极其严格，可以 raise ValueError。

                self.policy_net.load_state_dict(checkpoint["policy_net"])
                self.target_net.load_state_dict(checkpoint["target_net"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.epsilon = checkpoint["epsilon"]
                self.update_count = checkpoint.get("update_count", 0)
                self.best_tps = checkpoint.get(
                    "best_tps",
                    float("inf") if os.path.basename(path) == "best.pth" else -1.0,
                )
                buffer_path = path + ".buffer"
                if os.path.exists(buffer_path):
                    with open(buffer_path, "rb") as f:
                        self.memory.extend(pickle.load(f))
                print(
                    f"Loaded LSTM-RL agent from {path}, series: {saved_series}, steps: {self.update_count}"
                )
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Starting fresh.")


class RLNetworkAdapter:
    def __init__(
        self,
        args,
        model_path="checkpoints/rl_adapter.pth",
        best_model_path=None,
        agent_name=None,
        legacy_load_paths=None,
        device="cuda",
        k_candidates=None,
        threshold_candidates=None,
        init_seed: int | None = None,
        init_strategy: str = "resume",
        frozen=False,
        epsilon_decay=0.9995,
        reward_scale=0.01,
        batch_size=32,
        prefer_latest=False,
        force_threshold_override=None,
    ):
        self.args = args
        self.device = device
        self.frozen = frozen

        # ==========================================
        # 动作空间：(top-k, ARP 阈值) 或 (top-k, ARP 阈值, draft 长度 gamma)
        # ==========================================
        self.action_space = getattr(args, "rl_action_space", "topk_thr")
        if self.action_space not in {"topk_thr", "topk_thr_gamma"}:
            raise ValueError(f"Unsupported rl_action_space: {self.action_space}")
        gamma_arg = getattr(args, "rl_gamma_candidates", None)
        if gamma_arg:
            self.gamma_candidates = [
                int(x) for x in str(gamma_arg).replace(" ", "").split(",") if x
            ]
        else:
            self.gamma_candidates = list(GAMMA_CANDIDATES)
        if not self.gamma_candidates:
            raise ValueError("rl_gamma_candidates must not be empty")
        self.include_gamma_in_state = bool(
            getattr(args, "rl_include_gamma_in_state", False)
        )
        self.gamma_dim = (
            len(self.gamma_candidates) if self.action_space == "topk_thr_gamma" else 1
        )
        self.last_gamma = int(getattr(args, "gamma1", self.gamma_candidates[0]))
        # --rl_force_gamma: pin gamma to a value (ablation), None = policy-controlled.
        _force = getattr(args, "rl_force_gamma", None)
        self._force_gamma_idx = (
            self.gamma_candidates.index(int(_force))
            if _force is not None and int(_force) in self.gamma_candidates
            else None
        )

        self.task_dim = len(KNOWN_TASKS) + 1
        self.feature_dim = (
            3 + 1 + self.task_dim + (1 if self.include_gamma_in_state else 0)
        )  # [bw, lat, entropy, last_acc] + task [+ last_gamma]
        self.seq_len = 8

        self.state_history = deque(
            [np.zeros(self.feature_dim) for _ in range(self.seq_len)],
            maxlen=self.seq_len,
        )

        self.topk_candidates = (
            k_candidates if k_candidates is not None else TOPK_CANDIDATES
        )
        self.threshold_candidates = (
            threshold_candidates
            if threshold_candidates is not None
            else THRESHOLD_CANDIDATES
        )
        self.action_dim = (
            len(self.topk_candidates)
            * len(self.threshold_candidates)
            * self.gamma_dim
        )

        # --rl_force_threshold: pin the ARP threshold (ablation), None = policy.
        # 只用于消融，因此允许取候选集合之外的数值（主角色候选上界是 0.4，但要测
        # 0.6/0.8/0.9 是否更好）。argmax 仍限制在"最接近的候选档"切片上选择
        # top-k/γ，但实际生效的阈值用原始值。
        _force_thr = getattr(args, "rl_force_threshold", None)
        self._force_threshold_value = None if _force_thr is None else float(_force_thr)
        # 按角色分离：little 级是 opportunistic（可能接受未经目标模型验证的 token），
        # 若把主阈值同时强加到它身上，准确率变化会与主阈值混在一起（实测过一次，
        # 见文档 F21/F23）。force_threshold_override 允许显式指定本实例的值。
        if force_threshold_override is not None:
            self._force_threshold_value = (
                None if float(force_threshold_override) < 0 else float(force_threshold_override)
            )
        if _force_thr is None:
            self._force_threshold_idx = None
        else:
            _cands = [float(c) for c in self.threshold_candidates]
            self._force_threshold_idx = min(
                range(len(_cands)), key=lambda i: abs(_cands[i] - float(_force_thr))
            )

        self.model_path = model_path
        self.best_model_path = best_model_path or model_path
        self.legacy_load_paths = list(legacy_load_paths or [])
        agent_name = agent_name or os.path.basename(self.model_path).replace(".pth", "")

        # ==========================================
        # 奖励设计 (v2)
        # ==========================================
        # v1 的奖励 r = exp(min(N_acc/T,100)/20) * (N_acc/gamma)^2 同时存在三个问题：
        #   (1) exp+cap 是手调凸变换，在实测 ±17% 的主机负载抖动下会放大高方差动作，
        #       且 cap 恰好在快链路区间把梯度压平；
        #   (2) alpha^2 的跨样本动态范围实测 0.001~1.0（三个数量级），奖励方差被任务
        #       难度主导，而且它系统性奖励"少起草"（小 gamma），与快链路需求相反；
        #   (3) T 用墙钟计算时间，同一动作在忙机器上奖励更低 —— 学到的策略是"这台机器
        #       此刻的最优"，不是"这个网络条件下的最优"。
        # v2 用 ratio 目标 E[N]/E[T] 的 Lagrangian 松弛：r = N_acc - lambda*T，
        # 线性、无 cap、无 alpha^2，T 可换成与主机无关的计算时间模型。
        # 默认仍是 "legacy"，保证历史 checkpoint 与已发表数字完全可复现。
        self.reward_mode = getattr(args, "rl_reward_mode", "legacy")
        if self.reward_mode not in {"legacy", "linear", "lagrangian", "slo", "energy"}:
            raise ValueError(f"Unsupported rl_reward_mode: {self.reward_mode}")

        self.agent = DDQNAgent(
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            seq_len=self.seq_len,
            device=device,
            name=agent_name,
            init_seed=init_seed,
            epsilon_decay=epsilon_decay,
            # reward_scale 只是把奖励缩放到优化器舒适的量级，是**策略不变**的常数
            # （对奖励乘正常数不改变 argmax Q）。调用方（baselines.py）已按模式解析好
            # 默认值（0.01）。它必须对所有奖励模式一致：v1 的奖励经 exp 变换后是
            # O(1-150)，v2 的 N_acc - lambda*T 是 O(1-10)，若 v2 不缩放，TD 目标会比
            # 已调好的 DDQN 超参（lr 1e-4、Huber）大 ~70 倍 —— 实测 loss 中位数从
            # 0.15 涨到 10.8，主智能体发散。
            reward_scale=reward_scale,
            batch_size=batch_size,
            buffer_size=int(getattr(args, "rl_buffer_size", 0) or 5000),
            # 组合动作空间用因子化（branching dueling）Q 头：每个维度独立出
            # advantage，Q 可加 ⇒ argmax 精确分解，且每次决策给所有维度梯度。
            head_dims=(
                [
                    self.gamma_dim,
                    len(self.topk_candidates),
                    len(self.threshold_candidates),
                ]
                if getattr(args, "rl_factored_q", False)
                and self.action_space == "topk_thr_gamma"
                else None
            ),
        )

        self.max_bandwidth = 1000.0
        self.max_latency = 500.0

        # ==========================================
        # 状态特征缩放方式
        # ==========================================
        # 诊断结论（2026-09-11）：默认的线性缩放 norm_bw = bw/1000 会把整个
        # 0.5-50 Mbps 工作区间压进 [0.0005, 0.05] 这一个维度里，而其余特征
        # （entropy/10、last_acc、task one-hot）都铺满 [0,1]，结果是策略几乎
        # 完全忽略带宽：把真实 trace 的回放带宽从 1 Mbps 改成 5 Mbps，
        # 1040 次决策里 0 次改变（见 scripts/analyze_policy_bandwidth_sensitivity.py）。
        # 因此提供 "log" 缩放（对数量级敏感），以及延迟的 "centi"/"log" 缩放。
        # 默认仍是 "linear"，保证已有 checkpoint 与历史结果完全可比。
        self.bw_scaling = getattr(args, "state_bw_scaling", "linear")
        self.latency_scaling = getattr(args, "state_latency_scaling", "linear")
        if self.bw_scaling not in {"linear", "log"}:
            raise ValueError(f"Unsupported state_bw_scaling: {self.bw_scaling}")
        if self.latency_scaling not in {"linear", "centi", "log"}:
            raise ValueError(
                f"Unsupported state_latency_scaling: {self.latency_scaling}"
            )

        compute_costs = {}
        cost_json = getattr(args, "rl_compute_cost_json", None)
        if cost_json:
            if os.path.exists(cost_json):
                with open(cost_json) as fh:
                    compute_costs = json.load(fh)
            else:
                warnings.warn(
                    f"rl_compute_cost_json={cost_json} not found; "
                    "falling back to an empty compute-cost model (all zeros)."
                )
        self.reward_shaper = RewardShaper(
            RewardConfig(
                mode=self.reward_mode,
                lam=float(getattr(args, "rl_reward_lambda", 0.0) or 0.0),
                deadline_ms=float(getattr(args, "rl_reward_deadline_ms", 0.0) or 0.0),
                deadline_penalty=float(
                    getattr(args, "rl_reward_deadline_penalty", 1.0) or 1.0
                ),
                energy_weight=float(
                    getattr(args, "rl_reward_energy_weight", 0.0) or 0.0
                ),
                byte_price_s_per_mb=float(
                    getattr(args, "rl_byte_price", 0.0) or 0.0
                ),
                compute_mode=getattr(args, "rl_compute_time_mode", "wall"),
                compute_cost_s=compute_costs,
                legacy_alpha2=not bool(getattr(args, "rl_reward_no_alpha2", False)),
                legacy_warp=float(getattr(args, "rl_reward_legacy_warp", 20.0) or 20.0),
                legacy_cap=float(getattr(args, "rl_reward_legacy_cap", 100.0) or 100.0),
            )
        )
        self.reward_log = RewardLogger(
            window=int(getattr(args, "rl_reward_log_window", 200) or 200)
        )
        self._reward_calls = 0
        self.last_reward_components = {}
        self._action_calls = 0
        self._reward_by_gamma: dict = {}
        self.last_topk = None
        self.last_threshold = None
        self.last_bw = None
        self.last_ntt = None
        self._gamma_hist: deque = deque(maxlen=2000)
        self._topk_hist: deque = deque(maxlen=2000)
        self._thr_hist: deque = deque(maxlen=2000)

        self.last_state_seq = None
        self.last_action = None
        self.last_reward = None

        self.best_tps = -1.0

        if init_strategy not in {"fresh", "resume"}:
            raise ValueError(f"Unsupported RL initialization strategy: {init_strategy}")

        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        existing_checkpoint = os.path.exists(self.best_model_path) or os.path.exists(
            self.model_path
        )
        if init_strategy == "fresh" and existing_checkpoint:
            raise FileExistsError(
                f"Fresh RL initialization refuses to overwrite {self.model_path}"
            )

        if init_strategy == "fresh":
            print(
                f"[{agent_name}] Fresh deterministic initialization with seed {init_seed}"
            )
        elif prefer_latest and os.path.exists(self.model_path):
            self.agent.load(self.model_path)
        elif os.path.exists(self.best_model_path):
            self.agent.load(self.best_model_path)
        elif os.path.exists(self.model_path):
            self.agent.load(self.model_path)
        else:
            legacy_path = next(
                (path for path in self.legacy_load_paths if os.path.exists(path)), None
            )
            if legacy_path is not None:
                self.agent.load(legacy_path)
                print(
                    f"[{agent_name}] Migrating legacy RL checkpoint from {legacy_path} to {self.model_path}"
                )
                self.agent.save(self.model_path)
            else:
                print(
                    f"[{agent_name}] No checkpoint found at {self.model_path} or {self.best_model_path}"
                )
        self.best_tps = self.agent.best_tps

    def _get_current_feature_vector(
        self, bandwidth_mbps, latency_ms, entropy, last_acc_prob, task_name
    ):
        self._check_state_units(bandwidth_mbps, latency_ms)
        norm_bw = self._scale_bandwidth(bandwidth_mbps)
        norm_lat = self._scale_latency(latency_ms)
        norm_entropy = min(entropy / 10.0, 1.0)

        task_idx = TASK_MAP.get(task_name, UNKNOWN_TASK_ID)
        task_vec = np.zeros(self.task_dim, dtype=np.float32)
        task_vec[task_idx] = 1.0

        feats = [[norm_bw, norm_lat, norm_entropy, last_acc_prob], task_vec]
        if self.include_gamma_in_state:
            # 上一轮选择的 draft 长度（归一化到 [0,1]），让策略知道自己当前的档位。
            feats.append(
                np.array(
                    [self.last_gamma / max(self.gamma_candidates)],
                    dtype=np.float32,
                )
            )
        return np.concatenate(feats).astype(np.float32)

    def _scale_bandwidth(self, bandwidth_mbps: float) -> float:
        """把带宽（Mbps）映射到 [0,1]；log 缩放让每个数量级都有可分辨的跨度。"""
        bw = max(float(bandwidth_mbps), 0.0)
        if self.bw_scaling == "log":
            return min(np.log10(bw + 1.0) / np.log10(self.max_bandwidth + 1.0), 1.0)
        return min(bw / self.max_bandwidth, 1.0)

    def _scale_latency(self, latency_ms: float) -> float:
        """把延迟（毫秒）映射到 [0,1]。"""
        lat = max(float(latency_ms), 0.0)
        if self.latency_scaling == "log":
            return min(np.log10(lat + 1.0) / np.log10(self.max_latency + 1.0), 1.0)
        if self.latency_scaling == "centi":
            return min(lat / 100.0, 1.0)
        return min(lat / self.max_latency, 1.0)

    def _check_state_units(self, bandwidth_mbps: float, latency_ms: float):
        """
        单位自检：select_config 期望带宽单位为 Mbps、延迟单位为毫秒。
        CommunicationSimulator 内部以 bytes/second 和 seconds 存储，
        若调用方误传原始内部单位，会导致特征退化（带宽恒为 1、延迟恒为 0）。
        """
        if bandwidth_mbps > 10.0 * self.max_bandwidth:
            warnings.warn(
                f"[{self.agent.name}] bandwidth {bandwidth_mbps:.3e} exceeds the "
                f"plausible Mbps range (>{10.0 * self.max_bandwidth:.0f} Mbps); "
                f"it may have been passed in bytes/second (see "
                f"CommunicationSimulator.bandwidth_*_mbps).",
                stacklevel=3,
            )
        if 0.0 < latency_ms < 0.5:
            warnings.warn(
                f"[{self.agent.name}] latency {latency_ms:.3e} ms is implausibly "
                f"small; it may have been passed in seconds (see "
                f"CommunicationSimulator.ntt_*_ms).",
                stacklevel=3,
            )

    def _random_action_draw(self) -> bool:
        """True if the epsilon-greedy branch should be taken (same draw as select_action)."""
        return self.agent._random.random() < self.agent.epsilon

    def select_config(
        self,
        bandwidth_mbps: float,
        latency_ms: float,
        acc_probs: List[float],
        entropy: float,
        task_name: str = "unknown",
        training=True,
    ) -> Tuple[int, float]:
        """
        Args:
            bandwidth_mbps: 链路带宽，单位 **Mbps**（兆比特每秒）。
                注意 CommunicationSimulator 内部以 bytes/second 存储，
                请使用 comm_simulator.bandwidth_*_mbps 属性传入。
            latency_ms: 网络传输时间（NTT），单位 **毫秒**。
                CommunicationSimulator 内部以 seconds 存储，
                请使用 comm_simulator.ntt_*_ms 属性传入。
            acc_probs: 本步接受预测头输出的接受概率列表（可为空）。
            entropy: 当前 logits 的平均熵。
            task_name: 任务名（KNOWN_TASKS 之一或 "unknown"）。
        """
        last_acc = acc_probs[-1] if len(acc_probs) > 0 else 0.5
        current_feat = self._get_current_feature_vector(
            bandwidth_mbps, latency_ms, entropy, last_acc, task_name
        )
        self.state_history.append(current_feat)
        state_seq = np.array(self.state_history)

        if not self.frozen and (
            self.last_state_seq is not None
            and self.last_action is not None
            and self.last_reward is not None
        ):
            self.agent.store_transition(
                self.last_state_seq,
                self.last_action,
                self.last_reward,
                state_seq,
                done=False,
            )
            self.agent.update()

        action_idx = self.agent.select_action(
            state_seq, training=training and not self.frozen
        )

        # 反事实消融（--rl_force_gamma <值>）：把 γ 钉死，top-k/阈值仍由策略在同一
        # checkpoint 上选。用于把端到端收益拆成"γ 维度贡献"与"top-k/阈值维度贡献"
        # —— 否则无法排除"joint 智能体赢在它学会了用大 top-k"这一替代解释。
        if self._force_gamma_idx is not None and self.gamma_dim > 1:
            n_kt = len(self.topk_candidates) * len(self.threshold_candidates)
            lo = self._force_gamma_idx * n_kt
            if training and not self.frozen and self._random_action_draw():
                action_idx = lo + self.agent._random.randrange(n_kt)
            else:
                with torch.no_grad():
                    st = torch.FloatTensor(state_seq).unsqueeze(0).to(self.agent.device)
                    action_idx = lo + int(self.agent.policy_net(st)[0, lo:lo + n_kt].argmax())

        # 反事实消融（--rl_force_threshold <阈值>）：把 ARP 阈值钉死，top-k/γ 仍由策略
        # 选。阈值决定"草稿什么时候早停"，实测它把"草稿跑长"的概率从 19% 推到 26%
        # （奖励 +0.2 vs +8.2），是 F17 之后剩下的主要自由度。
        if self._force_threshold_idx is not None:
            n_thr = len(self.threshold_candidates)
            n_topk = len(self.topk_candidates)
            if self.gamma_dim > 1:
                n_kt = n_topk * n_thr
                gamma_idx = action_idx // n_kt
                rem = action_idx % n_kt
            else:
                gamma_idx, rem = 0, action_idx
            topk_idx = rem // n_thr
            if training and not self.frozen and self._random_action_draw():
                action_idx = (
                    gamma_idx * n_topk * n_thr
                    + topk_idx * n_thr
                    + self.agent._random.randrange(n_thr)
                )
            else:
                n_kt = n_topk * n_thr
                lo = gamma_idx * n_kt + topk_idx * n_thr
                with torch.no_grad():
                    st = torch.FloatTensor(state_seq).unsqueeze(0).to(self.agent.device)
                    action_idx = lo + int(
                        self.agent.policy_net(st)[0, lo:lo + n_thr].argmax()
                    )

        if self.gamma_dim > 1:
            n_kt = len(self.topk_candidates) * len(self.threshold_candidates)
            gamma_idx = action_idx // n_kt
            rem = action_idx % n_kt
        else:
            gamma_idx, rem = 0, action_idx
        topk_idx = rem // len(self.threshold_candidates)
        threshold_idx = rem % len(self.threshold_candidates)

        selected_topk = self.topk_candidates[topk_idx]
        selected_threshold = self.threshold_candidates[threshold_idx]
        selected_gamma = self.gamma_candidates[gamma_idx]
        # legacy 动作空间（gamma_dim == 1）不控制 gamma，此时 last_gamma 只是
        # gamma_candidates[0]，不代表系统实际使用的 draft 长度（那是 --gamma1）。
        # 用后者覆盖，轨迹/奖励归因里记录的才是真实生效的 gamma（否则 trace 里
        # 会永远显示 2，掩盖 --gamma1 16 这类配置，见诊断文档 F14/F17）。
        if self.gamma_dim <= 1:
            selected_gamma = int(getattr(self.args, "gamma1", selected_gamma))
        self.last_gamma = int(selected_gamma)
        self.last_topk = int(selected_topk)
        self.last_threshold = float(selected_threshold)
        if self._force_threshold_value is not None:
            self.last_threshold = float(self._force_threshold_value)
        self.last_bw = float(bandwidth_mbps)
        self.last_ntt = float(latency_ms)

        # 动作使用情况的可观测性：没有这个，"策略到底在用什么档位"只能靠猜
        # （之前正是因为缺它，才需要事后用反事实回放才发现带宽不可观测）。
        if self._action_calls and self._action_calls % 2000 == 0:
            hist = np.bincount(
                np.array(self._gamma_hist), minlength=len(self.gamma_candidates)
            ).tolist()
            print(
                f"[{self.agent.name}] action use (last {len(self._gamma_hist)}): "
                f"gamma mean={float(np.mean(self._gamma_hist)):.1f} hist={hist} "
                f"| topk mean={float(np.mean(self._topk_hist)):.0f} "
                f"| thr mean={float(np.mean(self._thr_hist)):.3f} "
                f"| eps={self.agent.epsilon:.3f}",
                flush=True,
            )
        self._gamma_hist.append(gamma_idx)
        self._topk_hist.append(topk_idx)
        self._thr_hist.append(threshold_idx)
        self._action_calls += 1

        # 按 γ 归因的奖励统计（每 500 次决策打印一次）。
        # 动机：两个 400 样本训练跑下来，策略在 γ 维度上始终接近均匀分布
        # （末期 γ=2/4/8/16 = 594/564/397/445），而按探针数据反解的奖励明明偏好
        # γ=16（每次决策 −0.035 vs γ=4 的 −0.860）。在改任何东西之前，必须先
        # **直接测出策略实际观测到的 E[r | γ]**：如果 γ=16 的实测奖励并不更好，
        # 那就是奖励/环境的问题；如果更好却学不到，才是估计器或探索的问题。
        if self._action_calls and self._action_calls % 500 == 0 and self._reward_by_gamma:
            parts = []
            for gi, gval in enumerate(self.gamma_candidates):
                n, s = self._reward_by_gamma.get(gi, (0, 0.0))
                if n:
                    parts.append(f"g{gval}: n={n} r={s / n:+.3f}")
            if parts:
                print(
                    f"[{self.agent.name}] reward by gamma: " + " | ".join(parts),
                    flush=True,
                )

        if not self.frozen:
            self.last_state_seq = state_seq
            self.last_action = action_idx
            self.last_reward = None

        if ACTION_TRACE_PATH:
            # 记录"实际生效"的动作：强制消融（--rl_force_gamma/threshold）会覆盖
            # 局部变量，用 last_* 才能反映系统真正使用的档位（F14 同类问题：
            # 之前 trace 一直显示被覆盖前的值，误导了两次结论）。
            self._trace_action(
                bandwidth_mbps, latency_ms, entropy, last_acc, task_name,
                action_idx, self.last_topk, self.last_threshold, training,
                self.last_gamma,
            )

        # 必须返回"实际生效"的值：强制消融（--rl_force_gamma/threshold）与 legacy
        # gamma 回填都只改了 self.last_*，如果这里返回局部变量，调用方
        # （baselines.py: self.draft_target_adapter.threshold = next_threshold）拿到的
        # 就是覆盖前的档位，消融实验会变成"测了个假值"（F18 的第一次结论正是这个
        # bug 的产物；同类问题此前已出现两次：F14 与轨迹记录）。
        return self.last_topk, self.last_threshold

    def _trace_action(
        self, bandwidth_mbps, latency_ms, entropy, last_acc, task_name,
        action_idx, selected_topk, selected_threshold, training,
        selected_gamma=None,
    ):
        record = {
            "agent": self.agent.name,
            "step": self.agent.update_count,
            "bw_mbps": bandwidth_mbps,
            "ntt_ms": latency_ms,
            "entropy": entropy,
            "last_acc": last_acc,
            "task": task_name,
            "action_idx": action_idx,
            "topk": selected_topk,
            "threshold": selected_threshold,
            "gamma": selected_gamma,
            "training": bool(training),
            "epsilon": self.agent.epsilon,
        }
        try:
            with open(ACTION_TRACE_PATH, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:  # tracing must never break a run
            warnings.warn(f"RL action trace write failed: {exc}")

    def compute_reward(
        self,
        *,
        accepted: float,
        generated: float,
        comm_s: float,
        compute_s: float,
        energy_j: float = 0.0,
        forward_counts: dict | None = None,
        opportunistic: bool | None = None,
        transferred_bytes: float = 0.0,
    ) -> float:
        """Reward of ONE decoding decision, plus its component bookkeeping.

        `compute_s` is only used when `rl_compute_time_mode == "wall"` (legacy);
        with `"model"` the compute time is reconstructed from `forward_counts`
        and the calibrated per-model costs, which makes the reward independent of
        the host load (see src/rl_reward.py).
        """
        reward, components = self.reward_shaper.compute(
            accepted=accepted,
            generated=generated,
            comm_s=comm_s,
            compute_s=compute_s,
            energy_j=energy_j,
            forward_counts=forward_counts,
            opportunistic=opportunistic,
            transferred_bytes=transferred_bytes,
        )
        self.reward_log.add(reward)
        self.last_reward_components = components
        self._reward_calls += 1
        # 归因到"本轮实际生效的 γ"：奖励在 select_config 之前计算，所以 last_gamma
        # 仍是上一轮决策选出的 γ —— 正是决定了本次测量区间长度的那个值。
        if self.gamma_dim > 1:
            gi = self.gamma_candidates.index(self.last_gamma)
            n, s = self._reward_by_gamma.get(gi, (0, 0.0))
            self._reward_by_gamma[gi] = (n + 1, s + reward)
        if REWARD_TRACE_PATH:
            try:
                with open(REWARD_TRACE_PATH, "a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "agent": self.agent.name,
                                "step": self.agent.update_count,
                                "bw_mbps": self.last_bw,
                                "ntt_ms": self.last_ntt,
                                "gamma": self.last_gamma,
                                "topk": self.last_topk,
                                "threshold": self.last_threshold,
                                "accepted": float(accepted),
                                "generated": float(generated),
                                "comm_s": components["comm_s"],
                                "compute_s": components["compute_s"],
                                "bytes": components.get("transferred_bytes", 0.0),
                                "byte_price_s": components.get("byte_price_s", 0.0),
                                "total_s": components.get("total_s", 0.0),
                                "reward": float(reward),
                            }
                        )
                        + "\n"
                    )
            except OSError as exc:  # tracing must never break a run
                warnings.warn(f"RL reward trace write failed: {exc}")
        if self._reward_calls % 500 == 0:
            print(f"[{self.agent.name}] {self.reward_log.format(components)}", flush=True)
        return reward

    def step(self, reward: float):
        if self.frozen:
            return
        self.last_reward = reward

    def save(self, current_tps: float | None = None):
        if self.frozen:
            return
        if (
            self.last_state_seq is not None
            and self.last_action is not None
            and self.last_reward is not None
        ):
            self.agent.store_transition(
                self.last_state_seq,
                self.last_action,
                self.last_reward,
                self.last_state_seq,
                done=True,
            )
            self.agent.update()
            self.last_state_seq = None
            self.last_action = None
            self.last_reward = None
        new_best = current_tps is not None and current_tps > self.best_tps
        if new_best:
            self.best_tps = current_tps
        self.agent.best_tps = self.best_tps
        self.agent.save(self.model_path)

        if new_best:
            self.agent.save(self.best_model_path)
            print(
                f"[{self.agent.name}] New Best TPS: {current_tps:.2f}! Saved to {self.best_model_path}"
            )

        # 每 100 次更新备份一次（可选）
        if self.agent.update_count % 100 == 0:
            self.agent.save(self.model_path)
