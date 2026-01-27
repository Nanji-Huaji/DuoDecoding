import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
from collections import deque
from typing import List, Tuple, Dict
import os

# 定义候选的 K 值
K_CANDIDATES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# 定义候选的阈值
THRESHOLD_CANDIDATES = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]

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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.adv_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        x = torch.relu(self.fc_embed(x)) 
        lstm_out, _ = self.lstm(x)
        last_timestep_out = lstm_out[:, -1, :]
        
        val = self.val_fc(last_timestep_out)
        adv = self.adv_fc(last_timestep_out)
        
        return val + adv - adv.mean(1, keepdim=True)

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
        target_update_freq=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        name="RL-Agent"
    ):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.update_count = 0
        self.name = name
        self.reward_history = deque(maxlen=100)

        self.policy_net = RecurrentQNetwork(feature_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = RecurrentQNetwork(feature_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = deque(maxlen=buffer_size)

    def select_action(self, state_seq, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
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

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 核心修复: 确保在 enable_grad 环境下运行，因为调用者(推理循环)通常在 no_grad 下
        with torch.enable_grad():
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            rewards = rewards * 0.01

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
                avg_reward = np.mean(self.reward_history) if self.reward_history else 0.0
                print(f"[{self.name}] Step: {self.update_count}, Loss: {loss.item():.4f}, Reward: {avg_reward:.4f}, Epsilon: {self.epsilon:.4f}")

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, path)
        buffer_path = path + ".buffer"
        try:
            with open(buffer_path, 'wb') as f:
                pickle.dump(list(self.memory)[-2000:], f) 
        except Exception:
            pass

    def load(self, path):
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint['epsilon']
                self.update_count = checkpoint.get('update_count', 0)
                print(f"Loaded LSTM-RL agent from {path}, steps: {self.update_count}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Starting fresh.")

class RLNetworkAdapter:
    def __init__(self, args, model_name="rl_adapter", device="cuda", k_candidates=None, threshold_candidates=None):
        self.args = args
        self.device = device
        
        # ==========================================
        # 核心修改 2: 状态定义变更 (时序特征)
        # ==========================================
        self.task_dim = len(KNOWN_TASKS) + 1
        self.feature_dim = 3 + 1 + self.task_dim # [bw, lat, entropy, last_acc] + task
        self.seq_len = 8 
        
        self.state_history = deque(
            [np.zeros(self.feature_dim) for _ in range(self.seq_len)], 
            maxlen=self.seq_len
        )
        
        self.k_candidates = k_candidates if k_candidates is not None else K_CANDIDATES
        self.threshold_candidates = threshold_candidates if threshold_candidates is not None else THRESHOLD_CANDIDATES
        self.action_dim = len(self.k_candidates) * len(self.threshold_candidates)
        
        self.agent = DDQNAgent(
            feature_dim=self.feature_dim, 
            action_dim=self.action_dim, 
            seq_len=self.seq_len,
            device=device,
            name=model_name
        )
        
        self.max_bandwidth = 1000.0 
        self.max_latency = 500.0 
        
        self.last_state_seq = None
        self.last_action = None
        self.last_reward = None
        
        self.model_path = os.path.join("checkpoints", f"{model_name}.pth")
        self.best_model_path = os.path.join("checkpoints", f"{model_name}_best.pth")
        self.best_tps = -1.0
        
        os.makedirs("checkpoints", exist_ok=True)
        # 实验评估时默认加载最优模型
        if os.path.exists(self.best_model_path):
            self.agent.load(self.best_model_path)
        else:
            self.agent.load(self.model_path)

    def _get_current_feature_vector(self, bandwidth_mbps, latency_ms, entropy, last_acc_prob, task_name):
        norm_bw = min(bandwidth_mbps / self.max_bandwidth, 1.0)
        norm_lat = min(latency_ms / self.max_latency, 1.0)
        norm_entropy = min(entropy / 10.0, 1.0)
        
        task_idx = TASK_MAP.get(task_name, UNKNOWN_TASK_ID)
        task_vec = np.zeros(self.task_dim, dtype=np.float32)
        task_vec[task_idx] = 1.0
        
        return np.concatenate([
            [norm_bw, norm_lat, norm_entropy, last_acc_prob], 
            task_vec
        ]).astype(np.float32)

    def select_config(self, bandwidth_mbps: float, latency_ms: float, acc_probs: List[float], entropy: float, task_name: str = "unknown", training=True) -> Tuple[int, float]:
        last_acc = acc_probs[-1] if len(acc_probs) > 0 else 0.5
        current_feat = self._get_current_feature_vector(bandwidth_mbps, latency_ms, entropy, last_acc, task_name)
        self.state_history.append(current_feat)
        state_seq = np.array(self.state_history)
        
        if self.last_state_seq is not None and self.last_action is not None and self.last_reward is not None:
            self.agent.store_transition(self.last_state_seq, self.last_action, self.last_reward, state_seq, done=False)
            self.agent.update()

        action_idx = self.agent.select_action(state_seq, training=training)
        
        k_idx = action_idx // len(self.threshold_candidates)
        t_idx = action_idx % len(self.threshold_candidates)
        
        selected_k = self.k_candidates[k_idx]
        selected_threshold = self.threshold_candidates[t_idx]
        
        self.last_state_seq = state_seq
        self.last_action = action_idx
        self.last_reward = None
        
        return selected_k, selected_threshold

    def step(self, reward: float):
        self.last_reward = reward

    def save(self, current_tps: float = None):
        # 始终保存最新的模型
        self.agent.save(self.model_path)
        
        # 如果提供了 TPS 且是目前最好的，则保存为 _best.pth
        if current_tps is not None and current_tps > self.best_tps:
            self.best_tps = current_tps
            self.agent.save(self.best_model_path)
            print(f"[{self.agent.name}] New Best TPS: {current_tps:.2f}! Saved to {self.best_model_path}")

        # 每 100 次更新备份一次（可选）
        if self.agent.update_count % 100 == 0:
            self.agent.save(self.model_path)

