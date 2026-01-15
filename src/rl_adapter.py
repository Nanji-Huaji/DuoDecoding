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
K_CANDIDATES = [0, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 32000]

# 定义任务列表以进行 One-Hot 编码
KNOWN_TASKS = ["mt_bench", "gsm8k", "cnndm", "xsum", "humaneval"]
TASK_MAP = {name: i for i, name in enumerate(KNOWN_TASKS)}
UNKNOWN_TASK_ID = len(KNOWN_TASKS)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

class DDQNAgent:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dim=256, 
        lr=1e-3, 
        gamma=0.99, 
        epsilon=1.0, 
        epsilon_decay=0.995, 
        epsilon_min=0.01, 
        buffer_size=10000, 
        batch_size=64,
        target_update_freq=10,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.update_count = 0

        # Q Networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay Buffer
        self.memory = deque(maxlen=buffer_size)

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # DDQN Logic
        # 1. Select action using Policy Net
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # 2. Evaluate action using Target Net
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        with torch.enable_grad():
            current_q_values = self.policy_net(states).gather(1, actions)

            loss = self.loss_fn(current_q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.update_count % 10 == 0:
                print(f"[RL Agent] Step: {self.update_count}, Loss: {loss.item():.4f}, Epsilon: {self.epsilon:.4f}")

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        # Save model
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, path)
        
        # Save buffer
        buffer_path = path + ".buffer"
        try:
            with open(buffer_path, 'wb') as f:
                pickle.dump(list(self.memory), f)
        except Exception as e:
            print(f"Failed to save replay buffer: {e}")

    def load(self, path):
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint['epsilon']
                self.update_count = checkpoint.get('update_count', 0)
                print(f"Loaded RL agent from {path}, steps: {self.update_count}")
                
                # Load buffer
                buffer_path = path + ".buffer"
                if os.path.exists(buffer_path):
                    with open(buffer_path, 'rb') as f:
                        memory_list = pickle.load(f)
                        self.memory.extend(memory_list)
                    print(f"Loaded {len(memory_list)} transitions from replay buffer.")
            except Exception as e:
                print(f"Failed to load checkpoint from {path} (possibly due to architecture/state_dim change): {e}")
                print("Starting training from scratch.")

class RLNetworkAdapter:
    def __init__(self, args, device="cuda"):
        self.args = args
        self.device = device
        
        # State definition: 
        # [bandwidth (norm), latency (norm), draft_acc_prob, entropy] + [one-hot task vector (6 dims)]
        self.task_dim = len(KNOWN_TASKS) + 1 # +1 for unknown
        self.state_dim = 4 + self.task_dim
        self.action_dim = len(K_CANDIDATES)
        
        self.agent = DDQNAgent(self.state_dim, self.action_dim, device=device)
        
        # Normalization factors (can be adjusted)
        self.max_bandwidth = 1000.0 # Mbps
        self.max_latency = 500.0 # ms
        
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        
        # Load pretrained model if exists
        self.model_path = os.path.join("checkpoints", "rl_adapter.pth")
        os.makedirs("checkpoints", exist_ok=True)
        self.agent.load(self.model_path)

    def get_state(self, bandwidth_mbps: float, latency_ms: float, draft_acc_prob: float, entropy: float, task_name: str = "unknown") -> np.ndarray:
        """
        Construct state vector.
        """
        # 1. Normalize Bandwidth
        norm_bw = min(bandwidth_mbps / self.max_bandwidth, 1.0)
        
        # 2. Normalize Latency
        norm_lat = min(latency_ms / self.max_latency, 1.0)
        
        # 3. Draft Acceptance Probability (already 0-1)
        acc_prob = draft_acc_prob
        
        # 4. Entropy 
        norm_entropy = min(entropy / 10.0, 1.0)
        
        # 5. Task One-Hot Encoding
        task_idx = TASK_MAP.get(task_name, UNKNOWN_TASK_ID)
        task_vec = np.zeros(self.task_dim, dtype=np.float32)
        task_vec[task_idx] = 1.0
        
        basic_features = np.array([norm_bw, norm_lat, acc_prob, norm_entropy], dtype=np.float32)
        
        return np.concatenate([basic_features, task_vec])

    def select_k(self, bandwidth_mbps: float, latency_ms: float, draft_acc_prob: float, entropy: float, task_name: str = "unknown", training=True) -> int:
        """
        Select top-k parameter k.
        """
        state = self.get_state(bandwidth_mbps, latency_ms, draft_acc_prob, entropy, task_name)
        
        # If we have a previous state/action/reward, store the transition
        if self.last_state is not None and self.last_action is not None and self.last_reward is not None:
            self.agent.store_transition(self.last_state, self.last_action, self.last_reward, state, done=False)
            self.agent.update()

        action_idx = self.agent.select_action(state, training=training)
        
        self.last_state = state
        self.last_action = action_idx
        self.last_reward = None # Reset reward
        
        return K_CANDIDATES[action_idx]

    def step(self, reward: float):
        """
        Observe reward.
        """
        self.last_reward = reward

    def save(self):
        self.agent.save(self.model_path)
        
        if self.agent.update_count % 100 == 0:
            self.agent.save(self.model_path)

