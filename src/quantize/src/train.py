import gymnasium as gym
import torch
from model import QNetwork
from utils import get_model
import json
from gymnasium import spaces

class QuantizeSpecDecEnv(gym.Env):
    def __init__(self, bandwidth_mbps: int | float, draft_model_path: str, target_model_path: str, dataset_path: str):
        self.bandwidth = bandwidth_mbps
        self.draft_model, _ = get_model(draft_model_path)
        self.target_model, self.tokenizer = get_model(target_model_path)
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)

        # action space: (k, quantization level)
        self.actions_map = [
            (1,1), (1,2), (2,1), (2,2), (3,1), (3,2), 
            (4,1), (4,2), (6,1), (6,2), (8,1), (8,2)
        ]

        self.action_space = spaces.Discrete(len(self.actions_map))


    def step(self, action):
        pass