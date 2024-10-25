import numpy as np
from typing import Tuple, Dict
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        """
        经验回放缓冲区
        Args:
            capacity: 缓冲区容量
            state_dim: 状态空间维度
            action_dim: 动作空间维度
        """
        self.capacity = capacity
        self.pointer = 0
        self.size = 0
        
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros((capacity, 1))
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros((capacity, 1))
        
    def add(self, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, done: bool):
        """添加一条经验"""
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done
        
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """采样一个批次的经验"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[indices]).to(device),
            torch.FloatTensor(self.actions[indices]).to(device),
            torch.FloatTensor(self.rewards[indices]).to(device),
            torch.FloatTensor(self.next_states[indices]).to(device),
            torch.FloatTensor(self.dones[indices]).to(device)
        )

