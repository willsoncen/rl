import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple

from .high_network import HighLevelNetwork
from .high_reward import HighLevelReward
from .collision_matcher import CollisionMatcher
from .space_time_allocator import SpaceTimeAllocator
from ..replay_buffer import ReplayBuffer

class HighLevelController:
    def __init__(self, config: Dict, device: torch.device):
        """
        高层控制器
        Args:
            config: 控制器配置
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # 状态和动作空间
        self.state_dim = config['environment']['state_dim']['high']
        self.action_dim = config['environment']['action_dim']['high']
        
        # 创建网络
        self.network = HighLevelNetwork(
            self.state_dim, 
            self.action_dim,
            config['controllers']['high_level']['hidden_size']
        ).to(device)
        
        self.target_network = HighLevelNetwork(
            self.state_dim,
            self.action_dim,
            config['controllers']['high_level']['hidden_size']
        ).to(device)
        
        self.target_network.load_state_dict(self.network.state_dict())
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config['controllers']['high_level']['learning_rate']
        )
        
        # 创建经验回放
        self.buffer = ReplayBuffer(
            config['controllers']['high_level']['buffer_size'],
            self.state_dim,
            1  # DQN的动作是离散的
        )
        
        # 创建奖励计算器
        self.reward_calculator = HighLevelReward(config)
        
        # 创建碰撞对匹配器和时空分配器
        self.collision_matcher = CollisionMatcher(config)
        self.space_time_allocator = SpaceTimeAllocator(config)
        
        # 训练参数
        self.epsilon = config['controllers']['high_level']['epsilon']
        self.gamma = config['controllers']['high_level']['gamma']
        self.tau = config['controllers']['high_level']['tau']
        self.batch_size = config['controllers']['high_level']['batch_size']
        
    def select_action(self, state: np.ndarray, vehicles: List[Dict], evaluate: bool = False) -> Dict:
        """
        选择动作
        Args:
            state: 环境状态
            vehicles: 车辆信息列表
            evaluate: 是否是评估模式
        Returns:
            Dict: 包含碰撞对和时空分配的决策结果
        """
        # 1. 匹配碰撞对
        collision_pairs = self.collision_matcher.match_collision_pairs(vehicles)
        
        # 2. 使用DQN网络选择动作(时空分配方案的索引)
        if not evaluate and np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.network(state_tensor)
                action_idx = q_values.argmax().item()
                
        # 3. 根据选择的动作进行时空分配
        allocations = self.space_time_allocator.allocate_space_time(collision_pairs)
        
        return {
            'collision_pairs': collision_pairs,
            'allocations': allocations,
            'action_idx': action_idx
        }
        
    def train_step(self) -> float:
        """训练一步"""
        if self.buffer.size < self.batch_size:
            return 0.0
            
        # 采样经验
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, self.device
        )
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # 计算当前Q值
        q_values = self.network(states).gather(1, actions.long())
        
        # 计算损失
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(
            self.network.parameters(),
            self.target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
            
        return loss.item()
        
    def save(self, path: str):
        """保存模型"""
        torch.save(self.network.state_dict(), path)
        
    def load(self, path: str):
        """加载模型"""
        self.network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.network.state_dict())
