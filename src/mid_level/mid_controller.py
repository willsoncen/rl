import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple

from .mid_network import Actor, Critic
from .mid_reward import MidLevelReward
from .speed_planner import SpeedPlanner
from ..replay_buffer import ReplayBuffer

class MidLevelController:
    def __init__(self, config: Dict, device: torch.device):
        """
        中层控制器
        Args:
            config: 控制器配置
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # 状态和动作空间
        self.state_dim = config['environment']['state_dim']['mid']
        self.action_dim = config['environment']['action_dim']['mid']
        
        # 创建网络
        self.actor = Actor(
            self.state_dim, self.action_dim,
            config['controllers']['mid_level']['hidden_size']
        ).to(device)
        
        self.actor_target = Actor(
            self.state_dim, self.action_dim,
            config['controllers']['mid_level']['hidden_size']
        ).to(device)
        
        self.critic = Critic(
            self.state_dim, self.action_dim,
            config['controllers']['mid_level']['hidden_size']
        ).to(device)
        
        self.critic_target = Critic(
            self.state_dim, self.action_dim,
            config['controllers']['mid_level']['hidden_size']
        ).to(device)
        
        # 复制参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config['controllers']['mid_level']['actor_lr']
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config['controllers']['mid_level']['critic_lr']
        )
        
        # 创建经验回放
        self.buffer = ReplayBuffer(
            config['controllers']['mid_level']['buffer_size'],
            self.state_dim,
            self.action_dim
        )
        
        # 创建奖励计算器和速度规划器
        self.reward_calculator = MidLevelReward(config)
        self.speed_planner = SpeedPlanner(config)
        
        # 训练参数
        self.gamma = config['controllers']['mid_level']['gamma']
        self.tau = config['controllers']['mid_level']['tau']
        self.batch_size = config['controllers']['mid_level']['batch_size']
        
    def select_action(self, state: np.ndarray, vehicle: Dict, 
                     collision_pairs: List[Tuple[Dict, Dict]], 
                     time_allocation: Dict,
                     nearby_vehicles: List[Dict],
                     evaluate: bool = False) -> np.ndarray:
        """选择动作"""
        # 1. 使用DDPG网络生成基础动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
            
        if not evaluate:
            noise = np.random.normal(0, 0.1, size=self.action_dim)
            action = np.clip(action + noise, -1.0, 1.0)
            
        # 2. 使用速度规划器调整动作
        speed_plan = self.speed_planner.plan_speed(vehicle, collision_pairs, time_allocation)
        
        # 3. 安全检查
        safe_plan = self.speed_planner.check_safety(vehicle, speed_plan, nearby_vehicles)
        
        # 4. 将规划结果转换为动作空间
        final_action = np.array([
            safe_plan['target_speed'] / self.speed_planner.max_speed,  # 归一化速度
            safe_plan['acceleration'] / self.speed_planner.max_accel   # 归一化加速度
        ])
        
        return np.clip(final_action, -1.0, 1.0)
        
    def train_step(self) -> float:
        """训练一步"""
        if self.buffer.size < self.batch_size:
            return 0.0
            
        # 采样经验
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, self.device
        )
        
        # 更新评论家
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新演员
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
            
        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
            
        return critic_loss.item()
        
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
