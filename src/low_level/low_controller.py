import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple

from .low_network import Actor, Critic
from .low_reward import LowLevelReward
from .action_executor import ActionExecutor
from ..replay_buffer import ReplayBuffer

class LowLevelController:
    def __init__(self, config: Dict, device: torch.device):
        """
        底层控制器
        Args:
            config: 控制器配置
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # 状态和动作空间
        self.state_dim = config['environment']['state_dim']['low']
        self.action_dim = config['environment']['action_dim']['low']
        
        # 创建网络
        self.actor = Actor(
            self.state_dim, self.action_dim,
            config['controllers']['low_level']['hidden_size']
        ).to(device)
        
        self.actor_target = Actor(
            self.state_dim, self.action_dim,
            config['controllers']['low_level']['hidden_size']
        ).to(device)
        
        self.critic1 = Critic(
            self.state_dim, self.action_dim,
            config['controllers']['low_level']['hidden_size']
        ).to(device)
        
        self.critic2 = Critic(
            self.state_dim, self.action_dim,
            config['controllers']['low_level']['hidden_size']
        ).to(device)
        
        self.critic1_target = Critic(
            self.state_dim, self.action_dim,
            config['controllers']['low_level']['hidden_size']
        ).to(device)
        
        self.critic2_target = Critic(
            self.state_dim, self.action_dim,
            config['controllers']['low_level']['hidden_size']
        ).to(device)
        
        # 复制参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config['controllers']['low_level']['actor_lr']
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(),
            lr=config['controllers']['low_level']['critic_lr']
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(),
            lr=config['controllers']['low_level']['critic_lr']
        )
        
        # 创建经验回放
        self.buffer = ReplayBuffer(
            config['controllers']['low_level']['buffer_size'],
            self.state_dim,
            self.action_dim
        )
        
        # 创建奖励计算器和动作执行器
        self.reward_calculator = LowLevelReward(config)
        self.action_executor = ActionExecutor(config)
        
        # TD3特定参数
        self.policy_noise = config['controllers']['low_level']['policy_noise']
        self.noise_clip = config['controllers']['low_level']['noise_clip']
        self.policy_freq = config['controllers']['low_level']['policy_freq']
        
        # 其他参数
        self.gamma = config['controllers']['low_level']['gamma']
        self.tau = config['controllers']['low_level']['tau']
        self.batch_size = config['controllers']['low_level']['batch_size']
        self.total_it = 0
        
    def select_action(self, state: np.ndarray, vehicle_info: Dict, 
                     evaluate: bool = False) -> Dict:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
            
            if not evaluate:
                noise = np.random.normal(0, 0.1, size=self.action_dim)
                action = np.clip(action + noise, -1.0, 1.0)
                
        # 执行动作
        return self.action_executor.execute_action(action, vehicle_info)
        
    def train_step(self) -> float:
        """训练一步"""
        self.total_it += 1
        
        if self.buffer.size < self.batch_size:
            return 0.0
            
        # 采样经验
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, self.device
        )
        
        # 为目标策略添加噪声
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise
                   ).clamp(-self.noise_clip, self.noise_clip)
                   
            next_actions = (self.actor_target(next_states) + noise
                         ).clamp(-1, 1)
                         
            # 目标Q值
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
        # 当前Q值
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # 计算critic损失
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # 更新critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 延迟策略更新
        if self.total_it % self.policy_freq == 0:
            # 计算actor损失
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            # 更新actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update_target()
            
        return (critic1_loss.item() + critic2_loss.item()) / 2
        
    def _soft_update_target(self):
        """软更新目标网络"""
        for param, target_param in zip(
            self.critic1.parameters(), self.critic1_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
            
        for param, target_param in zip(
            self.critic2.parameters(), self.critic2_target.parameters()
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
            
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
