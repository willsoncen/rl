import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        """
        底层Actor网络 (TD3)
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0]),
            nn.Dropout(0.1)
        )
        
        # 特征提取层
        self.feature_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.feature_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU(),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.Dropout(0.1)
            ))
        
        # 动作输出层
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Tanh()  # 输出范围[-1,1]
        )
        
        # 参数初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            state: 状态输入 [batch_size, state_dim]
        Returns:
            动作输出 [batch_size, action_dim]
        """
        x = self.state_encoder(state)
        
        for layer in self.feature_layers:
            x = layer(x)
            
        return self.action_head(x)

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        """
        底层Critic网络 (TD3)
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        # 状态和动作编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0] // 2)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0] // 2)
        )
        
        # 双Q网络
        self.q1_layers = self._build_critic_layers(hidden_dims)
        self.q2_layers = self._build_critic_layers(hidden_dims)
        
        # 参数初始化
        self.apply(self._init_weights)
        
    def _build_critic_layers(self, hidden_dims: List[int]) -> nn.ModuleList:
        layers = nn.ModuleList()
        prev_dim = hidden_dims[0]  # 状态和动作特征拼接后的维度
        
        for hidden_dim in hidden_dims[1:]:
            layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        return layers
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            state: 状态输入 [batch_size, state_dim]
            action: 动作输入 [batch_size, action_dim]
        Returns:
            两个Q值输出 [batch_size, 1]
        """
        state_features = self.state_encoder(state)
        action_features = self.action_encoder(action)
        x = torch.cat([state_features, action_features], dim=1)
        
        # Q1计算
        q1 = x
        for layer in self.q1_layers:
            q1 = layer(q1)
            
        # Q2计算
        q2 = x
        for layer in self.q2_layers:
            q2 = layer(q2)
            
        return q1, q2
