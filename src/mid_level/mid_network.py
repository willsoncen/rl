import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list):
        """
        中层Actor网络 (DDPG)
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        # 特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.1)
        )
        
        # 策略网络层
        layers = []
        prev_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        layers.extend([
            nn.Linear(prev_dim, action_dim),
            nn.Tanh()  # 将输出限制在[-1,1]范围
        ])
        
        self.policy_net = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            state: 状态输入
        Returns:
            动作输出
        """
        features = self.feature_net(state)
        return self.policy_net(features)

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list):
        """
        中层Critic网络 (DDPG)
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        # 状态编码层
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0])
        )
        
        # 动作编码层
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0])
        )
        
        # Q值网络层
        layers = []
        prev_dim = hidden_dims[0] * 2  # 状态和动作特征拼接
        
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.q_net = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            state: 状态输入
            action: 动作输入
        Returns:
            Q值输出
        """
        state_features = self.state_encoder(state)
        action_features = self.action_encoder(action)
        combined_features = torch.cat([state_features, action_features], dim=1)
        return self.q_net(combined_features)
