import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class HighLevelNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        """
        高层控制器网络 (DQN)
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
        
        # 优势流和价值流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, action_dim)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            state: 状态输入 [batch_size, state_dim]
        Returns:
            Q值输出 [batch_size, action_dim]
        """
        # 状态编码
        x = self.state_encoder(state)
        
        # 特征提取
        for layer in self.feature_layers:
            x = layer(x)
            
        # 分离优势流和价值流
        advantage = self.advantage_stream(x)
        value = self.value_stream(x)
        
        # 组合Q值 (Dueling DQN架构)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
