import torch
import numpy as np
from typing import Dict, List, Any

class ModelValidator:
    def __init__(self, config: Dict):
        self.config = config
        
    def validate_network_inputs(self, state: torch.Tensor, network_name: str) -> bool:
        """验证网络输入"""
        expected_dim = self.config['environment']['state_dim'][network_name]
        if state.shape[-1] != expected_dim:
            raise ValueError(f"Invalid input dimension for {network_name}. "
                           f"Expected {expected_dim}, got {state.shape[-1]}")
        
        if torch.isnan(state).any():
            raise ValueError(f"NaN values detected in {network_name} input")
            
        return True
        
    def validate_actions(self, actions: np.ndarray, level: str) -> bool:
        """验证动作输出"""
        if not (-1 <= actions).all() or not (actions <= 1).all():
            raise ValueError(f"Actions for {level} must be in range [-1, 1]")
            
        if np.isnan(actions).any():
            raise ValueError(f"NaN values detected in {level} actions")
            
        return True
        
    def validate_rewards(self, rewards: Dict[str, float]) -> bool:
        """验证奖励值"""
        for level, reward in rewards.items():
            if not isinstance(reward, (int, float)):
                raise ValueError(f"Invalid reward type for {level}")
                
            if np.isnan(reward) or np.isinf(reward):
                raise ValueError(f"Invalid reward value for {level}: {reward}")
                
        return True
