import numpy as np
from typing import Dict, List

class LowLevelReward:
    def __init__(self, config: Dict):
        """
        底层控制器奖励计算
        Args:
            config: 配置字典，包含奖励权重
        """
        self.weights = config['controllers']['low_level']['reward_weights']
        
    def calculate_reward(self, vehicle_info: Dict) -> float:
        """
        计算底层控制器的奖励
        Args:
            vehicle_info: 车辆信息，包含速度、加速度等
        Returns:
            float: 奖励值
        """
        # 基础移动奖励
        reward = self.weights['base_move'] if vehicle_info['speed'] > 0 else self.weights['stop_penalty']
        
        # 平滑控制奖励
        if abs(vehicle_info['acceleration']) < 2.0:
            reward += self.weights['smooth']
            
        # 速度维持奖励 (5-14 m/s, 约18-50km/h)
        if 5 <= vehicle_info['speed'] <= 14:
            reward += self.weights['speed']
            
        return reward
