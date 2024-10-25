import numpy as np
from typing import Dict, List

class HighLevelReward:
    def __init__(self, config: Dict):
        """
        高层控制器奖励计算
        Args:
            config: 配置字典，包含奖励权重
        """
        self.weights = config['reward_weights']
        self.control_radius = config['training']['control_radius']
        
    def calculate_reward(self, vehicles_info: List[Dict]) -> float:
        """
        计算高层控制器的奖励
        Args:
            vehicles_info: 车辆信息列表，每个字典包含车辆的位置、速度等信息
        Returns:
            float: 奖励值
        """
        if not vehicles_info:
            return 0.0
            
        total_reward = 0
        n_vehicles = len(vehicles_info)
        
        for vehicle in vehicles_info:
            # 基础移动奖励
            reward = self.weights['base_move'] if vehicle['speed'] > 0 else self.weights['stop_penalty']
            
            # 速度奖励 (8-14 m/s, 约30-50km/h)
            if 8 <= vehicle['speed'] <= 14:
                reward += self.weights['speed']
                
            # 接近奖励 (根据到控制区域的距离)
            dist_to_control = max(0, vehicle['distance'] - self.control_radius)
            approach_reward = self.weights['approach'] * (1 - dist_to_control / 300)
            reward += approach_reward
            
            total_reward += reward
            
        return total_reward / n_vehicles
