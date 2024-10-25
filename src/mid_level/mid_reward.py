import numpy as np
from typing import Dict, List

class MidLevelReward:
    def __init__(self, config: Dict):
        """
        中层控制器奖励计算
        Args:
            config: 配置字典，包含奖励权重
        """
        self.weights = config['controllers']['mid_level']['reward_weights']
        self.control_radius = config['training']['control_radius']
        
    def calculate_reward(self, vehicles_info: List[Dict]) -> float:
        """
        计算中层控制器的奖励
        Args:
            vehicles_info: 控制区域内的车辆信息列表
        Returns:
            float: 奖励值
        """
        if not vehicles_info:
            return 0.0
            
        total_reward = 0
        n_vehicles = len(vehicles_info)
        
        # 检查碰撞
        collision_penalty = self._check_collisions(vehicles_info)
        
        for vehicle in vehicles_info:
            # 基础移动奖励
            reward = self.weights['base_move'] if vehicle['speed'] > 0 else self.weights['stop_penalty']
            
            # 通过奖励
            if vehicle['distance'] < 10:  # 距离十字路口中心小于10米
                reward += self.weights['cross']
            else:
                # 进度奖励
                progress = (self.control_radius - vehicle['distance']) / self.control_radius
                reward += self.weights['progress'] * progress
                
            total_reward += reward
            
        # 添加碰撞惩罚
        total_reward += collision_penalty
        
        return total_reward / n_vehicles
        
    def _check_collisions(self, vehicles: List[Dict]) -> float:
        """检查车辆间是否存在碰撞"""
        collision_penalty = 0
        
        for i, v1 in enumerate(vehicles[:-1]):
            for v2 in vehicles[i+1:]:
                dist = np.sqrt((v1['x'] - v2['x'])**2 + (v1['y'] - v2['y'])**2)
                if dist < 5:  # 如果两车距离小于5米
                    collision_penalty += self.weights['collision']
                    
        return collision_penalty
