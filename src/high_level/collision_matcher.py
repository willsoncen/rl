import numpy as np
from typing import List, Dict, Tuple
from itertools import combinations

class CollisionMatcher:
    def __init__(self, config: Dict):
        """
        碰撞对匹配器
        Args:
            config: 配置字典
        """
        self.control_radius = config['training']['control_radius']
        self.time_threshold = 5.0  # 时间阈值(秒)
        
    def predict_arrival_time(self, vehicle: Dict) -> float:
        """预测车辆到达控制区域的时间"""
        dist_to_control = max(0, vehicle['distance'] - self.control_radius)
        if vehicle['speed'] <= 0:
            return float('inf')
        return dist_to_control / vehicle['speed']
        
    def match_collision_pairs(self, vehicles: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """
        匹配可能的碰撞对
        Args:
            vehicles: 控制区域外的车辆列表
        Returns:
            List[Tuple]: 碰撞对列表
        """
        collision_pairs = []
        vehicles_with_time = [(v, self.predict_arrival_time(v)) for v in vehicles]
        
        # 按到达时间排序
        vehicles_with_time.sort(key=lambda x: x[1])
        
        # 检查所有可能的对
        for i, (v1, t1) in enumerate(vehicles_with_time[:-1]):
            for v2, t2 in vehicles_with_time[i+1:]:
                # 如果到达时间差异小于阈值，且路径有交叉
                if abs(t1 - t2) < self.time_threshold and self.check_path_conflict(v1, v2):
                    collision_pairs.append((v1, v2))
                    
        return collision_pairs
        
    def check_path_conflict(self, v1: Dict, v2: Dict) -> bool:
        """检查两辆车的路径是否可能发生冲突"""
        # 简化版本：检查是否来自不同方向且目标方向有交叉
        return (v1['direction'] != v2['direction'] and 
                self.will_paths_cross(v1['route'], v2['route']))
                
    def will_paths_cross(self, route1: List[str], route2: List[str]) -> bool:
        """检查两条路径是否会相交"""
        # 简化版本：检查是否有共同的路段
        return bool(set(route1) & set(route2))
