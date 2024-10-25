import numpy as np
from typing import List, Dict, Tuple

class SpaceTimeAllocator:
    def __init__(self, config: Dict):
        """
        时空分配器
        Args:
            config: 配置字典
        """
        self.control_radius = config['training']['control_radius']
        self.time_slot = 2.0  # 时间槽大小(秒)
        self.space_grid = 5.0  # 空间网格大小(米)
        
    def allocate_space_time(self, collision_pairs: List[Tuple[Dict, Dict]]) -> Dict:
        """
        为碰撞对分配时空资源
        Args:
            collision_pairs: 碰撞对列表
        Returns:
            Dict: 分配结果，包含每辆车的通过时间和位置
        """
        allocations = {}
        occupied_slots = set()  # 已占用的时空槽
        
        for v1, v2 in collision_pairs:
            # 计算两辆车的预计到达时间
            t1 = self.predict_arrival_time(v1)
            t2 = self.predict_arrival_time(v2)
            
            # 为两辆车分配不同的时间槽
            slot1 = self.find_free_slot(occupied_slots, t1)
            slot2 = self.find_free_slot(occupied_slots, t2)
            
            # 记录分配结果
            allocations[v1['id']] = {
                'time': slot1 * self.time_slot,
                'position': self.calculate_position(v1, slot1)
            }
            allocations[v2['id']] = {
                'time': slot2 * self.time_slot,
                'position': self.calculate_position(v2, slot2)
            }
            
            # 标记时间槽为已占用
            occupied_slots.add(slot1)
            occupied_slots.add(slot2)
            
        return allocations
        
    def predict_arrival_time(self, vehicle: Dict) -> float:
        """预测车辆到达控制区域的时间"""
        dist = max(0, vehicle['distance'] - self.control_radius)
        return dist / max(vehicle['speed'], 1e-6)
        
    def find_free_slot(self, occupied_slots: set, preferred_time: float) -> int:
        """找到最近的空闲时间槽"""
        slot = int(preferred_time / self.time_slot)
        while slot in occupied_slots:
            slot += 1
        return slot
        
    def calculate_position(self, vehicle: Dict, time_slot: int) -> Dict:
        """计算车辆在指定时间槽的位置"""
        # 简化版本：根据车辆的路线和速度预测位置
        time = time_slot * self.time_slot
        distance = vehicle['speed'] * time
        return {
            'x': vehicle['x'] + distance * np.cos(vehicle['heading']),
            'y': vehicle['y'] + distance * np.sin(vehicle['heading'])
        }
