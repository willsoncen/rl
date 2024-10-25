import numpy as np
from typing import Dict, List, Tuple

class SpeedPlanner:
    def __init__(self, config: Dict):
        """
        速度规划器
        Args:
            config: 配置字典
        """
        self.control_radius = config['training']['control_radius']
        self.max_speed = 14.0  # 最大速度 (m/s)
        self.min_speed = 2.0   # 最小速度 (m/s)
        self.max_accel = 2.0   # 最大加速度 (m/s²)
        self.min_accel = -4.0  # 最大减速度 (m/s²)
        
    def plan_speed(self, vehicle: Dict, collision_pairs: List[Tuple[Dict, Dict]], 
                  time_allocation: Dict) -> Dict:
        """
        规划车辆速度
        Args:
            vehicle: 车辆信息
            collision_pairs: 碰撞对列表
            time_allocation: 时间分配信息
        Returns:
            Dict: 速度规划结果
        """
        # 获取车辆的目标到达时间
        target_time = time_allocation.get(vehicle['id'], {}).get('time', None)
        
        if target_time is None:
            # 如果没有时间分配，保持当前速度
            return {
                'target_speed': vehicle['speed'],
                'acceleration': 0.0
            }
            
        # 计算所需速度
        distance = vehicle['distance']  # 到十字路口的距离
        current_speed = vehicle['speed']
        
        # 计算理想速度
        ideal_speed = distance / target_time if target_time > 0 else self.max_speed
        ideal_speed = np.clip(ideal_speed, self.min_speed, self.max_speed)
        
        # 计算所需加速度
        desired_accel = (ideal_speed - current_speed) / 1.0  # 1秒内达到目标速度
        desired_accel = np.clip(desired_accel, self.min_accel, self.max_accel)
        
        return {
            'target_speed': ideal_speed,
            'acceleration': desired_accel
        }
        
    def check_safety(self, vehicle: Dict, plan: Dict, nearby_vehicles: List[Dict]) -> Dict:
        """
        检查速度规划的安全性
        Args:
            vehicle: 当前车辆信息
            plan: 速度规划结果
            nearby_vehicles: 附近的车辆
        Returns:
            Dict: 修正后的规划结果
        """
        safe_plan = plan.copy()
        
        for other in nearby_vehicles:
            # 计算两车距离
            dist = np.sqrt((vehicle['x'] - other['x'])**2 + (vehicle['y'] - other['y'])**2)
            
            if dist < 10:  # 安全距离阈值
                # 减速以保持安全距离
                safe_plan['target_speed'] = min(safe_plan['target_speed'], 
                                              other['speed'] * 0.8)  # 保持80%的前车速度
                safe_plan['acceleration'] = min(safe_plan['acceleration'], 0)  # 不允许加速
                
        return safe_plan
