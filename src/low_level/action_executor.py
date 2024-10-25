import numpy as np
from typing import Dict, List

class ActionExecutor:
    def __init__(self, config: Dict):
        """
        动作执行器
        Args:
            config: 配置字典
        """
        self.max_speed = 14.0  # 最大速度 (m/s)
        self.min_speed = 2.0   # 最小速度 (m/s)
        self.max_accel = 2.0   # 最大加速度 (m/s²)
        self.min_accel = -4.0  # 最大减速度 (m/s²)
        self.max_steer = 0.5   # 最大转向角 (rad)
        
    def execute_action(self, action: np.ndarray, vehicle_info: Dict) -> Dict:
        """
        执行动作
        Args:
            action: 网络输出的动作
            vehicle_info: 车辆信息
        Returns:
            Dict: 执行结果
        """
        # 将动作值从[-1,1]映射到实际范围
        target_speed = (action[0] + 1) / 2 * (self.max_speed - self.min_speed) + self.min_speed
        acceleration = action[1] * self.max_accel
        steering = action[2] * self.max_steer if len(action) > 2 else 0.0
        
        # 安全检查
        current_speed = vehicle_info['speed']
        safe_accel = self._check_safety(current_speed, target_speed, acceleration)
        
        return {
            'target_speed': target_speed,
            'acceleration': safe_accel,
            'steering': steering
        }
        
    def _check_safety(self, current_speed: float, target_speed: float, 
                     desired_accel: float) -> float:
        """安全检查"""
        # 限制加速度
        safe_accel = np.clip(desired_accel, self.min_accel, self.max_accel)
        
        # 确保速度在安全范围内
        next_speed = current_speed + safe_accel * 0.1  # 假设控制间隔为0.1秒
        if next_speed > self.max_speed:
            safe_accel = (self.max_speed - current_speed) / 0.1
        elif next_speed < self.min_speed:
            safe_accel = (self.min_speed - current_speed) / 0.1
            
        return safe_accel
