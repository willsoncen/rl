import os
import sys
import traci
import numpy as np
from typing import Dict, List, Tuple, Any
from .sumo_utils import parse_sumo_config, get_network_info

class SumoEnvironment:
    def __init__(self, config: Dict[str, Any]):
        """
        SUMO仿真环境
        Args:
            config: 环境配置
        """
        self.config = config
        self.sumo_cfg = config['environment']['sumo_cfg']
        self.gui = config['environment']['gui']
        self.control_radius = config['training']['control_radius']
        
        # 检查SUMO环境
        if 'SUMO_HOME' not in os.environ:
            raise ValueError("请设置SUMO_HOME环境变量")
            
        # 添加SUMO工具到Python路径
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        
        # 初始化SUMO命令
        sumo_binary = 'sumo-gui' if self.gui else 'sumo'
        self.sumo_cmd = [sumo_binary, '-c', self.sumo_cfg]
        
        # 获取路网信息
        self.net_info = get_network_info(os.path.join(
            os.path.dirname(self.sumo_cfg), 
            parse_sumo_config(self.sumo_cfg)['net-file']
        ))
        
        # 状态和动作空间维度
        self.state_dims = config['environment']['state_dim']
        self.action_dims = config['environment']['action_dim']
        
    def reset(self) -> Dict:
        """重置环境"""
        if 'SUMO_HOME' in os.environ:
            try:
                traci.close()
            except:
                pass
            traci.start(self.sumo_cmd)
            
        return self._get_state()
        
    def step(self, actions: List[Dict]) -> Tuple[Dict, Dict, bool, Dict]:
        """
        执行一步仿真
        Args:
            actions: 动作列表
        Returns:
            state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 1. 执行动作
        self._apply_actions(actions)
        
        # 2. 仿真一步
        traci.simulationStep()
        
        # 3. 获取新状态
        next_state = self._get_state()
        
        # 4. 计算奖励
        reward = self._get_reward()
        
        # 5. 检查是否结束
        done = self._is_done()
        
        # 6. 收集信息
        info = self._get_info()
        
        return next_state, reward, done, info
        
    def _get_state(self) -> Dict:
        """获取环境状态"""
        # 获取所有车辆
        vehicles = traci.vehicle.getIDList()
        
        # 分离控制区域内外的车辆
        vehicles_inside = []
        vehicles_outside = []
        
        for veh_id in vehicles:
            x, y = traci.vehicle.getPosition(veh_id)
            dist = np.sqrt(x*x + y*y)  # 到路口中心的距离
            
            vehicle_info = {
                'id': veh_id,
                'x': x,
                'y': y,
                'speed': traci.vehicle.getSpeed(veh_id),
                'acceleration': traci.vehicle.getAcceleration(veh_id),
                'distance': dist,
                'route': traci.vehicle.getRoute(veh_id),
                'direction': self._get_vehicle_direction(veh_id)
            }
            
            if dist <= self.control_radius:
                vehicles_inside.append(vehicle_info)
            else:
                vehicles_outside.append(vehicle_info)
                
        # 构建状态向量
        high_state = self._build_high_level_state(vehicles_outside)
        mid_state = self._build_mid_level_state(vehicles_inside)
        low_state = self._build_low_level_state(vehicles_inside)
        
        return {
            'high_state': high_state,
            'mid_state': mid_state,
            'low_state': low_state,
            'vehicles_inside': vehicles_inside,
            'vehicles_outside': vehicles_outside,
            'nearby_vehicles': vehicles  # 所有车辆信息
        }
        
    def _build_high_level_state(self, vehicles: List[Dict]) -> np.ndarray:
        """构建高层状态向量"""
        state = np.zeros(self.state_dims['high'])
        if not vehicles:
            return state
            
        # 编码车辆信息
        idx = 0
        for vehicle in vehicles[:min(len(vehicles), self.state_dims['high']//8)]:
            state[idx:idx+8] = [
                vehicle['x'] / 500.0,  # 归一化位置
                vehicle['y'] / 500.0,
                vehicle['speed'] / 20.0,  # 归一化速度
                vehicle['acceleration'] / 4.0,  # 归一化加速度
                vehicle['distance'] / 500.0,  # 归一化距离
                np.sin(vehicle['direction']),  # 方向编码
                np.cos(vehicle['direction']),
                1.0  # 车辆存在标志
            ]
            idx += 8
            
        return state
        
    def _build_mid_level_state(self, vehicles: List[Dict]) -> np.ndarray:
        """构建中层状态向量"""
        state = np.zeros(self.state_dims['mid'])
        if not vehicles:
            return state
            
        # 编码车辆信息
        idx = 0
        for vehicle in vehicles[:min(len(vehicles), self.state_dims['mid']//8)]:
            state[idx:idx+8] = [
                vehicle['x'] / 50.0,  # 归一化位置(控制区域内)
                vehicle['y'] / 50.0,
                vehicle['speed'] / 20.0,
                vehicle['acceleration'] / 4.0,
                vehicle['distance'] / 50.0,
                np.sin(vehicle['direction']),
                np.cos(vehicle['direction']),
                1.0
            ]
            idx += 8
            
        return state
        
    def _build_low_level_state(self, vehicles: List[Dict]) -> np.ndarray:
        """构建底层状态向量"""
        state = np.zeros(self.state_dims['low'])
        if not vehicles:
            return state
            
        # 编码车辆信息(更详细的局部信息)
        idx = 0
        for vehicle in vehicles[:min(len(vehicles), self.state_dims['low']//8)]:
            state[idx:idx+8] = [
                vehicle['x'] / 50.0,
                vehicle['y'] / 50.0,
                vehicle['speed'] / 20.0,
                vehicle['acceleration'] / 4.0,
                vehicle['distance'] / 50.0,
                np.sin(vehicle['direction']),
                np.cos(vehicle['direction']),
                1.0
            ]
            idx += 8
            
        return state
        
    def _get_vehicle_direction(self, veh_id: str) -> float:
        """获取车辆行驶方向"""
        return traci.vehicle.getAngle(veh_id) * np.pi / 180.0
        
    def _apply_actions(self, actions: List[Dict]):
        """应用动作到环境"""
        for action in actions:
            veh_id = action['id']
            if veh_id in traci.vehicle.getIDList():
                traci.vehicle.setSpeed(veh_id, action['target_speed'])
                traci.vehicle.setAcceleration(veh_id, action['acceleration'])
                if 'steering' in action:
                    traci.vehicle.setAngle(veh_id, action['steering'])
                    
    def _get_reward(self) -> Dict:
        """计算奖励"""
        vehicles = traci.vehicle.getIDList()
        high_reward = 0
        mid_reward = 0
        low_reward = 0
        
        for veh_id in vehicles:
            x, y = traci.vehicle.getPosition(veh_id)
            dist = np.sqrt(x*x + y*y)
            
            vehicle_info = {
                'speed': traci.vehicle.getSpeed(veh_id),
                'acceleration': traci.vehicle.getAcceleration(veh_id),
                'distance': dist,
                'x': x,
                'y': y
            }
            
            if dist > self.control_radius:
                # 高层奖励
                if vehicle_info['speed'] > 0:
                    high_reward += 100  # 基础移动奖励
                    if 8 <= vehicle_info['speed'] <= 14:
                        high_reward += 50  # 速度奖励
                    high_reward += 200 * (1 - dist/300)  # 接近奖励
                else:
                    high_reward -= 200  # 停止惩罚
            else:
                # 中层奖励
                if vehicle_info['speed'] > 0:
                    mid_reward += 100  # 基础移动奖励
                    if dist < 10:
                        mid_reward += 1000  # 通过奖励
                    else:
                        mid_reward += 300 * (1 - dist/self.control_radius)  # 进度奖励
                else:
                    mid_reward -= 200  # 停止惩罚
                    
                # 底层奖励
                if vehicle_info['speed'] > 0:
                    low_reward += 100  # 基础移动奖励
                    if abs(vehicle_info['acceleration']) < 2.0:
                        low_reward += 50  # 平滑控制奖励
                    if 5 <= vehicle_info['speed'] <= 14:
                        low_reward += 50  # 速度维持奖励
                else:
                    low_reward -= 200  # 停止惩罚
                    
        # 碰撞检查(中层)
        collision_penalty = self._check_collisions()
        mid_reward += collision_penalty
        
        # 归一化奖励
        n_vehicles = max(len(vehicles), 1)
        return {
            'high': high_reward / n_vehicles,
            'mid': mid_reward / n_vehicles,
            'low': low_reward / n_vehicles
        }
        
    def _check_collisions(self) -> float:
        """检查碰撞"""
        vehicles = traci.vehicle.getIDList()
        collision_penalty = 0
        
        for i, v1 in enumerate(vehicles[:-1]):
            x1, y1 = traci.vehicle.getPosition(v1)
            for v2 in vehicles[i+1:]:
                x2, y2 = traci.vehicle.getPosition(v2)
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < 5:  # 碰撞阈值
                    collision_penalty -= 500
                    
        return collision_penalty
        
    def _is_done(self) -> bool:
        """检查是否结束"""
        return (traci.simulation.getTime() >= self.config['training']['max_steps'] or
                len(traci.vehicle.getIDList()) == 0)
                
    def _get_info(self) -> Dict:
        """获取额外信息"""
        return {
            'time': traci.simulation.getTime(),
            'vehicles_passed': traci.simulation.getArrivedNumber(),
            'collisions': len(traci.simulation.getCollidingVehiclesIDList()),
            'waiting_time': sum(traci.vehicle.getWaitingTime(v) 
                              for v in traci.vehicle.getIDList())
        }
        
    def close(self):
        """关闭环境"""
        traci.close()
