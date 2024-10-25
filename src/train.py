import os
import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

from utils.logger import Logger
from utils.config_loader import ConfigLoader
from sumo.sumo_env import SumoEnvironment
from high_level.high_controller import HighLevelController
from mid_level.mid_controller import MidLevelController
from low_level.low_controller import LowLevelController

class Trainer:
    def __init__(self, device: torch.device):
        """
        训练器
        Args:
            device: 计算设备
        """
        self.config = ConfigLoader()
        self.logger = Logger("trainer")
        self.device = device
        
        # 初始化环境
        self.env = SumoEnvironment(self.config.config)
        
        # 初始化控制器
        self.high_controller = HighLevelController(
            self.config.config, device
        )
        self.mid_controller = MidLevelController(
            self.config.config, device
        )
        self.low_controller = LowLevelController(
            self.config.config, device
        )
        
        # 创建模型保存目录
        os.makedirs("models", exist_ok=True)
        
    def train(self, num_episodes: int = 1000):
        """
        训练主循环
        Args:
            num_episodes: 训练轮数
        """
        self.logger.info("开始训练...")
        
        for episode in tqdm(range(num_episodes)):
            # 重置环境
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 1. 高层控制器决策
                high_action = self.high_controller.select_action(
                    state['high_state'],
                    state['vehicles_outside']
                )
                
                # 2. 中层控制器决策
                mid_actions = []
                for vehicle in state['vehicles_inside']:
                    # 获取车辆相关的碰撞对和时空分配
                    vehicle_allocation = high_action['allocations'].get(vehicle['id'], {})
                    collision_pairs = [pair for pair in high_action['collision_pairs'] 
                                    if vehicle['id'] in [p['id'] for p in pair]]
                    
                    # 构建中层状态
                    mid_state = np.concatenate([
                        state['mid_state'],
                        [high_action['action_idx']]
                    ])
                    
                    # 中层决策
                    mid_action = self.mid_controller.select_action(
                        mid_state,
                        vehicle,
                        collision_pairs,
                        vehicle_allocation,
                        state['nearby_vehicles']
                    )
                    mid_actions.append(mid_action)
                
                # 3. 底层控制器决策
                low_actions = []
                for vehicle, mid_action in zip(state['vehicles_inside'], mid_actions):
                    # 构建底层状态
                    low_state = np.concatenate([
                        state['low_state'],
                        mid_action
                    ])
                    
                    # 底层决策
                    low_action = self.low_controller.select_action(
                        low_state,
                        vehicle
                    )
                    low_actions.append(low_action)
                
                # 4. 执行动作
                next_state, reward, done, info = self.env.step(low_actions)
                
                # 5. 存储经验
                # 高层经验
                self.high_controller.buffer.add(
                    state['high_state'],
                    high_action['action_idx'],
                    reward['high'],
                    next_state['high_state'],
                    done
                )
                
                # 中层经验
                for i, vehicle in enumerate(state['vehicles_inside']):
                    mid_state = np.concatenate([
                        state['mid_state'],
                        [high_action['action_idx']]
                    ])
                    next_mid_state = np.concatenate([
                        next_state['mid_state'],
                        [high_action['action_idx']]
                    ])
                    self.mid_controller.buffer.add(
                        mid_state,
                        mid_actions[i],
                        reward['mid'],
                        next_mid_state,
                        done
                    )
                
                # 底层经验
                for i, vehicle in enumerate(state['vehicles_inside']):
                    low_state = np.concatenate([
                        state['low_state'],
                        mid_actions[i]
                    ])
                    next_low_state = np.concatenate([
                        next_state['low_state'],
                        mid_actions[i]
                    ])
                    self.low_controller.buffer.add(
                        low_state,
                        low_actions[i],
                        reward['low'],
                        next_low_state,
                        done
                    )
                
                # 6. 训练
                high_loss = self.high_controller.train_step()
                mid_loss = self.mid_controller.train_step()
                low_loss = self.low_controller.train_step()
                
                # 更新状态和奖励
                state = next_state
                episode_reward += sum(reward.values())
                
                # 记录信息
                self.logger.info(
                    f"Step Info - "
                    f"High Loss: {high_loss:.4f}, "
                    f"Mid Loss: {mid_loss:.4f}, "
                    f"Low Loss: {low_loss:.4f}, "
                    f"Reward: {sum(reward.values()):.4f}"
                )
            
            # 每轮结束后记录
            self.logger.info(
                f"Episode {episode + 1} - "
                f"Total Reward: {episode_reward:.2f}"
            )
            
            # 定期保存模型
            if (episode + 1) % self.config.config['training']['save_interval'] == 0:
                self.save_models(episode + 1)
                
        # 训练结束，保存最终模型
        self.save_models('final')
        self.env.close()
        
    def save_models(self, epoch: str):
        """保存模型"""
        self.high_controller.save(f"models/high_level_{epoch}.pth")
        self.mid_controller.save(f"models/mid_level_{epoch}.pth")
        self.low_controller.save(f"models/low_level_{epoch}.pth")

if __name__ == "__main__":
    trainer = Trainer(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer.train()
