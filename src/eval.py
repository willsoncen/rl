import os
import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.logger import Logger
from utils.config_loader import ConfigLoader
from sumo.sumo_env import SumoEnvironment
from high_level.high_controller import HighLevelController
from mid_level.mid_controller import MidLevelController
from low_level.low_controller import LowLevelController

class Evaluator:
    def __init__(self, device: torch.device):
        """
        评估器
        Args:
            device: 计算设备
        """
        self.config = ConfigLoader()
        self.logger = Logger("evaluator")
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
        
        # 创建结果保存目录
        os.makedirs("results", exist_ok=True)
        
        # 评估指标
        self.metrics = {
            'rewards': [],
            'collisions': [],
            'waiting_times': [],
            'throughput': []
        }
        
    def load_models(self, model_path: str):
        """加载模型"""
        self.high_controller.load(f"{model_path}/high_level_final.pth")
        self.mid_controller.load(f"{model_path}/mid_level_final.pth")
        self.low_controller.load(f"{model_path}/low_level_final.pth")
        
    def evaluate(self, num_episodes: int = 10):
        """评估"""
        self.logger.info("开始评估...")
        
        for episode in tqdm(range(num_episodes)):
            episode_metrics = self._evaluate_episode()
            
            # 更新指标
            self.metrics['rewards'].append(episode_metrics['reward'])
            self.metrics['collisions'].append(episode_metrics['collisions'])
            self.metrics['waiting_times'].append(episode_metrics['waiting_time'])
            self.metrics['throughput'].append(episode_metrics['throughput'])
            
            # 记录结果
            self.logger.info(
                f"Episode {episode + 1} - "
                f"Reward: {episode_metrics['reward']:.2f}, "
                f"Collisions: {episode_metrics['collisions']}, "
                f"Waiting Time: {episode_metrics['waiting_time']:.2f}, "
                f"Throughput: {episode_metrics['throughput']}"
            )
            
        # 保存评估结果
        self._save_results()
        self.env.close()
        
    def _evaluate_episode(self) -> Dict:
        """评估一个episode"""
        state = self.env.reset()
        episode_reward = 0
        collisions = 0
        waiting_times = []
        vehicles_passed = 0
        done = False
        
        while not done:
            # 1. 高层控制器决策
            high_action = self.high_controller.select_action(
                state['high_state'],
                state['vehicles_outside'],
                evaluate=True
            )
            
            # 2. 中层控制器决策
            mid_actions = []
            for vehicle in state['vehicles_inside']:
                vehicle_allocation = high_action['allocations'].get(vehicle['id'], {})
                collision_pairs = [pair for pair in high_action['collision_pairs'] 
                                if vehicle['id'] in [p['id'] for p in pair]]
                
                mid_state = np.concatenate([
                    state['mid_state'],
                    [high_action['action_idx']]
                ])
                
                mid_action = self.mid_controller.select_action(
                    mid_state,
                    vehicle,
                    collision_pairs,
                    vehicle_allocation,
                    state['nearby_vehicles'],
                    evaluate=True
                )
                mid_actions.append(mid_action)
            
            # 3. 底层控制器决策
            low_actions = []
            for vehicle, mid_action in zip(state['vehicles_inside'], mid_actions):
                low_state = np.concatenate([
                    state['low_state'],
                    mid_action
                ])
                
                low_action = self.low_controller.select_action(
                    low_state,
                    vehicle,
                    evaluate=True
                )
                low_actions.append(low_action)
            
            # 4. 执行动作
            next_state, reward, done, info = self.env.step(low_actions)
            
            # 5. 更新指标
            episode_reward += sum(reward.values())
            collisions += info['collisions']
            waiting_times.append(info['waiting_time'])
            vehicles_passed += info['vehicles_passed']
            
            state = next_state
            
        return {
            'reward': episode_reward,
            'collisions': collisions,
            'waiting_time': np.mean(waiting_times),
            'throughput': vehicles_passed
        }
        
    def _save_results(self):
        """保存评估结果"""
        # 1. 保存数值结果
        results = {
            'mean_reward': np.mean(self.metrics['rewards']),
            'std_reward': np.std(self.metrics['rewards']),
            'total_collisions': sum(self.metrics['collisions']),
            'mean_waiting_time': np.mean(self.metrics['waiting_times']),
            'total_throughput': sum(self.metrics['throughput'])
        }
        
        with open('results/evaluation_results.txt', 'w') as f:
            for metric, value in results.items():
                f.write(f"{metric}: {value}\n")
                
        # 2. 绘制图表
        self._plot_metrics()
        
    def _plot_metrics(self):
        """绘制评估指标图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 奖励曲线
        axes[0, 0].plot(self.metrics['rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # 碰撞次数
        axes[0, 1].plot(self.metrics['collisions'])
        axes[0, 1].set_title('Collisions per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Collisions')
        
        # 等待时间
        axes[1, 0].plot(self.metrics['waiting_times'])
        axes[1, 0].set_title('Average Waiting Time')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Time (s)')
        
        # 通过车辆数
        axes[1, 1].plot(self.metrics['throughput'])
        axes[1, 1].set_title('Vehicle Throughput')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Vehicles')
        
        plt.tight_layout()
        plt.savefig('results/evaluation_metrics.png')
        plt.close()

if __name__ == "__main__":
    evaluator = Evaluator(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    evaluator.evaluate()
