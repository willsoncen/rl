import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class DataCollector:
    def __init__(self):
        self.data = {
            'episode': [],
            'step': [],
            'high_reward': [],
            'mid_reward': [],
            'low_reward': [],
            'total_reward': [],
            'collisions': [],
            'waiting_time': [],
            'throughput': [],
            'high_loss': [],
            'mid_loss': [],
            'low_loss': []
        }
        
    def add_step_data(self, episode: int, step: int, **kwargs):
        """添加每步数据"""
        self.data['episode'].append(episode)
        self.data['step'].append(step)
        
        for key, value in kwargs.items():
            if key in self.data:
                self.data[key].append(value)
                
    def save_to_csv(self, filename: str):
        """保存数据到CSV"""
        df = pd.DataFrame(self.data)
        df.to_csv(f"results/{filename}.csv", index=False)
        
    def plot_training_curves(self):
        """绘制训练曲线"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建图表
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
        
        # 奖励曲线
        df = pd.DataFrame({
            'Episode': self.data['episode'],
            'High Level': self.data['high_reward'],
            'Mid Level': self.data['mid_reward'],
            'Low Level': self.data['low_reward']
        })
        df_melted = df.melt('Episode', var_name='Level', value_name='Reward')
        sns.lineplot(data=df_melted, x='Episode', y='Reward', hue='Level', ax=axes[0,0])
        axes[0,0].set_title('Rewards by Level')
        
        # 损失曲线
        df = pd.DataFrame({
            'Episode': self.data['episode'],
            'High Level': self.data['high_loss'],
            'Mid Level': self.data['mid_loss'],
            'Low Level': self.data['low_loss']
        })
        df_melted = df.melt('Episode', var_name='Level', value_name='Loss')
        sns.lineplot(data=df_melted, x='Episode', y='Loss', hue='Level', ax=axes[0,1])
        axes[0,1].set_title('Training Loss by Level')
        
        # 碰撞统计
        sns.lineplot(data=pd.DataFrame(self.data), x='episode', y='collisions', ax=axes[1,0])
        axes[1,0].set_title('Collisions per Episode')
        
        # 等待时间
        sns.lineplot(data=pd.DataFrame(self.data), x='episode', y='waiting_time', ax=axes[1,1])
        axes[1,1].set_title('Average Waiting Time')
        
        # 吞吐量
        sns.lineplot(data=pd.DataFrame(self.data), x='episode', y='throughput', ax=axes[2,0])
        axes[2,0].set_title('Vehicle Throughput')
        
        # 总奖励
        sns.lineplot(data=pd.DataFrame(self.data), x='episode', y='total_reward', ax=axes[2,1])
        axes[2,1].set_title('Total Reward per Episode')
        
        plt.tight_layout()
        plt.savefig(f'results/training_curves_{timestamp}.png')
        plt.close()
