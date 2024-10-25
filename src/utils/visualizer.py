import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
from datetime import datetime

class TrainingVisualizer:
    def __init__(self):
        plt.style.use('seaborn')
        self.fig = None
        self.axes = None
        
    def plot_realtime(self, data: Dict[str, List[float]], episode: int):
        """实时绘制训练曲线"""
        if self.fig is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            plt.ion()  # 开启交互模式
            
        # 清除旧图
        for ax in self.axes.flat:
            ax.clear()
            
        # 绘制奖励
        self.axes[0,0].plot(data['rewards'], label='Total Reward')
        self.axes[0,0].set_title('Training Rewards')
        self.axes[0,0].set_xlabel('Episode')
        self.axes[0,0].set_ylabel('Reward')
        self.axes[0,0].legend()
        
        # 绘制损失
        self.axes[0,1].plot(data['high_loss'], label='High Level')
        self.axes[0,1].plot(data['mid_loss'], label='Mid Level')
        self.axes[0,1].plot(data['low_loss'], label='Low Level')
        self.axes[0,1].set_title('Training Loss')
        self.axes[0,1].set_xlabel('Episode')
        self.axes[0,1].set_ylabel('Loss')
        self.axes[0,1].legend()
        
        # 绘制碰撞次数
        self.axes[1,0].plot(data['collisions'])
        self.axes[1,0].set_title('Collisions')
        self.axes[1,0].set_xlabel('Episode')
        self.axes[1,0].set_ylabel('Count')
        
        # 绘制通过率
        throughput = np.array(data['throughput'])
        self.axes[1,1].plot(throughput / np.max(throughput))
        self.axes[1,1].set_title('Normalized Throughput')
        self.axes[1,1].set_xlabel('Episode')
        self.axes[1,1].set_ylabel('Rate')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
    def plot_trajectory(self, trajectories: List[Dict]):
        """绘制车辆轨迹"""
        fig = go.Figure()
        
        for traj in trajectories:
            fig.add_trace(go.Scatter(
                x=traj['x'],
                y=traj['y'],
                mode='lines+markers',
                name=f"Vehicle {traj['id']}"
            ))
            
        fig.update_layout(
            title='Vehicle Trajectories',
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            showlegend=True
        )
        
        fig.write_html(f'results/trajectories_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
