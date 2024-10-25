import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import List, Dict
import plotly.graph_objects as go
from datetime import datetime

class TrafficVisualizer:
    def __init__(self, control_radius: float = 50.0):
        """
        交通流可视化工具
        Args:
            control_radius: 控制区域半径
        """
        self.control_radius = control_radius
        plt.style.use('seaborn')
        
    def plot_vehicle_positions(self, vehicles: List[Dict], save_path: str = None):
        """绘制车辆位置分布"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制控制区域
        circle = plt.Circle((0, 0), self.control_radius, fill=False, 
                          linestyle='--', color='r', label='Control Area')
        ax.add_artist(circle)
        
        # 绘制车辆位置
        inside_x, inside_y = [], []
        outside_x, outside_y = [], []
        
        for vehicle in vehicles:
            x, y = vehicle['x'], vehicle['y']
            dist = np.sqrt(x*x + y*y)
            
            if dist <= self.control_radius:
                inside_x.append(x)
                inside_y.append(y)
            else:
                outside_x.append(x)
                outside_y.append(y)
                
        ax.scatter(inside_x, inside_y, c='g', label='Inside Control Area')
        ax.scatter(outside_x, outside_y, c='b', label='Outside Control Area')
        
        # 设置图表
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Vehicle Positions')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_traffic_flow(self, data: Dict[str, List[float]], save_path: str = None):
        """绘制交通流指标"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 车辆密度
        sns.lineplot(data=data['density'], ax=axes[0,0])
        axes[0,0].set_title('Vehicle Density')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Vehicles/km')
        
        # 平均速度
        sns.lineplot(data=data['speed'], ax=axes[0,1])
        axes[0,1].set_title('Average Speed')
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('Speed (m/s)')
        
        # 等待时间
        sns.lineplot(data=data['waiting_time'], ax=axes[1,0])
        axes[1,0].set_title('Average Waiting Time')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Time (s)')
        
        # 通过率
        sns.lineplot(data=data['throughput'], ax=axes[1,1])
        axes[1,1].set_title('Throughput Rate')
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Vehicles/min')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def create_animation(self, trajectory_data: List[Dict], save_path: str):
        """创建车辆轨迹动画"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def update(frame):
            ax.clear()
            
            # 绘制控制区域
            circle = plt.Circle((0, 0), self.control_radius, fill=False, 
                              linestyle='--', color='r')
            ax.add_artist(circle)
            
            # 绘制车辆
            for vehicle in trajectory_data[frame]:
                x, y = vehicle['x'], vehicle['y']
                color = 'g' if np.sqrt(x*x + y*y) <= self.control_radius else 'b'
                ax.scatter(x, y, c=color)
                
                # 绘制速度向量
                if 'speed' in vehicle and 'direction' in vehicle:
                    dx = vehicle['speed'] * np.cos(vehicle['direction'])
                    dy = vehicle['speed'] * np.sin(vehicle['direction'])
                    ax.arrow(x, y, dx, dy, head_width=1, head_length=1)
                    
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title(f'Time: {frame * 0.1:.1f}s')
            ax.grid(True)
            
        anim = FuncAnimation(fig, update, frames=len(trajectory_data), 
                           interval=100, blit=False)
        anim.save(save_path, writer='pillow')
        plt.close()
        
    def plot_collision_pairs(self, collision_pairs: List[Tuple[Dict, Dict]], 
                           time_allocations: Dict, save_path: str = None):
        """可视化碰撞对和时空分配"""
        fig = go.Figure()
        
        # 绘制控制区域
        theta = np.linspace(0, 2*np.pi, 100)
        x = self.control_radius * np.cos(theta)
        y = self.control_radius * np.sin(theta)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Control Area',
                               line=dict(dash='dash', color='red')))
        
        # 绘制碰撞对
        for v1, v2 in collision_pairs:
            # 第一辆车
            fig.add_trace(go.Scatter(
                x=[v1['x']], y=[v1['y']],
                mode='markers+text',
                name=f'Vehicle {v1["id"]}',
                text=[f'V{v1["id"]}'],
                marker=dict(size=10)
            ))
            
            # 第二辆车
            fig.add_trace(go.Scatter(
                x=[v2['x']], y=[v2['y']],
                mode='markers+text',
                name=f'Vehicle {v2["id"]}',
                text=[f'V{v2["id"]}'],
                marker=dict(size=10)
            ))
            
            # 连接线
            fig.add_trace(go.Scatter(
                x=[v1['x'], v2['x']], y=[v1['y'], v2['y']],
                mode='lines',
                line=dict(dash='dot'),
                showlegend=False
            ))
            
        # 添加时空分配信息
        for vehicle_id, allocation in time_allocations.items():
            fig.add_annotation(
                x=allocation['position']['x'],
                y=allocation['position']['y'],
                text=f't={allocation["time"]:.1f}s',
                showarrow=True,
                arrowhead=1
            )
            
        fig.update_layout(
            title='Collision Pairs and Time-Space Allocation',
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            
    def plot_control_performance(self, performance_data: Dict, save_path: str = None):
        """绘制控制性能指标"""
        fig = make_subplots(rows=3, cols=1, subplot_titles=(
            'High Level Control', 'Mid Level Control', 'Low Level Control'
        ))
        
        # 高层控制性能
        fig.add_trace(
            go.Scatter(y=performance_data['high_level']['collision_prediction_accuracy'],
                      name='Collision Prediction Accuracy'),
            row=1, col=1
        )
        
        # 中层控制性能
        fig.add_trace(
            go.Scatter(y=performance_data['mid_level']['speed_tracking_error'],
                      name='Speed Tracking Error'),
            row=2, col=1
        )
        
        # 底层控制性能
        fig.add_trace(
            go.Scatter(y=performance_data['low_level']['control_smoothness'],
                      name='Control Smoothness'),
            row=3, col=1
        )
        
        fig.update_layout(height=900, showlegend=True)
        
        if save_path:
            fig.write_html(save_path)
