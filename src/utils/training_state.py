import os
import json
import torch
from typing import Dict, Any
from datetime import datetime

class TrainingState:
    def __init__(self, save_dir: str = "training_states"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.state = {
            'episode': 0,
            'best_reward': float('-inf'),
            'total_steps': 0,
            'high_level_stats': {
                'avg_reward': 0,
                'avg_loss': 0,
                'collision_pairs': 0
            },
            'mid_level_stats': {
                'avg_reward': 0,
                'avg_loss': 0,
                'successful_passes': 0
            },
            'low_level_stats': {
                'avg_reward': 0,
                'avg_loss': 0,
                'smooth_control_rate': 0
            },
            'timestamp': None
        }
        
    def update(self, **kwargs):
        """更新训练状态"""
        for key, value in kwargs.items():
            if key in self.state:
                self.state[key] = value
            elif '.' in key:
                # 处理嵌套更新，如 'high_level_stats.avg_reward'
                main_key, sub_key = key.split('.')
                if main_key in self.state and isinstance(self.state[main_key], dict):
                    self.state[main_key][sub_key] = value
                    
        self.state['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def save(self):
        """保存训练状态"""
        filename = f"training_state_{self.state['episode']}.json"
        path = os.path.join(self.save_dir, filename)
        
        with open(path, 'w') as f:
            json.dump(self.state, f, indent=4)
            
    def load(self, path: str):
        """加载训练状态"""
        with open(path, 'r') as f:
            self.state = json.load(f)
            
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return self.state.copy()
