import os
import torch
import json
from typing import Dict
from datetime import datetime

class CheckpointManager:
    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.checkpoints = []
        self.best_reward = float('-inf')
        self.current_checkpoint = None
        
    def save_checkpoint(self, state_dict: Dict, metrics: Dict, episode: int):
        """保存检查点"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(self.save_dir, f"checkpoint_{episode}_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型状态
        torch.save(state_dict, os.path.join(checkpoint_dir, "model_state.pth"))
        
        # 保存训练指标
        with open(os.path.join(checkpoint_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
            
        self.checkpoints.append({
            'episode': episode,
            'path': checkpoint_dir,
            'metrics': metrics
        })
        
        # 更新最佳检查点
        if metrics['reward'] > self.best_reward:
            self.best_reward = metrics['reward']
            self.current_checkpoint = checkpoint_dir
            
        # 保存检查点信息
        self._save_checkpoint_info()
        
    def load_checkpoint(self, episode: int = None) -> Dict:
        """加载检查点"""
        if episode is None:
            # 加载最佳检查点
            checkpoint_path = self.current_checkpoint
        else:
            # 加载指定轮次的检查点
            checkpoint_path = next(
                (cp['path'] for cp in self.checkpoints if cp['episode'] == episode),
                None
            )
            
        if checkpoint_path is None:
            raise ValueError(f"找不到检查点: episode {episode}")
            
        state_dict = torch.load(os.path.join(checkpoint_path, "model_state.pth"))
        
        with open(os.path.join(checkpoint_path, "metrics.json"), 'r') as f:
            metrics = json.load(f)
            
        return {
            'state_dict': state_dict,
            'metrics': metrics
        }
        
    def _save_checkpoint_info(self):
        """保存检查点信息"""
        info = {
            'checkpoints': self.checkpoints,
            'best_checkpoint': self.current_checkpoint,
            'best_reward': self.best_reward
        }
        
        with open(os.path.join(self.save_dir, "checkpoint_info.json"), 'w') as f:
            json.dump(info, f, indent=4)
