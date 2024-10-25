import os
import json
import torch
from typing import Dict, Any
from datetime import datetime

class TrainingSaver:
    def __init__(self, save_dir: str = "training_states"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_state(self, state: Dict[str, Any], episode: int):
        """保存训练状态"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_state_{episode}_{timestamp}.pt"
        path = os.path.join(self.save_dir, filename)
        
        # 保存训练状态
        torch.save({
            'episode': episode,
            'models': {
                'high': state['high_controller'].state_dict(),
                'mid': state['mid_controller'].state_dict(),
                'low': state['low_controller'].state_dict()
            },
            'optimizers': {
                'high': state['high_optimizer'].state_dict(),
                'mid': state['mid_optimizer'].state_dict(),
                'low': state['low_optimizer'].state_dict()
            },
            'metrics': state['metrics'],
            'random_state': torch.get_rng_state()
        }, path)
        
    def load_state(self, path: str) -> Dict[str, Any]:
        """加载训练状态"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training state file not found: {path}")
            
        state = torch.load(path)
        return state
