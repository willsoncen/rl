import torch
import numpy as np
from typing import Dict, List
import logging
import traceback

class Debugger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.grad_stats = {}
        self.activation_stats = {}
        
    def check_gradients(self, model: torch.nn.Module, name: str):
        """检查梯度"""
        total_grad = 0
        max_grad = 0
        
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad += grad_norm
                max_grad = max(max_grad, grad_norm)
                
        self.grad_stats[name] = {
            'total_grad': total_grad,
            'max_grad': max_grad
        }
        
        if max_grad > 10:
            self.logger.warning(f"{name} 梯度爆炸: {max_grad}")
            
    def check_activations(self, tensor: torch.Tensor, name: str):
        """检查激活值"""
        with torch.no_grad():
            mean = tensor.mean().item()
            std = tensor.std().item()
            max_val = tensor.max().item()
            min_val = tensor.min().item()
            
            self.activation_stats[name] = {
                'mean': mean,
                'std': std,
                'max': max_val,
                'min': min_val
            }
            
            if std < 1e-7:
                self.logger.warning(f"{name} 激活值消失")
                
    def check_nan(self, tensor: torch.Tensor, name: str) -> bool:
        """检查NaN值"""
        if torch.isnan(tensor).any():
            self.logger.error(f"{name} 包含NaN值")
            return True
        return False
        
    def log_stats(self):
        """记录统计信息"""
        self.logger.info("梯度统计:")
        for name, stats in self.grad_stats.items():
            self.logger.info(f"{name}: total_grad={stats['total_grad']:.4f}, "
                           f"max_grad={stats['max_grad']:.4f}")
            
        self.logger.info("激活值统计:")
        for name, stats in self.activation_stats.items():
            self.logger.info(f"{name}: mean={stats['mean']:.4f}, "
                           f"std={stats['std']:.4f}, "
                           f"max={stats['max']:.4f}, "
                           f"min={stats['min']:.4f}")
