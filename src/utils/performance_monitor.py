import time
import psutil
import torch
import logging
from typing import Dict, List
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        """
        性能监控器
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.metrics = {
            'fps': deque(maxlen=window_size),
            'cpu_usage': deque(maxlen=window_size),
            'memory_usage': deque(maxlen=window_size),
            'gpu_memory': deque(maxlen=window_size),
            'training_time': deque(maxlen=window_size),
            'inference_time': deque(maxlen=window_size)
        }
        self.logger = logging.getLogger('performance')
        
    def update(self, **kwargs):
        """更新性能指标"""
        # 系统资源使用
        self.metrics['cpu_usage'].append(psutil.cpu_percent())
        self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
        
        # GPU使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.metrics['gpu_memory'].append(gpu_memory)
            
        # 更新自定义指标
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                
    def get_summary(self) -> Dict[str, float]:
        """获取性能统计摘要"""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'current': values[-1]
                }
        return summary
        
    def log_summary(self):
        """记录性能摘要"""
        summary = self.get_summary()
        self.logger.info("Performance Summary:")
        for metric, stats in summary.items():
            self.logger.info(f"{metric}: mean={stats['mean']:.2f}, "
                           f"min={stats['min']:.2f}, "
                           f"max={stats['max']:.2f}, "
                           f"current={stats['current']:.2f}")
