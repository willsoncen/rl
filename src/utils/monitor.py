import psutil
import torch
import time
from typing import Dict
import logging

class ResourceMonitor:
    def __init__(self):
        self.logger = logging.getLogger('monitor')
        
    def get_system_stats(self) -> Dict:
        """获取系统资源使用情况"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_memory_used': self._get_gpu_memory_used(),
            'disk_usage': psutil.disk_usage('/').percent
        }
        
    def _get_gpu_memory_used(self) -> float:
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        return 0.0
        
    def log_stats(self):
        """记录资源使用情况"""
        stats = self.get_system_stats()
        self.logger.info(
            f"System Stats - CPU: {stats['cpu_percent']}%, "
            f"Memory: {stats['memory_percent']}%, "
            f"GPU Memory: {stats['gpu_memory_used']:.2%}, "
            f"Disk: {stats['disk_usage']}%"
        )
