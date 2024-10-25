import time
import torch
from typing import Dict, List
from collections import defaultdict

class Profiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timing = None
        
    def start(self, name: str):
        """开始计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.current_timing = (name, time.perf_counter())
        
    def end(self):
        """结束计时"""
        if self.current_timing is None:
            return
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        name, start_time = self.current_timing
        elapsed_time = time.perf_counter() - start_time
        self.timings[name].append(elapsed_time)
        self.current_timing = None
        
    def summary(self) -> Dict[str, float]:
        """获取性能统计摘要"""
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'count': len(times)
            }
        return summary
