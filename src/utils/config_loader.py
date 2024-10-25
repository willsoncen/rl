import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str = "config.yaml"):
        """
        配置加载器
        Args:
            config_path: 配置文件路径
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练相关配置"""
        return self.config.get('training', {})
        
    def get_controller_config(self, level: str) -> Dict[str, Any]:
        """
        获取指定层级控制器的配置
        Args:
            level: 控制器层级 ('high_level', 'mid_level', 'low_level')
        """
        return self.config.get('controllers', {}).get(level, {})
        
    def get_cuda_config(self) -> bool:
        """获取CUDA配置"""
        return self.config.get('training', {}).get('cuda_enabled', False)
