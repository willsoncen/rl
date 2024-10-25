import os
import yaml
import json
from datetime import datetime
from typing import Dict, Any
import shutil

class ExperimentManager:
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        self.current_exp_dir = None
        self.config = None
        
    def create_experiment(self, name: str, config: Dict[str, Any]):
        """创建新实验"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{name}_{timestamp}"
        self.current_exp_dir = os.path.join(self.base_dir, exp_name)
        
        # 创建实验目录结构
        os.makedirs(self.current_exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.current_exp_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.current_exp_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.current_exp_dir, "results"), exist_ok=True)
        
        # 保存配置
        self.config = config
        self.save_config()
        
        return self.current_exp_dir
        
    def save_config(self):
        """保存配置"""
        if self.current_exp_dir and self.config:
            config_path = os.path.join(self.current_exp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
                
    def save_results(self, results: Dict[str, Any]):
        """保存结果"""
        if self.current_exp_dir:
            results_path = os.path.join(self.current_exp_dir, "results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
                
    def archive_experiment(self):
        """归档实验"""
        if self.current_exp_dir:
            archive_dir = os.path.join(self.base_dir, "archived")
            os.makedirs(archive_dir, exist_ok=True)
            
            exp_name = os.path.basename(self.current_exp_dir)
            archive_path = os.path.join(archive_dir, exp_name)
            
            shutil.move(self.current_exp_dir, archive_path)
            self.current_exp_dir = None
