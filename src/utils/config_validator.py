from typing import Dict, Any
import yaml
import jsonschema

class ConfigValidator:
    def __init__(self):
        self.schema = {
            "type": "object",
            "required": ["training", "environment", "controllers"],
            "properties": {
                "training": {
                    "type": "object",
                    "required": ["epochs", "batch_size", "learning_rate", 
                                "cuda_enabled", "max_steps", "control_radius"],
                    "properties": {
                        "epochs": {"type": "integer", "minimum": 1},
                        "batch_size": {"type": "integer", "minimum": 1},
                        "learning_rate": {"type": "number", "minimum": 0},
                        "cuda_enabled": {"type": "boolean"},
                        "max_steps": {"type": "integer", "minimum": 1},
                        "control_radius": {"type": "number", "minimum": 0}
                    }
                },
                "environment": {
                    "type": "object",
                    "required": ["sumo_cfg", "gui", "state_dim", "action_dim"],
                    "properties": {
                        "sumo_cfg": {"type": "string"},
                        "gui": {"type": "boolean"},
                        "state_dim": {
                            "type": "object",
                            "required": ["high", "mid", "low"],
                            "properties": {
                                "high": {"type": "integer", "minimum": 1},
                                "mid": {"type": "integer", "minimum": 1},
                                "low": {"type": "integer", "minimum": 1}
                            }
                        },
                        "action_dim": {
                            "type": "object",
                            "required": ["high", "mid", "low"],
                            "properties": {
                                "high": {"type": "integer", "minimum": 1},
                                "mid": {"type": "integer", "minimum": 1},
                                "low": {"type": "integer", "minimum": 1}
                            }
                        }
                    }
                }
            }
        }
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置"""
        try:
            jsonschema.validate(instance=config, schema=self.schema)
            return self._validate_additional_rules(config)
        except jsonschema.exceptions.ValidationError as e:
            print(f"配置验证失败: {str(e)}")
            return False
            
    def _validate_additional_rules(self, config: Dict[str, Any]) -> bool:
        """验证额外规则"""
        try:
            # 验证学习率范围
            if not (0 < config['training']['learning_rate'] <= 1):
                print("学习率应在(0,1]范围内")
                return False
                
            # 验证状态空间维度
            state_dims = config['environment']['state_dim']
            if not (state_dims['high'] >= state_dims['mid'] >= state_dims['low']):
                print("状态空间维度应满足 high >= mid >= low")
                return False
                
            # 验证动作空间维度
            action_dims = config['environment']['action_dim']
            if not (action_dims['high'] >= action_dims['mid'] >= action_dims['low']):
                print("动作空间维度应满足 high >= mid >= low")
                return False
                
            return True
            
        except KeyError as e:
            print(f"缺少必要的配置项: {str(e)}")
            return False
