import os
import yaml

def create_directory_structure():
    # 创建主目录结构
    directories = [
        'logs',
        'models',
        'data/raw',
        'data/processed',
        'src/utils',
        'src/high_level',
        'src/mid_level',
        'src/low_level',
        'src/cuda',
        'src/sumo'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    # 创建空文件
    files = [
        'requirements.txt',
        'environment.yml',
        'setup.py',
        'src/main.py',
        'src/train.py',
        'src/eval.py',
        'src/replay_buffer.py',
    ]
    
    for file in files:
        with open(file, 'a'):
            pass

    # 创建配置文件
    config = {
        'training': {
            'epochs': 1000,
            'batch_size': 64,
            'learning_rate': 0.001,
            'cuda_enabled': True
        },
        'controllers': {
            'high_level': {
                'algorithm': 'DQN',
                'hidden_size': [256, 256]
            },
            'mid_level': {
                'algorithm': 'DDPG',
                'hidden_size': [400, 300]
            },
            'low_level': {
                'algorithm': 'TD3',
                'hidden_size': [400, 300]
            }
        }
    }
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

if __name__ == '__main__':
    create_directory_structure()
    print("项目结构创建完成！")
