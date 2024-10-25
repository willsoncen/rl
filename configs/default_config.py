config = {
    'training': {
        'epochs': 1000,
        'batch_size': 64,
        'learning_rate': 0.001,
        'cuda_enabled': True,
        'max_steps': 3600,
        'save_interval': 100,
        'control_radius': 50  # 控制区域半径(米)
    },
    'environment': {
        'sumo_cfg': 'data/sumo/config.sumocfg',
        'gui': False,
        'state_dim': {
            'high': 128,  # 高层状态空间维度
            'mid': 64,   # 中层状态空间维度
            'low': 32    # 底层状态空间维度
        },
        'action_dim': {
            'high': 16,  # 高层动作空间维度(碰撞对匹配和时空分配)
            'mid': 8,    # 中层动作空间维度(速度和加速度控制)
            'low': 4     # 底层动作空间维度(具体执行动作)
        }
    },
    'controllers': {
        'high_level': {
            'algorithm': 'DQN',
            'hidden_size': [256, 256],
            'learning_rate': 0.001,
            'buffer_size': 100000,
            'batch_size': 64,
            'gamma': 0.99,
            'tau': 0.005,
            'epsilon': 0.1,
            'reward_weights': {
                'base_move': 100,
                'speed': 50,
                'approach': 200,
                'stop_penalty': -200
            }
        },
        'mid_level': {
            'algorithm': 'DDPG',
            'hidden_size': [400, 300],
            'actor_lr': 0.001,
            'critic_lr': 0.001,
            'buffer_size': 100000,
            'batch_size': 64,
            'gamma': 0.99,
            'tau': 0.005,
            'reward_weights': {
                'base_move': 100,
                'cross': 1000,
                'progress': 300,
                'collision': -500,
                'stop_penalty': -200
            }
        },
        'low_level': {
            'algorithm': 'TD3',
            'hidden_size': [400, 300],
            'actor_lr': 0.001,
            'critic_lr': 0.001,
            'buffer_size': 100000,
            'batch_size': 64,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_noise': 0.2,
            'noise_clip': 0.5,
            'policy_freq': 2,
            'reward_weights': {
                'base_move': 100,
                'smooth': 50,
                'speed': 50,
                'stop_penalty': -200
            }
        }
    }
}
