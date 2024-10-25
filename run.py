import argparse
import torch
from src.train import Trainer
from src.eval import Evaluator
from utils.logger import Logger
from utils.config_loader import ConfigLoader

def main():
    parser = argparse.ArgumentParser(description='分层强化学习交通控制系统')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                      help='运行模式：训练或评估')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='训练或评估的轮数')
    parser.add_argument('--gui', action='store_true',
                      help='是否使用SUMO GUI')
    parser.add_argument('--cuda', type=int, default=0,
                      help='使用的CUDA设备ID，-1表示使用CPU')
    parser.add_argument('--load_model', type=str, default=None,
                      help='加载预训练模型的路径')
    
    args = parser.parse_args()
    
    # 初始化日志
    logger = Logger("main")
    logger.info("启动分层强化学习交通控制系统...")
    
    # 设置设备
    if args.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
        logger.info(f"使用CUDA设备 {args.cuda}")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU设备")
    
    try:
        # 加载配置
        config = ConfigLoader()
        if args.gui:
            config.config['environment']['gui'] = True
            
        if args.mode == 'train':
            trainer = Trainer(device=device)
            trainer.train(num_episodes=args.episodes)
        else:
            evaluator = Evaluator(device=device)
            if args.load_model:
                evaluator.load_models(args.load_model)
            evaluator.evaluate(num_episodes=args.episodes)
            
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        raise e

if __name__ == '__main__':
    main()
