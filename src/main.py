from utils.config_loader import ConfigLoader
from utils.logger import Logger
from train import Trainer

def main():
    logger = Logger("main")
    logger.info("启动分层强化学习交通控制系统...")
    
    try:
        # 加载配置
        config = ConfigLoader()
        logger.info("配置加载成功")
        
        # 开始训练
        trainer = Trainer()
        trainer.train()
        
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        raise e

if __name__ == "__main__":
    main()

