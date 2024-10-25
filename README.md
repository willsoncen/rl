# 分层强化学习交通控制系统

基于分层强化学习的智能交通控制系统，使用三层控制架构实现车辆的安全高效通行。

## 系统架构

### 高层控制器 (DQN)
- 负责50米控制区域外的车辆管理
- 进行碰撞对匹配和时空资源分配
- 使用Dueling DQN网络优化决策

### 中层控制器 (DDPG)
- 负责50米控制区域内的车辆速度规划
- 基于高层分配的时空资源进行精确控制
- 使用DDPG算法实现连续动作控制

### 底层控制器 (TD3)
- 负责具体的车辆控制指令执行
- 确保车辆平稳安全地通过路口
- 使用TD3算法提高控制精度和稳定性

## 安装说明

1. 环境要求
