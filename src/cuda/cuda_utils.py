import torch

def get_device(cuda_id: int = 0) -> torch.device:
    """
    获取计算设备
    Args:
        cuda_id: CUDA设备ID
    Returns:
        torch.device: 计算设备
    """
    if cuda_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_id}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        return device
    return torch.device("cpu")

def to_cuda(data: dict, device: torch.device) -> dict:
    """
    将数据转移到CUDA设备
    Args:
        data: 包含张量的字典
        device: 目标设备
    Returns:
        dict: 转移后的数据
    """
    cuda_data = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            cuda_data[key] = value.to(device)
        elif isinstance(value, dict):
            cuda_data[key] = to_cuda(value, device)
        else:
            cuda_data[key] = value
    return cuda_data
