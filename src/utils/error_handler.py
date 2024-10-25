import sys
import traceback
import logging
from typing import Callable, Any
from functools import wraps

class ErrorHandler:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_counts = {}
        self.max_retries = 3
        
    def handle_error(self, error: Exception, context: str = ""):
        """处理异常"""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.logger.error(f"Error in {context}: {str(error)}")
        self.logger.error(traceback.format_exc())
        
        # 记录错误统计
        if self.error_counts[error_type] >= self.max_retries:
            self.logger.critical(f"Critical error: {error_type} occurred "
                               f"{self.error_counts[error_type]} times")
            return False
        return True
        
    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0):
        """重试装饰器"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise e
                        time.sleep(delay)
                return None
            return wrapper
        return decorator
