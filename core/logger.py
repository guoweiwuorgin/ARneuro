"""
ARneuro 日志系统
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from ..config import get_config


def setup_logger(
    name: str = "ARneuro",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
        format_str: 日志格式
    
    Returns:
        logging.Logger实例
    """
    # 获取配置
    config = get_config()
    
    # 使用配置或参数
    log_level = level or config.get("logging.level", "INFO")
    log_file_path = log_file or config.get("logging.file", "./logs/arneuro.log")
    log_format = format_str or config.get("logging.format", 
                                         "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    
    # 设置日志级别
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(log_format)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file_path:
        # 确保日志目录存在
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 避免日志传播到根记录器
    logger.propagate = False
    
    return logger


def get_module_logger(module_name: str) -> logging.Logger:
    """
    获取模块特定的日志记录器
    
    Args:
        module_name: 模块名称
    
    Returns:
        logging.Logger实例
    """
    return logging.getLogger(f"ARneuro.{module_name}")