"""
ARneuro 日志系统
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config.config_manager import get_config

_LOGGER_INITIALIZED = False

_LOGGER_INITIALIZED = False


def setup_logger(
    name: str = "ARneuro",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
    config_path: Optional[str] = None,
    force_reconfigure: bool = False,
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
        format_str: 日志格式
        config_path: 配置文件路径
        force_reconfigure: 是否强制重建handler
    
    Returns:
        logging.Logger实例
    """
    global _LOGGER_INITIALIZED

    # 获取配置
    config = get_config(config_path)
    
    # 使用配置或参数
    log_level = level or config.get("logging.level", "INFO")
    log_file_path = log_file or config.get("logging.file")
    if not log_file_path:
        logs_dir = Path(config.get("paths.logs_dir", "./logs"))
        log_file_path = str(logs_dir / "arneuro.log")
    log_format = format_str or config.get("logging.format", 
                                         "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    
    # 设置日志级别
    logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))

    if logger.handlers and not force_reconfigure:
        _LOGGER_INITIALIZED = True
        return logger

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
    _LOGGER_INITIALIZED = True
    
    return logger


def get_logger(config_path: Optional[str] = None) -> logging.Logger:
    """
    获取根日志记录器；未初始化时会按配置自动初始化。
    """
    if not _LOGGER_INITIALIZED:
        return setup_logger(config_path=config_path)
    return logging.getLogger("ARneuro")


def get_module_logger(module_name: str) -> logging.Logger:
    """
    获取模块特定的日志记录器
    
    Args:
        module_name: 模块名称
    
    Returns:
        logging.Logger实例
    """
    base_logger = get_logger()
    return base_logger.getChild(module_name)
