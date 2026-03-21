"""
ARneuro: 基于LLM的文献定量综述工具

一个完整的文献处理流水线，支持：
1. PubMed文献元数据获取
2. PDF下载（支持多种期刊格式）
3. GLM-OCR PDF转Markdown
4. 文档结构分割和特征提取
5. 脑激活表格处理
6. 结构化信息提取和汇总
"""

__version__ = "0.1.0"
__author__ = "ARneuro Team"
__email__ = ""

# 导入核心模块
from .core import setup_logger, ARneuroError
from .config import ConfigManager, get_config
from .data_fetch import PDFDownloader, PubMedFetcher
from .ocr_processing import GLMOCRProcessor

# 主工作流程
from .core.pipeline import ARneuroPipeline

__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    '__email__',
    
    # 核心模块
    'setup_logger',
    'ARneuroError',
    'ConfigManager',
    'get_config',
    
    # 数据获取
    'PDFDownloader',
    'PubMedFetcher',
    
    # OCR处理
    'GLMOCRProcessor',
    
    # 主工作流程
    'ARneuroPipeline'
]