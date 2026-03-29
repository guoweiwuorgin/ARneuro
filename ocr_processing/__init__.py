"""
ARneuro OCR处理模块
"""

from .glm_ocr import GLMOCRProcessor, OCRResult, BatchOCRResult
from .deepseek_ocr import DeepSeekOCRProcessor, DeepSeekOCRResult

__all__ = [
    'GLMOCRProcessor',
    'OCRResult',
    'BatchOCRResult',
    'DeepSeekOCRProcessor',
    'DeepSeekOCRResult'
]
