"""
ARneuro OCR处理模块
"""

from .glm_ocr import GLMOCRProcessor, OCRResult, BatchOCRResult

__all__ = [
    'GLMOCRProcessor',
    'OCRResult',
    'BatchOCRResult'
]