"""
ARneuro 异常定义
"""


class ARneuroError(Exception):
    """ARneuro基础异常"""
    pass


class ConfigError(ARneuroError):
    """配置错误"""
    pass


class PDFDownloadError(ARneuroError):
    """PDF下载错误"""
    
    def __init__(self, pmid: str, message: str, finder_used: str = None):
        self.pmid = pmid
        self.finder_used = finder_used
        super().__init__(f"PDF下载失败 PMID={pmid}: {message}")


class OCRProcessingError(ARneuroError):
    """OCR处理错误"""
    
    def __init__(self, pdf_path: str, message: str):
        self.pdf_path = pdf_path
        super().__init__(f"OCR处理失败 {pdf_path}: {message}")


class SegmentationError(ARneuroError):
    """文档分割错误"""
    
    def __init__(self, markdown_path: str, message: str, missing_sections: list = None):
        self.markdown_path = markdown_path
        self.missing_sections = missing_sections or []
        super().__init__(f"文档分割失败 {markdown_path}: {message}")


class ValidationError(ARneuroError):
    """数据验证错误"""
    pass


class FileNotFoundError(ARneuroError):
    """文件未找到错误"""
    pass


class NetworkError(ARneuroError):
    """网络错误"""
    pass


class ModelError(ARneuroError):
    """模型错误"""
    pass


class ProcessingTimeoutError(ARneuroError):
    """处理超时错误"""
    pass


class BatchProcessingError(ARneuroError):
    """批量处理错误"""
    
    def __init__(self, succeeded: int, failed: int, errors: list):
        self.succeeded = succeeded
        self.failed = failed
        self.errors = errors
        super().__init__(f"批量处理完成: {succeeded}成功, {failed}失败")


def handle_exception(func):
    """异常处理装饰器"""
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ARneuroError:
            raise
        except Exception as e:
            # 将通用异常转换为ARneuroError
            raise ARneuroError(f"未处理的异常: {str(e)}") from e
    
    return wrapper