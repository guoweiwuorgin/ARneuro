"""ARneuro literature-processing toolkit."""

__version__ = "0.1.0"
__author__ = "ARneuro Team"
__email__ = ""

from .config import ConfigManager, get_config
from .core import ARneuroError, setup_logger
from .data_fetch import PDFDownloader, PubMedFetcher


def __getattr__(name):
    """Lazily load optional heavyweight OCR and pipeline dependencies."""

    if name in {"GLMOCRProcessor", "DeepSeekOCRProcessor"}:
        from .ocr_processing import DeepSeekOCRProcessor, GLMOCRProcessor

        return {"GLMOCRProcessor": GLMOCRProcessor, "DeepSeekOCRProcessor": DeepSeekOCRProcessor}[name]
    if name == "ARneuroPipeline":
        from .core.pipeline import ARneuroPipeline

        return ARneuroPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "setup_logger",
    "ARneuroError",
    "ConfigManager",
    "get_config",
    "PDFDownloader",
    "PubMedFetcher",
    "GLMOCRProcessor",
    "DeepSeekOCRProcessor",
    "ARneuroPipeline",
]
