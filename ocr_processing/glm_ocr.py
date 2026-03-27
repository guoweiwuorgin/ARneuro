"""
GLM-OCR处理模块
支持本地 glmocr SDK 与在线 zai SDK 两种方式进行PDF转Markdown
"""

import base64
import importlib
import inspect
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from ..core import get_module_logger
from ..config.config_manager import get_config
from ..utils.file_utils import ensure_dir, is_valid_pdf
from ..utils.text_utils import clean_text, normalize_text, detect_language


class OCRStatus(Enum):
    """OCR处理状态"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class OCRQuality(Enum):
    """OCR质量等级"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


class GLMOCRBackend(str, Enum):
    """GLM-OCR执行后端"""
    LOCAL = "local"
    API = "api"


@dataclass
class OCRResult:
    """OCR处理结果"""
    pdf_path: Path
    status: OCRStatus
    markdown_path: Optional[Path] = None
    pages_processed: int = 0
    text_length: int = 0
    quality_score: float = 0.0
    quality_level: OCRQuality = OCRQuality.UNKNOWN
    language: str = "unknown"
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchOCRResult:
    """批量OCR处理结果"""
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[OCRResult] = field(default_factory=list)
    total_pages: int = 0
    total_text_length: int = 0


class GLMOCRProcessor:
    """GLM-OCR处理器：本地使用 glmocr SDK，在线使用 zai SDK。"""

    def __init__(self, config_path: Optional[str] = None, backend: Optional[str] = None):
        self.config = get_config(config_path)
        self.logger = get_module_logger("glm_ocr")

        configured_backend = backend or self.config.get("ocr_processing.backend", GLMOCRBackend.LOCAL.value)
        self.backend = GLMOCRBackend(str(configured_backend).lower())
        self.output_dir = Path(self.config.get("ocr_processing.output_dir", "./data/markdown"))
        self.batch_size = self.config.get("ocr_processing.batch_size", 4)
        self.language = self.config.get("ocr_processing.language", "ch+en")

        # 本地 glmocr SDK 配置
        self.model_path = Path(self.config.get("ocr_processing.model_path", "/storage/work/wuguowei/Bigmodel/GLM-OCR"))
        self.device = self.config.get("ocr_processing.device", "cuda")
        self.local_sdk_class = self.config.get("ocr_processing.local_sdk_class", "GLMOCR")
        self.local_sdk_method = self.config.get("ocr_processing.local_sdk_method", "predict")

        # 在线 zai SDK 配置
        self.api_key = self.config.get("ocr_processing.api_key") or self.config.get("glm_api_key")
        self.api_model = self.config.get("ocr_processing.api_model", "glm-ocr")
        self.api_timeout = int(self.config.get("ocr_processing.api_timeout", 300))
        self.api_return_crop_images = bool(self.config.get("ocr_processing.api_return_crop_images", False))
        self.api_need_layout_visualization = bool(self.config.get("ocr_processing.api_need_layout_visualization", False))
        self.api_start_page_id = self.config.get("ocr_processing.api_start_page_id")
        self.api_end_page_id = self.config.get("ocr_processing.api_end_page_id")
        self.api_use_base64 = bool(self.config.get("ocr_processing.api_use_base64", True))
        self.api_file_url = self.config.get("ocr_processing.api_file_url")
        self.api_user_id = self.config.get("ocr_processing.api_user_id")

        self._local_client: Optional[Any] = None
        self._api_client: Optional[Any] = None

        ensure_dir(self.output_dir)
        self._check_glmocr_availability()
        self.logger.info(f"GLM-OCR处理器初始化完成，后端: {self.backend.value}")

    def _get_zai_client_class(self):
        spec = importlib.util.find_spec("zai")
        if spec is None:
            return None
        module = importlib.import_module("zai")
        return getattr(module, "ZhipuAiClient", None)

    def _get_glmocr_module(self):
        spec = importlib.util.find_spec("glmocr")
        if spec is None:
            return None
        return importlib.import_module("glmocr")

    def _init_api_client(self) -> bool:
        if self._api_client is not None:
            return True
        if not self.api_key:
            self.logger.error("GLM-OCR API模式缺少api_key，请配置 ocr_processing.api_key 或 glm_api_key")
            return False

        client_class = self._get_zai_client_class()
        if client_class is None:
            self.logger.error("未检测到 zai SDK，请安装后再使用API模式：pip install zai")
            return False

        self._api_client = client_class(api_key=self.api_key)
        return True

    def _init_local_client(self) -> bool:
        if self._local_client is not None:
            return True

        module = self._get_glmocr_module()
        if module is None:
            self.logger.error("未检测到 glmocr SDK，请先安装 glmocr 后再使用本地模式")
            return False

        sdk_class = getattr(module, self.local_sdk_class, None)
        if sdk_class is None:
            # 兼容：若无类，直接使用模块
            self._local_client = module
            return True

        kwargs = {
            "model_path": str(self.model_path),
            "device": self.device,
            "lang": self.language,
        }
        sig = inspect.signature(sdk_class)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        self._local_client = sdk_class(**filtered_kwargs)
        return True

    def _check_glmocr_availability(self) -> bool:
        if self.backend == GLMOCRBackend.API:
            ok = self._init_api_client()
            if ok:
                self.logger.info("GLM-OCR API模式已启用（zai SDK）")
            return ok

        ok = self._init_local_client()
        if ok:
            self.logger.info("GLM-OCR本地模式已启用（glmocr SDK）")
        return ok

    def _build_api_request_payload(self, pdf_path: Path) -> Dict[str, Any]:
        file_value = self.api_file_url
        if self.api_use_base64 or not file_value:
            file_value = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")

        payload: Dict[str, Any] = {
            "model": self.api_model,
            "file": file_value,
        }
        if self.api_return_crop_images:
            payload["return_crop_images"] = True
        if self.api_need_layout_visualization:
            payload["need_layout_visualization"] = True
        if self.api_start_page_id is not None:
            payload["start_page_id"] = int(self.api_start_page_id)
        if self.api_end_page_id is not None:
            payload["end_page_id"] = int(self.api_end_page_id)
        if self.api_user_id:
            payload["user_id"] = self.api_user_id
        return payload

    def _extract_markdown_text(self, response_data: Any) -> str:
        if response_data is None:
            return ""
        if isinstance(response_data, str):
            return response_data
        if hasattr(response_data, "model_dump"):
            response_data = response_data.model_dump()
        elif hasattr(response_data, "dict"):
            response_data = response_data.dict()

        if isinstance(response_data, dict):
            for key in ["md_results", "markdown", "text", "content"]:
                value = response_data.get(key)
                if isinstance(value, str) and value.strip():
                    return value

        return ""

    def _parse_local_sdk_metadata(self, raw_response: Any, processing_time: float) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "processing_time": processing_time,
            "backend": GLMOCRBackend.LOCAL.value,
            "sdk": "glmocr",
        }

        if isinstance(raw_response, dict):
            metadata.update({
                "pages_detected": raw_response.get("pages_detected", 0),
                "confidence": raw_response.get("confidence", 0),
                "model": raw_response.get("model", "glmocr"),
            })
        return metadata

    def _parse_api_metadata(self, response_data: Any, processing_time: float) -> Dict[str, Any]:
        if hasattr(response_data, "model_dump"):
            response_data = response_data.model_dump()
        elif hasattr(response_data, "dict"):
            response_data = response_data.dict()

        data_info = response_data.get("data_info", {}) if isinstance(response_data, dict) else {}
        usage = response_data.get("usage", {}) if isinstance(response_data, dict) else {}

        return {
            "processing_time": processing_time,
            "backend": GLMOCRBackend.API.value,
            "sdk": "zai",
            "task_id": response_data.get("id") if isinstance(response_data, dict) else None,
            "pages_detected": data_info.get("num_pages", 0) if isinstance(data_info, dict) else 0,
            "usage": usage,
            "model": response_data.get("model", self.api_model) if isinstance(response_data, dict) else self.api_model,
            "raw_response": response_data if isinstance(response_data, dict) else str(response_data),
        }

    def _invoke_local_predict(self, pdf_path: Path) -> Any:
        client = self._local_client
        if client is None:
            raise RuntimeError("本地glmocr SDK未初始化")

        # 统一尝试 client.predict(...) 或 module.predict(...)
        method = getattr(client, self.local_sdk_method, None)
        if method is None:
            method = getattr(client, "predict", None)
        if method is None:
            raise RuntimeError(f"本地glmocr SDK未找到可用方法: {self.local_sdk_method}/predict")

        sig = inspect.signature(method)
        kwargs_candidates = {
            "file": str(pdf_path),
            "pdf_path": str(pdf_path),
            "input": str(pdf_path),
            "path": str(pdf_path),
            "output_format": "markdown",
            "lang": self.language,
            "language": self.language,
        }
        filtered_kwargs = {k: v for k, v in kwargs_candidates.items() if k in sig.parameters}

        if filtered_kwargs:
            return method(**filtered_kwargs)
        return method(str(pdf_path))

    def _run_glmocr_local_sdk(self, pdf_path: Path, output_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
        if not self._init_local_client():
            return False, "", {"error": "本地glmocr SDK初始化失败", "backend": GLMOCRBackend.LOCAL.value}

        try:
            start_time = time.time()
            raw_response = self._invoke_local_predict(pdf_path)
            processing_time = time.time() - start_time

            output_text = self._extract_markdown_text(raw_response)
            if not output_text.strip() and output_path.exists():
                output_text = output_path.read_text(encoding="utf-8")

            if not output_text.strip():
                return False, "", {
                    "error": "本地glmocr SDK返回为空",
                    "backend": GLMOCRBackend.LOCAL.value,
                }

            metadata = self._parse_local_sdk_metadata(raw_response, processing_time)
            return True, output_text, metadata
        except Exception as e:
            error_msg = f"调用本地glmocr SDK异常: {e}"
            self.logger.error(error_msg)
            return False, "", {"error": error_msg, "backend": GLMOCRBackend.LOCAL.value}

    def _run_glmocr_api_request(self, pdf_path: Path, output_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
        if not self._init_api_client():
            return False, "", {"error": "zai客户端初始化失败", "backend": GLMOCRBackend.API.value}

        try:
            payload = self._build_api_request_payload(pdf_path)
            start_time = time.time()
            response = self._api_client.layout_parsing.create(**payload)
            processing_time = time.time() - start_time

            output_text = self._extract_markdown_text(response)
            if not output_text.strip():
                return False, "", {
                    "error": f"GLM-OCR API返回为空: {response}",
                    "backend": GLMOCRBackend.API.value,
                }

            output_path.write_text(output_text, encoding="utf-8")
            metadata = self._parse_api_metadata(response, processing_time)
            return True, output_text, metadata
        except Exception as e:
            error_msg = f"调用zai GLM-OCR API异常: {e}"
            self.logger.error(error_msg)
            return False, "", {"error": error_msg, "backend": GLMOCRBackend.API.value}

    def _run_glmocr_command(self, pdf_path: Path, output_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
        if self.backend == GLMOCRBackend.API:
            return self._run_glmocr_api_request(pdf_path, output_path)
        return self._run_glmocr_local_sdk(pdf_path, output_path)

    def _assess_ocr_quality(self, text: str, metadata: Dict[str, Any]) -> Tuple[float, OCRQuality]:
        if not text:
            return 0.0, OCRQuality.POOR

        text_length = len(text)
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        length_score = min(30, text_length / 1000 * 3)
        line_score = (len(non_empty_lines) / len(lines) * 30) if lines else 0
        confidence = metadata.get("confidence", 0)
        confidence_score = confidence * 0.4
        total_score = length_score + line_score + confidence_score

        if total_score >= 90:
            quality = OCRQuality.EXCELLENT
        elif total_score >= 70:
            quality = OCRQuality.GOOD
        elif total_score >= 50:
            quality = OCRQuality.FAIR
        else:
            quality = OCRQuality.POOR

        return total_score, quality

    def _post_process_markdown(self, markdown_text: str) -> str:
        if not markdown_text:
            return ""

        text = clean_text(markdown_text, remove_empty_lines=True)
        text = normalize_text(text)

        lines = text.split('\n')
        processed_lines = []
        for line in lines:
            line = line.strip()
            if '|' in line and ('---' in line or '===' in line):
                cells = line.split('|')
                processed_cells = []
                for cell in cells:
                    cell = cell.strip()
                    if cell and all(c in '-=:' for c in cell):
                        processed_cells.append(cell)
                    else:
                        processed_cells.append('---')
                line = '|' + '|'.join(processed_cells) + '|'
            processed_lines.append(line)

        return '\n'.join(processed_lines)

    def process_pdf(self, pdf_path: Path, output_filename: Optional[str] = None) -> OCRResult:
        start_time = time.time()
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return OCRResult(pdf_path=pdf_path, status=OCRStatus.ERROR, error_message="PDF文件不存在", processing_time=time.time() - start_time)
        if not is_valid_pdf(pdf_path):
            return OCRResult(pdf_path=pdf_path, status=OCRStatus.ERROR, error_message="无效的PDF文件", processing_time=time.time() - start_time)

        safe_filename = output_filename or pdf_path.stem
        markdown_path = self.output_dir / f"{safe_filename}.md"

        if markdown_path.exists():
            try:
                existing_text = markdown_path.read_text(encoding="utf-8")
                if existing_text.strip():
                    quality_score, quality_level = self._assess_ocr_quality(existing_text, {})
                    language = detect_language(existing_text)
                    self.logger.info(f"PDF已处理，跳过: {pdf_path}")
                    return OCRResult(
                        pdf_path=pdf_path,
                        status=OCRStatus.SKIPPED,
                        markdown_path=markdown_path,
                        text_length=len(existing_text),
                        quality_score=quality_score,
                        quality_level=quality_level,
                        language=language,
                        processing_time=0.0,
                        metadata={"backend": self.backend.value},
                    )
            except Exception as e:
                self.logger.warning(f"读取现有Markdown文件失败: {e}")

        self.logger.info(f"开始使用 {self.backend.value} 模式处理PDF: {pdf_path}")
        success, output_text, metadata = self._run_glmocr_command(pdf_path, markdown_path)
        if not success:
            return OCRResult(
                pdf_path=pdf_path,
                status=OCRStatus.FAILED,
                error_message=metadata.get("error", "GLM-OCR处理失败"),
                processing_time=time.time() - start_time,
                metadata=metadata,
            )

        processed_text = self._post_process_markdown(output_text)
        try:
            markdown_path.write_text(processed_text, encoding="utf-8")
        except Exception as e:
            self.logger.error(f"保存Markdown文件失败: {e}")
            return OCRResult(
                pdf_path=pdf_path,
                status=OCRStatus.ERROR,
                error_message=f"保存文件失败: {e}",
                processing_time=time.time() - start_time,
                metadata=metadata,
            )

        quality_score, quality_level = self._assess_ocr_quality(processed_text, metadata)
        language = detect_language(processed_text)
        pages_processed = metadata.get("pages_detected", 0) or max(1, len(processed_text) // 3000)

        result = OCRResult(
            pdf_path=pdf_path,
            status=OCRStatus.SUCCESS,
            markdown_path=markdown_path,
            pages_processed=pages_processed,
            text_length=len(processed_text),
            quality_score=quality_score,
            quality_level=quality_level,
            language=language,
            processing_time=time.time() - start_time,
            metadata=metadata,
        )

        self.logger.info(
            f"PDF处理完成: {pdf_path.name} -> {quality_level.value} ({quality_score:.1f}%), "
            f"{pages_processed}页, {len(processed_text)}字符, 后端: {self.backend.value}"
        )
        return result

    def process_batch(self, pdf_files: List[Path], max_workers: int = 1) -> BatchOCRResult:
        total = len(pdf_files)
        results: List[OCRResult] = []
        self.logger.info(f"开始批量处理 {total} 个PDF文件")

        for i, pdf_file in enumerate(pdf_files, 1):
            self.logger.info(f"处理进度: {i}/{total} ({Path(pdf_file).name})")
            results.append(self.process_pdf(Path(pdf_file)))
            if i < total and max_workers == 1:
                time.sleep(0.5)

        succeeded = sum(1 for r in results if r.status == OCRStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == OCRStatus.FAILED)
        skipped = sum(1 for r in results if r.status == OCRStatus.SKIPPED)
        error = sum(1 for r in results if r.status == OCRStatus.ERROR)
        total_pages = sum(r.pages_processed for r in results)
        total_text_length = sum(r.text_length for r in results)

        self.logger.info(
            f"批量处理完成: {succeeded}成功, {skipped}跳过, {failed}失败, {error}错误\n"
            f"总页数: {total_pages}, 总字符数: {total_text_length}"
        )

        return BatchOCRResult(
            total=total,
            succeeded=succeeded,
            failed=failed + error,
            skipped=skipped,
            results=results,
            total_pages=total_pages,
            total_text_length=total_text_length,
        )

    def process_directory(self, pdf_dir: Path, pattern: str = "*.pdf") -> BatchOCRResult:
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            self.logger.error(f"目录不存在: {pdf_dir}")
            return BatchOCRResult()

        valid_pdf_files = [f for f in pdf_dir.glob(pattern) if is_valid_pdf(f)]
        self.logger.info(f"在目录中找到 {len(valid_pdf_files)} 个有效PDF文件")
        if not valid_pdf_files:
            return BatchOCRResult()
        return self.process_batch(valid_pdf_files)

    def get_processed_files(self) -> List[Path]:
        if not self.output_dir.exists():
            return []

        valid_files = []
        for md_file in self.output_dir.glob("*.md"):
            try:
                if md_file.read_text(encoding="utf-8")[:100].strip():
                    valid_files.append(md_file)
            except Exception:
                continue
        return valid_files

    def validate_markdown_file(self, md_file: Path) -> Tuple[bool, Dict[str, Any]]:
        md_file = Path(md_file)
        if not md_file.exists():
            return False, {"error": "文件不存在"}

        try:
            content = md_file.read_text(encoding="utf-8")
            if not content.strip():
                return False, {"error": "文件为空"}

            stats = {
                "file_size": md_file.stat().st_size,
                "text_length": len(content),
                "line_count": content.count('\n') + 1,
                "non_empty_lines": sum(1 for line in content.split('\n') if line.strip()),
                "language": detect_language(content),
                "has_tables": '|' in content and '---' in content,
                "has_headings": '#' in content,
            }
            quality_score, quality_level = self._assess_ocr_quality(content, {})
            stats["quality_score"] = quality_score
            stats["quality_level"] = quality_level.value
            return True, stats
        except Exception as e:
            return False, {"error": str(e)}

    def cleanup_temp_files(self) -> int:
        return 0
