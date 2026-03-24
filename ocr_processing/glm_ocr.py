"""
GLM-OCR处理模块
支持本地CLI与智谱AI在线API两种方式进行PDF转Markdown
"""

import base64
import hashlib
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import requests

from ..core import get_module_logger
from ..config import get_config
from ..utils.file_utils import ensure_dir, is_valid_pdf
from ..utils.text_utils import clean_text, normalize_text, detect_language


class OCRStatus(Enum):
    """OCR处理状态"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  # 已存在
    ERROR = "error"


class OCRQuality(Enum):
    """OCR质量等级"""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 70-89%
    FAIR = "fair"  # 50-69%
    POOR = "poor"  # <50%
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
    """GLM-OCR处理器，支持本地CLI与在线API两种模式。"""

    def __init__(self, config_path: Optional[str] = None, backend: Optional[str] = None):
        """
        初始化GLM-OCR处理器

        Args:
            config_path: 配置文件路径
            backend: 可选，强制指定执行后端（local/api）
        """
        self.config = get_config(config_path)
        self.logger = get_module_logger("glm_ocr")

        # 通用配置
        configured_backend = backend or self.config.get("ocr_processing.backend", GLMOCRBackend.LOCAL.value)
        self.backend = GLMOCRBackend(str(configured_backend).lower())
        self.output_dir = Path(self.config.get("ocr_processing.output_dir", "./data/markdown"))
        self.batch_size = self.config.get("ocr_processing.batch_size", 4)
        self.language = self.config.get("ocr_processing.language", "ch+en")

        # 本地CLI配置
        self.model_path = Path(self.config.get(
            "ocr_processing.model_path",
            "/storage/work/wuguowei/Bigmodel/GLM-OCR",
        ))
        self.device = self.config.get("ocr_processing.device", "cuda")
        self.glmocr_cli_path = self.config.get("ocr_processing.glmocr_cli_path", "glmocr")

        # API配置
        self.api_key = (
            self.config.get("ocr_processing.api_key")
            or self.config.get("glm_api_key")
        )
        self.api_base_url = self.config.get(
            "ocr_processing.api_base_url",
            "https://open.bigmodel.cn/api/paas/v4/layout_parsing",
        )
        self.api_model = self.config.get("ocr_processing.api_model", "glm-ocr")
        self.api_timeout = int(self.config.get("ocr_processing.api_timeout", 300))
        self.api_return_crop_images = bool(self.config.get("ocr_processing.api_return_crop_images", False))
        self.api_need_layout_visualization = bool(
            self.config.get("ocr_processing.api_need_layout_visualization", False)
        )
        self.api_start_page_id = self.config.get("ocr_processing.api_start_page_id")
        self.api_end_page_id = self.config.get("ocr_processing.api_end_page_id")
        self.api_use_base64 = bool(self.config.get("ocr_processing.api_use_base64", True))
        self.api_file_url = self.config.get("ocr_processing.api_file_url")
        self.api_user_id = self.config.get("ocr_processing.api_user_id")

        ensure_dir(self.output_dir)

        self._check_glmocr_availability()
        self.logger.info(f"GLM-OCR处理器初始化完成，后端: {self.backend.value}")

    def _check_glmocr_availability(self) -> bool:
        """检查当前配置的GLM-OCR后端是否可用。"""
        if self.backend == GLMOCRBackend.LOCAL:
            try:
                result = subprocess.run(
                    [self.glmocr_cli_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    self.logger.info(f"GLM-OCR本地CLI可用: {result.stdout.strip()}")
                    return True

                self.logger.warning(f"GLM-OCR本地CLI检查失败: {result.stderr}")
                return False
            except FileNotFoundError:
                self.logger.error(f"GLM-OCR命令未找到: {self.glmocr_cli_path}")
                self.logger.info("请确保GLM-OCR已安装并添加到PATH环境变量，或切换为API模式")
                return False
            except Exception as e:
                self.logger.error(f"检查GLM-OCR本地CLI失败: {e}")
                return False

        if not self.api_key:
            self.logger.error("GLM-OCR API模式缺少api_key，请配置 ocr_processing.api_key 或 glm_api_key")
            return False

        self.logger.info("GLM-OCR API模式已启用")
        return True

    def _build_api_request_payload(self, pdf_path: Path) -> Dict[str, Any]:
        """构造智谱AI GLM-OCR API请求体。"""
        file_value = self.api_file_url
        if self.api_use_base64 or not file_value:
            file_bytes = pdf_path.read_bytes()
            file_value = base64.b64encode(file_bytes).decode("utf-8")

        payload: Dict[str, Any] = {
            "model": self.api_model,
            "file": file_value,
            "return_crop_images": self.api_return_crop_images,
            "need_layout_visualization": self.api_need_layout_visualization,
            "request_id": f"glmocr_{hashlib.md5(str(pdf_path).encode('utf-8')).hexdigest()}",
        }

        if self.api_start_page_id:
            payload["start_page_id"] = int(self.api_start_page_id)
        if self.api_end_page_id:
            payload["end_page_id"] = int(self.api_end_page_id)
        if self.api_user_id:
            payload["user_id"] = self.api_user_id

        return payload

    def _run_glmocr_local_command(self, pdf_path: Path, output_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
        """运行本地GLM-OCR命令。"""
        try:
            cmd = [
                self.glmocr_cli_path,
                "predict",
                "--model_path", str(self.model_path),
                "--device", self.device,
                "--lang", self.language,
                "--output_format", "markdown",
                "--output", str(output_path),
                str(pdf_path),
            ]

            self.logger.debug(f"运行本地GLM-OCR命令: {' '.join(cmd)}")
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                encoding="utf-8",
                errors="ignore",
            )
            processing_time = time.time() - start_time

            if result.returncode == 0 and output_path.exists():
                output_text = output_path.read_text(encoding="utf-8")
                metadata = self._parse_ocr_metadata(result.stdout, result.stderr, processing_time)
                metadata["backend"] = GLMOCRBackend.LOCAL.value
                return True, output_text, metadata

            error_msg = f"GLM-OCR本地命令失败 (code={result.returncode}): {result.stderr}"
            self.logger.error(error_msg)
            return False, "", {"error": error_msg, "backend": GLMOCRBackend.LOCAL.value}
        except subprocess.TimeoutExpired:
            error_msg = "GLM-OCR本地处理超时 (300秒)"
            self.logger.error(error_msg)
            return False, "", {"error": error_msg, "backend": GLMOCRBackend.LOCAL.value}
        except Exception as e:
            error_msg = f"运行GLM-OCR本地命令异常: {e}"
            self.logger.error(error_msg)
            return False, "", {"error": error_msg, "backend": GLMOCRBackend.LOCAL.value}

    def _run_glmocr_api_request(self, pdf_path: Path, output_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
        """调用智谱AI官方GLM-OCR在线API。"""
        if not self.api_key:
            error_msg = "GLM-OCR API模式缺少api_key"
            return False, "", {"error": error_msg, "backend": GLMOCRBackend.API.value}

        try:
            payload = self._build_api_request_payload(pdf_path)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            self.logger.debug(f"调用GLM-OCR API: {self.api_base_url}")
            start_time = time.time()
            response = requests.post(
                self.api_base_url,
                headers=headers,
                json=payload,
                timeout=self.api_timeout,
            )
            processing_time = time.time() - start_time
            response.raise_for_status()

            response_data = response.json()
            output_text = response_data.get("md_results", "")
            if not output_text.strip():
                error_msg = f"GLM-OCR API返回为空: {response_data}"
                self.logger.error(error_msg)
                return False, "", {"error": error_msg, "backend": GLMOCRBackend.API.value}

            output_path.write_text(output_text, encoding="utf-8")
            metadata = self._parse_api_metadata(response_data, processing_time)
            metadata["backend"] = GLMOCRBackend.API.value
            return True, output_text, metadata
        except requests.HTTPError as e:
            response_text = e.response.text[:1000] if e.response is not None else ""
            error_msg = f"GLM-OCR API请求失败: {e}; 响应: {response_text}"
            self.logger.error(error_msg)
            return False, "", {"error": error_msg, "backend": GLMOCRBackend.API.value}
        except Exception as e:
            error_msg = f"调用GLM-OCR API异常: {e}"
            self.logger.error(error_msg)
            return False, "", {"error": error_msg, "backend": GLMOCRBackend.API.value}

    def _run_glmocr_command(self, pdf_path: Path, output_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
        """根据当前后端执行OCR。"""
        if self.backend == GLMOCRBackend.API:
            return self._run_glmocr_api_request(pdf_path, output_path)
        return self._run_glmocr_local_command(pdf_path, output_path)

    def _parse_api_metadata(self, response_data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """解析API返回的OCR元数据。"""
        data_info = response_data.get("data_info") or {}
        usage = response_data.get("usage") or {}
        return {
            "processing_time": processing_time,
            "task_id": response_data.get("id"),
            "request_id": response_data.get("request_id"),
            "pages_detected": data_info.get("num_pages", 0),
            "pages": data_info.get("pages", []),
            "usage": usage,
            "layout_details": response_data.get("layout_details", []),
            "layout_visualization": response_data.get("layout_visualization", []),
            "model": response_data.get("model", self.api_model),
        }

    def _parse_ocr_metadata(self, stdout: str, stderr: str, processing_time: float) -> Dict[str, Any]:
        """解析OCR元数据"""
        metadata = {
            "processing_time": processing_time,
            "stdout": stdout[:1000] if stdout else "",
            "stderr": stderr[:1000] if stderr else "",
        }

        try:
            page_pattern = r'(\d+)\s*pages?'
            page_match = re.search(page_pattern, stdout + stderr, re.IGNORECASE)
            if page_match:
                metadata["pages_detected"] = int(page_match.group(1))

            confidence_pattern = r'confidence[:\s]*([\d.]+)%?'
            confidence_match = re.search(confidence_pattern, stdout + stderr, re.IGNORECASE)
            if confidence_match:
                metadata["confidence"] = float(confidence_match.group(1))

            lang_pattern = r'language[:\s]*([a-zA-Z+]+)'
            lang_match = re.search(lang_pattern, stdout + stderr, re.IGNORECASE)
            if lang_match:
                metadata["detected_language"] = lang_match.group(1)
        except Exception as e:
            self.logger.debug(f"解析OCR元数据失败: {e}")

        return metadata

    def _assess_ocr_quality(self, text: str, metadata: Dict[str, Any]) -> Tuple[float, OCRQuality]:
        """评估OCR质量"""
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
        """后处理Markdown文本"""
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
        """处理单个PDF文件"""
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
        """批量处理PDF文件"""
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
        """处理目录中的所有PDF文件"""
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
        """获取已处理的Markdown文件列表"""
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
        """验证Markdown文件"""
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
        """清理临时文件，返回清理的数量"""
        return 0
