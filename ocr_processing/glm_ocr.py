"""
GLM-OCR处理模块
使用GLM-OCR CLI方式进行PDF转Markdown
"""

import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from ..core import OCRProcessingError, get_module_logger
from ..config import get_config
from ..utils.file_utils import ensure_dir, is_valid_pdf, get_file_size
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
    """GLM-OCR处理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化GLM-OCR处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = get_config(config_path)
        self.logger = get_module_logger("glm_ocr")
        
        # 配置参数
        self.model_path = Path(self.config.get("ocr_processing.model_path", 
                                             "/storage/work/wuguowei/Bigmodel/GLM-OCR"))
        self.device = self.config.get("ocr_processing.device", "cuda")
        self.batch_size = self.config.get("ocr_processing.batch_size", 4)
        self.output_dir = Path(self.config.get("ocr_processing.output_dir", "./data/markdown"))
        self.glmocr_cli_path = self.config.get("ocr_processing.glmocr_cli_path", "glmocr")
        self.language = self.config.get("ocr_processing.language", "ch+en")
        
        # 确保目录存在
        ensure_dir(self.output_dir)
        
        # 检查GLM-OCR是否可用
        self._check_glmocr_availability()
        
        self.logger.info(f"GLM-OCR处理器初始化完成，模型路径: {self.model_path}")
    
    def _check_glmocr_availability(self) -> bool:
        """检查GLM-OCR是否可用"""
        try:
            # 尝试运行glmocr命令
            result = subprocess.run(
                [self.glmocr_cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.logger.info(f"GLM-OCR可用: {result.stdout.strip()}")
                return True
            else:
                self.logger.warning(f"GLM-OCR检查失败: {result.stderr}")
                return False
                
        except FileNotFoundError:
            self.logger.error(f"GLM-OCR命令未找到: {self.glmocr_cli_path}")
            self.logger.info("请确保GLM-OCR已安装并添加到PATH环境变量")
            return False
        except Exception as e:
            self.logger.error(f"检查GLM-OCR失败: {e}")
            return False
    
    def _run_glmocr_command(self, pdf_path: Path, output_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
        """
        运行GLM-OCR命令
        
        Args:
            pdf_path: PDF文件路径
            output_path: 输出文件路径
        
        Returns:
            (是否成功, 输出文本, 元数据)
        """
        try:
            # 构建命令
            cmd = [
                self.glmocr_cli_path,
                "predict",
                "--model_path", str(self.model_path),
                "--device", self.device,
                "--lang", self.language,
                "--output_format", "markdown",
                "--output", str(output_path),
                str(pdf_path)
            ]
            
            self.logger.debug(f"运行GLM-OCR命令: {' '.join(cmd)}")
            
            # 运行命令
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                encoding='utf-8',
                errors='ignore'
            )
            processing_time = time.time() - start_time
            
            if result.returncode == 0:
                # 读取输出文件
                if output_path.exists():
                    with open(output_path, 'r', encoding='utf-8') as f:
                        output_text = f.read()
                    
                    # 解析元数据
                    metadata = self._parse_ocr_metadata(result.stdout, result.stderr, processing_time)
                    
                    return True, output_text, metadata
                else:
                    self.logger.error(f"GLM-OCR输出文件不存在: {output_path}")
                    return False, "", {}
            else:
                error_msg = f"GLM-OCR命令失败 (code={result.returncode}): {result.stderr}"
                self.logger.error(error_msg)
                return False, "", {"error": error_msg}
                
        except subprocess.TimeoutExpired:
            error_msg = f"GLM-OCR处理超时 (300秒)"
            self.logger.error(error_msg)
            return False, "", {"error": error_msg}
        except Exception as e:
            error_msg = f"运行GLM-OCR命令异常: {e}"
            self.logger.error(error_msg)
            return False, "", {"error": error_msg}
    
    def _parse_ocr_metadata(self, stdout: str, stderr: str, processing_time: float) -> Dict[str, Any]:
        """解析OCR元数据"""
        metadata = {
            "processing_time": processing_time,
            "stdout": stdout[:1000] if stdout else "",  # 限制长度
            "stderr": stderr[:1000] if stderr else "",
        }
        
        # 尝试从输出中提取信息
        try:
            # 查找页面数
            page_pattern = r'(\d+)\s*pages?'
            page_match = re.search(page_pattern, stdout + stderr, re.IGNORECASE)
            if page_match:
                metadata["pages_detected"] = int(page_match.group(1))
            
            # 查找置信度
            confidence_pattern = r'confidence[:\s]*([\d.]+)%?'
            confidence_match = re.search(confidence_pattern, stdout + stderr, re.IGNORECASE)
            if confidence_match:
                metadata["confidence"] = float(confidence_match.group(1))
            
            # 查找语言
            lang_pattern = r'language[:\s]*([a-zA-Z+]+)'
            lang_match = re.search(lang_pattern, stdout + stderr, re.IGNORECASE)
            if lang_match:
                metadata["detected_language"] = lang_match.group(1)
        
        except Exception as e:
            self.logger.debug(f"解析OCR元数据失败: {e}")
        
        return metadata
    
    def _assess_ocr_quality(self, text: str, metadata: Dict[str, Any]) -> Tuple[float, OCRQuality]:
        """
        评估OCR质量
        
        Args:
            text: OCR文本
            metadata: OCR元数据
        
        Returns:
            (质量分数, 质量等级)
        """
        if not text:
            return 0.0, OCRQuality.POOR
        
        # 基础质量指标
        text_length = len(text)
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # 1. 文本长度评分 (0-30分)
        length_score = min(30, text_length / 1000 * 3)  # 每1000字符得3分，最多30分
        
        # 2. 非空行比例评分 (0-30分)
        if lines:
            non_empty_ratio = len(non_empty_lines) / len(lines)
            line_score = non_empty_ratio * 30
        else:
            line_score = 0
        
        # 3. 置信度评分 (如果有) (0-40分)
        confidence = metadata.get("confidence", 0)
        confidence_score = confidence * 0.4  # 转换为40分制
        
        # 总分数
        total_score = length_score + line_score + confidence_score
        
        # 确定质量等级
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
        
        # 清理文本
        text = clean_text(markdown_text, remove_empty_lines=True)
        text = normalize_text(text)
        
        # 修复常见的Markdown格式问题
        # 1. 修复表格格式
        lines = text.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            # 移除多余的空格
            line = line.strip()
            
            # 修复表格分隔符
            if '|' in line:
                # 确保表格行有正确的分隔符
                if '---' in line or '===' in line:
                    # 这是表格分隔行，确保格式正确
                    cells = line.split('|')
                    processed_cells = []
                    for cell in cells:
                        cell = cell.strip()
                        if cell and all(c == '-' or c == '=' or c == ':' for c in cell):
                            # 保持原样
                            processed_cells.append(cell)
                        else:
                            # 添加分隔符
                            processed_cells.append('---')
                    line = '|' + '|'.join(processed_cells) + '|'
            
            processed_lines.append(line)
        
        # 重新组合文本
        text = '\n'.join(processed_lines)
        
        return text
    
    def process_pdf(self, pdf_path: Path, output_filename: Optional[str] = None) -> OCRResult:
        """
        处理单个PDF文件
        
        Args:
            pdf_path: PDF文件路径
            output_filename: 自定义输出文件名（不含扩展名）
        
        Returns:
            OCRResult
        """
        start_time = time.time()
        
        # 验证PDF文件
        if not pdf_path.exists():
            return OCRResult(
                pdf_path=pdf_path,
                status=OCRStatus.ERROR,
                error_message="PDF文件不存在",
                processing_time=time.time() - start_time
            )
        
        if not is_valid_pdf(pdf_path):
            return OCRResult(
                pdf_path=pdf_path,
                status=OCRStatus.ERROR,
                error_message="无效的PDF文件",
                processing_time=time.time() - start_time
            )
        
        # 确定输出文件名
        if output_filename:
            safe_filename = output_filename
        else:
            safe_filename = pdf_path.stem
        
        markdown_path = self.output_dir / f"{safe_filename}.md"
        
        # 检查是否已处理
        if markdown_path.exists():
            try:
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    existing_text = f.read()
                
                if existing_text.strip():
                    # 评估现有文件的质量
                    quality_score, quality_level = self._assess_ocr_quality(existing_text, {})
                    language = detect_language(existing_text)
                    
                    self.logger.info(f"PDF已处理，跳过: {pdf_path}")
                    return OCRResult(
                        pdf_path=pdf_path,
                        status=OCRStatus.SKIPPED,
                        markdown_path=markdown_path,
                        pages_processed=0,  # 未知
                        text_length=len(existing_text),
                        quality_score=quality_score,
                        quality_level=quality_level,
                        language=language,
                        processing_time=0.0
                    )
            except Exception as e:
                self.logger.warning(f"读取现有Markdown文件失败: {e}")
        
        # 处理PDF
        self.logger.info(f"开始处理PDF: {pdf_path}")
        
        success, output_text, metadata = self._run_glmocr_command(pdf_path, markdown_path)
        
        if not success:
            return OCRResult(
                pdf_path=pdf_path,
                status=OCRStatus.FAILED,
                error_message=metadata.get("error", "GLM-OCR处理失败"),
                processing_time=time.time() - start_time,
                metadata=metadata
            )
        
        # 后处理
        processed_text = self._post_process_markdown(output_text)
        
        # 保存处理后的文本
        try:
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
        except Exception as e:
            self.logger.error(f"保存Markdown文件失败: {e}")
            return OCRResult(
                pdf_path=pdf_path,
                status=OCRStatus.ERROR,
                error_message=f"保存文件失败: {e}",
                processing_time=time.time() - start_time,
                metadata=metadata
            )
        
        # 评估质量
        quality_score, quality_level = self._assess_ocr_quality(processed_text, metadata)
        language = detect_language(processed_text)
        
        # 统计页面数
        pages_processed = metadata.get("pages_detected", 0)
        if pages_processed == 0:
            # 尝试从文本中估计页面数
            pages_processed = max(1, len(processed_text) // 3000)  # 每页约3000字符
        
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
            metadata=metadata
        )
        
        self.logger.info(
            f"PDF处理完成: {pdf_path.name} -> {quality_level.value} ({quality_score:.1f}%), "
            f"{pages_processed}页, {len(processed_text)}字符"
        )
        
        return result
    
    def process_batch(self, pdf_files: List[Path], 
                     max_workers: int = 1) -> BatchOCRResult:
        """
        批量处理PDF文件
        
        Args:
            pdf_files: PDF文件列表
            max_workers: 最大工作线程数（目前只支持单线程）
        
        Returns:
            BatchOCRResult
        """
        total = len(pdf_files)
        results = []
        
        self.logger.info(f"开始批量处理 {total} 个PDF文件")
        
        for i, pdf_file in enumerate(pdf_files, 1):
            self.logger.info(f"处理进度: {i}/{total} ({pdf_file.name})")
            
            result = self.process_pdf(pdf_file)
            results.append(result)
            
            # 添加延迟以避免资源竞争
            if i < total and max_workers == 1:
                time.sleep(0.5)
        
        # 统计结果
        succeeded = sum(1 for r in results if r.status == OCRStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == OCRStatus.FAILED)
        skipped = sum(1 for r in results if r.status == OCRStatus.SKIPPED)
        error = sum(1 for r in results if r.status == OCRStatus.ERROR)
        
        total_pages = sum(r.pages_processed for r in results)
        total_text_length = sum(r.text_length for r in results)
        
        batch_result = BatchOCRResult(
            total=total,
            succeeded=succeeded,
            failed=failed + error,
            skipped=skipped,
            results=results,
            total_pages=total_pages,
            total_text_length=total_text_length
        )
        
        # 质量统计
        quality_counts = {}
        for result in results:
            if result.quality_level:
                level = result.quality_level.value
                quality_counts[level] = quality_counts.get(level, 0) + 1
        
        self.logger.info(
            f"批量处理完成: {succeeded}成功, {skipped}跳过, {failed}失败, {error}错误\n"
            f"总页数: {total_pages}, 总字符数: {total_text_length}\n"
            f"质量分布: {quality_counts}"
        )
        
        return batch_result
    
    def process_directory(self, pdf_dir: Path, pattern: str = "*.pdf") -> BatchOCRResult:
        """
        处理目录中的所有PDF文件
        
        Args:
            pdf_dir: PDF目录路径
            pattern: 文件匹配模式
        
        Returns:
            BatchOCRResult
        """
        if not pdf_dir.exists():
            self.logger.error(f"目录不存在: {pdf_dir}")
            return BatchOCRResult()
        
        # 查找PDF文件
        pdf_files = list(pdf_dir.glob(pattern))
        valid_pdf_files = [f for f in pdf_files if is_valid_pdf(f)]
        
        self.logger.info(f"在目录中找到 {len(valid_pdf_files)} 个有效PDF文件")
        
        if not valid_pdf_files:
            return BatchOCRResult()
        
        # 批量处理
        return self.process_batch(valid_pdf_files)
    
    def get_processed_files(self) -> List[Path]:
        """获取已处理的Markdown文件列表"""
        if not self.output_dir.exists():
            return []
        
        md_files = list(self.output_dir.glob("*.md"))
        valid_files = []
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read(100)  # 读取前100个字符
                    if content.strip():
                        valid_files.append(md_file)
            except:
                continue
        
        return valid_files
    
    def validate_markdown_file(self, md_file: Path) -> Tuple[bool, Dict[str, Any]]:
        """
        验证Markdown文件
        
        Args:
            md_file: Markdown文件路径
        
        Returns:
            (是否有效, 验证信息)
        """
        if not md_file.exists():
            return False, {"error": "文件不存在"}
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return False, {"error": "文件为空"}
            
            # 基本统计
            stats = {
                "file_size": md_file.stat().st_size,
                "text_length": len(content),
                "line_count": content.count('\n') + 1,
                "non_empty_lines": sum(1 for line in content.split('\n') if line.strip()),
                "language": detect_language(content),
                "has_tables": '|' in content and '---' in content,
                "has_headings": '#' in content
            }
            
            # 质量评估
            quality_score, quality_level = self._assess_ocr_quality(content, {})
            stats["quality_score"] = quality_score
            stats["quality_level"] = quality_level.value
            
            return True, stats
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def cleanup_temp_files(self) -> int:
        """清理临时文件，返回清理的数量"""
        # GLM-OCR可能会创建临时文件，这里可以添加清理逻辑
        # 目前返回0，需要根据实际使用情况实现
        return 0