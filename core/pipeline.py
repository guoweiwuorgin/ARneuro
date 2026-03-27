"""
ARneuro 主工作流程
整合PDF下载和OCR处理
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd

from .logger import get_module_logger
from .exceptions import ARneuroError, PDFDownloadError, OCRProcessingError
from ..config.config_manager import get_config
from ..data_fetch import PDFDownloader, PubMedFetcher, DownloadResult, BatchResult
from ..ocr_processing import GLMOCRProcessor, OCRResult, BatchOCRResult
from ..data_fetch.pdf_downloader import DownloadStatus
from ..ocr_processing.glm_ocr import OCRStatus
from ..utils.file_utils import ensure_dir
from ..utils.validation import validate_pmid, validate_csv_file


class ProcessingStage(Enum):
    """处理阶段"""
    INITIALIZED = "initialized"
    PDF_DOWNLOAD = "pdf_download"
    OCR_PROCESSING = "ocr_processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PaperProcessingStatus:
    """论文处理状态"""
    pmid: str
    stage: ProcessingStage
    pdf_path: Optional[Path] = None
    markdown_path: Optional[Path] = None
    pdf_download_result: Optional[DownloadResult] = None
    ocr_result: Optional[OCRResult] = None
    errors: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def processing_time(self) -> float:
        """处理时间"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def success(self) -> bool:
        """是否成功"""
        return self.stage == ProcessingStage.COMPLETED
    
    @property
    def has_pdf(self) -> bool:
        """是否有PDF"""
        return self.pdf_path is not None and self.pdf_path.exists()
    
    @property
    def has_markdown(self) -> bool:
        """是否有Markdown"""
        return self.markdown_path is not None and self.markdown_path.exists()


@dataclass
class PipelineResult:
    """流水线处理结果"""
    total_papers: int = 0
    processed_papers: int = 0
    successful_papers: int = 0
    failed_papers: int = 0
    skipped_papers: int = 0
    paper_statuses: Dict[str, PaperProcessingStatus] = field(default_factory=dict)
    pdf_download_stats: Dict[str, Any] = field(default_factory=dict)
    ocr_stats: Dict[str, Any] = field(default_factory=dict)
    total_processing_time: float = 0.0
    output_dir: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_papers': self.total_papers,
            'processed_papers': self.processed_papers,
            'successful_papers': self.successful_papers,
            'failed_papers': self.failed_papers,
            'skipped_papers': self.skipped_papers,
            'success_rate': f"{(self.successful_papers/self.total_papers*100):.1f}%" if self.total_papers > 0 else "0%",
            'pdf_download_stats': self.pdf_download_stats,
            'ocr_stats': self.ocr_stats,
            'total_processing_time': self.total_processing_time,
            'output_dir': str(self.output_dir) if self.output_dir else None
        }
    
    def save_report(self, report_path: Path) -> bool:
        """保存处理报告"""
        try:
            ensure_dir(report_path.parent)
            
            report_data = self.to_dict()
            
            # 添加详细状态
            detailed_status = {}
            for pmid, status in self.paper_statuses.items():
                detailed_status[pmid] = {
                    'stage': status.stage.value,
                    'success': status.success,
                    'has_pdf': status.has_pdf,
                    'has_markdown': status.has_markdown,
                    'processing_time': status.processing_time,
                    'errors': status.errors,
                    'download_source': (
                        status.pdf_download_result.downloader_used
                        if status.pdf_download_result else None
                    ),
                    'download_source_url': (
                        status.pdf_download_result.source_url
                        if status.pdf_download_result else None
                    ),
                    'manual_download_suggestions': (
                        status.pdf_download_result.manual_download_suggestions
                        if status.pdf_download_result else []
                    ),
                }
            
            report_data['detailed_status'] = detailed_status
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            get_module_logger("pipeline").error(f"保存报告失败: {e}")
            return False


class ARneuroPipeline:
    """ARneuro主工作流程"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化流水线
        
        Args:
            config_path: 配置文件路径
        """
        self.config = get_config(config_path)
        self.logger = get_module_logger("pipeline")
        
        # 初始化组件
        self.pdf_downloader = PDFDownloader(config_path)
        self.pubmed_fetcher = PubMedFetcher(config_path)
        self.ocr_processor = GLMOCRProcessor(config_path)
        
        # 输出目录
        self.output_dir = Path(self.config.get("paths.data_dir", "./data"))
        ensure_dir(self.output_dir)
        
        # 状态跟踪
        self.paper_statuses: Dict[str, PaperProcessingStatus] = {}
        self.csv_management_report: Dict[str, Any] = {}
        
        self.logger.info("ARneuro流水线初始化完成")
    
    def load_papers_from_csv(self, csv_file: Path) -> List[str]:
        """
        从CSV文件加载PMID列表
        
        Args:
            csv_file: CSV文件路径
        
        Returns:
            PMID列表
        """
        self.logger.info(f"从CSV文件加载PMID: {csv_file}")
        
        # 验证CSV文件
        required_columns = ['PMID']
        valid, error = validate_csv_file(csv_file, required_columns)
        
        if not valid:
            self.logger.error(f"CSV文件验证失败: {error}")
            return []
        
        # 提取PMID并生成CSV管理报告
        records, csv_report = self.pubmed_fetcher.parse_csv_with_management_report(csv_file)
        self.csv_management_report = csv_report
        report_path = self.output_dir / "reports" / "csv_document_management_report.json"
        self.pubmed_fetcher.save_management_report(csv_report, report_path)
        pmids = [record.pmid for record in records]
        
        if not pmids:
            self.logger.warning("CSV文件中没有有效的PMID")
            return []
        
        self.logger.info(f"加载 {len(pmids)} 个有效PMID")
        
        # 初始化状态
        for pmid in pmids:
            self.paper_statuses[pmid] = PaperProcessingStatus(
                pmid=pmid,
                stage=ProcessingStage.INITIALIZED
            )
        
        return pmids
    
    def process_single_paper(self, pmid: str, force_redownload: bool = False) -> PaperProcessingStatus:
        """
        处理单篇论文
        
        Args:
            pmid: PMID
            force_redownload: 是否强制重新下载
        
        Returns:
            处理状态
        """
        if pmid not in self.paper_statuses:
            self.paper_statuses[pmid] = PaperProcessingStatus(
                pmid=pmid,
                stage=ProcessingStage.INITIALIZED
            )
        
        status = self.paper_statuses[pmid]
        
        try:
            # 阶段1: PDF下载
            self.logger.info(f"处理PMID {pmid}: 开始PDF下载")
            status.stage = ProcessingStage.PDF_DOWNLOAD
            
            # 检查是否已存在PDF
            pdf_filename = f"{pmid}.pdf"
            pdf_path = self.pdf_downloader.output_dir / pdf_filename
            
            if pdf_path.exists() and not force_redownload:
                self.logger.info(f"PMID {pmid}: PDF已存在，跳过下载")
                download_result = DownloadResult(
                    pmid=pmid,
                    status=DownloadStatus.SKIPPED,
                    file_path=pdf_path,
                    file_size=pdf_path.stat().st_size
                )
            else:
                # 下载PDF
                download_result = self.pdf_downloader.download(pmid)
            
            status.pdf_download_result = download_result
            
            if download_result.status not in [DownloadStatus.SUCCESS, DownloadStatus.SKIPPED]:
                status.errors.append(f"PDF下载失败: {download_result.error_message}")
                status.stage = ProcessingStage.FAILED
                status.end_time = time.time()
                return status
            
            if download_result.file_path:
                status.pdf_path = download_result.file_path
            
            # 阶段2: OCR处理
            self.logger.info(f"处理PMID {pmid}: 开始OCR处理")
            status.stage = ProcessingStage.OCR_PROCESSING
            
            if status.pdf_path:
                ocr_result = self.ocr_processor.process_pdf(status.pdf_path, pmid)
                status.ocr_result = ocr_result
                
                if ocr_result.status == OCRStatus.SUCCESS:
                    status.markdown_path = ocr_result.markdown_path
                    status.stage = ProcessingStage.COMPLETED
                    self.logger.info(f"PMID {pmid}: 处理完成")
                else:
                    status.errors.append(f"OCR处理失败: {ocr_result.error_message}")
                    status.stage = ProcessingStage.FAILED
            else:
                status.errors.append("PDF路径不存在")
                status.stage = ProcessingStage.FAILED
        
        except Exception as e:
            self.logger.error(f"处理PMID {pmid}时发生异常: {e}")
            status.errors.append(f"处理异常: {str(e)}")
            status.stage = ProcessingStage.FAILED
        
        status.end_time = time.time()
        return status
    
    def process_batch(self, pmids: List[str], 
                     force_redownload: bool = False,
                     max_workers: int = 1) -> PipelineResult:
        """
        批量处理论文
        
        Args:
            pmids: PMID列表
            force_redownload: 是否强制重新下载
            max_workers: 最大工作线程数
        
        Returns:
            流水线处理结果
        """
        start_time = time.time()
        total = len(pmids)
        
        self.logger.info(f"开始批量处理 {total} 篇论文")
        
        # 初始化状态
        for pmid in pmids:
            if pmid not in self.paper_statuses:
                self.paper_statuses[pmid] = PaperProcessingStatus(
                    pmid=pmid,
                    stage=ProcessingStage.INITIALIZED
                )
        
        # 批量处理（目前只支持单线程）
        for i, pmid in enumerate(pmids, 1):
            self.logger.info(f"处理进度: {i}/{total} (PMID: {pmid})")
            
            status = self.process_single_paper(pmid, force_redownload)
            self.paper_statuses[pmid] = status
            
            # 进度报告
            if i % 10 == 0 or i == total:
                completed = sum(1 for s in self.paper_statuses.values() 
                              if s.stage == ProcessingStage.COMPLETED)
                failed = sum(1 for s in self.paper_statuses.values() 
                           if s.stage == ProcessingStage.FAILED)
                
                self.logger.info(
                    f"进度报告: {completed}完成, {failed}失败, {i-completed-failed}进行中"
                )
        
        # 统计结果
        total_processing_time = time.time() - start_time
        
        # PDF下载统计
        pdf_results = []
        for status in self.paper_statuses.values():
            if status.pdf_download_result:
                pdf_results.append(status.pdf_download_result)
        
        pdf_stats = self._calculate_pdf_stats(pdf_results)
        
        # OCR统计
        ocr_results = []
        for status in self.paper_statuses.values():
            if status.ocr_result:
                ocr_results.append(status.ocr_result)
        
        ocr_stats = self._calculate_ocr_stats(ocr_results)
        
        # 论文统计
        successful_papers = sum(1 for s in self.paper_statuses.values() 
                              if s.stage == ProcessingStage.COMPLETED)
        failed_papers = sum(1 for s in self.paper_statuses.values() 
                          if s.stage == ProcessingStage.FAILED)
        skipped_papers = total - successful_papers - failed_papers
        
        result = PipelineResult(
            total_papers=total,
            processed_papers=total,
            successful_papers=successful_papers,
            failed_papers=failed_papers,
            skipped_papers=skipped_papers,
            paper_statuses=self.paper_statuses,
            pdf_download_stats=pdf_stats,
            ocr_stats=ocr_stats,
            total_processing_time=total_processing_time,
            output_dir=self.output_dir
        )
        
        self.logger.info(
            f"批量处理完成: {successful_papers}成功, {failed_papers}失败, {skipped_papers}跳过\n"
            f"总处理时间: {total_processing_time:.1f}秒"
        )
        
        return result
    
    def _calculate_pdf_stats(self, pdf_results: List[DownloadResult]) -> Dict[str, Any]:
        """计算PDF下载统计"""
        if not pdf_results:
            return {}
        
        total = len(pdf_results)
        succeeded = sum(1 for r in pdf_results if r.status == DownloadStatus.SUCCESS)
        failed = sum(1 for r in pdf_results if r.status == DownloadStatus.FAILED)
        skipped = sum(1 for r in pdf_results if r.status == DownloadStatus.SKIPPED)
        error = sum(1 for r in pdf_results if r.status == DownloadStatus.ERROR)
        
        # 文件大小统计
        file_sizes = [r.file_size for r in pdf_results if r.file_size > 0]
        avg_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
        
        # 下载时间统计
        download_times = [r.download_time for r in pdf_results if r.download_time > 0]
        avg_time = sum(download_times) / len(download_times) if download_times else 0
        
        # 重试统计
        retry_counts = [r.retry_count for r in pdf_results]
        avg_retries = sum(retry_counts) / len(retry_counts) if retry_counts else 0
        
        source_distribution: Dict[str, int] = {}
        for result in pdf_results:
            source = result.downloader_used or result.source_details.get("source") or "unknown"
            source_distribution[source] = source_distribution.get(source, 0) + 1

        return {
            'total': total,
            'succeeded': succeeded,
            'failed': failed,
            'skipped': skipped,
            'error': error,
            'success_rate': f"{(succeeded/total*100):.1f}%" if total > 0 else "0%",
            'source_distribution': source_distribution,
            'avg_file_size': f"{avg_size/1024/1024:.2f} MB",
            'avg_download_time': f"{avg_time:.2f}秒",
            'avg_retries': f"{avg_retries:.2f}",
            'total_file_size': f"{sum(file_sizes)/1024/1024:.2f} MB"
        }
    
    def _calculate_ocr_stats(self, ocr_results: List[OCRResult]) -> Dict[str, Any]:
        """计算OCR处理统计"""
        if not ocr_results:
            return {}
        
        total = len(ocr_results)
        succeeded = sum(1 for r in ocr_results if r.status == OCRStatus.SUCCESS)
        failed = sum(1 for r in ocr_results if r.status == OCRStatus.FAILED)
        skipped = sum(1 for r in ocr_results if r.status == OCRStatus.SKIPPED)
        error = sum(1 for r in ocr_results if r.status == OCRStatus.ERROR)
        
        # 质量统计
        quality_counts = {}
        for result in ocr_results:
            if result.quality_level:
                level = result.quality_level.value
                quality_counts[level] = quality_counts.get(level, 0) + 1
        
        # 页面和文本统计
        total_pages = sum(r.pages_processed for r in ocr_results)
        total_text = sum(r.text_length for r in ocr_results)
        
        # 处理时间统计
        processing_times = [r.processing_time for r in ocr_results if r.processing_time > 0]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # 语言统计
        language_counts = {}
        for result in ocr_results:
            if result.language:
                lang = result.language
                language_counts[lang] = language_counts.get(lang, 0) + 1
        
        return {
            'total': total,
            'succeeded': succeeded,
            'failed': failed,
            'skipped': skipped,
            'error': error,
            'success_rate': f"{(succeeded/total*100):.1f}%" if total > 0 else "0%",
            'quality_distribution': quality_counts,
            'language_distribution': language_counts,
            'total_pages': total_pages,
            'total_text_length': total_text,
            'avg_processing_time': f"{avg_time:.2f}秒",
            'avg_pages_per_doc': f"{total_pages/total:.1f}" if total > 0 else "0",
            'avg_text_per_doc': f"{total_text/total:.0f}" if total > 0 else "0"
        }
    
    def get_processed_papers(self) -> List[PaperProcessingStatus]:
        """获取已处理的论文状态"""
        return [status for status in self.paper_statuses.values() 
                if status.stage != ProcessingStage.INITIALIZED]
    
    def get_successful_papers(self) -> List[PaperProcessingStatus]:
        """获取成功处理的论文"""
        return [status for status in self.paper_statuses.values() 
                if status.stage == ProcessingStage.COMPLETED]
    
    def get_failed_papers(self) -> List[PaperProcessingStatus]:
        """获取处理失败的论文"""
        return [status for status in self.paper_statuses.values() 
                if status.stage == ProcessingStage.FAILED]
    
    def export_results(self, output_dir: Optional[Path] = None) -> bool:
        """
        导出处理结果
        
        Args:
            output_dir: 输出目录
        
        Returns:
            是否成功
        """
        if output_dir is None:
            output_dir = self.output_dir / "results"
        
        ensure_dir(output_dir)
        
        try:
            # 导出状态报告
            report_path = output_dir / "processing_report.json"
            
            # 创建结果对象
            result = PipelineResult(
                total_papers=len(self.paper_statuses),
                processed_papers=len(self.get_processed_papers()),
                successful_papers=len(self.get_successful_papers()),
                failed_papers=len(self.get_failed_papers()),
                skipped_papers=len(self.paper_statuses) - len(self.get_processed_papers()),
                paper_statuses=self.paper_statuses,
                output_dir=output_dir
            )
            
            # 保存报告
            result.save_report(report_path)

            # 额外写入CSV解析管理报告
            if self.csv_management_report:
                csv_mgmt_report_path = output_dir / "csv_document_management_report.json"
                with open(csv_mgmt_report_path, 'w', encoding='utf-8') as f:
                    json.dump(self.csv_management_report, f, ensure_ascii=False, indent=2)
            
            # 导出成功论文列表
            successful_papers = self.get_successful_papers()
            if successful_papers:
                success_list = []
                for status in successful_papers:
                    success_list.append({
                        'pmid': status.pmid,
                        'pdf_path': str(status.pdf_path) if status.pdf_path else None,
                        'markdown_path': str(status.markdown_path) if status.markdown_path else None,
                        'processing_time': status.processing_time,
                        'ocr_quality': status.ocr_result.quality_level.value if status.ocr_result else None,
                        'ocr_quality_score': status.ocr_result.quality_score if status.ocr_result else None
                    })
                
                success_df = pd.DataFrame(success_list)
                success_csv = output_dir / "successful_papers.csv"
                success_df.to_csv(success_csv, index=False)
            
            # 导出失败论文列表
            failed_papers = self.get_failed_papers()
            if failed_papers:
                fail_list = []
                for status in failed_papers:
                    manual_urls = []
                    download_source = None
                    attempts = []
                    if status.pdf_download_result:
                        manual_urls = status.pdf_download_result.manual_download_suggestions
                        download_source = (
                            status.pdf_download_result.downloader_used
                            or status.pdf_download_result.source_details.get("source")
                        )
                        attempts = status.pdf_download_result.attempts
                    fail_list.append({
                        'pmid': status.pmid,
                        'stage': status.stage.value,
                        'errors': '; '.join(status.errors),
                        'processing_time': status.processing_time,
                        'download_source': download_source,
                        'manual_download_urls': ' | '.join(manual_urls or []),
                        'download_attempts': json.dumps(attempts, ensure_ascii=False)
                    })
                
                fail_df = pd.DataFrame(fail_list)
                fail_csv = output_dir / "failed_papers.csv"
                fail_df.to_csv(fail_csv, index=False)
            
            self.logger.info(f"结果已导出到: {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出结果失败: {e}")
            return False
    
    def cleanup(self) -> int:
        """
        清理临时文件
        
        Returns:
            清理的文件数量
        """
        cleaned = 0
        
        # 清理无效PDF文件
        cleaned += self.pdf_downloader.cleanup_invalid_files()
        
        # 清理OCR临时文件
        cleaned += self.ocr_processor.cleanup_temp_files()
        
        if cleaned > 0:
            self.logger.info(f"清理了 {cleaned} 个临时文件")
        
        return cleaned
    
    def run_full_pipeline(self, csv_file: Path, 
                         force_redownload: bool = False) -> PipelineResult:
        """
        运行完整流水线
        
        Args:
            csv_file: 输入CSV文件路径
            force_redownload: 是否强制重新下载
        
        Returns:
            流水线处理结果
        """
        self.logger.info("开始运行完整流水线")
        
        # 步骤1: 加载PMID
        pmids = self.load_papers_from_csv(csv_file)
        
        if not pmids:
            self.logger.error("没有有效的PMID，流水线终止")
            return PipelineResult()
        
        # 步骤2: 批量处理
        result = self.process_batch(pmids, force_redownload)
        
        # 步骤3: 导出结果
        self.export_results()
        
        # 步骤4: 清理
        self.cleanup()
        
        self.logger.info("完整流水线运行完成")
        return result
