"""
PDF下载模块
基于现有的fetch_pdfs.py重构，支持多种期刊格式
"""

import os
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..core import get_module_logger
from ..config.config_manager import get_config
from ..utils.file_utils import ensure_dir, sanitize_filename, is_valid_pdf


class DownloadStatus(Enum):
    """下载状态"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  # 已存在
    ERROR = "error"


@dataclass
class DownloadResult:
    """下载结果"""
    pmid: str
    status: DownloadStatus
    file_path: Optional[Path] = None
    file_size: int = 0
    downloader_used: Optional[str] = None
    source_url: Optional[str] = None
    source_details: Dict[str, Any] = field(default_factory=dict)
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    manual_download_suggestions: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    retry_count: int = 0
    download_time: float = 0.0


@dataclass
class BatchResult:
    """批量下载结果"""
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[DownloadResult] = field(default_factory=list)
    error_file: Optional[Path] = None
    report_file: Optional[Path] = None
    manual_checklist_file: Optional[Path] = None


class PDFDownloader:
    """PDF下载器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化PDF下载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = get_config(config_path)
        self.logger = get_module_logger("pdf_downloader")
        
        # 配置参数
        self.output_dir = Path(self.config.get("pdf_download.output_dir", "./data/pdfs"))
        self.max_retries = self.config.get("pdf_download.max_retries", 3)
        self.timeout = self.config.get("pdf_download.timeout", 30)
        self.user_agent = self.config.get("pdf_download.user_agent", 
                                         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        # 注册finder
        self.finder_registry = self._register_finders()
        
        # 请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # 确保输出目录存在
        ensure_dir(self.output_dir)
        
        self.logger.info(f"PDF下载器初始化完成，输出目录: {self.output_dir}")
    
    def _register_finders(self) -> Dict[str, Callable]:
        """注册PDF查找器"""
        finders = {
            'generic_citation': self._generic_citation_finder,
            'pubmed_central': self._pubmed_central_finder,
            'acs': self._acs_finder,
            'nejm': self._nejm_finder,
            'science_direct': self._science_direct_finder,
            'future_medicine': self._future_medicine_finder,
            'uchicago_press': self._uchicago_press_finder,
            'direct_pdf': self._direct_pdf_finder
        }
        
        # 根据配置启用finder
        enabled_finders = self.config.get("pdf_download.finders", list(finders.keys()))
        return {name: finders[name] for name in enabled_finders if name in finders}
    
    def _get_main_url(self, url: str) -> str:
        """获取主URL（协议+域名）"""
        parts = url.split('/')
        return '/'.join(parts[:3])
    
    def _generic_citation_finder(self, req: requests.Response, soup: BeautifulSoup) -> Optional[str]:
        """通用引用标签查找器"""
        possible_links = soup.find_all('meta', attrs={'name': 'citation_pdf_url'})
        if possible_links:
            self.logger.debug("使用generic_citation查找器")
            return possible_links[0].get('content')
        return None
    
    def _pubmed_central_finder(self, req: requests.Response, soup: BeautifulSoup) -> Optional[str]:
        """PubMed Central查找器"""
        # 方法1: 查找/pmc/articles链接
        possible_links = soup.find_all('a', attrs={'href': re.compile('/pmc/articles')})
        if possible_links:
            self.logger.debug("使用pubmed_central查找器（方法1）")
            return f"https://www.ncbi.nlm.nih.gov{possible_links[0].get('href')}"
        
        # 方法2: 查找PDF链接
        possible_links = soup.find_all('a', string=re.compile('pdf', re.IGNORECASE))
        for link in possible_links:
            href = link.get('href', '')
            if 'pdf' in href.lower() and 'epdf' not in href.lower():
                self.logger.debug("使用pubmed_central查找器（方法2）")
                return self._get_main_url(req.url) + href
        
        return None
    
    def _acs_finder(self, req: requests.Response, soup: BeautifulSoup) -> Optional[str]:
        """ACS出版物查找器"""
        possible_links = [x for x in soup.find_all('a') 
                         if isinstance(x.get('title'), str) and 
                         ('high-res pdf' in x.get('title').lower() or 
                          'low-res pdf' in x.get('title').lower())]
        
        if possible_links:
            self.logger.debug("使用acs查找器")
            return self._get_main_url(req.url) + possible_links[0].get('href')
        
        return None
    
    def _nejm_finder(self, req: requests.Response, soup: BeautifulSoup) -> Optional[str]:
        """新英格兰医学杂志查找器"""
        possible_links = [x for x in soup.find_all('a') 
                         if isinstance(x.get('data-download-type'), str) and 
                         (x.get('data-download-type').lower() == 'article pdf')]
        
        if possible_links:
            self.logger.debug("使用nejm查找器")
            return self._get_main_url(req.url) + possible_links[0].get('href')
        
        return None
    
    def _science_direct_finder(self, req: requests.Response, soup: BeautifulSoup) -> Optional[str]:
        """Science Direct查找器"""
        try:
            # 查找重定向URL
            inputs = soup.find_all('input')
            if not inputs:
                return None
            
            new_uri = requests.utils.unquote(inputs[0].get('value', ''))
            if not new_uri:
                return None
            
            # 获取重定向页面
            redirect_req = self.session.get(new_uri, allow_redirects=True, timeout=self.timeout)
            redirect_soup = BeautifulSoup(redirect_req.content, 'html.parser')
            
            # 查找PDF链接
            possible_links = redirect_soup.find_all('meta', attrs={'name': 'citation_pdf_url'})
            if possible_links:
                pdf_url = possible_links[0].get('content')
                # 获取实际的PDF链接
                pdf_req = self.session.get(pdf_url, timeout=self.timeout)
                pdf_soup = BeautifulSoup(pdf_req.content, 'html.parser')
                pdf_links = pdf_soup.find_all('a')
                
                if pdf_links:
                    self.logger.debug("使用science_direct查找器")
                    return pdf_links[0].get('href')
        
        except Exception as e:
            self.logger.warning(f"Science Direct查找器错误: {e}")
        
        return None
    
    def _future_medicine_finder(self, req: requests.Response, soup: BeautifulSoup) -> Optional[str]:
        """Future Medicine查找器"""
        possible_links = soup.find_all('a', attrs={'href': re.compile("/doi/pdf")})
        if possible_links:
            self.logger.debug("使用future_medicine查找器")
            return self._get_main_url(req.url) + possible_links[0].get('href')
        return None
    
    def _uchicago_press_finder(self, req: requests.Response, soup: BeautifulSoup) -> Optional[str]:
        """芝加哥大学出版社查找器"""
        possible_links = [x for x in soup.find_all('a') 
                         if isinstance(x.get('href'), str) and 
                         'pdf' in x.get('href') and '.edu/doi/' in x.get('href')]
        
        if possible_links:
            self.logger.debug("使用uchicago_press查找器")
            return self._get_main_url(req.url) + possible_links[0].get('href')
        
        return None
    
    def _direct_pdf_finder(self, req: requests.Response, soup: BeautifulSoup) -> Optional[str]:
        """直接PDF链接查找器"""
        content_type = req.headers.get('Content-Type', '')
        if 'application/pdf' in content_type:
            self.logger.debug("使用direct_pdf查找器")
            return req.url
        
        # 检查URL是否以.pdf结尾
        if req.url.lower().endswith('.pdf'):
            self.logger.debug("使用direct_pdf查找器（URL以.pdf结尾）")
            return req.url
        
        return None
    
    def _save_pdf(self, pdf_url: str, file_path: Path) -> bool:
        """保存PDF文件"""
        try:
            response = self.session.get(pdf_url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # 验证PDF文件
            if is_valid_pdf(file_path):
                self.logger.debug(f"PDF保存成功: {file_path}")
                return True
            else:
                self.logger.warning(f"保存的文件不是有效的PDF: {file_path}")
                file_path.unlink(missing_ok=True)
                return False
                
        except Exception as e:
            self.logger.error(f"保存PDF失败 {pdf_url}: {e}")
            file_path.unlink(missing_ok=True)
            return False
    
    def _fetch_pdf_url(self, pmid: str) -> Tuple[Optional[str], Optional[str]]:
        """
        获取PDF URL
        
        Returns:
            (PDF URL, 使用的finder名称)
        """
        result = self._fetch_pdf_url_with_details(pmid)
        return result[0], result[1]

    def _fetch_pdf_url_with_details(
        self, pmid: str
    ) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
        """
        以多来源方式获取PDF URL，并保留来源记录（用于系统报告）。

        Returns:
            (pdf_url, source_name, attempts, source_details)
        """
        attempts: List[Dict[str, Any]] = []
        uri = (
            "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
            f"?dbfrom=pubmed&id={pmid}&retmode=ref&cmd=prlinks"
        )

        try:
            req = self.session.get(uri, timeout=self.timeout)
            
            # 检查是否支持Ovid（不支持）
            if 'ovid' in req.url:
                self.logger.warning(f"PMID {pmid}: Ovid不支持")
                attempts.append({
                    "source": "pubmed_prlinks",
                    "url": req.url,
                    "status": "unsupported",
                    "message": "Ovid平台不支持自动下载",
                })
                return None, None, attempts, {}

            soup = BeautifulSoup(req.content, 'html.parser')
            
            # 尝试所有注册的finder
            for finder_name, finder_func in self.finder_registry.items():
                try:
                    pdf_url = finder_func(req, soup)
                    if pdf_url:
                        self.logger.info(f"PMID {pmid}: 使用 {finder_name} 找到PDF链接")
                        source = {
                            "source": "pubmed_prlinks",
                            "finder": finder_name,
                            "landing_url": req.url,
                            "pdf_url": pdf_url,
                            "access_type": "publisher_or_open",
                        }
                        attempts.append({**source, "status": "found", "message": "找到可下载PDF链接"})
                        return pdf_url, finder_name, attempts, source
                    attempts.append({
                        "source": "pubmed_prlinks",
                        "finder": finder_name,
                        "landing_url": req.url,
                        "status": "not_found",
                        "message": "未找到PDF链接",
                    })
                except Exception as e:
                    self.logger.warning(f"PMID {pmid}: {finder_name} 查找器错误: {e}")
                    attempts.append({
                        "source": "pubmed_prlinks",
                        "finder": finder_name,
                        "landing_url": req.url,
                        "status": "error",
                        "message": str(e),
                    })
                    continue

            # 备用来源1：Europe PMC开放全文接口（合法公开来源）
            epmc_url = (
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                f"?query=EXT_ID:{pmid}%20AND%20SRC:MED&format=json&resultType=core"
            )
            try:
                epmc_resp = self.session.get(epmc_url, timeout=self.timeout)
                epmc_resp.raise_for_status()
                epmc_data = epmc_resp.json()
                results = epmc_data.get("resultList", {}).get("result", [])
                for item in results:
                    for ft in item.get("fullTextUrlList", {}).get("fullTextUrl", []):
                        if str(ft.get("documentStyle", "")).lower() == "pdf":
                            pdf_url = ft.get("url")
                            if pdf_url:
                                source = {
                                    "source": "europe_pmc",
                                    "finder": "europe_pmc_fulltext",
                                    "landing_url": epmc_url,
                                    "pdf_url": pdf_url,
                                    "access_type": "open_access",
                                }
                                attempts.append({**source, "status": "found", "message": "Europe PMC开放链接可用"})
                                self.logger.info(f"PMID {pmid}: 使用Europe PMC找到PDF链接")
                                return pdf_url, "europe_pmc_fulltext", attempts, source
                attempts.append({
                    "source": "europe_pmc",
                    "finder": "europe_pmc_fulltext",
                    "landing_url": epmc_url,
                    "status": "not_found",
                    "message": "Europe PMC未提供PDF链接",
                })
            except Exception as e:
                attempts.append({
                    "source": "europe_pmc",
                    "finder": "europe_pmc_fulltext",
                    "landing_url": epmc_url,
                    "status": "error",
                    "message": str(e),
                })

            # 备用来源2：PMCID开放访问PDF（合法公开来源）
            idconv_url = (
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
                f"?ids={pmid}&format=json"
            )
            try:
                idconv_resp = self.session.get(idconv_url, timeout=self.timeout)
                idconv_resp.raise_for_status()
                id_data = idconv_resp.json()
                records = id_data.get("records", [])
                if records and records[0].get("pmcid"):
                    pmcid = records[0]["pmcid"]
                    pmc_pdf_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"
                    source = {
                        "source": "pmc_idconv",
                        "finder": "pmc_open_pdf",
                        "landing_url": idconv_url,
                        "pdf_url": pmc_pdf_url,
                        "access_type": "open_access",
                        "pmcid": pmcid,
                    }
                    attempts.append({**source, "status": "found", "message": "通过PMCID构造开放PDF链接"})
                    self.logger.info(f"PMID {pmid}: 使用PMCID开放访问链接")
                    return pmc_pdf_url, "pmc_open_pdf", attempts, source
                attempts.append({
                    "source": "pmc_idconv",
                    "finder": "pmc_open_pdf",
                    "landing_url": idconv_url,
                    "status": "not_found",
                    "message": "未检索到PMCID或开放PDF",
                })
            except Exception as e:
                attempts.append({
                    "source": "pmc_idconv",
                    "finder": "pmc_open_pdf",
                    "landing_url": idconv_url,
                    "status": "error",
                    "message": str(e),
                })

            self.logger.warning(f"PMID {pmid}: 未找到PDF链接")
            return None, None, attempts, {}

        except Exception as e:
            self.logger.error(f"PMID {pmid}: 获取PDF URL失败: {e}")
            attempts.append({
                "source": "pubmed_prlinks",
                "status": "error",
                "message": str(e),
            })
            return None, None, attempts, {}
    
    def download(self, pmid: str, filename: Optional[str] = None) -> DownloadResult:
        """
        下载单个PDF
        
        Args:
            pmid: PMID
            filename: 自定义文件名（不含扩展名）
        
        Returns:
            DownloadResult
        """
        start_time = time.time()
        
        # 验证PMID
        if not re.match(r'^\d+$', str(pmid)):
            return DownloadResult(
                pmid=pmid,
                status=DownloadStatus.ERROR,
                error_message="无效的PMID格式"
            )
        
        # 确定文件名
        if filename:
            safe_filename = sanitize_filename(filename)
        else:
            safe_filename = str(pmid)
        
        file_path = self.output_dir / f"{safe_filename}.pdf"
        
        # 检查文件是否已存在
        if file_path.exists() and is_valid_pdf(file_path):
            file_size = file_path.stat().st_size
            self.logger.info(f"PMID {pmid}: PDF已存在，跳过下载")
            return DownloadResult(
                pmid=pmid,
                status=DownloadStatus.SKIPPED,
                file_path=file_path,
                file_size=file_size,
                download_time=time.time() - start_time
            )
        
        # 尝试下载
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                pdf_url, finder_used, attempts, source_details = self._fetch_pdf_url_with_details(pmid)
                
                if not pdf_url:
                    error_msg = "未找到PDF链接"
                    if retry_count < self.max_retries:
                        self.logger.warning(f"PMID {pmid}: {error_msg}, 重试 {retry_count + 1}/{self.max_retries}")
                        retry_count += 1
                        time.sleep(1)  # 等待后重试
                        continue
                    else:
                        return DownloadResult(
                            pmid=pmid,
                            status=DownloadStatus.FAILED,
                            error_message=error_msg,
                            attempts=attempts,
                            source_details=source_details,
                            manual_download_suggestions=self._build_manual_suggestions(pmid, attempts),
                            retry_count=retry_count,
                            download_time=time.time() - start_time
                        )
                
                # 下载PDF
                if self._save_pdf(pdf_url, file_path):
                    file_size = file_path.stat().st_size
                    self.logger.info(f"PMID {pmid}: 下载成功，大小: {file_size}字节")
                    
                    return DownloadResult(
                        pmid=pmid,
                        status=DownloadStatus.SUCCESS,
                        file_path=file_path,
                        file_size=file_size,
                        downloader_used=finder_used,
                        source_url=pdf_url,
                        source_details=source_details,
                        attempts=attempts,
                        retry_count=retry_count,
                        download_time=time.time() - start_time
                    )
                else:
                    error_msg = "PDF保存失败"
                    if retry_count < self.max_retries:
                        self.logger.warning(f"PMID {pmid}: {error_msg}, 重试 {retry_count + 1}/{self.max_retries}")
                        retry_count += 1
                        time.sleep(1)
                        continue
                    else:
                        return DownloadResult(
                            pmid=pmid,
                            status=DownloadStatus.FAILED,
                            error_message=error_msg,
                            attempts=attempts,
                            source_details=source_details,
                            manual_download_suggestions=self._build_manual_suggestions(pmid, attempts),
                            retry_count=retry_count,
                            download_time=time.time() - start_time
                        )
                        
            except Exception as e:
                error_msg = str(e)
                if retry_count < self.max_retries:
                    self.logger.warning(f"PMID {pmid}: 下载失败: {error_msg}, 重试 {retry_count + 1}/{self.max_retries}")
                    retry_count += 1
                    time.sleep(1)
                else:
                    self.logger.error(f"PMID {pmid}: 下载最终失败: {error_msg}")
                    return DownloadResult(
                        pmid=pmid,
                        status=DownloadStatus.ERROR,
                        error_message=error_msg,
                        attempts=attempts if 'attempts' in locals() else [],
                        source_details=source_details if 'source_details' in locals() else {},
                        manual_download_suggestions=self._build_manual_suggestions(
                            pmid, attempts if 'attempts' in locals() else []
                        ),
                        retry_count=retry_count,
                        download_time=time.time() - start_time
                    )
        
        # 不应该执行到这里
        return DownloadResult(
            pmid=pmid,
            status=DownloadStatus.ERROR,
            error_message="未知错误",
            attempts=[],
            source_details={},
            manual_download_suggestions=self._build_manual_suggestions(pmid, []),
            retry_count=retry_count,
            download_time=time.time() - start_time
        )

    def _build_manual_suggestions(self, pmid: str, attempts: List[Dict[str, Any]]) -> List[str]:
        """构建人工下载建议（合法公开来源优先）。"""
        suggestions = [
            f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            f"https://www.ncbi.nlm.nih.gov/pmc/?term={pmid}",
            f"https://europepmc.org/search?query=EXT_ID:{pmid}%20SRC:MED",
        ]
        for attempt in attempts:
            landing = attempt.get("landing_url")
            if landing and landing not in suggestions:
                suggestions.append(landing)
        return suggestions[:10]
    
    def download_batch(self, pmids: List[str], 
                      error_file: Optional[str] = None) -> BatchResult:
        """
        批量下载PDF
        
        Args:
            pmids: PMID列表
            error_file: 错误记录文件路径
        
        Returns:
            BatchResult
        """
        total = len(pmids)
        results = []
        
        self.logger.info(f"开始批量下载 {total} 个PDF")
        
        # 创建错误文件
        if error_file:
            error_path = Path(error_file)
            ensure_dir(error_path.parent)
        else:
            error_path = self.output_dir / "failed_downloads.tsv"
        
        failed_pmids = []
        
        for i, pmid in enumerate(pmids, 1):
            self.logger.info(f"处理进度: {i}/{total} (PMID: {pmid})")
            
            result = self.download(pmid)
            results.append(result)
            
            # 记录失败
            if result.status in [DownloadStatus.FAILED, DownloadStatus.ERROR]:
                failed_pmids.append(pmid)
        
        # 统计结果
        succeeded = sum(1 for r in results if r.status == DownloadStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == DownloadStatus.FAILED)
        skipped = sum(1 for r in results if r.status == DownloadStatus.SKIPPED)
        error = sum(1 for r in results if r.status == DownloadStatus.ERROR)
        
        # 保存错误记录
        if failed_pmids:
            try:
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write("PMID\tError\n")
                    for pmid in failed_pmids:
                        # 查找对应的错误信息
                        for result in results:
                            if result.pmid == pmid and result.error_message:
                                f.write(f"{pmid}\t{result.error_message}\n")
                                break
                self.logger.info(f"错误记录已保存: {error_path}")
            except Exception as e:
                self.logger.error(f"保存错误记录失败: {e}")
        
        report_file, manual_checklist_file = self._save_download_system_report(results)

        batch_result = BatchResult(
            total=total,
            succeeded=succeeded,
            failed=failed + error,
            skipped=skipped,
            results=results,
            error_file=error_path if failed_pmids else None,
            report_file=report_file,
            manual_checklist_file=manual_checklist_file
        )
        
        self.logger.info(
            f"批量下载完成: {succeeded}成功, {skipped}跳过, {failed}失败, {error}错误"
        )
        
        return batch_result

    def _save_download_system_report(self, results: List[DownloadResult]) -> Tuple[Optional[Path], Optional[Path]]:
        """保存下载系统报告（JSON）与人工补充清单（CSV）。"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"download_system_report_{timestamp}.json"
            manual_file = self.output_dir / f"manual_download_checklist_{timestamp}.csv"

            source_stats: Dict[str, int] = {}
            rows = []
            for result in results:
                source_name = result.downloader_used or result.source_details.get("source") or "unknown"
                source_stats[source_name] = source_stats.get(source_name, 0) + 1
                rows.append({
                    "pmid": result.pmid,
                    "status": result.status.value if hasattr(result.status, "value") else str(result.status),
                    "file_path": str(result.file_path) if result.file_path else "",
                    "source": source_name,
                    "source_url": result.source_url or "",
                    "error_message": result.error_message or "",
                    "manual_suggestions": result.manual_download_suggestions,
                    "attempts": result.attempts,
                    "download_time": result.download_time,
                    "retry_count": result.retry_count,
                })

            report_data = {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total": len(results),
                "succeeded": sum(1 for r in results if r.status == DownloadStatus.SUCCESS),
                "failed": sum(1 for r in results if r.status in [DownloadStatus.FAILED, DownloadStatus.ERROR]),
                "skipped": sum(1 for r in results if r.status == DownloadStatus.SKIPPED),
                "source_distribution": source_stats,
                "results": rows,
            }

            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)

            # 人工下载清单
            with open(manual_file, "w", encoding="utf-8") as f:
                f.write("PMID,Status,Error,Manual_URL_1,Manual_URL_2,Manual_URL_3\n")
                for result in results:
                    if result.status in [DownloadStatus.SUCCESS, DownloadStatus.SKIPPED]:
                        continue
                    urls = (result.manual_download_suggestions or [])[:3]
                    while len(urls) < 3:
                        urls.append("")
                    safe_error = (result.error_message or "").replace(",", "，")
                    f.write(f"{result.pmid},{result.status.value},{safe_error},{urls[0]},{urls[1]},{urls[2]}\n")

            self.logger.info(f"下载系统报告已生成: {report_file}")
            self.logger.info(f"人工补充清单已生成: {manual_file}")
            return report_file, manual_file
        except Exception as e:
            self.logger.error(f"生成下载系统报告失败: {e}")
            return None, None
    
    def resume_failed_downloads(self, error_file: str) -> BatchResult:
        """
        恢复失败的下载
        
        Args:
            error_file: 错误记录文件路径
        
        Returns:
            BatchResult
        """
        error_path = Path(error_file)
        if not error_path.exists():
            self.logger.warning(f"错误文件不存在: {error_file}")
            return BatchResult()
        
        # 读取失败的PMID
        failed_pmids = []
        try:
            with open(error_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:  # 跳过标题行
                    parts = line.strip().split('\t')
                    if parts:
                        failed_pmids.append(parts[0])
        except Exception as e:
            self.logger.error(f"读取错误文件失败: {e}")
            return BatchResult()
        
        if not failed_pmids:
            self.logger.info("没有需要恢复的失败下载")
            return BatchResult()
        
        self.logger.info(f"恢复 {len(failed_pmids)} 个失败的下载")
        
        # 重新下载
        return self.download_batch(failed_pmids, error_file)
    
    def get_downloaded_files(self) -> List[Path]:
        """获取已下载的PDF文件列表"""
        if not self.output_dir.exists():
            return []
        
        pdf_files = list(self.output_dir.glob("*.pdf"))
        valid_files = [f for f in pdf_files if is_valid_pdf(f)]
        
        return valid_files
    
    def cleanup_invalid_files(self) -> int:
        """清理无效的PDF文件，返回清理的数量"""
        if not self.output_dir.exists():
            return 0
        
        pdf_files = list(self.output_dir.glob("*.pdf"))
        cleaned = 0
        
        for pdf_file in pdf_files:
            if not is_valid_pdf(pdf_file):
                try:
                    pdf_file.unlink()
                    self.logger.info(f"清理无效PDF: {pdf_file}")
                    cleaned += 1
                except Exception as e:
                    self.logger.error(f"清理文件失败 {pdf_file}: {e}")
        
        return cleaned
