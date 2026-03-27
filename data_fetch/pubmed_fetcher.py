"""
PubMed数据获取模块
"""

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd

from ..core import get_module_logger
from ..config.config_manager import get_config
from ..utils.file_utils import ensure_dir
from ..utils.validation import validate_pmid, validate_csv_file


@dataclass
class PubMedRecord:
    """PubMed记录"""
    pmid: str
    title: Optional[str] = None
    authors: Optional[str] = None
    citation: Optional[str] = None
    first_author: Optional[str] = None
    journal: Optional[str] = None
    publication_year: Optional[str] = None
    create_date: Optional[str] = None
    pmcid: Optional[str] = None
    nihms_id: Optional[str] = None
    abstract: Optional[str] = None
    keywords: Optional[str] = None
    doi: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'PMID': self.pmid,
            'Title': self.title or '',
            'Authors': self.authors or '',
            'Citation': self.citation or '',
            'First Author': self.first_author or '',
            'Journal/Book': self.journal or '',
            'Publication Year': self.publication_year or '',
            'Create Date': self.create_date or '',
            'PMCID': self.pmcid or '',
            'NIHMS ID': self.nihms_id or '',
            'Abstract': self.abstract or '',
            'Keywords': self.keywords or '',
            'DOI': self.doi or ''
        }


class PubMedFetcher:
    """PubMed数据获取器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化PubMed获取器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = get_config(config_path)
        self.logger = get_module_logger("pubmed_fetcher")
        
        # 配置参数
        self.cache_dir = Path(self.config.get("paths.cache_dir", "./cache"))
        ensure_dir(self.cache_dir)
        
        self.logger.info("PubMed获取器初始化完成")
    
    def fetch_from_csv(self, csv_file: Path, required_columns: List[str] = None) -> List[PubMedRecord]:
        """
        从CSV文件读取PubMed记录
        
        Args:
            csv_file: CSV文件路径
            required_columns: 必需的列名
        
        Returns:
            PubMed记录列表
        """
        # 验证CSV文件
        valid, error = validate_csv_file(csv_file, required_columns)
        if not valid:
            self.logger.error(f"CSV文件验证失败: {error}")
            return []
        
        records = []
        
        try:
            # 尝试使用pandas读取
            try:
                df = pd.read_csv(csv_file)
            except:
                # 尝试TSV格式
                df = pd.read_csv(csv_file, sep='\t')
            
            # 标准化列名
            df.columns = df.columns.str.strip()
            
            # 检查必需列
            if required_columns:
                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    self.logger.error(f"缺少必需列: {missing}")
                    return []
            
            # 处理每一行
            for _, row in df.iterrows():
                try:
                    record = self._create_record_from_row(row)
                    if record:
                        records.append(record)
                except Exception as e:
                    pmid = str(row.get('PMID', 'unknown'))
                    self.logger.warning(f"处理PMID {pmid}失败: {e}")
            
            self.logger.info(f"从CSV读取 {len(records)} 条记录")
            
        except Exception as e:
            self.logger.error(f"读取CSV文件失败: {e}")
        
        return records

    def parse_csv_with_management_report(
        self,
        csv_file: Path,
        pmid_column: str = "PMID",
    ) -> Tuple[List[PubMedRecord], Dict[str, Any]]:
        """
        解析CSV并生成文档管理系统报告（用于下载前检查）。

        Returns:
            (有效记录列表, 报告字典)
        """
        records: List[PubMedRecord] = []
        invalid_rows: List[Dict[str, Any]] = []
        duplicate_pmids: List[str] = []
        seen_pmids = set()

        valid, error = validate_csv_file(csv_file, [pmid_column])
        if not valid:
            report = {
                "source_csv": str(csv_file),
                "status": "invalid_csv",
                "error": error,
                "total_rows": 0,
                "valid_records": 0,
                "invalid_rows": [],
                "duplicate_pmids": [],
            }
            return [], report

        try:
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                df = pd.read_csv(csv_file, sep="\t")

            df.columns = df.columns.str.strip()
            total_rows = len(df)

            for idx, row in df.iterrows():
                raw_pmid = str(row.get(pmid_column, "")).strip()
                row_no = int(idx) + 2  # 加上header行

                valid_pmid, pmid_error = validate_pmid(raw_pmid)
                if not valid_pmid:
                    invalid_rows.append({
                        "row": row_no,
                        "pmid": raw_pmid,
                        "reason": pmid_error,
                    })
                    continue

                if raw_pmid in seen_pmids:
                    duplicate_pmids.append(raw_pmid)
                    continue

                seen_pmids.add(raw_pmid)
                record = self._create_record_from_row(row)
                if record:
                    records.append(record)
                else:
                    invalid_rows.append({
                        "row": row_no,
                        "pmid": raw_pmid,
                        "reason": "无法从该行构建PubMedRecord",
                    })

            report = {
                "source_csv": str(csv_file),
                "status": "ok",
                "total_rows": total_rows,
                "valid_records": len(records),
                "invalid_count": len(invalid_rows),
                "duplicate_count": len(duplicate_pmids),
                "invalid_rows": invalid_rows,
                "duplicate_pmids": duplicate_pmids,
                "valid_pmids": [r.pmid for r in records],
            }
            return records, report
        except Exception as e:
            report = {
                "source_csv": str(csv_file),
                "status": "parse_error",
                "error": str(e),
                "total_rows": 0,
                "valid_records": 0,
                "invalid_rows": [],
                "duplicate_pmids": [],
            }
            return [], report

    def save_management_report(self, report: Dict[str, Any], output_file: Path) -> bool:
        """保存CSV解析后的文档管理报告。"""
        try:
            ensure_dir(output_file.parent)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self.logger.info(f"文档管理报告已保存: {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"保存文档管理报告失败: {e}")
            return False
    
    def _create_record_from_row(self, row: pd.Series) -> Optional[PubMedRecord]:
        """从数据行创建PubMed记录"""
        pmid = str(row.get('PMID', '')).strip()
        
        # 验证PMID
        valid, error = validate_pmid(pmid)
        if not valid:
            self.logger.warning(f"无效PMID: {pmid} - {error}")
            return None
        
        # 创建记录
        record = PubMedRecord(pmid=pmid)
        
        # 映射字段
        field_mappings = {
            'title': ['Title', 'Article Title', 'title'],
            'authors': ['Authors', 'Author List', 'authors'],
            'citation': ['Citation', 'citation'],
            'first_author': ['First Author', 'first_author'],
            'journal': ['Journal/Book', 'Journal', 'journal'],
            'publication_year': ['Publication Year', 'Year', 'year'],
            'create_date': ['Create Date', 'Date', 'create_date'],
            'pmcid': ['PMCID', 'pmcid'],
            'nihms_id': ['NIHMS ID', 'nihms_id'],
            'abstract': ['Abstract', 'abstract'],
            'keywords': ['Keywords', 'keywords'],
            'doi': ['DOI', 'doi']
        }
        
        for field_name, possible_columns in field_mappings.items():
            value = None
            for col in possible_columns:
                if col in row and pd.notna(row[col]):
                    value = str(row[col]).strip()
                    break
            
            if value:
                setattr(record, field_name, value)
        
        return record
    
    def fetch_abstracts(self, pmids: List[str], delay: float = 0.1) -> List[PubMedRecord]:
        """
        从PubMed获取摘要（需要metapub库）
        
        Args:
            pmids: PMID列表
            delay: 请求延迟（秒）
        
        Returns:
            PubMed记录列表
        """
        try:
            from metapub import PubMedFetcher
        except ImportError:
            self.logger.error("需要安装metapub库: pip install metapub")
            return []
        
        records = []
        fetcher = PubMedFetcher()
        
        for i, pmid in enumerate(pmids, 1):
            try:
                # 验证PMID
                valid, error = validate_pmid(pmid)
                if not valid:
                    self.logger.warning(f"跳过无效PMID {pmid}: {error}")
                    continue
                
                self.logger.info(f"获取摘要进度: {i}/{len(pmids)} (PMID: {pmid})")
                
                # 获取文章信息
                article = fetcher.article_by_pmid(pmid)
                
                # 创建记录
                record = PubMedRecord(pmid=pmid)
                
                # 填充字段
                if hasattr(article, 'title'):
                    record.title = article.title
                
                if hasattr(article, 'authors'):
                    record.authors = ', '.join(article.authors) if article.authors else None
                
                if hasattr(article, 'citation'):
                    record.citation = article.citation
                
                if hasattr(article, 'journal'):
                    record.journal = article.journal
                
                if hasattr(article, 'year'):
                    record.publication_year = str(article.year)
                
                if hasattr(article, 'abstract'):
                    record.abstract = article.abstract
                
                if hasattr(article, 'doi'):
                    record.doi = article.doi
                
                if hasattr(article, 'pmc'):
                    record.pmcid = article.pmc
                
                records.append(record)
                
                # 延迟以避免请求过快
                if i < len(pmids):
                    time.sleep(delay)
                    
            except Exception as e:
                self.logger.warning(f"获取PMID {pmid}摘要失败: {e}")
                # 创建基本记录
                record = PubMedRecord(pmid=pmid)
                records.append(record)
        
        self.logger.info(f"获取 {len(records)} 条记录的摘要")
        return records
    
    def save_records_to_csv(self, records: List[PubMedRecord], output_file: Path) -> bool:
        """
        保存PubMed记录到CSV文件
        
        Args:
            records: PubMed记录列表
            output_file: 输出文件路径
        
        Returns:
            是否成功
        """
        if not records:
            self.logger.warning("没有记录可保存")
            return False
        
        try:
            ensure_dir(output_file.parent)
            
            # 转换为字典列表
            data = [record.to_dict() for record in records]
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 保存到CSV
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            self.logger.info(f"保存 {len(records)} 条记录到 {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存记录失败: {e}")
            return False
    
    def extract_pmids_from_csv(self, csv_file: Path, pmid_column: str = 'PMID') -> List[str]:
        """
        从CSV文件提取PMID列表
        
        Args:
            csv_file: CSV文件路径
            pmid_column: PMID列名
        
        Returns:
            PMID列表
        """
        pmids = []
        
        try:
            df = pd.read_csv(csv_file)
            
            if pmid_column not in df.columns:
                self.logger.error(f"CSV文件中没有 {pmid_column} 列")
                return []
            
            # 提取PMID
            for pmid in df[pmid_column].dropna().astype(str):
                pmid = pmid.strip()
                valid, _ = validate_pmid(pmid)
                if valid:
                    pmids.append(pmid)
                else:
                    self.logger.warning(f"跳过无效PMID: {pmid}")
            
            self.logger.info(f"从CSV提取 {len(pmids)} 个有效PMID")
            
        except Exception as e:
            self.logger.error(f"提取PMID失败: {e}")
        
        return pmids
    
    def validate_records(self, records: List[PubMedRecord]) -> Dict[str, Any]:
        """
        验证PubMed记录
        
        Args:
            records: PubMed记录列表
        
        Returns:
            验证结果统计
        """
        total = len(records)
        
        if total == 0:
            return {
                'total': 0,
                'valid': 0,
                'invalid': 0,
                'with_abstract': 0,
                'with_title': 0,
                'with_journal': 0
            }
        
        valid_count = 0
        with_abstract = 0
        with_title = 0
        with_journal = 0
        
        for record in records:
            # 验证PMID
            valid, _ = validate_pmid(record.pmid)
            if valid:
                valid_count += 1
            
            # 统计信息
            if record.abstract and record.abstract.strip():
                with_abstract += 1
            
            if record.title and record.title.strip():
                with_title += 1
            
            if record.journal and record.journal.strip():
                with_journal += 1
        
        return {
            'total': total,
            'valid': valid_count,
            'invalid': total - valid_count,
            'with_abstract': with_abstract,
            'with_title': with_title,
            'with_journal': with_journal,
            'abstract_coverage': f"{(with_abstract/total)*100:.1f}%" if total > 0 else "0%",
            'title_coverage': f"{(with_title/total)*100:.1f}%" if total > 0 else "0%",
            'journal_coverage': f"{(with_journal/total)*100:.1f}%" if total > 0 else "0%"
        }
    
    def create_sample_csv(self, output_file: Path, num_samples: int = 10) -> bool:
        """
        创建示例CSV文件
        
        Args:
            output_file: 输出文件路径
            num_samples: 样本数量
        
        Returns:
            是否成功
        """
        try:
            # 示例数据
            sample_data = []
            
            for i in range(1, num_samples + 1):
                sample_data.append({
                    'PMID': f'3000000{i}',
                    'Title': f'Sample Article Title {i}',
                    'Authors': f'Author A{i}, Author B{i}, Author C{i}',
                    'Citation': f'J Sample Res. {2020+i};{i}({i}):{i*100}-{i*100+50}',
                    'First Author': f'Author A{i}',
                    'Journal/Book': 'Journal of Sample Research',
                    'Publication Year': str(2020 + i),
                    'Create Date': f'2020-0{i}-0{i}',
                    'PMCID': f'PMC{i}00000{i}',
                    'NIHMS ID': f'NIHMS{i}00000{i}',
                    'Abstract': f'This is a sample abstract for article {i}. It demonstrates the structure of a typical abstract.',
                    'Keywords': f'sample{i}, test{i}, example{i}',
                    'DOI': f'10.1000/sample.{i}'
                })
            
            # 创建DataFrame并保存
            df = pd.DataFrame(sample_data)
            ensure_dir(output_file.parent)
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"创建示例CSV文件: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建示例CSV失败: {e}")
            return False
