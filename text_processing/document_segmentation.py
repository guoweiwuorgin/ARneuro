"""
Document segmentation module for ARneuro.

This module handles parsing markdown files and extracting structured content.
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional
from ..core.logger import get_logger

logger = get_logger(__name__)


class DocumentSegmenter:
    """
    Segment markdown documents into structured sections, tables, and metadata.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the document segmenter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def parse_markdown_file(self, file_path: str) -> Tuple[Dict, List, List, List]:
        """
        Parse a Markdown file, extract different sections, and store:
          - All tables (Markdown & HTML)
          - Table descriptions (preceding each table, up to 4 valid lines)
          - Table annotations (the 4 valid lines following each table)
          - Section titles and contents
          - Page count and metadata
        
        Args:
            file_path (str): Path to the Markdown file.
            
        Returns:
            tuple: 
                - document_structure (dict): Maps section titles to their content.
                - tables (list): A list of all extracted tables (as strings).
                - tables_info (list): The preceding descriptions for each table.
                - tables_annotation (list): The next 4 lines after each table.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Markdown file not found: {file_path}")
        
        logger.info(f"Parsing markdown file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        document_structure = {}
        current_title = "Document_Root"
        current_content = []
        tables = []
        tables_info = []
        tables_annotation = []
        
        page_count = 0
        
        # 预编译正则，适应特定的MD文本特征
        page_header_re = re.compile(r'^##\s+Page\s+\d+', re.IGNORECASE)
        total_pages_re = re.compile(r'\*\*Total Pages:\*\*\s*(\d+)', re.IGNORECASE)
        section_re = re.compile(r'^(#{1,6})\s+(.*)')
        separator_re = re.compile(r'^={10,}$|^-{10,}$')
        html_table_start_re = re.compile(r'<table\b[^>]*>', re.IGNORECASE)
        html_table_end_re = re.compile(r'</table>', re.IGNORECASE)

        def get_table_info(start_idx: int) -> str:
            """向前追溯最多4行有效文本作为表格的描述（忽略空行和分页符）"""
            info = []
            look_back = 4
            idx = start_idx - 1
            while look_back > 0 and idx >= 0:
                l = lines[idx].strip()
                if not l or separator_re.match(l) or page_header_re.match(l):
                    idx -= 1
                    continue
                if 'table' in l.lower():
                    info.insert(0, l)
                    break
                else:
                    info.insert(0, l)
                look_back -= 1
                idx -= 1
            return '\n'.join(info)

        def get_table_annotation(start_idx: int) -> Tuple[str, int]:
            """向后获取最多4行有效文本作为表格注释，并返回结束行号"""
            ann = []
            count = 0
            idx = start_idx
            total = len(lines)
            while count < 4 and idx < total:
                l = lines[idx].rstrip('\n')
                stripped = l.strip()
                # 过滤掉干扰性的结构符，避免占掉提取的名额
                if not stripped or separator_re.match(stripped) or page_header_re.match(stripped):
                    idx += 1
                    continue
                # 如果遇到了新的段落标题，立即停止获取注释
                if section_re.match(stripped):
                    break
                ann.append(l)
                count += 1
                idx += 1
            return '\n'.join(ann), idx

        i = 0
        total_lines = len(lines)
        
        while i < total_lines:
            line = lines[i].rstrip('\n')
            stripped_line = line.strip()
            
            # 检测元数据行 "Total Pages: X"
            tp_match = total_pages_re.search(stripped_line)
            if tp_match:
                document_structure['Metadata_Total_Pages'] = int(tp_match.group(1))
                i += 1
                continue
                
            # 检测并跳过分页标记，从而使得被分页切断的自然段能完美拼接
            if page_header_re.match(stripped_line):
                page_count += 1
                i += 1
                continue
                
            # 跳过无意义的横杠分割线
            if separator_re.match(stripped_line):
                i += 1
                continue

            # 检测 HTML 类型的表格
            if html_table_start_re.match(stripped_line):
                table_lines = []
                info_str = get_table_info(i)
                
                while i < total_lines:
                    table_lines.append(lines[i].rstrip('\n'))
                    if html_table_end_re.search(lines[i]):
                        i += 1
                        break
                    i += 1
                    
                tables.append('\n'.join(table_lines))
                tables_info.append(info_str)
                
                ann_str, _ = get_table_annotation(i)
                tables_annotation.append(ann_str)
                
                current_content.append(f'[TABLE: table_{len(tables)}]')
                continue

            # 检测传统的 Markdown 格式表格
            if stripped_line.startswith('|'):
                table_lines = []
                info_str = get_table_info(i)
                
                while i < total_lines and lines[i].strip().startswith('|'):
                    table_lines.append(lines[i].strip())
                    i += 1
                    
                tables.append('\n'.join(table_lines))
                tables_info.append(info_str)
                
                ann_str, _ = get_table_annotation(i)
                tables_annotation.append(ann_str)
                
                current_content.append(f'[TABLE: table_{len(tables)}]')
                continue

            # 检测章节标题（由于之前排除了 page_header_re，所以不会混淆 Page X 标题）
            sec_match = section_re.match(stripped_line)
            if sec_match:
                # 存储上一个段落
                if current_title:
                    text = '\n'.join(current_content).strip()
                    if text:
                        document_structure[current_title] = text
                
                current_title = sec_match.group(2).strip()
                current_content = []
                i += 1
                continue
            
            # 存储常规正文内容
            if stripped_line != "":
                current_content.append(line)
                
            i += 1
            
        # 结尾边界处理
        if current_title:
            text = '\n'.join(current_content).strip()
            if text:
                document_structure[current_title] = text

        # 将独立的 Page 标记数量赋予特殊的 key，用以统计结果验证
        document_structure['Parsed_Page_Count'] = page_count
        
        logger.info(f"Parsed {len(document_structure)} sections and {len(tables)} tables")
        return document_structure, tables, tables_info, tables_annotation
    
    def validate_sections(self, document_structure: Dict) -> Dict:
        """
        Validate that required sections (Methods and Results) are present.
        
        Args:
            document_structure: Dictionary of section titles to content
            
        Returns:
            dict: Validation results with missing sections and warnings
        """
        required_sections = ['Methods', 'Results']
        section_titles = [title.lower() for title in document_structure.keys()]
        
        validation_result = {
            'has_methods': False,
            'has_results': False,
            'missing_sections': [],
            'warnings': []
        }
        
        # Check for Methods section
        methods_keywords = ['methods', 'materials and methods', 'experimental methods', 'methodology']
        for keyword in methods_keywords:
            if any(keyword in title.lower() for title in document_structure.keys()):
                validation_result['has_methods'] = True
                break
        
        # Check for Results section
        results_keywords = ['results', 'findings']
        for keyword in results_keywords:
            if any(keyword in title.lower() for title in document_structure.keys()):
                validation_result['has_results'] = True
                break
        
        # Record missing sections
        if not validation_result['has_methods']:
            validation_result['missing_sections'].append('Methods')
        if not validation_result['has_results']:
            validation_result['missing_sections'].append('Results')
        
        # Add warnings if sections are missing
        if validation_result['missing_sections']:
            validation_result['warnings'].append(
                f"Missing required sections: {', '.join(validation_result['missing_sections'])}"
            )
        
        return validation_result
    
    def save_segmentation_results(self, 
                                 document_structure: Dict, 
                                 tables: List, 
                                 tables_info: List, 
                                 tables_annotation: List,
                                 output_dir: str,
                                 filename: str = "segmentation_results.json") -> str:
        """
        Save segmentation results to JSON file.
        
        Args:
            document_structure: Dictionary of section titles to content
            tables: List of extracted tables
            tables_info: List of table descriptions
            tables_annotation: List of table annotations
            output_dir: Directory to save results
            filename: Output filename
            
        Returns:
            str: Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'document_structure': document_structure,
            'tables': tables,
            'tables_info': tables_info,
            'tables_annotation': tables_annotation,
            'metadata': {
                'num_sections': len(document_structure),
                'num_tables': len(tables),
                'validation': self.validate_sections(document_structure)
            }
        }
        
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Segmentation results saved to: {output_path}")
        return output_path