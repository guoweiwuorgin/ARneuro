"""
Table extractor module for ARneuro.

This module extracts tables from markdown documents.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from core.logger import get_logger

logger = get_logger(__name__)


class TableExtractor:
    """
    Extract tables from markdown documents.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the table extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def extract_tables_from_markdown(self, 
                                    markdown_file: str,
                                    context_lines: int = 5) -> List[Dict]:
        """
        Extract tables from markdown file with context.
        
        Args:
            markdown_file: Path to markdown file
            context_lines: Number of context lines before and after table
            
        Returns:
            List[Dict]: List of table information dictionaries
        """
        if not os.path.exists(markdown_file):
            raise FileNotFoundError(f"Markdown file not found: {markdown_file}")
        
        logger.info(f"Extracting tables from: {markdown_file}")
        
        with open(markdown_file, 'r', encoding='utf-8') as f:
            md_text = f.read()
        
        return self.extract_tables_from_text(md_text, context_lines)
    
    def extract_tables_from_text(self, 
                                md_text: str,
                                context_lines: int = 5) -> List[Dict]:
        """
        Extract tables from markdown text with context.
        
        Args:
            md_text: Markdown text
            context_lines: Number of context lines before and after table
            
        Returns:
            List[Dict]: List of table information dictionaries
        """
        import re
        
        lines = md_text.split('\n')
        tables_info = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if line starts a markdown table (starts and ends with pipe)
            if line.strip().startswith('|') and line.strip().endswith('|'):
                table_start = i
                table_lines = []
                
                # Collect all consecutive table rows
                while i < len(lines) and (lines[i].strip().startswith('|') or lines[i].strip() == ''):
                    table_lines.append(lines[i])
                    i += 1
                
                # Join table lines
                table_text = '\n'.join(table_lines)
                
                # Get context before table
                before_start = max(0, table_start - context_lines)
                before_context = '\n'.join(lines[before_start:table_start])
                
                # Get context after table
                after_end = min(len(lines), i + context_lines)
                after_context = '\n'.join(lines[i:after_end])
                
                # Full context
                full_context = f"{before_context}\n[TABLE {len(tables_info) + 1}]\n{after_context}"
                
                # Parse table structure
                table_structure = self._parse_table_structure(table_text)
                
                tables_info.append({
                    'table_index': len(tables_info),
                    'table_text': table_text,
                    'table_structure': table_structure,
                    'before_context': before_context,
                    'after_context': after_context,
                    'full_context': full_context,
                    'num_rows': table_structure['num_rows'],
                    'num_columns': table_structure['num_columns'],
                    'has_header': table_structure['has_header'],
                    'header_row': table_structure['header_row']
                })
            else:
                i += 1
        
        logger.info(f"Extracted {len(tables_info)} tables from text")
        return tables_info
    
    def _parse_table_structure(self, table_text: str) -> Dict:
        """
        Parse table structure from markdown text.
        
        Args:
            table_text: Markdown table text
            
        Returns:
            Dict: Table structure information
        """
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        if not lines:
            return {
                'num_rows': 0,
                'num_columns': 0,
                'has_header': False,
                'header_row': None,
                'data_rows': []
            }
        
        # Find separator row (contains only dashes, colons, and pipes)
        separator_idx = -1
        for i, line in enumerate(lines):
            # Remove pipes and check if only dashes, colons, and spaces remain
            clean_line = line.replace('|', '').strip()
            if re.match(r'^[:-\s]+$', clean_line):
                separator_idx = i
                break
        
        # Parse headers
        has_header = separator_idx > 0
        header_row = None
        
        if has_header:
            header_line = lines[separator_idx - 1]
            headers = [cell.strip() for cell in header_line.split('|') if cell.strip()]
        else:
            # Use first row as headers
            header_line = lines[0]
            headers = [cell.strip() for cell in header_line.split('|') if cell.strip()]
        
        # Parse data rows
        data_start = separator_idx + 1 if separator_idx >= 0 else 1
        data_rows = []
        
        for i in range(data_start, len(lines)):
            if '|' in lines[i]:
                cells = [cell.strip() for cell in lines[i].split('|') if cell.strip()]
                if len(cells) == len(headers):
                    data_rows.append(cells)
        
        return {
            'num_rows': len(data_rows),
            'num_columns': len(headers),
            'has_header': has_header,
            'header_row': headers,
            'data_rows': data_rows
        }
    
    def filter_tables_by_content(self, 
                                tables_info: List[Dict],
                                keywords: List[str]) -> List[Dict]:
        """
        Filter tables by content keywords.
        
        Args:
            tables_info: List of table information dictionaries
            keywords: List of keywords to search for
            
        Returns:
            List[Dict]: Filtered tables
        """
        filtered_tables = []
        
        for table_info in tables_info:
            table_text = table_info['table_text'].lower()
            context_text = table_info['full_context'].lower()
            
            # Check if any keyword appears in table or context
            for keyword in keywords:
                if keyword.lower() in table_text or keyword.lower() in context_text:
                    filtered_tables.append(table_info)
                    break
        
        logger.info(f"Filtered {len(filtered_tables)} tables matching keywords: {keywords}")
        return filtered_tables
    
    def categorize_tables(self, tables_info: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize tables by content type.
        
        Args:
            tables_info: List of table information dictionaries
            
        Returns:
            Dict: Tables categorized by type
        """
        categories = {
            'brain_activation': [],
            'demographic': [],
            'statistical': [],
            'methodological': [],
            'other': []
        }
        
        brain_keywords = ['brain', 'activation', 'coordinate', 'mni', 'talairach', 'x', 'y', 'z', 'voxel', 'cluster']
        demo_keywords = ['age', 'gender', 'sex', 'participant', 'subject', 'demographic']
        stat_keywords = ['p-value', 'p value', 't-value', 't value', 'f-value', 'f value', 'statistic', 'correlation', 'regression']
        method_keywords = ['method', 'procedure', 'protocol', 'parameter', 'setting', 'equipment']
        
        for table_info in tables_info:
            table_text = table_info['table_text'].lower()
            context_text = table_info['full_context'].lower()
            all_text = f"{table_text} {context_text}"
            
            categorized = False
            
            # Check for brain activation tables
            for keyword in brain_keywords:
                if keyword in all_text:
                    categories['brain_activation'].append(table_info)
                    categorized = True
                    break
            
            if not categorized:
                # Check for demographic tables
                for keyword in demo_keywords:
                    if keyword in all_text:
                        categories['demographic'].append(table_info)
                        categorized = True
                        break
            
            if not categorized:
                # Check for statistical tables
                for keyword in stat_keywords:
                    if keyword in all_text:
                        categories['statistical'].append(table_info)
                        categorized = True
                        break
            
            if not categorized:
                # Check for methodological tables
                for keyword in method_keywords:
                    if keyword in all_text:
                        categories['methodological'].append(table_info)
                        categorized = True
                        break
            
            if not categorized:
                categories['other'].append(table_info)
        
        # Log categorization results
        for category, tables in categories.items():
            logger.info(f"Category '{category}': {len(tables)} tables")
        
        return categories
    
    def save_table_extraction_results(self, 
                                     tables_info: List[Dict],
                                     output_dir: str,
                                     filename: str = "table_extraction.json") -> str:
        """
        Save table extraction results to JSON file.
        
        Args:
            tables_info: List of table information dictionaries
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            str: Path to saved file
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare results for JSON serialization
        results = []
        for table_info in tables_info:
            # Create a serializable copy
            table_result = {
                'table_index': table_info['table_index'],
                'table_text': table_info['table_text'],
                'table_structure': table_info['table_structure'],
                'before_context': table_info['before_context'],
                'after_context': table_info['after_context'],
                'num_rows': table_info['num_rows'],
                'num_columns': table_info['num_columns'],
                'has_header': table_info['has_header']
            }
            results.append(table_result)
        
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_tables': len(results),
                'tables': results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved table extraction results to: {output_path}")
        return output_path
    
    def export_tables_to_csv(self, 
                            tables_info: List[Dict],
                            output_dir: str) -> List[str]:
        """
        Export tables to CSV files.
        
        Args:
            tables_info: List of table information dictionaries
            output_dir: Output directory
            
        Returns:
            List[str]: List of paths to saved CSV files
        """
        import os
        import pandas as pd
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        for table_info in tables_info:
            table_idx = table_info['table_index']
            structure = table_info['table_structure']
            
            # Create DataFrame from table structure
            if structure['data_rows']:
                df = pd.DataFrame(
                    structure['data_rows'],
                    columns=structure['header_row'] if structure['has_header'] else None
                )
                
                # Save to CSV
                csv_path = os.path.join(output_dir, f"table_{table_idx + 1}.csv")
                df.to_csv(csv_path, index=False)
                saved_files.append(csv_path)
                
                # Save metadata
                metadata = {
                    'table_index': table_idx,
                    'num_rows': len(df),
                    'num_columns': len(df.columns),
                    'columns': list(df.columns),
                    'context_before': table_info['before_context'],
                    'context_after': table_info['after_context']
                }
                
                metadata_path = os.path.join(output_dir, f"table_{table_idx + 1}_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(saved_files)} tables to CSV files in: {output_dir}")
        return saved_files