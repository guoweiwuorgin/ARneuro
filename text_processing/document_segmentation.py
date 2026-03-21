"""
Document segmentation module for ARneuro.

This module handles parsing markdown files and extracting structured content.
"""

import os
import json
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
          - All tables
          - Table descriptions (preceding each table, up to 4 lines)
          - Table annotations (the 4 lines following each table)
          - Section titles and contents
        
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
        current_title = None
        current_content = []
        tables = []              # Store all tables
        tables_info = []         # Store preceding descriptions for each table
        tables_annotation = []   # Store the 4 lines following each table
        
        i = 0
        total_lines = len(lines)
        
        # Maintain a sliding window of up to 4 previous lines
        previous_lines = []
        
        while i < total_lines:
            line = lines[i].rstrip('\n')
            
            # Update sliding window
            if len(previous_lines) < 4:
                previous_lines.append(line)
            else:
                previous_lines.pop(0)
                previous_lines.append(line)
            
            # Detect section headers (starting with ## )
            if line.startswith('## '):
                if current_title:
                    # Store content of the previous section
                    document_structure[current_title] = '\n'.join(current_content).strip()
                current_title = line[3:].strip()  # new section title
                current_content = []
                i += 1
                continue
            
            # Detect table start
            if line.strip().startswith('|'):
                table_lines = []
                table_start_index = i
                
                # Collect all consecutive table rows
                while i < total_lines and (lines[i].strip().startswith('|') or lines[i].strip() == ''):
                    table_lines.append(lines[i].strip())
                    i += 1
                
                # Store complete table as one string
                table_str = '\n'.join(table_lines)
                tables.append(table_str)
                
                # Extract preceding description (up to 4 lines)
                info_lines = []
                look_back = 4
                current_index = table_start_index - 1
                
                while look_back > 0 and current_index >= 0:
                    current_line = lines[current_index].strip()
                    if current_line and ('table' in current_line.lower()):
                        # Found a relevant description line containing 'table'
                        info_lines.insert(0, current_line)
                        break
                    elif current_line:
                        # Found a non-empty line
                        info_lines.insert(0, current_line)
                    current_index -= 1
                    look_back -= 1
                
                if info_lines:
                    tables_info.append('\n'.join(info_lines))
                else:
                    tables_info.append('')
                
                # Extract the next 4 lines following the table
                annotation_lines = []
                annotation_count = 0
                
                while annotation_count < 3 and i < total_lines:
                    next_line = lines[i].rstrip('\n')
                    annotation_lines.append(next_line)
                    i += 1
                    annotation_count += 1
                
                tables_annotation.append('\n'.join(annotation_lines))
                
                # Place a placeholder in the content to indicate where the table was
                current_content.append(f'[TABLE: table_{len(tables)}]')
                continue
            
            # If not a header or table, treat it as regular content
            current_content.append(line)
            i += 1
            
        # After the loop, store content of the final section
        if current_title:
            document_structure[current_title] = '\n'.join(current_content).strip()
            
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