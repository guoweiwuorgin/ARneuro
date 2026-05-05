"""
Text reorganization module for ARneuro.

This module handles reorganizing document content into predefined categories.
"""

import json
from typing import Dict, List, Tuple, Optional
from core.logger import get_logger

logger = get_logger(__name__)


class TextReorganizer:
    """
    Reorganize document content into predefined categories.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the text reorganizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def reorganize_document(self, 
                           document_structure: Dict[str, str], 
                           section_mapping: Dict[str, str]) -> Tuple[Dict, Dict]:
        """
        Rearrange text from the original document structure into predefined categories,
        and store unassigned text in an 'Unassigned' field.
        
        Args:
            document_structure: Dictionary where each key is a section title 
                               and the value is the corresponding text content.
            section_mapping: Mapping from section titles to predefined categories.
            
        Returns:
            tuple:
                reorganized_structure (dict): Dictionary with each predefined 
                                              category as a key and its 
                                              corresponding text content as a value,
                                              plus an 'Unassigned' key for text that 
                                              does not match any category.
                metadata (dict): Dictionary containing word/character statistics 
                                 for each category, along with overall totals.
        """
        # Predefined categories of interest
        categories = ['Introduction', 'Methods', 'Results', 'Discussion', 'References']
        
        # Initialize a dictionary with empty strings for each category,
        # plus one additional field for unassigned text
        reorganized_structure = {category: '' for category in categories}
        reorganized_structure['Unassigned'] = ''
        
        # Assign each section of document_structure to the corresponding category
        # or to 'Unassigned' if it is not mapped or not recognized as a valid category
        for section_title, content in document_structure.items():
            category = section_mapping.get(section_title)
            if category in categories:
                reorganized_structure[category] += content + '\n\n'
            else:
                reorganized_structure['Unassigned'] += content + '\n\n'
        
        # Calculate metadata for each category
        metadata = self._calculate_metadata(reorganized_structure, document_structure)
        
        logger.info(f"Reorganized document into {len(categories)} categories")
        return reorganized_structure, metadata
    
    def _calculate_metadata(self, 
                           reorganized_structure: Dict[str, str], 
                           original_structure: Dict[str, str]) -> Dict:
        """
        Calculate word/character statistics for each category.
        
        Args:
            reorganized_structure: Reorganized document structure
            original_structure: Original document structure
            
        Returns:
            dict: Metadata with word/character counts
        """
        metadata = {}
        
        # Calculate stats for each category (including 'Unassigned')
        for category, content in reorganized_structure.items():
            num_words = len(content.split())
            num_chars = len(content)
            metadata[category] = {
                'num_words': num_words,
                'num_chars': num_chars
            }
        
        # Calculate overall word/character totals for the entire document
        total_content = '\n\n'.join(original_structure.values())
        total_num_words = len(total_content.split())
        total_num_chars = len(total_content)
        
        metadata['Total_Words'] = total_num_words
        metadata['Total_Chars'] = total_num_chars
        
        # Calculate how many words have not been counted in the predefined categories
        # (excluding 'Unassigned' from the counted words to isolate missing coverage if needed)
        counted_words = sum(
            info['num_words']
            for key, info in metadata.items()
            if isinstance(info, dict) and key != 'Unassigned'
        )
        
        # The difference between the total words and counted words is considered unassigned
        uncounted_words = total_num_words - counted_words
        metadata['Uncounted_Words'] = uncounted_words
        
        return metadata
    
    def validate_reorganization(self, 
                               reorganized_structure: Dict[str, str], 
                               metadata: Dict) -> Dict:
        """
        Validate that required sections (Methods and Results) have content.
        
        Args:
            reorganized_structure: Reorganized document structure
            metadata: Document metadata
            
        Returns:
            dict: Validation results
        """
        validation_result = {
            'has_methods': False,
            'has_results': False,
            'missing_content': [],
            'warnings': []
        }
        
        # Check Methods section
        if 'Methods' in reorganized_structure and reorganized_structure['Methods'].strip():
            validation_result['has_methods'] = True
        
        # Check Results section
        if 'Results' in reorganized_structure and reorganized_structure['Results'].strip():
            validation_result['has_results'] = True
        
        # Record missing content
        if not validation_result['has_methods']:
            validation_result['missing_content'].append('Methods')
        if not validation_result['has_results']:
            validation_result['missing_content'].append('Results')
        
        # Add warnings if content is missing
        if validation_result['missing_content']:
            validation_result['warnings'].append(
                f"Missing content in required sections: {', '.join(validation_result['missing_content'])}"
            )
        
        # Check if too much content is unassigned
        if 'Unassigned' in metadata and metadata['Unassigned']['num_words'] > 0:
            unassigned_percentage = (metadata['Unassigned']['num_words'] / metadata['Total_Words']) * 100
            if unassigned_percentage > 30:  # More than 30% unassigned
                validation_result['warnings'].append(
                    f"High percentage of unassigned content: {unassigned_percentage:.1f}%"
                )
        
        return validation_result
    
    def save_reorganized_document(self, 
                                 reorganized_structure: Dict[str, str], 
                                 metadata: Dict,
                                 validation: Dict,
                                 output_dir: str,
                                 filename: str = "reorganized_document.json") -> str:
        """
        Save reorganized document to JSON file.
        
        Args:
            reorganized_structure: Reorganized document structure
            metadata: Document metadata
            validation: Validation results
            output_dir: Directory to save results
            filename: Output filename
            
        Returns:
            str: Path to saved file
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'reorganized_structure': reorganized_structure,
            'metadata': metadata,
            'validation': validation
        }
        
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Reorganized document saved to: {output_path}")
        return output_path
    
    def extract_tables_with_context(self, 
                                   tables: List[str], 
                                   tables_info: List[str], 
                                   tables_annotation: List[str]) -> List[Dict]:
        """
        Extract tables with their surrounding context.
        
        Args:
            tables: List of extracted tables
            tables_info: List of table descriptions
            tables_annotation: List of table annotations
            
        Returns:
            list: List of dictionaries with table data and context
        """
        table_data = []
        
        for i, (table, info, annotation) in enumerate(zip(tables, tables_info, tables_annotation)):
            table_entry = {
                'table_id': f'table_{i+1}',
                'table_content': table,
                'preceding_context': info,
                'following_context': annotation,
                'num_rows': len([line for line in table.split('\n') if line.strip().startswith('|')]),
                'num_columns': self._count_table_columns(table)
            }
            table_data.append(table_entry)
        
        return table_data
    
    def _count_table_columns(self, table: str) -> int:
        """
        Count the number of columns in a markdown table.
        
        Args:
            table: Markdown table string
            
        Returns:
            int: Number of columns
        """
        lines = table.strip().split('\n')
        if not lines:
            return 0
        
        # Find the first row that looks like a table header
        for line in lines:
            if line.strip().startswith('|'):
                # Count pipe characters to estimate columns
                # In markdown tables, columns = pipe_count - 1
                pipe_count = line.count('|')
                return max(0, pipe_count - 1)
        
        return 0
