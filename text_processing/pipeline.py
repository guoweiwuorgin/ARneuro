"""
Text processing pipeline for ARneuro.

This module provides a complete pipeline for document segmentation and reorganization.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from ..core.logger import get_logger
from ..core.llm_client import LLMClientManager
from .document_segmentation import DocumentSegmenter
from .heading_classifier import HeadingClassifier
from .text_reorganizer import TextReorganizer

logger = get_logger(__name__)


class TextProcessingPipeline:
    """
    Complete text processing pipeline for document segmentation and reorganization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the text processing pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.segmenter = DocumentSegmenter(config)
        self.classifier = HeadingClassifier(config=config)
        self.reorganizer = TextReorganizer(config)
        
        # Initialize LLM client manager if API keys are provided
        self.llm_manager = None
        if any(key in config for key in ['deepseek_api_key', 'openai_api_key', 'glm_api_key', 'huoshan_api_key']):
            self.llm_manager = LLMClientManager(config)
            if self.llm_manager:
                self.classifier.set_llm_client(self.llm_manager)
    
    def process_document(self, 
                        markdown_file: str,
                        output_dir: Optional[str] = None,
                        use_llm: bool = True,
                        llm_client_type: str = 'deepseek') -> Dict:
        """
        Process a markdown document through the complete pipeline.
        
        Args:
            markdown_file: Path to markdown file
            output_dir: Directory to save results (default: same as input file)
            use_llm: Whether to use LLM for classification
            llm_client_type: Type of LLM client to use
            
        Returns:
            dict: Complete processing results
        """
        logger.info(f"Processing document: {markdown_file}")
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(markdown_file)
        
        # Step 1: Segment the document
        document_structure, tables, tables_info, tables_annotation = self.segmenter.parse_markdown_file(markdown_file)
        
        # Step 2: Validate required sections
        validation = self.segmenter.validate_sections(document_structure)
        
        # Step 3: Classify headings
        section_titles = list(document_structure.keys())
        
        if use_llm and self.llm_manager:
            try:
                # Use LLM for classification
                headings_str = ','.join(section_titles)
                classification = self.llm_manager.classify_headings_with_llm(
                    headings_str, 
                    client_type=llm_client_type
                )
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}, using rule-based classification")
                classification = self.classifier.classify_headings(section_titles)
        else:
            # Use rule-based classification
            classification = self.classifier.classify_headings(section_titles)
        
        # Step 4: Map sections to categories
        section_mapping = self.classifier.map_sections_to_categories(section_titles, classification)
        
        # Step 5: Reorganize document
        reorganized_structure, metadata = self.reorganizer.reorganize_document(
            document_structure, 
            section_mapping
        )
        
        # Step 6: Validate reorganization
        reorg_validation = self.reorganizer.validate_reorganization(reorganized_structure, metadata)
        
        # Step 7: Extract tables with context
        table_data = self.reorganizer.extract_tables_with_context(tables, tables_info, tables_annotation)
        
        # Step 8: Save results
        results = self._save_results(
            markdown_file,
            output_dir,
            document_structure,
            tables,
            tables_info,
            tables_annotation,
            classification,
            section_mapping,
            reorganized_structure,
            metadata,
            validation,
            reorg_validation,
            table_data
        )
        
        logger.info(f"Document processing completed: {markdown_file}")
        return results
    
    def _save_results(self,
                     markdown_file: str,
                     output_dir: str,
                     document_structure: Dict,
                     tables: List,
                     tables_info: List,
                     tables_annotation: List,
                     classification: Dict,
                     section_mapping: Dict,
                     reorganized_structure: Dict,
                     metadata: Dict,
                     validation: Dict,
                     reorg_validation: Dict,
                     table_data: List) -> Dict:
        """
        Save all processing results to files.
        
        Returns:
            dict: Paths to saved files
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(markdown_file))[0]
        
        # Save segmentation results
        segmentation_file = self.segmenter.save_segmentation_results(
            document_structure,
            tables,
            tables_info,
            tables_annotation,
            output_dir,
            f"{base_name}_segmentation.json"
        )
        
        # Save classification results
        classification_file = os.path.join(output_dir, f"{base_name}_classification.json")
        with open(classification_file, 'w', encoding='utf-8') as f:
            json.dump({
                'classification': classification,
                'section_mapping': section_mapping
            }, f, indent=2, ensure_ascii=False)
        
        # Save reorganization results
        reorganization_file = self.reorganizer.save_reorganized_document(
            reorganized_structure,
            metadata,
            reorg_validation,
            output_dir,
            f"{base_name}_reorganized.json"
        )
        
        # Save table data
        table_file = os.path.join(output_dir, f"{base_name}_tables.json")
        with open(table_file, 'w', encoding='utf-8') as f:
            json.dump(table_data, f, indent=2, ensure_ascii=False)
        
        # Create summary
        summary = {
            'input_file': markdown_file,
            'output_files': {
                'segmentation': segmentation_file,
                'classification': classification_file,
                'reorganization': reorganization_file,
                'tables': table_file
            },
            'statistics': {
                'num_sections': len(document_structure),
                'num_tables': len(tables),
                'validation': validation,
                'reorganization_validation': reorg_validation,
                'metadata_summary': {
                    'total_words': metadata.get('Total_Words', 0),
                    'total_chars': metadata.get('Total_Chars', 0),
                    'unassigned_words': metadata.get('Uncounted_Words', 0)
                }
            }
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, f"{base_name}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_dir}")
        return summary
    
    def batch_process(self,
                     input_dir: str,
                     output_dir: Optional[str] = None,
                     use_llm: bool = True,
                     llm_client_type: str = 'deepseek',
                     file_pattern: str = "*.md") -> List[Dict]:
        """
        Process multiple markdown files in batch.
        
        Args:
            input_dir: Directory containing markdown files
            output_dir: Directory to save results
            use_llm: Whether to use LLM for classification
            llm_client_type: Type of LLM client to use
            file_pattern: File pattern to match
            
        Returns:
            list: List of processing results for each file
        """
        import glob
        
        if output_dir is None:
            output_dir = os.path.join(input_dir, "processed")
        
        # Find markdown files
        pattern = os.path.join(input_dir, file_pattern)
        markdown_files = glob.glob(pattern)
        
        if not markdown_files:
            logger.warning(f"No markdown files found matching pattern: {pattern}")
            return []
        
        logger.info(f"Found {len(markdown_files)} markdown files to process")
        
        results = []
        for i, markdown_file in enumerate(markdown_files):
            logger.info(f"Processing file {i+1}/{len(markdown_files)}: {markdown_file}")
            
            try:
                # Create subdirectory for this file
                file_base = os.path.splitext(os.path.basename(markdown_file))[0]
                file_output_dir = os.path.join(output_dir, file_base)
                
                # Process the document
                result = self.process_document(
                    markdown_file,
                    output_dir=file_output_dir,
                    use_llm=use_llm,
                    llm_client_type=llm_client_type
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {markdown_file}: {e}")
                # Continue with next file
        
        # Save batch summary
        batch_summary = {
            'input_directory': input_dir,
            'output_directory': output_dir,
            'total_files': len(markdown_files),
            'successful_files': len(results),
            'failed_files': len(markdown_files) - len(results),
            'results': results
        }
        
        batch_summary_file = os.path.join(output_dir, "batch_summary.json")
        with open(batch_summary_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing completed. Results saved to: {output_dir}")
        return results