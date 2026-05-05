"""
Table processing pipeline for ARneuro.

This module provides a complete pipeline for processing tables from research papers.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
from core.logger import get_logger
from core.llm_client import LLMClientManager
from .table_extractor import TableExtractor
from .brain_activation_processor import BrainActivationProcessor

logger = get_logger(__name__)


class TableProcessingPipeline:
    """
    Complete table processing pipeline for research papers.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the table processing pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.table_extractor = TableExtractor(config)
        self.brain_processor = BrainActivationProcessor(config=config)
        
        # Initialize LLM client manager if API keys are provided
        self.llm_manager = None
        if any(key in config for key in ['deepseek_api_key', 'openai_api_key', 'glm_api_key', 'kimichat_api_key']):
            self.llm_manager = LLMClientManager(config)
    
    def process_structured_content_file(self,
                                      structured_content_file: str,
                                      output_dir: Optional[str] = None,
                                      source_markdown: Optional[str] = None,
                                      process_brain_tables: bool = True,
                                      llm_client_type: str = 'deepseek',
                                      model_name: str = 'deepseek-chat') -> Dict[str, Any]:
        """Process tables from `*_structured_content.json` produced by text_processing."""
        if not os.path.exists(structured_content_file):
            raise FileNotFoundError(f"Structured content file not found: {structured_content_file}")

        with open(structured_content_file, 'r', encoding='utf-8') as f:
            structured_content = json.load(f)

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(structured_content_file), "table_processing")

        if source_markdown is None:
            stem = Path(structured_content_file).name.replace('_structured_content.json', '')
            candidate = os.path.join(os.path.dirname(structured_content_file), f"{stem}.md")
            source_markdown = candidate if os.path.exists(candidate) else None

        logger.info(f"Processing tables from structured content: {structured_content_file}")
        tables_info = self.table_extractor.extract_tables_from_structured_content(
            structured_content=structured_content,
            source_markdown=source_markdown
        )
        return self._process_tables_payload(
            input_file=structured_content_file,
            output_dir=output_dir,
            tables_info=tables_info,
            process_brain_tables=process_brain_tables,
            llm_client_type=llm_client_type,
            model_name=model_name,
            source_markdown=source_markdown
        )

    def _process_tables_payload(self,
                               input_file: str,
                               output_dir: str,
                               tables_info: List[Dict],
                               process_brain_tables: bool,
                               llm_client_type: str,
                               model_name: str,
                               source_markdown: Optional[str] = None) -> Dict[str, Any]:
        """Shared table post-processing for both markdown and structured inputs."""
        results = {
            "input_file": input_file,
            "source_markdown": source_markdown,
            "output_directory": output_dir,
            "metadata": {}
        }
        results["tables_extracted"] = len(tables_info)
        results["tables_info"] = tables_info

        try:
            llm_client = None
            effective_model_name = model_name
            if not self.llm_manager:
                raise ValueError("LLM client manager is not initialized. Please provide API key config.")
            llm_client, effective_model_name = self.llm_manager.get_client(
                client_type=llm_client_type,
                model_name=model_name
            )
            categorized_tables = self.table_extractor.categorize_tables(
                tables_info,
                llm_client=llm_client,
                model_name=effective_model_name,
                llm_client_type=llm_client_type
            )
            results["categorized_tables"] = {
                category: len(tables)
                for category, tables in categorized_tables.items()
            }
        except Exception as e:
            logger.error(f"Table categorization failed: {e}")
            results["categorized_tables"] = {}

        if process_brain_tables and tables_info and self.llm_manager:
            try:
                self.brain_processor.set_llm_client(llm_client)
                self.brain_processor.config.update({
                    'model_name': effective_model_name,
                    'llm_client_type': llm_client_type
                })
                brain_tables_results = self._process_brain_activation_tables(
                    categorized_tables=categorized_tables if "categorized_tables" in locals() else None,
                    tables_info=tables_info,
                    llm_client=llm_client
                )
                results["brain_activation_tables"] = brain_tables_results
            except Exception as e:
                logger.error(f"Brain table processing failed: {e}")
                results["brain_activation_tables"] = {"error": str(e)}

        try:
            saved_files = self._save_results(results, output_dir)
            results["output_files"] = saved_files
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            results["output_files"] = {"error": str(e)}

        logger.info(f"Table processing completed for: {input_file}")
        return results

    def process_markdown_file(self, 
                             markdown_file: str,
                             output_dir: Optional[str] = None,
                             process_brain_tables: bool = True,
                             llm_client_type: str = 'deepseek',
                             model_name: str = 'deepseek-chat') -> Dict[str, Any]:
        """Process tables from a markdown file."""
        logger.info(f"Processing tables from: {markdown_file}")

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(markdown_file), "table_processing")

        tables_info = self.table_extractor.extract_tables_from_markdown(
            markdown_file,
            context_lines=5
        )
        return self._process_tables_payload(
            input_file=markdown_file,
            output_dir=output_dir,
            tables_info=tables_info,
            process_brain_tables=process_brain_tables,
            llm_client_type=llm_client_type,
            model_name=model_name,
            source_markdown=markdown_file
        )

    def _process_brain_activation_tables(self,
                                        categorized_tables: Optional[Dict[str, List[Dict]]],
                                        tables_info: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process brain activation tables.
        """
        brain_tables = []

        candidate_tables = []
        if categorized_tables and isinstance(categorized_tables, dict):
            candidate_tables = categorized_tables.get('brain_activation', [])

        # Backward-compatible fallback when categorization is unavailable
        if not candidate_tables and tables_info:
            logger.warning("No categorized brain activation tables found; fallback to all extracted tables")
            candidate_tables = tables_info

        for table_info in candidate_tables:
            table_text = table_info['table_text']
            table_context = table_info['full_context']

            brain_table_info = {
                'table_index': table_info['table_index'],
                'table_text': table_text,
                'table_context': table_context,
                'assessment': {
                    'contains_coordinates': True,
                    'Task_name': 'Unknown',
                    'reason': 'Selected from categorized_tables.brain_activation',
                    'Table_header': ''
                }
            }

            try:
                parsed_df = self.brain_processor.fix_and_parse_table(
                    table_text=table_text,
                    table_description=table_context,
                    task_name='Unknown'
                )

                if not parsed_df.empty:
                    brain_table_info['parsed_data'] = {
                        'num_rows': len(parsed_df),
                        'num_columns': len(parsed_df.columns),
                        'columns': list(parsed_df.columns),
                        'all_rows': parsed_df.to_dict(orient='records')
                    }
                else:
                    brain_table_info['parsed_data'] = {'error': 'Failed to parse table'}

            except Exception as e:
                logger.error(f"Failed to parse brain table {table_info['table_index']}: {e}")
                brain_table_info['parsed_data'] = {'error': str(e)}

            brain_tables.append(brain_table_info)
        
        return {
            'total_brain_tables': len(brain_tables),
            'brain_tables': brain_tables
        }
    
    def _save_results(self, results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """
        Save processing results to files.
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save full results as JSON
        results_file = os.path.join(output_dir, "table_processing_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        saved_files["full_results"] = results_file
        
        # Save table extraction results
        if "tables_info" in results and results["tables_info"]:
            extraction_file = self.table_extractor.save_table_extraction_results(
                results["tables_info"],
                output_dir,
                "table_extraction.json"
            )
            saved_files["table_extraction"] = extraction_file
            
            # Export tables to CSV
            csv_files = self.table_extractor.export_tables_to_csv(
                results["tables_info"],
                os.path.join(output_dir, "tables_csv")
            )
            saved_files["tables_csv"] = csv_files
        
        # Save brain activation table results
        if "brain_activation_tables" in results and "brain_tables" in results["brain_activation_tables"]:
            brain_tables = results["brain_activation_tables"]["brain_tables"]
            if brain_tables:
                brain_file = os.path.join(output_dir, "brain_activation_tables.json")
                with open(brain_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'total_brain_tables': len(brain_tables),
                        'brain_tables': brain_tables
                    }, f, indent=2, ensure_ascii=False)
                saved_files["brain_activation_tables"] = brain_file
                
                # Save parsed brain tables as CSV
                brain_csv_dir = os.path.join(output_dir, "brain_tables_csv")
                os.makedirs(brain_csv_dir, exist_ok=True)
                
                for brain_table in brain_tables:
                    if 'parsed_data' in brain_table and 'all_rows' in brain_table['parsed_data']:
                        table_idx = brain_table['table_index']
                        df = pd.DataFrame(brain_table['parsed_data']['all_rows'])
                        csv_path = os.path.join(brain_csv_dir, f"brain_table_{table_idx + 1}.csv")
                        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved results to: {output_dir}")
        return saved_files
    
    def batch_process(self,
                     input_dir: str,
                     output_dir: Optional[str] = None,
                     file_pattern: str = "*.md",
                     process_brain_tables: bool = True,
                     llm_client_type: str = 'deepseek',
                     model_name: str = 'deepseek-chat') -> List[Dict[str, Any]]:
        """
        Process tables from multiple markdown files in batch.
        
        Args:
            input_dir: Directory containing markdown files
            output_dir: Directory to save results
            file_pattern: File pattern to match
            process_brain_tables: Whether to process brain activation tables
            llm_client_type: Type of LLM client to use
            model_name: Model name to use
            
        Returns:
            list: List of processing results for each file
        """
        import glob
        
        if output_dir is None:
            output_dir = os.path.join(input_dir, "table_processing_batch")
        
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
                
                # Process the file
                result = self.process_markdown_file(
                    markdown_file=markdown_file,
                    output_dir=file_output_dir,
                    process_brain_tables=process_brain_tables,
                    llm_client_type=llm_client_type,
                    model_name=model_name
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {markdown_file}: {e}")
                results.append({
                    "input_file": markdown_file,
                    "error": str(e)
                })
        
        # Save batch summary
        batch_summary = {
            "input_directory": input_dir,
            "output_directory": output_dir,
            "total_files": len(markdown_files),
            "successful_files": len([r for r in results if "error" not in r]),
            "failed_files": len([r for r in results if "error" in r]),
            "total_tables_extracted": sum(r.get("tables_extracted", 0) for r in results if "error" not in r),
            "results_summary": [
                {
                    "file": r.get("input_file"),
                    "success": "error" not in r,
                    "tables_extracted": r.get("tables_extracted", 0),
                    "brain_tables_found": r.get("brain_activation_tables", {}).get("total_brain_tables", 0) 
                                          if "brain_activation_tables" in r else 0,
                    "error": r.get("error") if "error" in r else None
                }
                for r in results
            ]
        }
        
        batch_summary_file = os.path.join(output_dir, "batch_processing_summary.json")
        with open(batch_summary_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing completed. Results saved to: {output_dir}")
        return results
