"""
Feature extraction pipeline for ARneuro.

This module provides a complete pipeline for extracting features from research papers.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from ..core.logger import get_logger
from ..core.llm_client import LLMClientManager
from .task_feature_extractor import TaskFeatureExtractor
from .cognitive_atlas_extractor import CognitiveAtlasExtractor

logger = get_logger(__name__)


class FeatureExtractionPipeline:
    """
    Complete feature extraction pipeline for research papers.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the feature extraction pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.task_extractor = TaskFeatureExtractor(config)
        self.cognitive_extractor = CognitiveAtlasExtractor(config)
        
        # Initialize LLM client manager if API keys are provided
        self.llm_manager = None
        if any(key in config for key in ['deepseek_api_key', 'openai_api_key', 'glm_api_key', 'huoshan_api_key']):
            self.llm_manager = LLMClientManager(config)
    
    def extract_from_paper(self,
                          paper_id: str,
                          paper_text: str,
                          methods_section: Optional[str] = None,
                          results_section: Optional[str] = None,
                          output_dir: Optional[str] = None,
                          llm_client_type: str = 'deepseek',
                          model_name: str = 'gpt-4o-mini') -> Dict[str, Any]:
        """
        Extract features from a research paper.
        
        Args:
            paper_id: Unique identifier for the paper
            paper_text: Full text of the paper
            methods_section: Methods section text (if available)
            results_section: Results section text (if available)
            output_dir: Directory to save results
            llm_client_type: Type of LLM client to use
            model_name: Model name to use
            
        Returns:
            dict: Complete feature extraction results
        """
        logger.info(f"Extracting features from paper: {paper_id}")
        
        # Set output directory
        if output_dir is None:
            output_dir = "feature_extraction_results"
        
        results = {
            "paper_id": paper_id,
            "metadata": {
                "total_text_length": len(paper_text),
                "has_methods": methods_section is not None,
                "has_results": results_section is not None,
                "methods_length": len(methods_section) if methods_section else 0,
                "results_length": len(results_section) if results_section else 0
            }
        }
        
        # Step 1: Extract task features from methods section
        if methods_section and self.llm_manager:
            try:
                task_features = self._extract_task_features(
                    paper_id, methods_section, llm_client_type, model_name
                )
                results["task_features"] = task_features
            except Exception as e:
                logger.error(f"Task feature extraction failed: {e}")
                results["task_features"] = {"error": str(e)}
        else:
            results["task_features"] = {"error": "No methods section or LLM client available"}
        
        # Step 2: Extract cognitive atlas concepts
        try:
            cognitive_concepts = self._extract_cognitive_concepts(paper_text)
            results["cognitive_concepts"] = cognitive_concepts
        except Exception as e:
            logger.error(f"Cognitive concept extraction failed: {e}")
            results["cognitive_concepts"] = {"error": str(e)}
        
        # Step 3: Map features to cognitive concepts
        if "task_features" in results and "consistent_assignments" in results["task_features"]:
            try:
                feature_concept_map = self._map_features_to_concepts(
                    results["task_features"]["consistent_assignments"]
                )
                results["feature_concept_map"] = feature_concept_map
            except Exception as e:
                logger.error(f"Feature-concept mapping failed: {e}")
                results["feature_concept_map"] = {"error": str(e)}
        
        # Step 4: Save results
        if output_dir:
            try:
                saved_files = self._save_results(results, paper_id, output_dir)
                results["output_files"] = saved_files
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
                results["output_files"] = {"error": str(e)}
        
        logger.info(f"Feature extraction completed for paper: {paper_id}")
        return results
    
    def _extract_task_features(self,
                              paper_id: str,
                              methods_text: str,
                              llm_client_type: str,
                              model_name: str) -> Dict[str, Any]:
        """
        Extract task features from methods section.
        """
        if not self.llm_manager:
            raise ValueError("LLM client manager not initialized")
        
        # Get LLM client
        client, actual_model_name = self.llm_manager.get_client(
            client_type=llm_client_type,
            model_name=model_name
        )
        
        # Extract task name and context from methods
        # For now, use the first 2000 chars as context
        task_context = methods_text[:2000]
        task_name = f"Paper_{paper_id}_Methods"
        
        # Extract features
        features = self.task_extractor.extract_features(
            task_name=task_name,
            context=task_context,
            llm_client=client,
            model_name=actual_model_name
        )
        
        return features
    
    def _extract_cognitive_concepts(self, paper_text: str) -> Dict[str, Any]:
        """
        Extract cognitive concepts from paper text.
        """
        # Get language concepts
        concepts_df = self.cognitive_extractor.get_language_concepts()
        
        # Extract concepts mentioned in text
        extracted_concepts = self.cognitive_extractor.extract_concepts_from_text(
            text=paper_text,
            concepts_df=concepts_df
        )
        
        return {
            "total_concepts_available": len(concepts_df),
            "concepts_found": len(extracted_concepts),
            "concepts_data": extracted_concepts.to_dict(orient='records') if not extracted_concepts.empty else []
        }
    
    def _map_features_to_concepts(self, features: Dict[str, int]) -> Dict[str, Any]:
        """
        Map extracted features to cognitive concepts.
        """
        # Get concepts
        concepts_df = self.cognitive_extractor.get_language_concepts()
        
        # Map features to concepts
        feature_concept_map = self.cognitive_extractor.map_features_to_concepts(
            features=features,
            concepts_df=concepts_df
        )
        
        return {
            "total_features_mapped": len(feature_concept_map),
            "mapping": feature_concept_map
        }
    
    def _save_results(self, results: Dict[str, Any], paper_id: str, output_dir: str) -> Dict[str, str]:
        """
        Save extraction results to files.
        """
        import os
        
        # Create output directory
        paper_output_dir = os.path.join(output_dir, paper_id)
        os.makedirs(paper_output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save full results as JSON
        results_file = os.path.join(paper_output_dir, f"{paper_id}_features.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        saved_files["full_results"] = results_file
        
        # Save task features separately
        if "task_features" in results:
            task_file = os.path.join(paper_output_dir, f"{paper_id}_task_features.json")
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(results["task_features"], f, indent=2, ensure_ascii=False)
            saved_files["task_features"] = task_file
        
        # Save cognitive concepts as CSV if available
        if "cognitive_concepts" in results and "concepts_data" in results["cognitive_concepts"]:
            concepts_data = results["cognitive_concepts"]["concepts_data"]
            if concepts_data:
                concepts_df = pd.DataFrame(concepts_data)
                concepts_file = os.path.join(paper_output_dir, f"{paper_id}_cognitive_concepts.csv")
                concepts_df.to_csv(concepts_file, index=False)
                saved_files["cognitive_concepts"] = concepts_file
        
        # Save feature-concept mapping
        if "feature_concept_map" in results:
            mapping_file = os.path.join(paper_output_dir, f"{paper_id}_feature_concept_mapping.json")
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(results["feature_concept_map"], f, indent=2, ensure_ascii=False)
            saved_files["feature_concept_mapping"] = mapping_file
        
        logger.info(f"Saved results to: {paper_output_dir}")
        return saved_files
    
    def batch_extract(self,
                     papers_data: List[Dict[str, Any]],
                     output_dir: str,
                     llm_client_type: str = 'deepseek',
                     model_name: str = 'gpt-4o-mini') -> List[Dict[str, Any]]:
        """
        Extract features from multiple papers in batch.
        
        Args:
            papers_data: List of paper data dictionaries with keys:
                        - 'paper_id': Unique identifier
                        - 'paper_text': Full text
                        - 'methods_section': Methods text (optional)
                        - 'results_section': Results text (optional)
            output_dir: Directory to save results
            llm_client_type: Type of LLM client to use
            model_name: Model name to use
            
        Returns:
            list: List of extraction results for each paper
        """
        logger.info(f"Starting batch extraction for {len(papers_data)} papers")
        
        results = []
        for i, paper_data in enumerate(papers_data):
            paper_id = paper_data.get('paper_id', f'paper_{i+1}')
            logger.info(f"Processing paper {i+1}/{len(papers_data)}: {paper_id}")
            
            try:
                result = self.extract_from_paper(
                    paper_id=paper_id,
                    paper_text=paper_data.get('paper_text', ''),
                    methods_section=paper_data.get('methods_section'),
                    results_section=paper_data.get('results_section'),
                    output_dir=output_dir,
                    llm_client_type=llm_client_type,
                    model_name=model_name
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to extract features from paper {paper_id}: {e}")
                results.append({
                    "paper_id": paper_id,
                    "error": str(e)
                })
        
        # Save batch summary
        batch_summary = {
            "total_papers": len(papers_data),
            "successful_extractions": len([r for r in results if "error" not in r]),
            "failed_extractions": len([r for r in results if "error" in r]),
            "results_summary": [
                {
                    "paper_id": r["paper_id"],
                    "success": "error" not in r,
                    "has_task_features": "task_features" in r and "error" not in r.get("task_features", {}),
                    "has_cognitive_concepts": "cognitive_concepts" in r and "error" not in r.get("cognitive_concepts", {}),
                    "error": r.get("error") if "error" in r else None
                }
                for r in results
            ]
        }
        
        batch_summary_file = os.path.join(output_dir, "batch_extraction_summary.json")
        with open(batch_summary_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch extraction completed. Results saved to: {output_dir}")
        return results