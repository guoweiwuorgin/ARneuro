"""
Cognitive Atlas extraction module for ARneuro.

This module extracts cognitive concepts from the Cognitive Atlas.
"""

import pandas as pd
from typing import List, Dict, Optional, Any
from ..core.logger import get_logger

logger = get_logger(__name__)


class CognitiveAtlasExtractor:
    """
    Extract cognitive concepts from the Cognitive Atlas.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the cognitive atlas extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Try to import cognitiveatlas
        try:
            from cognitiveatlas.api import get_concept
            self.get_concept = get_concept
            self.has_cognitiveatlas = True
        except ImportError:
            logger.warning("Cognitive Atlas package not installed. Install with: pip install cognitiveatlas")
            self.has_cognitiveatlas = False
            self.get_concept = None
    
    def get_language_concepts(self) -> pd.DataFrame:
        """
        Fetch all Cognitive Atlas concepts classified under 'language'
        and collect their definitions.
        
        Returns:
            DataFrame with columns: ['concept_id', 'name', 'definitions', 'cognitive_category']
        """
        if not self.has_cognitiveatlas:
            logger.error("Cognitive Atlas package not available")
            return pd.DataFrame(columns=["concept_id", "name", "definitions", "cognitive_category"])
        
        try:
            # Pull all concepts, then filter by concept_class == 'language'
            all_concepts = self.get_concept()
            df_all = all_concepts.pandas
            
            # Defensive: some installs return None; ensure DataFrame
            if df_all is None or df_all.empty:
                logger.warning("No concepts returned from Cognitive Atlas")
                return pd.DataFrame(columns=["concept_id", "name", "definitions", "cognitive_category"])
            
            # Filter for language concepts (ctp_C6 is language category)
            lang_df = df_all[df_all["id_concept_class"] == "ctp_C6"].copy()
            
            logger.info(f"Found {len(lang_df)} language concepts in Cognitive Atlas")
            
            # For each language concept, re-query by id to get definitions
            rows: List[Dict] = []
            for cid, cname in zip(lang_df["id"], lang_df["name"]):
                detail = self.get_concept(id=cid)
                
                # Extract definitions
                defs = []
                rec = detail.json
                dtxt = rec.get("definition_text") or rec.get("def") or ""
                if isinstance(dtxt, str) and dtxt.strip():
                    defs.append(dtxt.strip())
                
                # Get additional definitions if available
                if "definitions" in rec:
                    for def_item in rec["definitions"]:
                        if isinstance(def_item, dict) and "definition_text" in def_item:
                            def_text = def_item["definition_text"]
                            if def_text and def_text.strip():
                                defs.append(def_text.strip())
                
                # Remove duplicates while preserving order
                if defs:
                    seen = set()
                    unique_defs = []
                    for d in defs:
                        if d not in seen:
                            unique_defs.append(d)
                            seen.add(d)
                    def_joined = "\n---\n".join(unique_defs)
                else:
                    def_joined = ""
                
                # Get category
                category = ""
                if "conceptclasses" in rec and rec["conceptclasses"]:
                    category = rec["conceptclasses"][0].get("name", "")
                
                rows.append({
                    "concept_id": cid,
                    "name": cname,
                    "definitions": def_joined,
                    "cognitive_category": category
                })
            
            out = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
            logger.info(f"Extracted {len(out)} language concepts with definitions")
            return out
            
        except Exception as e:
            logger.error(f"Error fetching Cognitive Atlas concepts: {e}")
            return pd.DataFrame(columns=["concept_id", "name", "definitions", "cognitive_category"])
    
    def extract_concepts_from_text(self, 
                                  text: str, 
                                  concepts_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract Cognitive Atlas concepts mentioned in text.
        
        Args:
            text: Text to analyze
            concepts_df: DataFrame of concepts (if None, fetch language concepts)
            
        Returns:
            DataFrame: Concepts found in text with relevance scores
        """
        if concepts_df is None:
            concepts_df = self.get_language_concepts()
        
        if concepts_df.empty:
            logger.warning("No concepts available for extraction")
            return pd.DataFrame(columns=["concept_id", "name", "definitions", "cognitive_category", "relevance_score"])
        
        # Simple keyword matching for concept extraction
        text_lower = text.lower()
        results = []
        
        for _, row in concepts_df.iterrows():
            concept_name = str(row["name"]).lower()
            definitions = str(row["definitions"]).lower()
            
            # Calculate relevance score based on mentions
            score = 0
            
            # Check if concept name appears in text
            if concept_name in text_lower:
                score += 2
            
            # Check if concept appears in definitions (indicating related concepts)
            for word in concept_name.split():
                if len(word) > 3 and word in text_lower:
                    score += 1
            
            # Check definitions for keywords from text
            text_words = set(text_lower.split())
            def_words = set(definitions.split())
            common_words = text_words.intersection(def_words)
            if len(common_words) > 0:
                score += 0.5 * len(common_words)
            
            if score > 0:
                results.append({
                    "concept_id": row["concept_id"],
                    "name": row["name"],
                    "definitions": row["definitions"],
                    "cognitive_category": row["cognitive_category"],
                    "relevance_score": min(score, 10)  # Cap at 10
                })
        
        # Sort by relevance score
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values("relevance_score", ascending=False).reset_index(drop=True)
        
        logger.info(f"Found {len(results_df)} relevant concepts in text")
        return results_df
    
    def map_features_to_concepts(self, 
                                features: Dict[str, int],
                                concepts_df: Optional[pd.DataFrame] = None) -> Dict[str, List[Dict]]:
        """
        Map extracted features to Cognitive Atlas concepts.
        
        Args:
            features: Dictionary of feature names to values (0/1)
            concepts_df: DataFrame of concepts
            
        Returns:
            Dict: Mapping of features to related concepts
        """
        if concepts_df is None:
            concepts_df = self.get_language_concepts()
        
        if concepts_df.empty:
            return {}
        
        feature_concept_map = {}
        
        for feature_name, feature_value in features.items():
            if feature_value == 0:  # Only map present features
                # Simple keyword matching between feature name and concept names/definitions
                related_concepts = []
                feature_words = set(feature_name.lower().replace('/', ' ').replace('_', ' ').split())
                
                for _, row in concepts_df.iterrows():
                    concept_name = str(row["name"]).lower()
                    definitions = str(row["definitions"]).lower()
                    
                    # Check for word overlap
                    concept_words = set(concept_name.split())
                    def_words = set(definitions.split())
                    
                    all_concept_words = concept_words.union(def_words)
                    overlap = feature_words.intersection(all_concept_words)
                    
                    if overlap:
                        relevance = len(overlap) / max(len(feature_words), 1)
                        if relevance > 0.3:  # Threshold
                            related_concepts.append({
                                "concept_id": row["concept_id"],
                                "name": row["name"],
                                "definitions": row["definitions"],
                                "cognitive_category": row["cognitive_category"],
                                "relevance_score": relevance
                            })
                
                if related_concepts:
                    # Sort by relevance
                    related_concepts.sort(key=lambda x: x["relevance_score"], reverse=True)
                    feature_concept_map[feature_name] = related_concepts[:5]  # Top 5
        
        return feature_concept_map
    
    def save_concepts(self, 
                     concepts_df: pd.DataFrame, 
                     output_path: str) -> str:
        """
        Save concepts to CSV file.
        
        Args:
            concepts_df: DataFrame of concepts
            output_path: Path to save CSV
            
        Returns:
            str: Path to saved file
        """
        import os
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        concepts_df.to_csv(output_path, index=False)
        logger.info(f"Saved concepts to: {output_path}")
        return output_path
    
    def load_concepts(self, input_path: str) -> pd.DataFrame:
        """
        Load concepts from CSV file.
        
        Args:
            input_path: Path to CSV file
            
        Returns:
            DataFrame: Loaded concepts
        """
        try:
            concepts_df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(concepts_df)} concepts from: {input_path}")
            return concepts_df
        except Exception as e:
            logger.error(f"Error loading concepts from {input_path}: {e}")
            return pd.DataFrame()