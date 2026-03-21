"""
Task feature extraction module for ARneuro.

This module extracts linguistic and cognitive features from task descriptions.
Based on ExtractFeatureJson.py with hierarchical feature ontology.
"""

import json
import copy
from typing import Dict, List, Any, Optional, Tuple
from ..core.logger import get_logger

logger = get_logger(__name__)


class TaskFeatureExtractor:
    """
    A theoretically grounded, hierarchically structured feature-extraction wrapper 
    for task annotations via an LLM.
    
    Based on the original ExtractFeatureJson.py with rhythm-related constructs.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the task feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_temperature = 0.0
        self.max_tokens = 2048
        
        # Hierarchical feature ontology (DAG)
        # Each node maps to its list of direct parents (for upward closure).
        # 0 = PRESENT, 1 = ABSENT.
        self.feature_graph: Dict[str, List[str]] = {
            # Top-level roots
            "language": [],
            "processing": [],
            "task_structure": [],
            
            # Representational (competence-like) strata
            "phonetics/phonology": ["language"],
            "morphology": ["language"],
            "syntax": ["language"],
            "semantics": ["language"],
            "pragmatics/discourse": ["language"],
            
            # Prosody & Rhythm (under phonology)
            "prosody": ["phonetics/phonology"],
            "rhythm/metrical_structure": ["prosody", "phonetics/phonology"],
            
            # Rhythm subcomponents
            "rhythm_class_judgement": ["rhythm/metrical_structure"],
            "metrical_foot_structure": ["rhythm/metrical_structure"],
            "poetic_meter_detection": ["rhythm/metrical_structure"],
            "pvi_based_rhythm_metric": ["rhythm/metrical_structure"],
            "speech_rate_normalization": ["rhythm/metrical_structure", "processing"],
            "beat_entrainment/synchrony": ["rhythm/metrical_structure", "processing"],
            
            # Phonetics/Phonology leaves
            "phoneme_discrimination": ["phonetics/phonology"],
            "phoneme_categorization": ["phonetics/phonology"],
            "phonological_awareness": ["phonetics/phonology"],
            "phonological_working_memory": ["phonetics/phonology"],
            "phonological_rule_application": ["phonetics/phonology"],
            "tone_perception/production": ["phonetics/phonology"],
            "vowel_formant_manipulation": ["phonetics/phonology"],
            "consonant_place_manner": ["phonetics/phonology"],
            "stress_accent_detection": ["phonetics/phonology"],
            "intonation_contour_perception": ["phonetics/phonology"],
            
            # Morphology leaves
            "morphological_decomposition": ["morphology"],
            "morphological_productivity": ["morphology"],
            "derivational_morphology": ["morphology"],
            "inflectional_morphology": ["morphology"],
            "compound_processing": ["morphology"],
            
            # Syntax leaves
            "syntactic_parsing": ["syntax"],
            "grammaticality_judgement": ["syntax"],
            "sentence_comprehension": ["syntax"],
            "sentence_production": ["syntax"],
            "agreement_processing": ["syntax"],
            "filler_gap_dependency": ["syntax"],
            "relative_clause_processing": ["syntax"],
            "wh_question_processing": ["syntax"],
            "garden_path/reanalysis": ["syntax"],
            
            # Semantics leaves
            "lexical_semantics": ["semantics"],
            "compositionality": ["semantics"],
            "semantic_relations/taxonomic": ["semantics"],
            "semantic_relations/thematic": ["semantics"],
            "entailment/implicature_sensitivity": ["semantics"],
            "metaphor/figurative": ["semantics"],
            "semantic_anomaly_detection": ["semantics"],
            
            # Pragmatics/Discourse leaves
            "reference/anaphora": ["pragmatics/discourse"],
            "coherence/narrative": ["pragmatics/discourse"],
            "speech_acts/implicature": ["pragmatics/discourse"],
            "information_structure": ["pragmatics/discourse"],
            
            # Processing leaves
            "comprehension": ["processing"],
            "production": ["processing"],
            "prediction/anticipation": ["processing"],
            "working_memory/maintenance": ["processing"],
            "parsing/structure_building": ["processing"],
            "semantic_integration": ["processing"],
            "monitoring/error_detection": ["processing"],
            "executive_control": ["processing"],
            "attention_selective": ["processing"],
            "attention_divided": ["processing"],
            "attention_sustained": ["processing"],
            "task_switching": ["processing"],
            "inhibitory_control": ["processing"],
            "entrainment/temporal_alignment": ["processing"],
            
            # Task structure & modality leaves
            "input_audio": ["task_structure"],
            "input_visual": ["task_structure"],
            "input_crossmodal": ["task_structure"],
            "output_motor_response": ["task_structure"],
            "time_pressure": ["task_structure"],
            "materials_letter/grapheme": ["task_structure"],
            "materials_morpheme": ["task_structure"],
            "materials_nonword": ["task_structure"],
            "materials_word": ["task_structure"],
            "materials_phrase": ["task_structure"],
            "materials_sentence": ["task_structure"],
            "materials_paragraph/narrative": ["task_structure"],
            "picture/object_naming": ["task_structure"],
            
            # Decision paradigms
            "semantic_priming_sensitivity": ["task_structure"],
            "semantic_plausibility_judgement": ["task_structure"],
            "syntactic_violation_detection": ["task_structure"],
            "prosody_perception": ["task_structure"],
            "prosody_production": ["task_structure"],
        }
        
        # Leaf nodes (directly queried from LLM)
        self.leaf_nodes: Dict[str, int] = {
            # Rhythm leaves
            "rhythm_class_judgement": 1,
            "metrical_foot_structure": 1,
            "poetic_meter_detection": 1,
            "pvi_based_rhythm_metric": 1,
            "speech_rate_normalization": 1,
            "beat_entrainment/synchrony": 1,
            
            # Phonetics/Phonology leaves
            "phoneme_discrimination": 1,
            "phoneme_categorization": 1,
            "phonological_awareness": 1,
            "phonological_working_memory": 1,
            "phonological_rule_application": 1,
            "tone_perception/production": 1,
            "vowel_formant_manipulation": 1,
            "consonant_place_manner": 1,
            "stress_accent_detection": 1,
            "intonation_contour_perception": 1,
            
            # Morphology leaves
            "morphological_decomposition": 1,
            "morphological_productivity": 1,
            "derivational_morphology": 1,
            "inflectional_morphology": 1,
            "compound_processing": 1,
            
            # Syntax leaves
            "syntactic_parsing": 1,
            "grammaticality_judgement": 1,
            "sentence_comprehension": 1,
            "sentence_production": 1,
            "agreement_processing": 1,
            "filler_gap_dependency": 1,
            "relative_clause_processing": 1,
            "wh_question_processing": 1,
            "garden_path/reanalysis": 1,
            
            # Semantics leaves
            "lexical_semantics": 1,
            "compositionality": 1,
            "semantic_relations/taxonomic": 1,
            "semantic_relations/thematic": 1,
            "entailment/implicature_sensitivity": 1,
            "metaphor/figurative": 1,
            "semantic_anomaly_detection": 1,
            
            # Pragmatics/Discourse leaves
            "reference/anaphora": 1,
            "coherence/narrative": 1,
            "speech_acts/implicature": 1,
            "information_structure": 1,
            
            # Processing leaves
            "comprehension": 1,
            "production": 1,
            "prediction/anticipation": 1,
            "working_memory/maintenance": 1,
            "parsing/structure_building": 1,
            "semantic_integration": 1,
            "monitoring/error_detection": 1,
            "executive_control": 1,
            "attention_selective": 1,
            "attention_divided": 1,
            "attention_sustained": 1,
            "task_switching": 1,
            "inhibitory_control": 1,
            "entrainment/temporal_alignment": 1,
            
            # Task structure & modality leaves
            "input_audio": 1,
            "input_visual": 1,
            "input_crossmodal": 1,
            "output_motor_response": 1,
            "time_pressure": 1,
            "materials_letter/grapheme": 1,
            "materials_morpheme": 1,
            "materials_nonword": 1,
            "materials_word": 1,
            "materials_phrase": 1,
            "materials_sentence": 1,
            "materials_paragraph/narrative": 1,
            "picture/object_naming": 1,
            
            # Decision paradigms
            "semantic_priming_sensitivity": 1,
            "semantic_plausibility_judgement": 1,
            "syntactic_violation_detection": 1,
            "prosody_perception": 1,
            "prosody_production": 1,
        }
        
        # Derived/internal nodes that will be induced by closure
        self.derived_nodes: List[str] = [
            "language", "processing", "task_structure",
            "phonetics/phonology", "morphology", "syntax", "semantics", "pragmatics/discourse",
            "prosody", "rhythm/metrical_structure"
        ]
        
        # Build feature examples
        self.feature_examples: Dict[str, Dict[str, str]] = self._build_feature_examples()
        
        logger.info(f"TaskFeatureExtractor initialized with {len(self.leaf_nodes)} leaf features")
    
    def _build_feature_examples(self) -> Dict[str, Dict[str, str]]:
        """
        Build per-feature examples to guide the LLM.
        
        Returns:
            Dict[str, Dict[str, str]]: Feature examples
        """
        examples: Dict[str, Dict[str, str]] = {}
        
        def add(key: str, pos: str, neg: str, heuristic: str):
            examples[key] = {
                "positive_example": pos,
                "negative_example": neg,
                "decision_heuristic": heuristic
            }
        
        # Rhythm-specific examples
        add(
            "rhythm_class_judgement",
            "Participants hear sentences from different languages and judge whether each token sounds stress-timed, syllable-timed, or mora-timed.",
            "Participants classify sentences by speaker gender; rhythm class is not judged.",
            "Mark PRESENT only if the task explicitly requires rhythm-class categorization or uses rhythm class as an independent variable."
        )
        
        add(
            "metrical_foot_structure",
            "Subjects tap to indicate foot boundaries while listening to English trochaic vs. iambic nonce words.",
            "Subjects listen to words and press a button when they hear a target phoneme; foot structure is irrelevant.",
            "Mark PRESENT if the task manipulates or measures metrical foot parsing, foot-boundary detection, or foot-based grouping."
        )
        
        add(
            "pvi_based_rhythm_metric",
            "We compute nPVI on vowel durations across utterances to quantify rhythm variability.",
            "We measure reaction times to lexical decisions; no rhythm metrics are computed.",
            "Mark PRESENT if the study explicitly computes Pairwise Variability Index (PVI, nPVI, rPVI) or cites it as a rhythm measure."
        )
        
        # Add more examples as needed...
        
        return examples
    
    def extract_features(self,
                        task_name: str,
                        context: str,
                        llm_client: Any,
                        model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Extract hierarchical features from task description.
        
        Args:
            task_name: Short name of the task
            context: Description of the task
            llm_client: LLM client instance
            model_name: Model name to use
            
        Returns:
            Dict[str, Any]: Feature extraction results
        """
        logger.info(f"Extracting features for task: {task_name}")
        
        # Extract representational features
        rep_features = self._extract_representational_features(task_name, context, llm_client, model_name)
        
        # Extract processing features
        proc_features = self._extract_processing_features(task_name, context, llm_client, model_name)
        
        # Extract task design features
        design_features = self._extract_task_design_features(task_name, context, llm_client, model_name)
        
        # Merge leaf judgments
        leaf_assignments: Dict[str, int] = {}
        leaf_assignments.update({k: rep_features.get(k, 1) for k in rep_features.keys()})
        leaf_assignments.update({k: proc_features.get(k, 1) for k in proc_features.keys()})
        leaf_assignments.update({k: design_features.get(k, 1) for k in design_features.keys()})
        
        # Apply hierarchical upward closure
        closed_assignments = self._upward_closure(leaf_assignments)
        
        # Enforce cross-level consistency
        consistent_assignments = self._enforce_cross_level_consistency(copy.deepcopy(closed_assignments))
        
        return {
            "task_name": task_name,
            "representational_features": rep_features,
            "processing_features": proc_features,
            "task_design_features": design_features,
            "hierarchy_closed": {k: closed_assignments[k] for k in sorted(closed_assignments.keys())},
            "consistent_assignments": {k: consistent_assignments[k] for k in sorted(consistent_assignments.keys())},
            "metadata": {
                "num_features_total": len(consistent_assignments),
                "num_features_present": sum(1 for v in consistent_assignments.values() if v == 0),
                "num_features_absent": sum(1 for v in consistent_assignments.values() if v == 1)
            }
        }
    
    def _extract_representational_features(self,
                                          task_name: str,
                                          context: str,
                                          client: Any,
                                          model_name: str) -> Dict[str, int]:
        """
        Extract representational (linguistic) features.
        """
        # Filter leaf nodes for representational features
        rep_schema = {
            k: v for k, v in self.leaf_nodes.items()
            if any(parent in ["language", "phonetics/phonology", "morphology", "syntax", 
                             "semantics", "pragmatics/discourse", "prosody", "rhythm/metrical_structure"]
                  for parent in self.feature_graph.get(k, []))
        }
        
        instruction = """
            You are annotating the LINGUISTIC REPRESENTATIONAL features of a cognitive task.
            Focus on which linguistic representations (phonology, morphology, syntax, semantics, pragmatics, rhythm)
            are centrally manipulated or measured.
            Ignore incidental involvement.
        """
        
        return self._extract_feature_json(task_name, context, client, model_name, rep_schema, instruction)
    
    def _extract_processing_features(self,
                                    task_name: str,
                                    context: str,
                                    client: Any,
                                    model_name: str) -> Dict[str, int]:
        """
        Extract processing (cognitive) features.
        """
        # Filter leaf nodes for processing features
        proc_schema = {
            k: v for k, v in self.leaf_nodes.items()
            if "processing" in self.feature_graph.get(k, [])
        }
        
        instruction = """
            You are annotating the COGNITIVE PROCESSING features of a task.
            Focus on which cognitive processes (comprehension, production, working memory, attention, control, etc.)
            are engaged or measured.
            Ignore incidental involvement.
        """
        
        return self._extract_feature_json(task_name, context, client, model_name, proc_schema, instruction)
    
    def _extract_task_design_features(self,
                                     task_name: str,
                                     context: str,
                                     client: Any,
                                     model_name: str) -> Dict[str, int]:
        """
        Extract task design and modality features.
        """
        # Filter leaf nodes for task design features
        design_schema = {
            k: v for k, v in self.leaf_nodes.items()
            if "task_structure" in self.feature_graph.get(k, [])
        }
        
        instruction = """
            You are annotating the TASK DESIGN & MODALITY features.
            Focus on input/output modalities, materials, timing constraints, and decision paradigms.
            Ignore incidental details.
        """
        
        return self._extract_feature_json(task_name, context, client, model_name, design_schema, instruction)
    
    def _extract_feature_json(self,
                             task_name: str,
                             context: str,
                             client: Any,
                             model_name: str,
                             schema: Dict[str, int],
                             task_instruction: str) -> Dict[str, int]:
        """
        Use an LLM to extract features as JSON.
        """
        if not task_name or not context or not str(context).strip():
            return {k: 1 for k in schema.keys()}
        
        # Compose per-feature examples block
        examples_block_lines: List[str] = []
        for k in schema.keys():
            ex = self.feature_examples.get(k)
            if ex:
                examples_block_lines.append(
                    f"- {k} :: POS: {ex['positive_example']} | NEG: {ex['negative_example']} | HEURISTIC: {ex['decision_heuristic']}"
                )
        
        examples_block = "\n".join(examples_block_lines) if examples_block_lines else "(no per-feature examples available)."
        
        system_prompt = f"""You are a precise theoretical-linguistics task annotator.
            Decide whether each requested feature is PRESENT (0) or ABSENT (1) in the described task.
            Think step-by-step internally, but DO NOT reveal your reasoning. Output ONLY valid JSON.
            Values MUST be integers 0 or 1. If uncertain, choose 1 (ABSENT).    
            Per-Feature Minimal Examples:
            {examples_block}

            Instructions:
            {task_instruction}

            Schema (keys must match exactly; values are 0 if the feature is present in this task, 1 if absent):
            {json.dumps(schema, ensure_ascii=False)}"""
        
        user_prompt = f"""
        Task Name: {task_name}

        Task Description:
        {context}

        Decision Rubric (silent internal reasoning only; output must be JSON):
        • A feature is PRESENT (0) only if it is a central target of manipulation/measurement or an explicit factor in task design.
        • Incidental involvement does NOT count as PRESENT.
        • Use the minimal examples below to calibrate your judgment.

        Return ONLY the JSON object. No comments. No markdown. No rationale.
        """
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.default_temperature,
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
                stream=False
            )
            
            parsed = json.loads(response.choices[0].message.content)
            
            # Normalize: ensure all schema keys exist and values are 0/1 ints
            out: Dict[str, int] = {}
            for k in schema.keys():
                v = parsed.get(k, 1)
                try:
                    v_int = int(v)
                    out[k] = 0 if v_int == 0 else 1
                except Exception:
                    s = str(v).strip().lower()
                    if s in {"present", "yes", "true", "y", "1"}:
                        out[k] = 0
                    elif s in {"absent", "no", "false", "n", "0"}:
                        out[k] = 1
                    else:
                        out[k] = 1
            return out
            
        except Exception as e:
            logger.error(f"LLM feature extraction failed: {e}")
            # Fallback to all-absent
            return {k: 1 for k in schema.keys()}
    
    def _upward_closure(self, leaf_assignments: Dict[str, int]) -> Dict[str, int]:
        """
        Apply upward closure: if a child is present (0), all ancestors become present (0).
        """
        # Start with leaf assignments
        assignments = leaf_assignments.copy()
        
        # Initialize all nodes with default value 1 (absent)
        for node in self.feature_graph.keys():
            if node not in assignments:
                assignments[node] = 1
        
        # Apply upward closure iteratively
        changed = True
        while changed:
            changed = False
            for child, parents in self.feature_graph.items():
                if assignments[child] == 0:  # Child is present
                    for parent in parents:
                        if assignments[parent] == 1:  # Parent is absent
                            assignments[parent] = 0  # Make parent present
                            changed = True
        
        return assignments
    
    def _enforce_cross_level_consistency(self, assignments: Dict[str, int]) -> Dict[str, int]:
        """
        Enforce additional cross-level consistency constraints.
        """
        # Rhythm → Prosody constraint (already handled by upward closure)
        # Add additional constraints here if needed
        
        return assignments
    
    def get_flat_features(self,
                         task_name: str,
                         context: str,
                         llm_client: Any,
                         model_name: str = "gpt-4o-mini") -> Dict[str, int]:
        """
        Get flat feature vector (consistent assignments only).
        
        Returns:
            Dict[str, int]: Flat feature vector with 0/1 values
        """
        result = self.extract_features(task_name, context, llm_client, model_name)
        return result["consistent_assignments"]