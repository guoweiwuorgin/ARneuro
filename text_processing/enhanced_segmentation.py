"""
Enhanced document segmentation module for ARneuro.

This module provides enhanced document segmentation using:
1. Extended title mapping (方案A) - including numbered headings
2. Content feature recognition (方案B)
3. LLM-based intelligent classification (新思路) - paragraph by paragraph
"""

import json
import re
import time
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.logger import get_logger

logger = get_logger(__name__)


class EnhancedSegmenter:
    """
    Enhanced document segmenter combining rules, features, and LLM.
    """
    
    # Canonical section categories
    TARGET_CATEGORIES = [
        "Title", "Author", "Keywords", "Abstract",
        "Introduction", "Methods", "Results", "Discussion",
        "References", "Acknowledgements", "Other"
    ]
    
    # Core required sections (Discussion is NOT required)
    REQUIRED_CORE = {"Introduction", "Methods", "Results", "References"}
    
    # Minimum required for success (Introduction, Methods, Results)
    MIN_REQUIRED = {"Introduction", "Methods", "Results"}
    
    # Extended heading mapping (方案A)
    SECTION_MAPPING = {
        # Introduction alternatives
        "background": "Introduction",
        "background and objectives": "Introduction",
        "background and aims": "Introduction",
        "background and purpose": "Introduction",
        "purpose": "Introduction",
        "purpose of the study": "Introduction",
        "objective": "Introduction",
        "objectives": "Introduction",
        "aims": "Introduction",
        "aim": "Introduction",
        "aim of the study": "Introduction",
        "context": "Introduction",
        "rationale": "Introduction",
        "the aim of this study": "Introduction",
        "study background": "Introduction",
        "introduction": "Introduction",
        "overview": "Introduction",
        
        # Methods alternatives
        "methods": "Methods",
        "materials and methods": "Methods",
        "materials & methods": "Methods",
        "methodology": "Methods",
        "experimental procedures": "Methods",
        "experimental methods": "Methods",
        "experimental design": "Methods",
        "study design": "Methods",
        "participants": "Methods",
        "subjects": "Methods",
        "patients": "Methods",
        "subjects and methods": "Methods",
        "patients and methods": "Methods",
        "participants and methods": "Methods",
        "data acquisition": "Methods",
        "data collection": "Methods",
        "statistical analysis": "Methods",
        "statistical analyses": "Methods",
        "image acquisition": "Methods",
        "mri acquisition": "Methods",
        "fmri methods": "Methods",
        "mri data acquisition": "Methods",
        "mri data analysis": "Methods",
        "procedure": "Methods",
        "procedures": "Methods",
        "measures": "Methods",
        "assessments": "Methods",
        "clinical evaluation": "Methods",
        "neuropsychological assessment": "Methods",
        "case report": "Methods",
        "case presentation": "Methods",
        "case presentations": "Methods",
        "case description": "Methods",
        "case descriptions": "Methods",
        "report of a case": "Methods",
        "report of cases": "Methods",
        "case 1": "Methods",
        "case 2": "Methods",
        "case 3": "Methods",
        "case study": "Methods",
        "stimuli": "Methods",
        "stimulus": "Methods",
        "tasks": "Methods",
        "task": "Methods",
        "experimental paradigm": "Methods",
        "paradigm": "Methods",
        "functional localizer scans": "Methods",
        "rapid event-related scans": "Methods",
        "er scans": "Methods",
        "scanning parameters": "Methods",
        "imaging protocol": "Methods",
        "voxel-based morphometry": "Methods",
        "region of interest analysis": "Methods",
        "region of interest": "Methods",
        "roi analysis": "Methods",
        "data analysis": "Methods",
        "analysis methods": "Methods",
        "analytical methods": "Methods",
        "magnetic resonance imaging": "Methods",
        "functional mri": "Methods",
        "fmri acquisition": "Methods",
        "assessment": "Methods",
        "lesion analysis": "Methods",
        "neuroimaging": "Methods",
        "behavioral testing": "Methods",
        "behavioral assessment": "Methods",
        "cognitive assessment": "Methods",
        "language assessment": "Methods",
        "neuropsychological testing": "Methods",
        "inclusion criteria": "Methods",
        "exclusion criteria": "Methods",
        "ethics": "Methods",
        "ethical approval": "Methods",
        "informed consent": "Methods",
        "definition of": "Methods",
        "measurement of": "Methods",
        "volume of": "Methods",
        "image processing": "Methods",
        "image analysis": "Methods",
        "preprocessing": "Methods",
        "data preprocessing": "Methods",
        "experimental setup": "Methods",
        "apparatus": "Methods",
        "materials": "Methods",
        "sample": "Methods",
        "sample size": "Methods",
        "recruitment": "Methods",
        "protocol": "Methods",
        
        # Results alternatives
        "results": "Results",
        "findings": "Results",
        "outcomes": "Results",
        "results and discussion": "Results",
        "clinical findings": "Results",
        "brain activation": "Results",
        "hemodynamic response": "Results",
        "volumes of": "Results",
        "asymmetry coefficient": "Results",
        "behavioral data": "Results",
        "behavioral results": "Results",
        "fmri data": "Results",
        "fmri results": "Results",
        "imaging results": "Results",
        "neuroimaging findings": "Results",
        "activation patterns": "Results",
        "correlation analysis": "Results",
        "group differences": "Results",
        "statistical results": "Results",
        "main results": "Results",
        "primary outcomes": "Results",
        "secondary outcomes": "Results",
        
        # Discussion alternatives
        "discussion": "Discussion",
        "conclusions": "Discussion",
        "conclusion": "Discussion",
        "summary and conclusions": "Discussion",
        "general discussion": "Discussion",
        "implications": "Discussion",
        "clinical implications": "Discussion",
        "comment": "Discussion",
        "commentary": "Discussion",
        "interpretation": "Discussion",
        "limitations": "Discussion",
        "future directions": "Discussion",
        "future research": "Discussion",
        "discussion and conclusions": "Discussion",
        "discussion and summary": "Discussion",
        "principal findings": "Discussion",
        "main findings": "Discussion",
        "significance": "Discussion",
        "discussion and implications": "Discussion",
        
        # References alternatives
        "references": "References",
        "bibliography": "References",
        "literature cited": "References",
        "works cited": "References",
        "citations": "References",
        
        # Acknowledgements alternatives
        "acknowledgements": "Acknowledgements",
        "acknowledgments": "Acknowledgements",
        "acknowledgment": "Acknowledgements",
        "funding": "Acknowledgements",
        "funding sources": "Acknowledgements",
        "financial support": "Acknowledgements",
        "conflict of interest": "Acknowledgements",
        "conflicts of interest": "Acknowledgements",
        "competing interests": "Acknowledgements",
        "disclosures": "Acknowledgements",
        "author contributions": "Acknowledgements",
        "supplementary information": "Acknowledgements",
        "supplementary material": "Acknowledgements",
        "supplementary materials": "Acknowledgements",
        "supplementary data": "Acknowledgements",
        "data availability": "Acknowledgements",
        "availability of data": "Acknowledgements",
        
        # Keywords alternatives
        "keywords": "Keywords",
        "key words": "Keywords",
        "index terms": "Keywords",
    }
    
    # Patterns for numbered headings like "## 4. Discussion", "## (1) Methods"
    NUMBERED_HEADING_PATTERNS = [
        r'^\d+\.?\s*(.+)$',
        r'^\(\d+\)\s*(.+)$',
        r'^\d+\)\s*(.+)$',
        r'^section\s+\d+:?\s*(.+)$',
        r'^part\s+\d+:?\s*(.+)$',
        r'^[ivxlcdm]+\.?\s*(.+)$',
    ]
    
    # Content feature keywords (方案B)
    CONTENT_FEATURES = {
        "Introduction": [
            "previous studies", "it has been shown", "little is known",
            "the aim of", "we investigated", "we examined", "the purpose",
            "has been reported", "is well known", "research has shown",
            "in recent years", "growing evidence", "remains unclear",
            "has been demonstrated", "is thought to be", "it is well established",
            "few studies", "has not been", "is poorly understood",
            "the role of", "the relationship between", "is associated with",
            "the present study", "this study", "we sought to",
            "to investigate", "to determine", "to examine", "to explore",
            "was to", "were to", "has been suggested", "it has been proposed",
            "is of interest", "of particular interest",
            "the mechanism", "the pathophysiology", "the neural basis",
            "brain regions", "cortical", "lateralization",
            "background:", "purpose:", "objective:", "aims:",
            "developmental", "reading disability", "dyslexia",
            "language processing", "semantic", "phonological",
        ],
        "Methods": [
            "participants were", "subjects were", "patients were",
            "we performed", "data were collected", "ethical approval",
            "informed consent", "statistical analysis", "was used to",
            "were recruited", "exclusion criteria", "inclusion criteria",
            "measured using", "assessed by", "administered",
            "the protocol", "approval was", "ethics committee",
            "institutional review board", "written consent",
            "sample consisted", "aged", "years old", "right-handed",
            "magnetic resonance", "functional mri", "fmri scan",
            "image acquisition", "voxel size", "repetition time",
            "echo time", "flip angle", "slice thickness",
            "statistical parametric", "spm", "software",
            "analysis of variance", "anova", "t-test", "chi-square",
            "p <", "p =", "significance level", "correlation",
            "regression", "linear model", "mixed model",
            "task was", "stimuli were", "presented using",
            "instruction", "experimental design",
            "data processing", "preprocessing", "normalization",
            "smoothing", "threshold", "cluster size",
            "region of interest", "roi", "atlas",
            "template", "standard space", "talairach", "mni",
            "scanning was", "acquisition", "sequence",
            "case report", "case presentation", "we present",
            "a year-old", "male patient", "female patient",
            "clinical examination", "neurological examination",
            "diagnosis", "treatment", "therapy",
            "computed tomography", "ct scan", "pet scan",
            "single photon", "spect", "eeg", "electroencephalography",
        ],
        "Results": [
            "we found", "significant", "p < 0.05", "p = 0",
            "table 1", "figure 1", "mean ±", "confidence interval",
            "was associated with", "showed that", "demonstrated",
            "no significant difference", "was significantly",
            "activation was", "activation in", "activated",
            "deactivation", "signal change", "percent change",
            "hemodynamic", "bold signal", "bold response",
            "correlation was", "correlation between", "r =",
            "t =", "f =", "z =", "df =",
            "effect size", "cohen", "partial eta",
            "group comparison", "group difference",
            "patients showed", "controls showed", "compared to",
            "in contrast", "whereas", "however",
            "mean score", "standard deviation", "sd =",
            "median", "interquartile", "range",
            "accuracy", "reaction time", "performance",
            "increased", "decreased", "higher", "lower",
            "greater", "smaller", "reduced", "elevated",
            "volume was", "area was", "asymmetry",
            "left hemisphere", "right hemisphere", "bilateral",
            "temporal", "frontal", "parietal", "occipital",
            "as shown in", "as presented in", "as illustrated",
            "the results", "our results", "these results",
        ],
        "Discussion": [
            "our findings", "in conclusion", "limitations",
            "clinical implications", "future studies", "novel",
            "consistent with", "in line with", "these results suggest",
            "one possible explanation", "a limitation", "further research",
            "the present study", "our results", "this study",
            "we found that", "the main finding", "the key finding",
            "is consistent", "is in agreement", "supports",
            "contrasts with", "differs from", "is contrary to",
            "may be due to", "could be explained", "might reflect",
            "is likely", "it is possible", "one explanation",
            "a strength", "a weakness", "an important",
            "should be interpreted", "must be considered",
            "requires further", "warrants further", "needs further",
            "in summary", "taken together", "overall",
            "the mechanism", "the pathway", "the network",
            "functional significance", "clinical relevance",
            "therapeutic", "intervention", "treatment",
            "developmental", "maturation", "plasticity",
            "lateralization", "hemispheric", "dominance",
            "these findings", "our data suggest", "this suggests",
            "we conclude", "we propose", "we hypothesize",
        ],
    }
    
    # Patterns for Results and Discussion combined section
    RESULTS_DISCUSSION_PATTERNS = [
        "results and discussion",
        "results & discussion",
        "discussion and results",
        "findings and discussion",
        "results and implications",
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the enhanced segmenter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.llm_call_count = 0
        self.LLM_CALL_LIMIT = 5
        self.LLM_WAIT_TIME = 1.5
    
    def _normalize_heading(self, heading: str) -> str:
        """
        Normalize heading for mapping.
        Handles numbered headings like "4. Discussion" or "(1) Methods"
        
        Args:
            heading: Original heading string
            
        Returns:
            str: Normalized heading
        """
        normalized = heading.strip()
        
        for pattern in self.NUMBERED_HEADING_PATTERNS:
            match = re.match(pattern, normalized, re.IGNORECASE)
            if match:
                normalized = match.group(1).strip()
                break
        
        normalized = re.sub(r'^\d+\.?\s*', '', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
        
        return normalized
    
    def _rule_category(self, heading: str) -> str:
        """
        Classify heading using rule-based approach.
        
        Args:
            heading: Heading string
            
        Returns:
            str: Category name
        """
        normalized = self._normalize_heading(heading)
        
        if normalized in self.SECTION_MAPPING:
            return self.SECTION_MAPPING[normalized]
        
        t = normalized
        
        rules = {
            "Keywords": ["keywords", "key words"],
            "Abstract": ["abstract", "summary"],
            "Introduction": ["introduction", "background"],
            "Methods": [
                "methods", "materials and methods", "methodology", "experimental",
                "participants", "subjects", "statistical", "case report",
                "case presentation", "case description", "stimuli", "task",
                "procedure", "assessment", "mri acquisition", "fmri",
                "data analysis", "image analysis", "lesion analysis",
                "voxel-based", "region of interest"
            ],
            "Results": ["results", "findings", "outcomes"],
            "Discussion": ["discussion", "conclusion", "limitations", "implications"],
            "References": ["references", "bibliography"],
            "Acknowledgements": ["acknowledg", "funding", "conflict of interest", "competing interests"],
            "Author": ["author", "affiliation"],
        }
        
        for k, vals in rules.items():
            if any(v in t for v in vals):
                return k
        
        return "Other"
    
    def _is_results_discussion_combined(self, blocks: List[Dict]) -> bool:
        """
        Check if the document has a combined Results and Discussion section.
        """
        for block in blocks:
            heading = block.get("heading", "").lower().strip()
            for pattern in self.RESULTS_DISCUSSION_PATTERNS:
                if pattern in heading:
                    return True
        return False
    
    def _handle_combined_results_discussion(self, blocks: List[Dict]) -> List[Dict]:
        """
        Handle combined Results and Discussion sections by duplicating content.
        """
        result_blocks = []
        
        for block in blocks:
            heading = block.get("heading", "").lower().strip()
            is_combined = False
            
            for pattern in self.RESULTS_DISCUSSION_PATTERNS:
                if pattern in heading:
                    is_combined = True
                    break
            
            if is_combined:
                results_block = block.copy()
                results_block["category"] = "Results"
                results_block["original_combined"] = True
                result_blocks.append(results_block)
                
                discussion_block = block.copy()
                discussion_block["category"] = "Discussion"
                discussion_block["original_combined"] = True
                result_blocks.append(discussion_block)
            else:
                result_blocks.append(block)
        
        return result_blocks
    
    def classify_by_content_features(self, blocks: List[Dict]) -> None:
        """
        Classify blocks using content feature keywords (方案B).
        """
        for block in blocks:
            if block.get("category", "Other") != "Other":
                continue
            
            content = block.get("content", "")
            if not content:
                continue
            
            content_check = content[:500].lower()
            
            scores = {}
            for section, features in self.CONTENT_FEATURES.items():
                score = sum(1 for f in features if f in content_check)
                if score > 0:
                    scores[section] = score
            
            if scores:
                best_section = max(scores, key=scores.get)
                if scores[best_section] >= 2:
                    block["category"] = best_section
    
    def classify_with_rules_only(self, blocks: List[Dict]) -> Tuple[List[Dict], str]:
        """
        Phase 1: Classify blocks using only rules (no LLM).
        
        Args:
            blocks: List of document blocks
            
        Returns:
            Tuple[List[Dict], str]: (classified blocks, strategy used)
        """
        strategy_parts = []
        
        # Handle combined Results and Discussion sections
        is_combined = self._is_results_discussion_combined(blocks)
        if is_combined:
            blocks = self._handle_combined_results_discussion(blocks)
            strategy_parts.append("combined_rd_handling")
        
        # Initialize block indices
        for i, block in enumerate(blocks):
            block["block_index"] = i
        
        # Use heading mapping
        for block in blocks:
            heading = block.get("heading", "")
            if block.get("original_combined"):
                continue
            if not heading:
                block["category"] = "Other"
                continue
            
            normalized = self._normalize_heading(heading)
            if normalized in self.SECTION_MAPPING:
                block["category"] = self.SECTION_MAPPING[normalized]
                continue
            
            block["category"] = self._rule_category(heading)
        
        strategy_parts.append("title_mapping")
        
        # Handle first block (usually Title)
        if blocks and blocks[0].get("category") == "Other":
            blocks[0]["category"] = "Title"
        
        # Use content features
        self.classify_by_content_features(blocks)
        strategy_parts.append("content_features")
        
        # Handle special cases
        self._handle_special_cases(blocks)
        
        strategy = "+".join(strategy_parts)
        return blocks, strategy
    
    def _check_success(self, blocks: List[Dict]) -> Tuple[bool, set, set]:
        """
        Check if segmentation is successful.
        
        Args:
            blocks: Classified blocks
            
        Returns:
            Tuple[bool, set, set]: (is_success, detected_sections, missing_sections)
        """
        detected_sections = {
            b["category"]
            for b in blocks
            if b.get("category") and (b.get("content") or b.get("block_kind") == "table_placeholder")
        }
        missing_sections = self.MIN_REQUIRED - detected_sections
        is_success = len(missing_sections) == 0
        return is_success, detected_sections, missing_sections
    
    def _get_unclassified_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """
        Get blocks that are still classified as "Other" and need LLM classification.
        
        Args:
            blocks: Classified blocks
            
        Returns:
            List[Dict]: Unclassified blocks
        """
        unclassified = []
        for block in blocks:
            if block.get("category") == "Other":
                content = block.get("content", "")
                if content and len(content.strip()) >= 50:
                    # Extract first 3 sentences
                    sentences = re.split(r'(?<=[.!?])\s+', content.strip())
                    opening = ' '.join(sentences[:3])
                    
                    unclassified.append({
                        "block_index": block["block_index"],
                        "heading": block.get("heading", ""),
                        "opening_text": opening[:500],
                    })
        return unclassified
    
    def classify_single_paragraph_with_llm(self, 
                                            opening_text: str,
                                            llm_client,
                                            model_name: str = "deepseek-chat") -> str:
        """
        Classify a single paragraph opening using LLM with rate limiting.
        """
        # Rate limiting: wait after every 5 calls
        if self.llm_call_count > 0 and self.llm_call_count % self.LLM_CALL_LIMIT == 0:
            logger.debug(f"Rate limiting: waiting {self.LLM_WAIT_TIME}s after {self.LLM_CALL_LIMIT} calls")
            time.sleep(self.LLM_WAIT_TIME)
        
        system_prompt = """You are an expert at analyzing academic paper structure.

Given the first 1-3 sentences of a single paragraph from a research paper, classify it into exactly one of the following categories:

- "Title": Paper title
- "Author": Author names and affiliations
- "Keywords": Keywords list
- "Abstract": Abstract/summary
- "Introduction": Background, prior work, study aims, research context, rationale
- "Methods": Study design, participants, procedures, materials, statistical methods, case reports, clinical findings
- "Results": Findings, data, statistical outcomes, figure/table descriptions, brain activation results
- "Discussion": Interpretation, implications, limitations, conclusions, future directions
- "References": Citation lists, bibliography
- "Acknowledgements": Funding, thanks, author contributions, supplementary materials
- "Other": Anything that doesn't fit above categories

Classification guidelines:
1. Introduction: broad context, prior research citations, phrases like "little is known", "has been shown", "the aim of this study"
2. Methods: procedures or participant characteristics, phrases like "we performed", "were recruited", "participants were", "statistical analysis"
3. Results: findings and statistics (p-values, means, SDs), phrases like "we found", "was significant", "as shown in Table/Figure"
4. Discussion: interpretation or limitations, phrases like "our findings suggest", "in conclusion", "these results indicate", "a limitation"
5. Abstract: concise summary covering purpose, methods, and/or findings within a few sentences
6. References: numbered or author-year citations, bibliography entries
7. Acknowledgements: funding sources, author contributions, conflict of interest statements
8. Title: a short standalone phrase with no full sentence structure
9. Author: names, degrees, institutional affiliations, or email addresses
10. Keywords: a short list of terms, often prefixed with "Keywords:" or "Key words:"

Output format:
Return only a valid JSON object with a single key "category":
{"category": "<one of the categories above>"}

Do not include any explanation or additional text outside the JSON."""
        
        cleaned_text = opening_text.replace('\n', ' ').strip()
        if len(cleaned_text) > 500:
            cleaned_text = cleaned_text[:500] + "..."
        
        user_prompt = f"Classify this paragraph:\n\n{cleaned_text}"
        
        self.llm_call_count += 1
        
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            
            result_text = result_text.strip()
            json_match = re.search(r'\{[^{}]*"category"[^{}]*\}', result_text)
            if json_match:
                result_text = json_match.group()
            
            result = json.loads(result_text)
            category = result.get("category", "Other")
            
            if category in self.TARGET_CATEGORIES:
                return category
            else:
                logger.warning(f"Invalid category from LLM: {category}")
                return "Other"
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return "Other"
    
    def classify_unclassified_with_llm(self,
                                        unclassified_blocks: List[Dict],
                                        llm_client,
                                        model_name: str = "deepseek-chat") -> Dict[int, str]:
        """
        Classify unclassified blocks using LLM with rate limiting.
        
        Args:
            unclassified_blocks: List of unclassified block info
            llm_client: LLM client instance
            model_name: Model name to use
            
        Returns:
            Dict[int, str]: block_index -> category mapping
        """
        if not unclassified_blocks:
            return {}
        
        classifications = {}
        total = len(unclassified_blocks)
        
        logger.info(f"LLM classification: processing {total} unclassified paragraphs")
        
        for i, p in enumerate(unclassified_blocks):
            if (i + 1) % 10 == 0:
                logger.info(f"LLM classification: processing paragraph {i+1}/{total}")
            
            category = self.classify_single_paragraph_with_llm(
                p["opening_text"],
                llm_client,
                model_name
            )
            
            classifications[p["block_index"]] = category
        
        return classifications
    
    def enhanced_classify_blocks(self, 
                                  blocks: List[Dict],
                                  llm_client=None,
                                  model_name: str = "deepseek-chat") -> Tuple[List[Dict], str]:
        """
        Enhanced block classification with two-phase approach.
        
        Phase 1: Rules only (title mapping + content features)
        Phase 2: LLM for unclassified blocks (if needed)
        
        Args:
            blocks: List of document blocks
            llm_client: LLM client (optional)
            model_name: Model name for LLM
            
        Returns:
            Tuple[List[Dict], str]: (classified blocks, strategy used)
        """
        # Phase 1: Rules only
        blocks, strategy = self.classify_with_rules_only(blocks)
        
        # Check success
        is_success, detected_sections, missing_sections = self._check_success(blocks)
        
        if is_success:
            logger.info(f"Phase 1 success: all required sections found")
            return blocks, strategy
        
        # Phase 2: LLM for unclassified blocks (if available)
        if llm_client and missing_sections:
            logger.info(f"Phase 2: Using LLM for unclassified blocks (missing: {missing_sections})")
            
            # Get only unclassified blocks
            unclassified = self._get_unclassified_blocks(blocks)
            
            if unclassified:
                # Reset LLM call count for each document
                self.llm_call_count = 0
                
                llm_classifications = self.classify_unclassified_with_llm(
                    unclassified, llm_client, model_name
                )
                
                if llm_classifications:
                    # Apply LLM classifications
                    for block in blocks:
                        idx = block.get("block_index", -1)
                        if idx in llm_classifications:
                            current = block.get("category", "Other")
                            llm_category = llm_classifications[idx]
                            
                            if current == "Other":
                                block["category"] = llm_category
                    
                    strategy += "+llm_classification"
        
        return blocks, strategy
    
    def _handle_special_cases(self, blocks: List[Dict]) -> None:
        """
        Handle special cases in classification.
        """
        for i, block in enumerate(blocks):
            if block.get("category") != "Other":
                continue
            
            prev_cat = blocks[i-1]["category"] if i > 0 else None
            next_cat = blocks[i+1]["category"] if i < len(blocks)-1 else None
            
            if prev_cat and prev_cat == next_cat and prev_cat in self.REQUIRED_CORE:
                block["category"] = prev_cat
                continue
            
            content_len = len(block.get("content", ""))
            if content_len < 200:
                if next_cat == "References" or prev_cat == "References":
                    block["category"] = "Acknowledgements"
    
    def build_section_content(self, blocks: List[Dict]) -> Dict:
        """
        Build structured content from classified blocks.
        """
        from .document_segmentation import _format_blocks_once

        out = {k: "" for k in self.TARGET_CATEGORIES}
        by_cat = defaultdict(list)
        
        has_combined_rd = any(b.get("original_combined") for b in blocks)
        
        for b in blocks:
            by_cat[b["category"]].append(b)
        
        for cat, arr in by_cat.items():
            out[cat] = _format_blocks_once(arr)
        
        method_blocks = [b for b in blocks if b["category"] == "Methods"]
        method_hierarchy = []
        stack = []
        
        for mb in sorted(method_blocks, key=lambda x: x.get("line_start", 0)):
            node = {
                "heading": mb.get("heading", ""),
                "level": mb.get("heading_level", 0),
                "content": mb.get("content", ""),
                "children": []
            }
            while stack and stack[-1]["level"] >= node["level"]:
                stack.pop()
            if stack:
                stack[-1]["children"].append(node)
            else:
                method_hierarchy.append(node)
            stack.append(node)
        
        out["Methods_Hierarchy"] = method_hierarchy
        
        return out
    
    def segment_document(self, 
                         file_path: str, 
                         llm_client=None,
                         model_name: str = "deepseek-chat") -> Tuple[Dict, Dict]:
        """
        Segment a document using enhanced classification.
        
        Args:
            file_path: Path to markdown file
            llm_client: LLM client (optional)
            model_name: Model name for LLM
            
        Returns:
            Tuple[Dict, Dict]: (structured content, metadata)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        from .document_segmentation import DocumentSegmenter
        base_segmenter = DocumentSegmenter()
        blocks, tables, meta = base_segmenter._extract_blocks(lines)
        
        # Enhanced classification
        blocks, strategy = self.enhanced_classify_blocks(blocks, llm_client, model_name)
        
        # Build structured content
        structured = self.build_section_content(blocks)
        structured["Tables"] = tables
        
        # Check success
        is_success, detected_sections, missing_sections = self._check_success(blocks)
        
        # Build metadata
        metadata = {
            "source_file": file_path,
            "strategy": strategy,
            "sections_detected": sorted(detected_sections),
            "required_sections_found": sorted(self.REQUIRED_CORE.intersection(detected_sections)),
            "required_sections_complete": self.REQUIRED_CORE.issubset(detected_sections),
            "segmentation_success": is_success,
            "has_combined_results_discussion": self._is_results_discussion_combined(blocks),
            "page_info": meta,
            "block_count": len(blocks),
            "table_count": len(tables),
            "blocks": [{k: v for k, v in b.items() if k != "content"} for b in blocks],
        }
        
        return structured, metadata
