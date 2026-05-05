"""
Heading classification module for ARneuro.

This module handles classification of document headings into predefined categories.
"""

import json
from typing import Dict, List, Optional, Tuple
from ..core.logger import get_logger

logger = get_logger(__name__)


class HeadingClassifier:
    """
    Classify document headings into predefined categories using LLM.
    """
    
    def __init__(self, llm_client=None, config: Optional[Dict] = None):
        """
        Initialize the heading classifier.
        
        Args:
            llm_client: LLM client instance (e.g., OpenAI client)
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}
        
    def set_llm_client(self, llm_client):
        """
        Set the LLM client for classification.
        
        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client
        
    def classify_headings(self, headings: List[str], client_type: str = 'deepseek') -> Dict:
        """
        Classify a list of headings into predefined categories using an AI language model.
        
        Args:
            headings: List of heading strings
            client_type: Type of LLM client to use
            
        Returns:
            dict: Dictionary containing classified headings by category
        """
        if not headings:
            return self._get_empty_classification()
        
        # Convert headings list to comma-separated string
        headings_str = ','.join(headings)
        
        # Define the system prompt with examples
        system_prompt = """
            You are a precise classifier for academic paper section headings. Your task is to analyze a given list of headings and classify each item into one of the following categories:

            - "Title"
            - "Author"
            - "Keywords"
            - "Abstract"
            - "Introduction"
            - "Methods"
            - "Results"
            - "Discussion"
            - "References"
            - "Acknowledgements"
            - "Other"

            Rules for classification:

            1. "Methods" includes any variations such as "Materials and Methods," "Experimental Methods," etc.
            2. Subsections of methods (e.g., "Subjects," "Data Analysis," "Experimental Design") should also be classified as "Methods."
            3. Be aware of numbered or lettered headings (e.g., "2. Materials and Methods," "2.1. Subjects") and classify them appropriately.
            4. If a heading doesn't clearly fit into any category, classify it as "Other."
            5. The first non-categorized item is usually the "Title."
            6. "Summary" can be classified as "Abstract."
            7. "Conclusion" should be classified as "Discussion."
            8. Titles and authors' names typically appear consecutively; if there is a non-categorized item between the title and abstract, it is often the authors' names.
            9. "Methods," "Results," and "Discussion" sections generally follow a natural sequence in the paper, but in some cases, "Methods" might appear after "Discussion"—be aware of this possibility when classifying.
            10. If a heading is ambiguous but falls between two defined sections, classify it as belonging to the preceding section.
            **Output format:**

            Your response should be **only** a valid JSON object with the following structure:

            {
                "Title": [...],
                "Author": [...],
                "Keywords": [...],
                "Abstract": [...],
                "Introduction": [...],
                "Methods": [...],
                "Results": [...],
                "Discussion": [...],
                "References": [...],
                "Acknowledgements": [...],
                "Other": [...]
            }

        Each category should contain a list of headings that belong to that category. Headings should be included exactly as they appear in the input, preserving case and any numbering.
        """
        
        user_prompt = f"""Classify the following heading:\n\n{headings_str} . Do not respond as 'Here is the classification of the given headings' or give other explaination, just give JSON response 
                        Do not provide your reasoning for each piece of information extracted or inferred.
                        Respond with the JSON structure specified in the system prompt. If some string would affect the JSON structral, delete them """
        
        # Check if we have an LLM client
        if not self.llm_client:
            logger.warning("No LLM client provided, using rule-based classification")
            return self._classify_headings_rule_based(headings)
        
        try:
            # Make API call
            response = self.llm_client.chat.completions.create(
                model=self.config.get('model_name', 'deepseek-chat'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1,  # Reduced for more consistent outputs
                response_format={'type': 'json_object'},
                max_tokens=8192
            )
            
            classification = response.choices[0].message.content
            return json.loads(classification)
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            logger.warning("Falling back to rule-based classification")
            return self._classify_headings_rule_based(headings)
    
    def _classify_headings_rule_based(self, headings: List[str]) -> Dict:
        """
        Rule-based classification of headings as fallback.
        
        Args:
            headings: List of heading strings
            
        Returns:
            dict: Dictionary containing classified headings by category
        """
        classification = self._get_empty_classification()
        
        if not headings:
            return classification
        
        # Simple rule-based classification
        for heading in headings:
            heading_lower = heading.lower()
            
            # Check for title (usually first non-empty heading)
            if not classification['Title'] and heading.strip():
                classification['Title'].append(heading)
                continue
            
            # Check for author (contains common author indicators)
            author_indicators = ['author', 'by ', 'et al', '&', 'and ']
            if any(indicator in heading_lower for indicator in author_indicators):
                classification['Author'].append(heading)
                continue
            
            # Check for abstract
            abstract_indicators = ['abstract', 'summary']
            if any(indicator in heading_lower for indicator in abstract_indicators):
                classification['Abstract'].append(heading)
                continue
            
            # Check for keywords
            keyword_indicators = ['keyword', 'key word']
            if any(indicator in heading_lower for indicator in keyword_indicators):
                classification['Keywords'].append(heading)
                continue
            
            # Check for introduction
            intro_indicators = ['introduction', 'background']
            if any(indicator in heading_lower for indicator in intro_indicators):
                classification['Introduction'].append(heading)
                continue
            
            # Check for methods
            methods_indicators = ['method', 'material', 'experiment', 'procedure', 'design', 'subject', 'participant', 'data analysis']
            if any(indicator in heading_lower for indicator in methods_indicators):
                classification['Methods'].append(heading)
                continue
            
            # Check for results
            results_indicators = ['result', 'finding', 'outcome']
            if any(indicator in heading_lower for indicator in results_indicators):
                classification['Results'].append(heading)
                continue
            
            # Check for discussion
            discussion_indicators = ['discussion', 'conclusion', 'implication', 'limitation', 'future work']
            if any(indicator in heading_lower for indicator in discussion_indicators):
                classification['Discussion'].append(heading)
                continue
            
            # Check for references
            ref_indicators = ['reference', 'bibliography', 'citation']
            if any(indicator in heading_lower for indicator in ref_indicators):
                classification['References'].append(heading)
                continue
            
            # Check for acknowledgements
            ack_indicators = ['acknowledgement', 'acknowledgment', 'thank', 'funding']
            if any(indicator in heading_lower for indicator in ack_indicators):
                classification['Acknowledgements'].append(heading)
                continue
            
            # If none matched, put in Other
            classification['Other'].append(heading)
        
        return classification
    
    def _get_empty_classification(self) -> Dict:
        """
        Get an empty classification dictionary with all categories.
        
        Returns:
            dict: Empty classification dictionary
        """
        return {
            "Title": [],
            "Author": [],
            "Keywords": [],
            "Abstract": [],
            "Introduction": [],
            "Methods": [],
            "Results": [],
            "Discussion": [],
            "References": [],
            "Acknowledgements": [],
            "Other": []
        }
    
    def map_sections_to_categories(self, 
                                  section_titles: List[str], 
                                  classification: Dict) -> Dict[str, str]:
        """
        Map each section title to its corresponding category.
        
        Args:
            section_titles: List of section titles from document
            classification: Dictionary of classified headings
            
        Returns:
            dict: Mapping from section title to category
        """
        mapping = {}
        
        for section_title in section_titles:
            if not section_title.strip():
                continue
                
            found = False
            for category, headings in classification.items():
                # Check if section_title is in the list of headings for this category
                if isinstance(headings, list):
                    if section_title in headings:
                        mapping[section_title] = category
                        found = True
                        break
                elif isinstance(headings, str):
                    # Split comma-separated string
                    headings_list = [h.strip() for h in headings.split(',') if h.strip()]
                    if section_title in headings_list:
                        mapping[section_title] = category
                        found = True
                        break
            
            if not found:
                mapping[section_title] = None
        
        return mapping