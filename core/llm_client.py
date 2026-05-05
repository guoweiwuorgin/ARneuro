"""
LLM client module for ARneuro.

This module provides LLM client management for various providers.
"""

import os
from typing import Optional, Tuple
from openai import OpenAI
from .logger import get_logger

logger = get_logger(__name__)


class LLMClientManager:
    """
    Manage LLM clients for different providers.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the LLM client manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.clients = {}
        
    def get_client(self, 
                  client_type: str = 'deepseek',
                  model_name: Optional[str] = None,
                  api_key: Optional[str] = None,
                  base_url: Optional[str] = None) -> Tuple[OpenAI, str]:
        """
        Get the appropriate API client based on the client type.
        
        Args:
            client_type: Type of client to use ('deepseek', 'gpt4', 'glm', or 'kimichat')
            model_name: The model name to use
            api_key: The API key for authentication
            base_url: The base URL for API requests
            
        Returns:
            tuple: (client object, model name)
        """
        # Check if we already have this client cached
        cache_key = f"{client_type}_{model_name}"
        if cache_key in self.clients:
            return self.clients[cache_key]
        
        if client_type == 'deepseek':
            model_name = model_name or "deepseek-chat"
            api_key = api_key or self.config.get('deepseek_api_key')
            if not api_key:
                raise ValueError("DeepSeek API key not provided in config")
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
            )
            
        elif client_type == 'gpt4':
            model_name = model_name or "gpt-4o-mini"
            api_key = api_key or os.environ.get("OPENAI_API_KEY") or self.config.get('openai_api_key')
            if not api_key:
                raise ValueError("OpenAI API key not provided in config or environment")
            
            client = OpenAI(api_key=api_key)
            
        elif client_type == 'kimichat':
            model_name = model_name or "kimi-for-coding"
            api_key = api_key or self.config.get('kimichat_api_key')
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.kimi.com/coding/v1"
            )
            
        elif client_type == 'glm':
            model_name = model_name or self.config.get('glm_model_name')
            api_key = api_key or self.config.get('glm_api_key')
            if not api_key:
                raise ValueError("GLM API key not provided in config")
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://open.bigmodel.cn/api/paas/v4"
            )
            
        else:
            raise ValueError(f"Unsupported client type: {client_type}")
        
        # Cache the client
        self.clients[cache_key] = (client, model_name)
        logger.info(f"Created LLM client for {client_type} with model {model_name}")
        
        return client, model_name
    
    def classify_headings_with_llm(self, 
                                  headings: str, 
                                  client_type: str = 'deepseek',
                                  **kwargs) -> dict:
        """
        Classify headings using LLM.
        
        Args:
            headings: Comma-separated headings string
            client_type: Type of LLM client to use
            **kwargs: Additional arguments for get_client
            
        Returns:
            dict: Classification results
        """
        import json
        
        client, model_name = self.get_client(client_type, **kwargs)
        
        # System prompt for classification
        system_prompt = """You are a precise classifier for academic paper section headings. Your task is to analyze a given list of headings and classify each item into one of the following categories:

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

        Each category should contain a list of headings that belong to that category. Headings should be included exactly as they appear in the input, preserving case and any numbering."""
        
        user_prompt = f"""Classify the following heading:\n\n{headings} . Do not respond as 'Here is the classification of the given headings' or give other explaination, just give JSON response 
                        Do not provide your reasoning for each piece of information extracted or inferred.
                        Respond with the JSON structure specified in the system prompt. If some string would affect the JSON structral, delete them,for example **\escape** """
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1,
                response_format={'type': 'json_object'},
                max_tokens=8192
            )
            
            classification = response.choices[0].message.content
            return json.loads(classification)
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            raise
    
    def analyze_abstract(self, 
                        abstract: str, 
                        client_type: str = 'deepseek',
                        **kwargs) -> dict:
        """
        Analyze an abstract for neuroscience research properties.
        
        Args:
            abstract: Abstract text
            client_type: Type of LLM client to use
            **kwargs: Additional arguments for get_client
            
        Returns:
            dict: Analysis results
        """
        import json
        
        client, model_name = self.get_client(client_type, **kwargs)
        
        # System prompt for abstract analysis
        system_prompt = """You are a language neuroscience expert screening abstracts to identify original research on neural mechanisms of language. Analyze each abstract using these criteria:

            1. **Original Study Verification**  
            - Confirm if the abstract describes original research (methodology, participants, experiments) rather than reviews/commentaries.

            2. **Core Research Theme Validation**  
            - Verify the primary focus is neural mechanisms of language networks/functions. Include:  
                • Language lateralization studies  
                • Functional mapping research  
                • Language assessment methods  
            - Exclude studies where language is secondary to emotion/cognition research.

            3. **Linguistic Elements Identification**  
            - Identify specific language components studied from:  
                ["Articulatory phonetics", "Semantic", "Syntax", "Phonology", "Rhyme", "Orthography", "Morphology", "Pragmatics", "Prosody"]  
            - Mark "None" if no specific linguistic elements are addressed.

            4. **Technical Methodology Assessment**  
            - Check for neuroimaging methods: fMRI, PET, fNIRS, etc.  
            - Exclude non-imaging studies (e.g., behavioral-only).

            5. **Experimental Paradigm Classification**  
            - Categorize as:  
                "Task-based", "Resting-state", "dMRI", "sMRI", or combination.

            6. **Clinical Relevance Evaluation**  
            - Note if patients are studied (specify disorders) or mark "No".

            7. **Research Focus Determination**  
            - Characterize primary focus:  
                • Task-induced activation  
                • Functional connectivity  
                • Lateralization methods  
                • Network neuroscience approaches

            8. **Neural Substrate Validation**  
            - Confirm if the study investigates biological foundations of language processing.

            Return assessment in this JSON format:
            {
            "IsOriginalStudy": "Yes/No", 
            "ResearchTheme": "Summary",
            "LinguisticElements": ["List","Relevant","Components"], 
            "TechnicalMethods": ["Neuroimaging","Techniques"],
            "ResearchParadigm": "Paradigm Type",
            "DiseaseStudy": "Yes (Disorders) / No",
            "ResearchFocus": "Primary Approach",
            "LanguageNeuralSubstrateResearch": "Yes/No with rationale"
            }"""
        
        user_prompt = f"""Screen the following abstract and provide the output strictly in JSON format without any additional explanation:\n\n{abstract}\n\n
                        Do not provide your reasoning for each piece of information extracted or inferred. 
                        Respond with the JSON structure specified in the system prompt."""
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.9,
                response_format={'type': 'json_object'},
                max_tokens=8192
            )
            
            analysis = response.choices[0].message.content
            return json.loads(analysis)
            
        except Exception as e:
            logger.error(f"Abstract analysis failed: {e}")
            raise