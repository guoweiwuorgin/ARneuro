"""
Real-world usage example for ARneuro.

This example shows how to use ARneuro with actual research data.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def demonstrate_real_usage():
    """
    Demonstrate real-world usage scenarios.
    """
    print("=" * 60)
    print("ARneuro Real-World Usage Examples")
    print("=" * 60)
    
    print("\nScenario 1: Literature Review for Meta-Analysis")
    print("-" * 40)
    
    print("""
    Use Case: You're conducting a meta-analysis on language lateralization studies.
    
    Steps:
    1. Collect PubMed IDs of relevant studies
    2. Download PDFs using ARneuro's PDF downloader
    3. Convert PDFs to text using GLM-OCR
    4. Extract brain activation coordinates from tables
    5. Analyze methodological features
    6. Generate structured data for meta-analysis
    
    Code Example:
    
    ```python
    from ARneuro.core.pipeline import ARneuroPipeline
    from ARneuro.config.config_manager import ConfigManager
    
    # Load configuration
    config = ConfigManager().load_config()
    
    # Update with your settings
    config.update({
        'llm': {'deepseek_api_key': 'your_key'},
        'glm_ocr': {'path': '/path/to/GLM-OCR'}
    })
    
    # Create pipeline
    pipeline = ARneuroPipeline(config)
    
    # PubMed IDs for language lateralization studies
    pubmed_ids = [
        '12345678',  # Study 1
        '23456789',  # Study 2
        '34567890',  # Study 3
        # ... add more IDs
    ]
    
    # Process all studies
    results = pipeline.process_pubmed_ids(
        pubmed_ids=pubmed_ids,
        output_dir='./meta_analysis_results'
    )
    
    # Extract brain coordinates for analysis
    all_coordinates = []
    for paper_id, result in results.items():
        if 'brain_activation_tables' in result:
            for table in result['brain_activation_tables']:
                if 'parsed_data' in table:
                    coordinates = extract_coordinates(table['parsed_data'])
                    all_coordinates.extend(coordinates)
    
    # Save for meta-analysis
    save_coordinates_for_meta_analysis(all_coordinates)
    ```
    """)
    
    print("\nScenario 2: Methodological Feature Extraction")
    print("-" * 40)
    
    print("""
    Use Case: Analyzing methodological trends in fMRI language studies.
    
    Steps:
    1. Process a corpus of language fMRI papers
    2. Extract methodological features (task design, stimuli, analysis methods)
    3. Identify trends over time
    4. Compare across research groups
    
    Code Example:
    
    ```python
    from ARneuro.feature_extraction.feature_pipeline import FeatureExtractionPipeline
    
    # Initialize feature extraction pipeline
    feature_pipeline = FeatureExtractionPipeline(config)
    
    # Load processed papers
    processed_papers = load_processed_papers('./processed_papers/')
    
    # Extract features from each paper
    all_features = []
    for paper in processed_papers:
        features = feature_pipeline.extract_from_paper(
            paper_id=paper['id'],
            paper_text=paper['full_text'],
            methods_section=paper['methods'],
            results_section=paper['results']
        )
        all_features.append(features)
    
    # Analyze feature trends
    analyze_methodological_trends(all_features)
    
    # Generate visualization
    plot_feature_frequency(all_features)
    ```
    """)
    
    print("\nScenario 3: Automated Systematic Review")
    print("-" * 40)
    
    print("""
    Use Case: Automating systematic review screening and data extraction.
    
    Steps:
    1. Import search results from PubMed/other databases
    2. Screen abstracts using LLM-based classification
    3. Extract full-text data from included studies
    4. Generate PRISMA flow diagram data
    5. Create structured data extraction tables
    
    Code Example:
    
    ```python
    from ARneuro.core.llm_client import LLMClientManager
    
    # Initialize LLM client
    llm_manager = LLMClientManager(config)
    
    # Screen abstracts
    abstracts = load_abstracts_from_csv('./search_results.csv')
    
    screened_studies = []
    for abstract in abstracts:
        # Use LLM to screen for inclusion criteria
        screening_result = llm_manager.analyze_abstract(
            abstract=abstract['text'],
            client_type='deepseek'
        )
        
        if meets_inclusion_criteria(screening_result):
            screened_studies.append({
                'pmid': abstract['pmid'],
                'title': abstract['title'],
                'abstract': abstract['text'],
                'screening_result': screening_result
            })
    
    # Process included studies
    included_pmids = [s['pmid'] for s in screened_studies]
    pipeline.process_pubmed_ids(
        pubmed_ids=included_pmids,
        output_dir='./systematic_review_data'
    )
    
    # Generate PRISMA data
    generate_prisma_data(screened_studies, included_pmids)
    ```
    """)
    
    print("\nScenario 4: Educational Tool for Neuroscience")
    print("-" * 40)
    
    print("""
    Use Case: Teaching tool for neuroscience students to analyze research papers.
    
    Steps:
    1. Students upload PDFs of assigned papers
    2. ARneuro extracts key information
    3. Interactive visualization of brain activation maps
    4. Methodological analysis exercises
    5. Comparison across studies
    
    Code Example:
    
    ```python
    from ARneuro.table_processing.table_pipeline import TableProcessingPipeline
    from ARneuro.text_processing.pipeline import TextProcessingPipeline
    
    # Educational analysis workflow
    def analyze_paper_for_education(pdf_path, student_name):
        # Process the paper
        text_results = text_pipeline.process_document(pdf_path)
        table_results = table_pipeline.process_markdown_file(pdf_path)
        
        # Extract educational content
        educational_data = {
            'student': student_name,
            'paper_title': extract_title(text_results),
            'main_findings': extract_findings(text_results),
            'brain_regions': extract_brain_regions(table_results),
            'methods_summary': summarize_methods(text_results),
            'critical_questions': generate_discussion_questions(text_results)
        }
        
        return educational_data
    
    # Process papers for a class
    student_papers = load_student_submissions()
    all_analyses = []
    
    for student, paper_path in student_papers.items():
        analysis = analyze_paper_for_education(paper_path, student)
        all_analyses.append(analysis)
    
    # Compare across students
    compare_analyses(all_analyses)
    ```
    """)
    
    print("\nIntegration with Existing Workflows")
    print("-" * 40)
    
    print("""
    ARneuro can be integrated with:
    
    1. **Jupyter Notebooks**: For interactive analysis
    2. **Web Applications**: As a backend API
    3. **Data Pipelines**: With pandas, numpy, scikit-learn
    4. **Visualization Tools**: Matplotlib, Plotly, NeuroSynth
    5. **Database Systems**: SQL, MongoDB, etc.
    
    Example Integration:
    
    ```python
    # Integration with pandas for data analysis
    import pandas as pd
    from ARneuro.core.pipeline import ARneuroPipeline
    
    # Process papers and convert to DataFrame
    pipeline = ARneuroPipeline(config)
    results = pipeline.process_pubmed_ids(['12345678', '23456789'])
    
    # Convert to pandas DataFrame
    df = pd.DataFrame([
        {
            'pmid': pid,
            'title': r.get('metadata', {}).get('title'),
            'year': r.get('metadata', {}).get('year'),
            'num_brain_tables': len(r.get('brain_activation_tables', [])),
            'methods_present': r.get('validation', {}).get('has_methods', False),
            'results_present': r.get('validation', {}).get('has_results', False)
        }
        for pid, r in results.items()
    ])
    
    # Analyze with pandas
    print(df.describe())
    print(df.groupby('year').size())
    ```
    """)
    
    print("\nBest Practices")
    print("-" * 40)
    
    print("""
    1. **Start Small**: Test with 2-3 papers before large batches
    2. **Validate Outputs**: Manually check results for accuracy
    3. **Handle Errors**: Implement proper error handling and logging
    4. **Cache Results**: Save intermediate results to avoid reprocessing
    5. **Monitor Resources**: OCR and LLM processing can be resource-intensive
    6. **Respect Rate Limits**: Be mindful of API rate limits for LLM services
    7. **Data Privacy**: Ensure compliance with data privacy regulations
    8. **Reproducibility**: Save configuration and version information
    
    Configuration Tips:
    
    ```yaml
    # config.yaml
    project:
      name: "Your Project Name"
      version: "1.0.0"
    
    paths:
      data_dir: "./data"
      output_dir: "./output"
      cache_dir: "./cache"  # For caching intermediate results
    
    processing:
      batch_size: 5  # Smaller batches for stability
      max_workers: 2  # Conservative parallel processing
      timeout: 600  # 10 minute timeout per paper
    
    llm:
      deepseek_api_key: ${DEEPSEEK_API_KEY}  # Use environment variables
      default_temperature: 0.1  # Lower temperature for consistent results
      max_tokens: 4096  # Limit token usage
    
    validation:
      require_methods: true
      require_results: true
      min_text_length: 500  # Minimum text for processing
    ```
    """)
    
    print("\n" + "=" * 60)
    print("Getting Started")
    print("=" * 60)
    
    print("""
    Quick Start:
    
    1. Install ARneuro:
       ```bash
       pip install -e .
       ```
    
    2. Configure your settings:
       ```bash
       cp config/config.example.yaml config/config.yaml
       # Edit config.yaml with your settings
       ```
    
    3. Run a test:
       ```bash
       python examples/basic_usage.py
       ```
    
    4. Process your first paper:
       ```python
       from ARneuro.core.pipeline import ARneuroPipeline
       
       pipeline = ARneuroPipeline()
       results = pipeline.process_pdf('path/to/your/paper.pdf')
       ```
    
    Support and Resources:
    
    - Documentation: See README.md and docstrings
    - Issues: Report bugs on GitHub
    - Examples: More examples in the examples/ directory
    - Configuration: Detailed configuration options in config/
    
    Happy researching with ARneuro!
    """)

if __name__ == "__main__":
    demonstrate_real_usage()