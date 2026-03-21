"""
Basic usage example for ARneuro.

This example demonstrates how to use the ARneuro pipeline for literature review.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ARneuro.core.pipeline import ARneuroPipeline
from ARneuro.config.config_manager import ConfigManager
from ARneuro.core.logger import setup_logger

def main():
    """
    Main function demonstrating basic ARneuro usage.
    """
    print("=" * 60)
    print("ARneuro - Literature Quantitative Review Tool")
    print("=" * 60)
    
    # Setup logging
    setup_logger()
    
    # Load configuration
    print("\n1. Loading configuration...")
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Update configuration with example values
    config.update({
        'data_dir': './data',
        'output_dir': './output',
        'glm_ocr_path': '/storage/work/wuguowei/Bigmodel/GLM-OCR',
        'deepseek_api_key': 'your_api_key_here',  # Replace with actual key
        'openai_api_key': 'your_api_key_here',    # Replace with actual key
    })
    
    # Create pipeline
    print("\n2. Creating ARneuro pipeline...")
    pipeline = ARneuroPipeline(config)
    
    # Example 1: Process a single paper
    print("\n3. Example: Process a single paper")
    print("-" * 40)
    
    # Create example data directory
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    
    # Create example markdown file
    example_md = """# Neural Mechanisms of Language Processing

## Abstract
This study investigates the neural correlates of syntactic processing using fMRI.

## Introduction
Language processing involves multiple brain regions...

## Methods
### Participants
20 right-handed native English speakers participated.

### Task Design
Participants performed a syntactic violation detection task during fMRI scanning.

### fMRI Acquisition
Images were acquired using a 3T scanner...

## Results
### Behavioral Results
Participants showed high accuracy in detecting syntactic violations.

### fMRI Results
Significant activation was found in Broca's area and Wernicke's area.

| Brain Region | X | Y | Z | t-value | p-value |
|--------------|---|---|---|---------|---------|
| Broca's area | 45 | 12 | 30 | 5.67 | 0.001 |
| Wernicke's area | -55 | -40 | 20 | 4.89 | 0.003 |

## Discussion
The findings support the role of left hemisphere regions in syntactic processing...

## References
1. Smith et al. (2020). Language and the brain.
"""
    
    # Save example markdown file
    example_file = './data/example_paper.md'
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(example_md)
    
    print(f"Created example markdown file: {example_file}")
    
    # Example 2: Text processing
    print("\n4. Example: Text processing pipeline")
    print("-" * 40)
    
    try:
        from ARneuro.text_processing.pipeline import TextProcessingPipeline
        
        text_pipeline = TextProcessingPipeline(config)
        
        # Process the example document
        print(f"Processing document: {example_file}")
        text_results = text_pipeline.process_document(
            markdown_file=example_file,
            output_dir='./output/text_processing',
            use_llm=False  # Don't use LLM for this example
        )
        
        print(f"Text processing completed:")
        print(f"  - Sections extracted: {text_results['statistics']['num_sections']}")
        print(f"  - Tables extracted: {text_results['statistics']['num_tables']}")
        print(f"  - Results saved to: {text_results['output_files']['segmentation']}")
        
    except Exception as e:
        print(f"Text processing example failed: {e}")
    
    # Example 3: Table processing
    print("\n5. Example: Table processing pipeline")
    print("-" * 40)
    
    try:
        from ARneuro.table_processing.table_pipeline import TableProcessingPipeline
        
        table_pipeline = TableProcessingPipeline(config)
        
        # Process tables from the example document
        print(f"Processing tables from: {example_file}")
        table_results = table_pipeline.process_markdown_file(
            markdown_file=example_file,
            output_dir='./output/table_processing',
            process_brain_tables=True,
            llm_client_type='deepseek'  # Will use rule-based without API key
        )
        
        print(f"Table processing completed:")
        print(f"  - Tables extracted: {table_results.get('tables_extracted', 0)}")
        if 'brain_activation_tables' in table_results:
            brain_tables = table_results['brain_activation_tables']
            print(f"  - Brain activation tables found: {brain_tables.get('total_brain_tables', 0)}")
        
    except Exception as e:
        print(f"Table processing example failed: {e}")
    
    # Example 4: Feature extraction
    print("\n6. Example: Feature extraction pipeline")
    print("-" * 40)
    
    try:
        from ARneuro.feature_extraction.feature_pipeline import FeatureExtractionPipeline
        
        feature_pipeline = FeatureExtractionPipeline(config)
        
        # Extract features from example text
        example_text = """This fMRI study examined syntactic processing using a violation detection task. 
        Participants listened to sentences with syntactic violations while undergoing fMRI scanning. 
        The task required participants to detect grammatical errors in real-time."""
        
        print("Extracting features from example text...")
        feature_results = feature_pipeline.extract_from_paper(
            paper_id='example_paper_001',
            paper_text=example_text,
            methods_section="Participants performed a syntactic violation detection task during fMRI scanning.",
            output_dir='./output/feature_extraction',
            llm_client_type='deepseek'  # Will use rule-based without API key
        )
        
        print(f"Feature extraction completed:")
        print(f"  - Paper ID: {feature_results.get('paper_id')}")
        if 'task_features' in feature_results:
            task_features = feature_results['task_features']
            if 'metadata' in task_features:
                metadata = task_features['metadata']
                print(f"  - Total features: {metadata.get('num_features_total', 0)}")
                print(f"  - Features present: {metadata.get('num_features_present', 0)}")
        
    except Exception as e:
        print(f"Feature extraction example failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ARneuro Example Completed")
    print("=" * 60)
    print("\nOutput directories created:")
    print("  - ./data/ - Example input data")
    print("  - ./output/ - Processing results")
    print("\nTo run with actual data:")
    print("  1. Update API keys in the configuration")
    print("  2. Add your PDF files to ./data/pdfs/")
    print("  3. Run the full pipeline with your data")
    print("\nFor more examples, see the documentation.")

if __name__ == "__main__":
    main()