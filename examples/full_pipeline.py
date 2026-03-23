"""
Full pipeline example for ARneuro.

This example demonstrates the complete ARneuro pipeline from PDF to features.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ARneuro.core.pipeline import ARneuroPipeline
from ARneuro.config.config_manager import ConfigManager
from ARneuro.core.logger import setup_logger

def run_full_pipeline():
    """
    Run the complete ARneuro pipeline.
    """
    print("=" * 60)
    print("ARneuro Full Pipeline Example")
    print("=" * 60)
    
    # Setup logging
    setup_logger(level="INFO")
    
    # Load configuration
    print("\n1. Loading configuration...")
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Update with example configuration
    config.update({
        'project': {
            'name': 'ARneuro Example Project',
            'description': 'Example pipeline for language neuroscience literature review',
            'version': '1.0.0'
        },
        'paths': {
            'data_dir': './data',
            'output_dir': './output',
            'temp_dir': './temp',
            'log_dir': './logs'
        },
        'processing': {
            'batch_size': 10,
            'max_workers': 4,
            'timeout': 300
        },
        'llm': {
            'deepseek_api_key': 'your_api_key_here',  # Replace with actual key
            'openai_api_key': 'your_api_key_here',    # Replace with actual key
            'default_client': 'deepseek',
            'default_model': 'deepseek-chat'
        },
        'ocr_processing': {
            'backend': 'local',
            'model_path': '/storage/work/wuguowei/Bigmodel/GLM-OCR',
            'api_key': 'your_glm_api_key_here',
            'batch_size': 5
        },
        'validation': {
            'require_methods': True,
            'require_results': True,
            'min_text_length': 1000
        }
    })
    
    # Create directories
    print("\n2. Creating directories...")
    for dir_path in ['data', 'output', 'temp', 'logs']:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  - Created: {dir_path}/")
    
    # Create pipeline
    print("\n3. Creating ARneuro pipeline...")
    pipeline = ARneuroPipeline(config)
    
    # Example 1: PDF Download (simulated)
    print("\n4. Example: PDF Download Module")
    print("-" * 40)
    
    try:
        from ARneuro.data_fetch.pdf_downloader import PDFDownloader
        
        pdf_downloader = PDFDownloader(config)
        
        # Example PubMed IDs
        pubmed_ids = ['12345678', '23456789', '34567890']
        
        print(f"Simulating PDF download for {len(pubmed_ids)} PubMed IDs:")
        for pmid in pubmed_ids:
            print(f"  - PubMed ID: {pmid}")
        
        # In a real scenario, this would download PDFs
        print("\nNote: PDF download requires valid PubMed IDs and journal access.")
        print("For this example, we'll simulate downloaded PDFs.")
        
    except Exception as e:
        print(f"PDF download example setup failed: {e}")
    
    # Example 2: OCR Processing (simulated)
    print("\n5. Example: OCR Processing Module")
    print("-" * 40)
    
    try:
        from ARneuro.ocr_processing.glm_ocr import GLMOCRProcessor
        
        # Check if GLM-OCR path exists
        ocr_config = config.get('ocr_processing', {})
        glm_path = ocr_config.get('model_path', '')
        ocr_backend = ocr_config.get('backend', 'local')
        if ocr_backend == 'api':
            print('GLM-OCR API mode enabled.')
            ocr_processor = GLMOCRProcessor(config, backend='api')
            print('GLM-OCR API processor initialized successfully.')
        elif os.path.exists(glm_path):
            print(f"GLM-OCR found at: {glm_path}")
            ocr_processor = GLMOCRProcessor(config, backend='local')
            print('GLM-OCR local processor initialized successfully.')
        else:
            print(f"GLM-OCR not found at: {glm_path}")
            print('Using simulated OCR processing for this example.')
        
        # Simulate OCR processing
        print("\nSimulating OCR processing workflow:")
        print("  1. Load PDF files")
        print("  2. Extract images from PDF")
        print("  3. Run GLM-OCR locally or by online API")
        print("  4. Convert OCR results to markdown")
        print("  5. Save processed text")
        
    except Exception as e:
        print(f"OCR processing example setup failed: {e}")
    
    # Example 3: Complete Text Processing
    print("\n6. Example: Complete Text Processing")
    print("-" * 40)
    
    try:
        from ARneuro.text_processing.pipeline import TextProcessingPipeline
        
        text_pipeline = TextProcessingPipeline(config)
        
        # Create a more complex example markdown
        complex_md = """# The Neural Basis of Syntactic Processing: An fMRI Study

## Authors
Jane Smith, John Doe, Alice Johnson

## Abstract
This functional magnetic resonance imaging (fMRI) study investigated the neural correlates of syntactic processing during sentence comprehension. Twenty-four right-handed native English speakers performed a grammaticality judgment task while undergoing fMRI scanning.

## Introduction
Syntax, the set of rules governing sentence structure, is a fundamental component of human language. Previous neuroimaging studies have implicated left hemisphere regions, particularly Broca's area and the posterior superior temporal gyrus, in syntactic processing.

## Methods
### Participants
Twenty-four right-handed native English speakers (12 male, 12 female; mean age = 24.3 years, SD = 3.2) participated in the study.

### Experimental Design
The experiment used a 2 × 2 factorial design with factors: Sentence Type (grammatical vs. ungrammatical) and Complexity (simple vs. complex).

### fMRI Acquisition
Functional images were acquired on a 3T Siemens Prisma scanner using a 32-channel head coil. T2*-weighted echo-planar images were obtained with the following parameters: TR = 2000 ms, TE = 30 ms, flip angle = 90°, voxel size = 3 × 3 × 3 mm³.

### Behavioral Task
Participants performed a grammaticality judgment task. On each trial, a sentence was presented visually, and participants indicated whether the sentence was grammatical or not by pressing one of two buttons.

## Results
### Behavioral Results
Participants showed high accuracy in grammaticality judgments (mean accuracy = 92.4%, SD = 4.1%). Reaction times were significantly longer for ungrammatical sentences (p < 0.001).

### fMRI Results
Significant activation was observed in left inferior frontal gyrus (Broca's area) and left posterior superior temporal gyrus for syntactic violations compared to grammatical sentences.

| Contrast | Brain Region | BA | X | Y | Z | t-value | Cluster Size | p-value |
|----------|--------------|----|---|---|---|---------|--------------|---------|
| Ungrammatical > Grammatical | L. Inferior Frontal Gyrus | 44/45 | -48 | 12 | 24 | 6.32 | 245 | <0.001 |
| Ungrammatical > Grammatical | L. Posterior Superior Temporal Gyrus | 22 | -56 | -42 | 8 | 5.87 | 187 | <0.001 |
| Complex > Simple | L. Inferior Frontal Gyrus | 45 | -46 | 18 | 20 | 4.56 | 123 | 0.002 |

## Discussion
The results confirm the involvement of left hemisphere fronto-temporal regions in syntactic processing. The increased activation in Broca's area for ungrammatical sentences supports its role in syntactic reanalysis and repair.

## Limitations
The study used only written stimuli; future research should investigate auditory sentence processing.

## Conclusion
This study provides further evidence for the neural basis of syntactic processing in the human brain.

## References
1. Friederici, A. D. (2011). The brain basis of language processing. *Nature Reviews Neuroscience*.
2. Hagoort, P. (2005). On Broca, brain, and binding. *Trends in Cognitive Sciences*.
"""
        
        # Save complex example
        complex_file = './data/complex_paper.md'
        with open(complex_file, 'w', encoding='utf-8') as f:
            f.write(complex_md)
        
        print(f"Created complex example markdown file: {complex_file}")
        
        # Process the document
        print("\nProcessing complex document...")
        results = text_pipeline.process_document(
            markdown_file=complex_file,
            output_dir='./output/complex_text_processing',
            use_llm=False  # Use rule-based for example
        )
        
        print(f"\nText processing results:")
        print(f"  - Sections extracted: {results['statistics']['num_sections']}")
        print(f"  - Tables extracted: {results['statistics']['num_tables']}")
        
        if 'validation' in results['statistics']:
            validation = results['statistics']['validation']
            print(f"  - Has Methods section: {validation.get('has_methods', False)}")
            print(f"  - Has Results section: {validation.get('has_results', False)}")
        
        print(f"  - Output files saved to: ./output/complex_text_processing/")
        
    except Exception as e:
        print(f"Complex text processing failed: {e}")
    
    # Example 4: Integrated Pipeline
    print("\n7. Example: Integrated ARneuro Pipeline")
    print("-" * 40)
    
    print("The complete ARneuro pipeline integrates all modules:")
    print("\n1. PDF Fetching → Download research papers")
    print("2. OCR Processing → Convert PDFs to text")
    print("3. Text Processing → Segment and organize content")
    print("4. Table Processing → Extract and analyze tables")
    print("5. Feature Extraction → Identify linguistic/cognitive features")
    print("6. Output Generation → Save structured results")
    
    print("\nTo run the full pipeline with your data:")
    print("""
    # Import the pipeline
    from ARneuro.core.pipeline import ARneuroPipeline
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Update with your settings
    config.update({
        'llm': {
            'deepseek_api_key': 'your_actual_key',
            'openai_api_key': 'your_actual_key'
        },
        'ocr_processing': {
            'backend': 'local',
            'model_path': '/your/path/to/GLM-OCR',
            'api_key': 'your_glm_api_key_here'
        }
    })
    
    # Create and run pipeline
    pipeline = ARneuroPipeline(config)
    
    # Process PubMed IDs
    results = pipeline.process_pubmed_ids(
        pubmed_ids=['12345678', '23456789'],
        output_dir='./results'
    )
    
    # Or process existing PDFs
    results = pipeline.process_pdfs(
        pdf_dir='./data/pdfs',
        output_dir='./results'
    )
    """)
    
    # Summary
    print("\n" + "=" * 60)
    print("ARneuro Full Pipeline Example Completed")
    print("=" * 60)
    
    print("\nGenerated files and directories:")
    print("  ./data/ - Example input files")
    print("  ./output/ - Processing results")
    print("    ├── complex_text_processing/ - Text segmentation results")
    print("    └── [other output directories]")
    
    print("\nNext steps:")
    print("  1. Install required dependencies: pip install -r requirements.txt")
    print("  2. Configure API keys in config.yaml or environment variables")
    print("  3. Choose local GLM-OCR or configure the online API key")
    print("  4. Add your PDF files or PubMed IDs")
    print("  5. Run the pipeline with your data")
    
    print("\nFor more information, see the README.md file.")

if __name__ == "__main__":
    run_full_pipeline()