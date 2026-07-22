"""
Text processing module for ARneuro.

This module handles document segmentation, heading classification, and text reorganization.
"""

from .document_segmentation import DocumentSegmenter
from .heading_classifier import HeadingClassifier
from .enhanced_segmentation import EnhancedSegmenter
from .text_reorganizer import TextReorganizer
from .pipeline import TextProcessingPipeline
from .final_segmentation import FinalSeg, FinalSegResult
from .other_method_recovery import (
    OtherMethodRecoveryResult,
    OtherMethodSectionRecoverer,
    recover_methods_from_other,
    split_other_into_blocks,
)

__all__ = [
    'DocumentSegmenter', 
    'HeadingClassifier',
    'EnhancedSegmenter',
    'TextReorganizer',
    'TextProcessingPipeline',
    'FinalSeg',
    'FinalSegResult',
    'OtherMethodRecoveryResult',
    'OtherMethodSectionRecoverer',
    'recover_methods_from_other',
    'split_other_into_blocks',
]
