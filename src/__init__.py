"""
Image Similarity Search - Core Module
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .feature_extractor import FeatureExtractor
from .similarity_search import SimilaritySearch
from .preprocessing import preprocess_image, ImagePreprocessor
from .utils import allowed_file, save_uploaded_file, get_image_paths

__all__ = [
    'FeatureExtractor',
    'SimilaritySearch',
    'preprocess_image',
    'ImagePreprocessor',
    'allowed_file',
    'save_uploaded_file',
    'get_image_paths'
]