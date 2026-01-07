"""
Unified XAI Platform
A multi-modal classification platform with Explainable AI capabilities
"""

__version__ = "1.0.0"
__author__ = "TD Group"

from utils.compatibility import (
    CompatibilityRegistry,
    get_compatible_models,
    get_compatible_xai,
    get_class_names
)

__all__ = [
    'CompatibilityRegistry',
    'get_compatible_models',
    'get_compatible_xai',
    'get_class_names'
]
