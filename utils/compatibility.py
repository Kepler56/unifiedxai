"""
Compatibility Registry
Manages model and XAI technique compatibility based on input types
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class InputType(Enum):
    """Supported input types"""
    AUDIO = "audio"
    IMAGE = "image"


# Input type identifiers
INPUT_TYPES = {
    'audio': InputType.AUDIO,
    'image': InputType.IMAGE
}


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    display_name: str
    input_type: InputType
    description: str
    last_conv_layer: str
    input_size: Tuple[int, int, int] = (224, 224, 3)


@dataclass
class XAIInfo:
    """Information about an XAI technique"""
    name: str
    display_name: str
    description: str
    compatible_input_types: List[InputType]
    requires_conv_layer: bool = False


# Registry of all available models
MODELS_REGISTRY: Dict[str, ModelInfo] = {
    # Audio models (Deepfake Detection)
    'VGG16': ModelInfo(
        name='VGG16',
        display_name='VGG16',
        input_type=InputType.AUDIO,
        description='VGG16 transfer learning model for audio spectrogram classification',
        last_conv_layer='block5_conv3'
    ),
    'MobileNet': ModelInfo(
        name='MobileNet',
        display_name='MobileNet',
        input_type=InputType.AUDIO,
        description='MobileNet v1 with 91.5% accuracy - matches the saved model from Audio_classifier.ipynb',
        last_conv_layer='conv_pw_13_relu'
    ),
    'ResNet50': ModelInfo(
        name='ResNet50',
        display_name='ResNet50',
        input_type=InputType.AUDIO,
        description='Deep residual network for audio classification',
        last_conv_layer='conv5_block3_out'
    ),
    'InceptionV3': ModelInfo(
        name='InceptionV3',
        display_name='InceptionV3',
        input_type=InputType.AUDIO,
        description='Google Inception architecture for spectrogram analysis',
        last_conv_layer='mixed10'
    ),
    
    # Image models (Lung Cancer Detection)
    'AlexNet': ModelInfo(
        name='AlexNet',
        display_name='AlexNet',
        input_type=InputType.IMAGE,
        description='Classic AlexNet architecture for chest X-ray classification',
        last_conv_layer='conv5'
    ),
    'DenseNet': ModelInfo(
        name='DenseNet',
        display_name='DenseNet121',
        input_type=InputType.IMAGE,
        description='DenseNet121 with dense connections for better feature propagation',
        last_conv_layer='conv5_block16_concat'
    )
}


# Registry of all available XAI techniques
XAI_REGISTRY: Dict[str, XAIInfo] = {
    'LIME': XAIInfo(
        name='LIME',
        display_name='LIME (Local Interpretable Model-agnostic Explanations)',
        description='Explains predictions by perturbing superpixels and learning local approximations',
        compatible_input_types=[InputType.AUDIO, InputType.IMAGE],
        requires_conv_layer=False
    ),
    'GradCAM': XAIInfo(
        name='GradCAM',
        display_name='Grad-CAM (Gradient-weighted Class Activation Mapping)',
        description='Visualizes important regions using gradients from the last convolutional layer',
        compatible_input_types=[InputType.AUDIO, InputType.IMAGE],
        requires_conv_layer=True
    ),
    'SHAP': XAIInfo(
        name='SHAP',
        display_name='SHAP (SHapley Additive exPlanations)',
        description='Uses Shapley values to explain feature contributions to predictions',
        compatible_input_types=[InputType.AUDIO, InputType.IMAGE],
        requires_conv_layer=False
    )
}


# Class labels for each input type
CLASS_LABELS = {
    InputType.AUDIO: ['Real', 'Fake'],
    InputType.IMAGE: ['Benign', 'Malignant']
}


class CompatibilityRegistry:
    """
    Registry for managing model and XAI technique compatibility.
    Automatically filters options based on input type.
    """
    
    def __init__(self):
        self.models = MODELS_REGISTRY
        self.xai_techniques = XAI_REGISTRY
        self.class_labels = CLASS_LABELS
    
    def get_models_for_input_type(self, input_type: InputType) -> Dict[str, ModelInfo]:
        """
        Get all models compatible with a given input type.
        
        Args:
            input_type: The input type (AUDIO or IMAGE)
        
        Returns:
            Dictionary of compatible models
        """
        return {
            name: info for name, info in self.models.items()
            if info.input_type == input_type
        }
    
    def get_xai_for_input_type(self, input_type: InputType) -> Dict[str, XAIInfo]:
        """
        Get all XAI techniques compatible with a given input type.
        
        Args:
            input_type: The input type (AUDIO or IMAGE)
        
        Returns:
            Dictionary of compatible XAI techniques
        """
        return {
            name: info for name, info in self.xai_techniques.items()
            if input_type in info.compatible_input_types
        }
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            ModelInfo object or None
        """
        return self.models.get(model_name)
    
    def get_xai_info(self, xai_name: str) -> Optional[XAIInfo]:
        """
        Get information about a specific XAI technique.
        
        Args:
            xai_name: Name of the XAI technique
        
        Returns:
            XAIInfo object or None
        """
        return self.xai_techniques.get(xai_name)
    
    def get_class_labels(self, input_type: InputType) -> List[str]:
        """
        Get class labels for a given input type.
        
        Args:
            input_type: The input type
        
        Returns:
            List of class labels
        """
        return self.class_labels.get(input_type, ['Class 0', 'Class 1'])
    
    def get_last_conv_layer(self, model_name: str) -> Optional[str]:
        """
        Get the last convolutional layer name for a model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Layer name or None
        """
        model_info = self.get_model_info(model_name)
        return model_info.last_conv_layer if model_info else None
    
    def is_xai_compatible(self, xai_name: str, input_type: InputType) -> bool:
        """
        Check if an XAI technique is compatible with an input type.
        
        Args:
            xai_name: Name of the XAI technique
            input_type: The input type
        
        Returns:
            True if compatible, False otherwise
        """
        xai_info = self.get_xai_info(xai_name)
        return xai_info is not None and input_type in xai_info.compatible_input_types
    
    def get_model_names_list(self, input_type: InputType) -> List[str]:
        """
        Get list of model names for an input type.
        
        Args:
            input_type: The input type
        
        Returns:
            List of model names
        """
        models = self.get_models_for_input_type(input_type)
        return list(models.keys())
    
    def get_xai_names_list(self, input_type: InputType) -> List[str]:
        """
        Get list of XAI technique names for an input type.
        
        Args:
            input_type: The input type
        
        Returns:
            List of XAI names
        """
        xai = self.get_xai_for_input_type(input_type)
        return list(xai.keys())


# Convenience functions
def get_compatible_models(input_type: str) -> List[str]:
    """
    Get list of compatible model names for an input type.
    
    Args:
        input_type: 'audio' or 'image'
    
    Returns:
        List of model names
    """
    registry = CompatibilityRegistry()
    it = INPUT_TYPES.get(input_type.lower())
    if it is None:
        return []
    return registry.get_model_names_list(it)


def get_compatible_xai(input_type: str) -> List[str]:
    """
    Get list of compatible XAI technique names for an input type.
    
    Args:
        input_type: 'audio' or 'image'
    
    Returns:
        List of XAI names
    """
    registry = CompatibilityRegistry()
    it = INPUT_TYPES.get(input_type.lower())
    if it is None:
        return []
    return registry.get_xai_names_list(it)


def get_class_names(input_type: str) -> List[str]:
    """
    Get class names for an input type.
    
    Args:
        input_type: 'audio' or 'image'
    
    Returns:
        List of class names
    """
    registry = CompatibilityRegistry()
    it = INPUT_TYPES.get(input_type.lower())
    if it is None:
        return ['Class 0', 'Class 1']
    return registry.get_class_labels(it)


def get_model_description(model_name: str) -> str:
    """
    Get description for a model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Model description
    """
    registry = CompatibilityRegistry()
    model_info = registry.get_model_info(model_name)
    return model_info.description if model_info else "No description available"


def get_xai_description(xai_name: str) -> str:
    """
    Get description for an XAI technique.
    
    Args:
        xai_name: Name of the XAI technique
    
    Returns:
        XAI description
    """
    registry = CompatibilityRegistry()
    xai_info = registry.get_xai_info(xai_name)
    return xai_info.description if xai_info else "No description available"
