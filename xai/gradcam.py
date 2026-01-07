"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Visualizes which regions of an image are important for classification decisions
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class GradCAM:
    """
    Grad-CAM implementation for convolutional neural networks.
    Compatible with any Keras/TensorFlow model with convolutional layers.
    """
    
    def __init__(self, model, last_conv_layer_name: str = None, class_names: list = None):
        """
        Initialize Grad-CAM explainer.
        
        Args:
            model: Keras/TensorFlow model
            last_conv_layer_name: Name of the last convolutional layer
            class_names: List of class names for display
        """
        self.model = model
        self.class_names = class_names or ['Class 0', 'Class 1']
        self.last_conv_layer_name = last_conv_layer_name or self._find_last_conv_layer()
        self.grad_model = self._build_grad_model()
    
    def _find_last_conv_layer(self) -> str:
        """
        Automatically find the last convolutional layer in the model.
        
        Returns:
            Name of the last conv layer
        """
        for layer in reversed(self.model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Activation)):
                if hasattr(layer, 'output'):
                    if len(layer.output.shape) == 4:  # Conv layer output shape
                        return layer.name
        
        # If no conv layer found, try to find in nested models
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'layers'):  # Nested model
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        return f"{layer.name}/{sub_layer.name}"
        
        raise ValueError("Could not find a convolutional layer in the model")
    
    def _build_grad_model(self) -> tf.keras.Model:
        """
        Build a model that outputs both the conv layer activations and predictions.
        
        Returns:
            Gradient model for Grad-CAM computation
        """
        try:
            last_conv_layer = self.model.get_layer(self.last_conv_layer_name)
            grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [last_conv_layer.output, self.model.output]
            )
            return grad_model
        except ValueError:
            # Handle nested models (e.g., transfer learning)
            if '/' in self.last_conv_layer_name:
                parent_name, child_name = self.last_conv_layer_name.split('/')
                parent_layer = self.model.get_layer(parent_name)
                last_conv_layer = parent_layer.get_layer(child_name)
                
                # Build intermediate model
                intermediate_model = tf.keras.models.Model(
                    parent_layer.input,
                    last_conv_layer.output
                )
                
                # Build gradient model
                grad_model = tf.keras.models.Model(
                    [self.model.inputs],
                    [intermediate_model(self.model.inputs), self.model.output]
                )
                return grad_model
            raise
    
    def compute_heatmap(self, 
                        image: np.ndarray, 
                        class_idx: int = None,
                        normalize: bool = True) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM heatmap for an image.
        
        Args:
            image: Input image (H, W, C) or (1, H, W, C)
            class_idx: Target class index (None for predicted class)
            normalize: Whether to normalize heatmap to 0-1 range
        
        Returns:
            Tuple of (heatmap, class_index, confidence)
        """
        # Prepare image
        if len(image.shape) == 3:
            img_array = np.expand_dims(image, axis=0)
        else:
            img_array = image
        
        # Normalize if needed
        if img_array.max() > 1:
            img_array = img_array / 255.0
        
        img_tensor = tf.cast(img_array, tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = self.grad_model(img_tensor)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            class_output = predictions[:, class_idx]
        
        # Get gradients of the predicted class with respect to conv layer output
        grads = tape.gradient(class_output, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        if normalize and tf.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.reduce_max(heatmap)
        
        confidence = float(predictions[0][class_idx])
        
        return heatmap.numpy(), int(class_idx), confidence
    
    def overlay_heatmap(self,
                        heatmap: np.ndarray,
                        original_image: np.ndarray,
                        alpha: float = 0.4,
                        colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            heatmap: Grad-CAM heatmap
            original_image: Original input image
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap to use
        
        Returns:
            Superimposed image
        """
        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Convert to uint8 and apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Prepare original image
        if original_image.max() <= 1:
            original_uint8 = np.uint8(255 * original_image)
        else:
            original_uint8 = np.uint8(original_image)
        
        # Superimpose
        superimposed = cv2.addWeighted(original_uint8, 1 - alpha, heatmap_colored, alpha, 0)
        
        return superimposed
    
    def explain(self, 
                image: np.ndarray, 
                class_idx: int = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate Grad-CAM explanation for an image.
        
        Args:
            image: Input image (H, W, C)
            class_idx: Target class index (None for predicted class)
        
        Returns:
            Tuple of (heatmap, superimposed_image, explanation_dict)
        """
        heatmap, pred_class, confidence = self.compute_heatmap(image, class_idx)
        superimposed = self.overlay_heatmap(heatmap, image)
        
        explanation_dict = {
            'predicted_class': pred_class,
            'class_name': self.class_names[pred_class],
            'confidence': confidence,
            'last_conv_layer': self.last_conv_layer_name
        }
        
        return heatmap, superimposed, explanation_dict
    
    def visualize(self,
                  image: np.ndarray,
                  class_idx: int = None,
                  figsize: Tuple[int, int] = (15, 5),
                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate and visualize Grad-CAM explanation.
        
        Args:
            image: Input image
            class_idx: Target class index
            figsize: Figure size
            save_path: Optional path to save the figure
        
        Returns:
            Matplotlib figure
        """
        heatmap, superimposed, exp_dict = self.explain(image, class_idx)
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Original image
        if image.max() <= 1:
            display_image = image
        else:
            display_image = image / 255.0
        
        axes[0].imshow(display_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Superimposed
        axes[2].imshow(superimposed)
        axes[2].set_title('Grad-CAM Overlay')
        axes[2].axis('off')
        
        # Info panel
        axes[3].axis('off')
        info_text = f"""
Grad-CAM Analysis
─────────────────
Predicted Class: {exp_dict['class_name']}
Confidence: {exp_dict['confidence']:.2%}
Conv Layer: {exp_dict['last_conv_layer']}

The heatmap shows regions
that most influenced the
model's prediction.

Red = High importance
Blue = Low importance
        """
        axes[3].text(0.1, 0.5, info_text, transform=axes[3].transAxes,
                     fontsize=10, verticalalignment='center', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def gradcam_explain_image(model, image: np.ndarray, last_conv_layer: str = None,
                          class_names: list = None, class_idx: int = None) -> plt.Figure:
    """
    Convenience function to generate Grad-CAM explanation.
    
    Args:
        model: Classification model
        image: Input image
        last_conv_layer: Name of last conv layer
        class_names: List of class names
        class_idx: Target class index
    
    Returns:
        Matplotlib figure with explanation
    """
    explainer = GradCAM(model, last_conv_layer, class_names)
    return explainer.visualize(image, class_idx)


# Predefined conv layer names for common models
CONV_LAYER_NAMES = {
    # Audio models
    'VGG16': 'block5_conv3',
    'MobileNetV2': 'Conv_1',
    'ResNet50': 'conv5_block3_out',
    'InceptionV3': 'mixed10',
    'CustomCNN': 'conv2d_2',
    
    # Image models
    'AlexNet': 'conv5',
    'DenseNet': 'conv5_block16_concat'
}


def get_conv_layer_name(model_name: str) -> Optional[str]:
    """
    Get the appropriate conv layer name for a model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Conv layer name or None
    """
    return CONV_LAYER_NAMES.get(model_name)
