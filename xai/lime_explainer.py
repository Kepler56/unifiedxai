"""
LIME (Local Interpretable Model-agnostic Explanations) Explainer
Provides interpretable explanations for image classification models
"""

import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from typing import Tuple, Optional, Callable
import warnings

warnings.filterwarnings('ignore')


class LimeExplainer:
    """
    LIME explainer for image-based classification models.
    Works with both audio spectrograms and chest X-ray images.
    """
    
    def __init__(self, model, class_names: list = None):
        """
        Initialize the LIME explainer.
        
        Args:
            model: Keras/TensorFlow model with predict method
            class_names: List of class names for display
        """
        self.model = model
        self.class_names = class_names or ['Class 0', 'Class 1']
        self.explainer = lime_image.LimeImageExplainer()
        self.last_explanation = None
    
    def predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function wrapper for LIME.
        
        Args:
            images: Batch of images (N, H, W, C)
        
        Returns:
            Model predictions
        """
        # Normalize images if needed
        if images.max() > 1:
            images = images / 255.0
        return self.model.predict(images, verbose=0)
    
    def explain(self, 
                image: np.ndarray, 
                num_samples: int = 1000,
                num_features: int = 10,
                hide_color: int = 0,
                positive_only: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate LIME explanation for an image.
        
        Args:
            image: Input image (H, W, C) - can be 0-255 or 0-1 range
            num_samples: Number of perturbed samples to generate
            num_features: Number of superpixels to include in explanation
            hide_color: Color to use for hiding superpixels
            positive_only: Show only positive contributions
        
        Returns:
            Tuple of (explanation_image, mask, explanation_dict)
        """
        # Ensure image is in correct format
        if image.max() <= 1:
            image_for_lime = (image * 255).astype('uint8')
        else:
            image_for_lime = image.astype('uint8')
        
        # Normalize for prediction
        image_normalized = image_for_lime.astype('float64') / 255.0
        
        # Get model prediction
        prediction = self.predict_fn(np.expand_dims(image_normalized, axis=0))
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Generate LIME explanation
        self.last_explanation = self.explainer.explain_instance(
            image_normalized,
            self.predict_fn,
            top_labels=len(self.class_names),
            hide_color=hide_color,
            num_samples=num_samples
        )
        
        # Get image and mask for the predicted class
        temp, mask = self.last_explanation.get_image_and_mask(
            predicted_class,
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=False
        )
        
        explanation_dict = {
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': float(confidence),
            'num_features': num_features,
            'num_samples': num_samples
        }
        
        return temp, mask, explanation_dict
    
    def visualize(self,
                  image: np.ndarray,
                  num_samples: int = 1000,
                  num_features: int = 10,
                  figsize: Tuple[int, int] = (12, 5),
                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate and visualize LIME explanation.
        
        Args:
            image: Input image
            num_samples: Number of perturbed samples
            num_features: Number of superpixels to show
            figsize: Figure size
            save_path: Optional path to save the figure
        
        Returns:
            Matplotlib figure
        """
        temp, mask, exp_dict = self.explain(image, num_samples, num_features)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        if image.max() <= 1:
            display_image = image
        else:
            display_image = image / 255.0
        
        axes[0].imshow(display_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # LIME explanation with boundaries
        axes[1].imshow(mark_boundaries(temp, mask))
        axes[1].set_title(f'LIME Explanation\n({num_features} features)')
        axes[1].axis('off')
        
        # Mask overlay
        masked_image = display_image.copy()
        mask_overlay = np.zeros_like(display_image)
        mask_overlay[mask == 1] = [0, 1, 0]  # Green for positive regions
        blended = 0.7 * masked_image + 0.3 * mask_overlay
        
        axes[2].imshow(np.clip(blended, 0, 1))
        axes[2].set_title(f'Prediction: {exp_dict["class_name"]}\nConfidence: {exp_dict["confidence"]:.2%}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def get_feature_importance(self, top_k: int = 10) -> list:
        """
        Get feature importance scores from the last explanation.
        
        Args:
            top_k: Number of top features to return
        
        Returns:
            List of (feature_id, importance_score) tuples
        """
        if self.last_explanation is None:
            raise ValueError("No explanation generated yet. Call explain() first.")
        
        predicted_class = np.argmax(
            self.predict_fn(np.expand_dims(self.last_explanation.image, axis=0))[0]
        )
        
        # Get local explanation
        local_exp = self.last_explanation.local_exp[predicted_class]
        
        # Sort by absolute importance
        sorted_exp = sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)
        
        return sorted_exp[:top_k]


def lime_explain_image(model, image: np.ndarray, class_names: list = None, 
                       num_samples: int = 1000, num_features: int = 10) -> plt.Figure:
    """
    Convenience function to generate LIME explanation.
    
    Args:
        model: Classification model
        image: Input image
        class_names: List of class names
        num_samples: Number of perturbed samples
        num_features: Number of features to show
    
    Returns:
        Matplotlib figure with explanation
    """
    explainer = LimeExplainer(model, class_names)
    return explainer.visualize(image, num_samples, num_features)
