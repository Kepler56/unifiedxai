"""
SHAP (SHapley Additive exPlanations) Explainer
Provides model-agnostic explanations based on Shapley values
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
from skimage.segmentation import slic
from typing import Tuple, Optional, Callable
import warnings

warnings.filterwarnings('ignore')


class ShapExplainer:
    """
    SHAP explainer for image classification models.
    Uses KernelExplainer for model-agnostic explanations with superpixel segmentation.
    """
    
    def __init__(self, model, class_names: list = None, background_samples: int = 50):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Keras/TensorFlow model with predict method
            class_names: List of class names for display
            background_samples: Number of background samples for SHAP
        """
        self.model = model
        self.class_names = class_names or ['Class 0', 'Class 1']
        self.background_samples = background_samples
        self.explainer = None
        self.segments = None
    
    def _segment_image(self, image: np.ndarray, n_segments: int = 50) -> np.ndarray:
        """
        Segment image into superpixels using SLIC.
        
        Args:
            image: Input image
            n_segments: Number of segments
        
        Returns:
            Segmentation mask
        """
        if image.max() <= 1:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        self.segments = slic(image_uint8, n_segments=n_segments, compactness=10, sigma=1)
        return self.segments
    
    def _mask_image(self, image: np.ndarray, mask: np.ndarray, background: float = 0) -> np.ndarray:
        """
        Apply binary mask to image using superpixels.
        
        Args:
            image: Input image
            mask: Binary mask for superpixels
            background: Background color value
        
        Returns:
            Masked image
        """
        masked = image.copy()
        for i, val in enumerate(mask):
            if val == 0:
                masked[self.segments == i] = background
        return masked
    
    def predict_fn(self, masks: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Prediction function for SHAP that takes binary masks.
        
        Args:
            masks: Binary masks for superpixels (N, num_segments)
            image: Original image
        
        Returns:
            Model predictions
        """
        predictions = []
        for mask in masks:
            masked_img = self._mask_image(image, mask)
            if masked_img.max() > 1:
                masked_img = masked_img / 255.0
            pred = self.model.predict(np.expand_dims(masked_img, axis=0), verbose=0)
            predictions.append(pred[0])
        return np.array(predictions)
    
    def explain(self,
                image: np.ndarray,
                n_segments: int = 50,
                nsamples: int = 100) -> Tuple[np.ndarray, dict]:
        """
        Generate SHAP explanation for an image.
        
        Args:
            image: Input image (H, W, C)
            n_segments: Number of superpixels
            nsamples: Number of samples for SHAP
        
        Returns:
            Tuple of (shap_values, explanation_dict)
        """
        # Segment the image
        segments = self._segment_image(image, n_segments)
        n_features = len(np.unique(segments))
        
        # Create prediction function
        def predict_wrapper(masks):
            return self.predict_fn(masks, image)
        
        # Create background (all segments visible)
        background = np.ones((1, n_features))
        
        # Create SHAP explainer
        self.explainer = shap.KernelExplainer(predict_wrapper, background)
        
        # Generate explanation
        test_mask = np.ones((1, n_features))
        shap_values = self.explainer.shap_values(test_mask, nsamples=nsamples)
        
        # Get prediction
        if image.max() > 1:
            img_normalized = image / 255.0
        else:
            img_normalized = image
        
        prediction = self.model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        explanation_dict = {
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': float(confidence),
            'n_segments': n_features,
            'nsamples': nsamples
        }
        
        return shap_values, explanation_dict
    
    def create_shap_heatmap(self, 
                            image: np.ndarray, 
                            shap_values: np.ndarray,
                            class_idx: int) -> np.ndarray:
        """
        Create a heatmap from SHAP values.
        
        Args:
            image: Original image
            shap_values: SHAP values for each segment
            class_idx: Target class index
        
        Returns:
            Heatmap array
        """
        if self.segments is None:
            raise ValueError("No segmentation available. Call explain() first.")
        
        # Get SHAP values for the target class
        if isinstance(shap_values, list):
            class_shap = shap_values[class_idx][0]
        else:
            class_shap = shap_values[0]
        
        # Create heatmap
        heatmap = np.zeros(image.shape[:2])
        for i, val in enumerate(class_shap):
            heatmap[self.segments == i] = val
        
        # Normalize
        if np.abs(heatmap).max() > 0:
            heatmap = heatmap / np.abs(heatmap).max()
        
        return heatmap
    
    def visualize(self,
                  image: np.ndarray,
                  n_segments: int = 50,
                  nsamples: int = 100,
                  figsize: Tuple[int, int] = (15, 5),
                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate and visualize SHAP explanation.
        
        Args:
            image: Input image
            n_segments: Number of superpixels
            nsamples: Number of SHAP samples
            figsize: Figure size
            save_path: Optional path to save the figure
        
        Returns:
            Matplotlib figure
        """
        shap_values, exp_dict = self.explain(image, n_segments, nsamples)
        
        # Create heatmap
        heatmap = self.create_shap_heatmap(image, shap_values, exp_dict['predicted_class'])
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Original image
        if image.max() <= 1:
            display_image = image
        else:
            display_image = image / 255.0
        
        axes[0].imshow(display_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation
        axes[1].imshow(display_image)
        axes[1].imshow(self.segments, alpha=0.3, cmap='nipy_spectral')
        axes[1].set_title(f'Superpixels ({exp_dict["n_segments"]})')
        axes[1].axis('off')
        
        # SHAP heatmap
        im = axes[2].imshow(heatmap, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2].set_title('SHAP Values')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Overlay
        overlay = display_image.copy()
        # Red for positive, blue for negative
        positive_mask = heatmap > 0.1
        negative_mask = heatmap < -0.1
        overlay_colored = np.zeros_like(overlay)
        overlay_colored[positive_mask] = [1, 0, 0]  # Red
        overlay_colored[negative_mask] = [0, 0, 1]  # Blue
        blended = 0.7 * overlay + 0.3 * overlay_colored
        
        axes[3].imshow(np.clip(blended, 0, 1))
        axes[3].set_title(f'Prediction: {exp_dict["class_name"]}\nConfidence: {exp_dict["confidence"]:.2%}')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def shap_explain_image(model, image: np.ndarray, class_names: list = None,
                       n_segments: int = 50, nsamples: int = 100) -> plt.Figure:
    """
    Convenience function to generate SHAP explanation.
    
    Args:
        model: Classification model
        image: Input image
        class_names: List of class names
        n_segments: Number of superpixels
        nsamples: Number of SHAP samples
    
    Returns:
        Matplotlib figure with explanation
    """
    explainer = ShapExplainer(model, class_names)
    return explainer.visualize(image, n_segments, nsamples)


class DeepShapExplainer:
    """
    Deep SHAP explainer using DeepExplainer for faster explanations.
    Requires TensorFlow/Keras models.
    """
    
    def __init__(self, model, background_data: np.ndarray, class_names: list = None):
        """
        Initialize Deep SHAP explainer.
        
        Args:
            model: Keras/TensorFlow model
            background_data: Background dataset for SHAP
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names or ['Class 0', 'Class 1']
        
        # Normalize background data
        if background_data.max() > 1:
            background_data = background_data / 255.0
        
        self.explainer = shap.DeepExplainer(model, background_data)
    
    def explain(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Generate Deep SHAP explanation.
        
        Args:
            image: Input image
        
        Returns:
            Tuple of (shap_values, explanation_dict)
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        if image.max() > 1:
            image = image / 255.0
        
        shap_values = self.explainer.shap_values(image)
        
        prediction = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        explanation_dict = {
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': float(confidence)
        }
        
        return shap_values, explanation_dict
    
    def visualize(self, image: np.ndarray, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Visualize Deep SHAP explanation.
        
        Args:
            image: Input image
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
        
        shap_values, exp_dict = self.explain(image_batch)
        
        # Get SHAP values for predicted class
        class_shap = shap_values[exp_dict['predicted_class']][0]
        
        # Sum across color channels
        shap_sum = np.sum(np.abs(class_shap), axis=-1)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        if image.max() <= 1:
            display_image = image if len(image.shape) == 3 else image[0]
        else:
            display_image = image / 255.0 if len(image.shape) == 3 else image[0] / 255.0
        
        axes[0].imshow(display_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        im = axes[1].imshow(shap_sum, cmap='hot')
        axes[1].set_title('SHAP Importance')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        axes[2].imshow(display_image)
        axes[2].imshow(shap_sum, cmap='hot', alpha=0.5)
        axes[2].set_title(f'Prediction: {exp_dict["class_name"]}\nConfidence: {exp_dict["confidence"]:.2%}')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
