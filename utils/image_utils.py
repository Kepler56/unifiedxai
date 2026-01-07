"""
Image Processing Utilities
Handles image loading, preprocessing, and augmentation for chest X-ray classification
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import os
import warnings

warnings.filterwarnings('ignore')


class ImageProcessor:
    """
    Image processing class for chest X-ray images.
    Used for lung cancer detection.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image processor.
        
        Args:
            target_size: Target size for images
        """
        self.target_size = target_size
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Image as numpy array (RGB)
        """
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img)
    
    def resize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            img: Input image
        
        Returns:
            Resized image
        """
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(self.target_size, Image.Resampling.LANCZOS)
        return np.array(pil_img)
    
    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-1 range.
        
        Args:
            img: Input image
        
        Returns:
            Normalized image
        """
        if img.max() > 1:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)
    
    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Useful for enhancing X-ray images.
        
        Args:
            img: Input image (RGB)
        
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def preprocess(self, 
                   img: np.ndarray, 
                   normalize: bool = True,
                   apply_clahe: bool = False) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Args:
            img: Input image
            normalize: Whether to normalize
            apply_clahe: Whether to apply CLAHE enhancement
        
        Returns:
            Preprocessed image
        """
        # Resize
        img = self.resize_image(img)
        
        # Apply CLAHE if requested
        if apply_clahe:
            img = self.apply_clahe(img)
        
        # Normalize
        if normalize:
            img = self.normalize_image(img)
        
        return img
    
    def preprocess_for_model(self, img: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            img: Input image
            normalize: Whether to normalize
        
        Returns:
            Preprocessed image batch
        """
        img = self.preprocess(img, normalize)
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        return img_batch
    
    def process_uploaded_file(self, uploaded_file) -> np.ndarray:
        """
        Process Streamlit uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
        
        Returns:
            Processed image array
        """
        img = Image.open(uploaded_file)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        return self.preprocess(img_array)


def load_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Convenience function to load and resize image.
    
    Args:
        image_path: Path to image
        target_size: Target size
    
    Returns:
        Processed image
    """
    processor = ImageProcessor(target_size)
    img = processor.load_image(image_path)
    return processor.resize_image(img)


def preprocess_image(img: np.ndarray, 
                     target_size: Tuple[int, int] = (224, 224),
                     normalize: bool = True) -> np.ndarray:
    """
    Convenience function to preprocess image.
    
    Args:
        img: Input image
        target_size: Target size
        normalize: Whether to normalize
    
    Returns:
        Preprocessed image
    """
    processor = ImageProcessor(target_size)
    return processor.preprocess(img, normalize)


def save_uploaded_image(uploaded_file, save_dir: str = 'uploaded_images') -> str:
    """
    Save uploaded image file to disk.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        save_dir: Directory to save the file
    
    Returns:
        Path to saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def display_image_grid(images: list, 
                        titles: list = None,
                        figsize: Tuple[int, int] = None,
                        cols: int = 4) -> plt.Figure:
    """
    Display multiple images in a grid.
    
    Args:
        images: List of images
        titles: List of titles
        figsize: Figure size
        cols: Number of columns
    
    Returns:
        Matplotlib figure
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    
    if figsize is None:
        figsize = (4 * cols, 4 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        if img.max() <= 1:
            ax.imshow(img)
        else:
            ax.imshow(img.astype(np.uint8))
        
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis('off')
    
    # Hide unused axes
    for ax in axes[n:]:
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def get_image_info(image_path: str) -> dict:
    """
    Get information about an image.
    
    Args:
        image_path: Path to image
    
    Returns:
        Dictionary with image info
    """
    img = Image.open(image_path)
    
    info = {
        'filename': os.path.basename(image_path),
        'format': img.format,
        'mode': img.mode,
        'size': img.size,
        'width': img.width,
        'height': img.height
    }
    
    return info


def detect_input_type(file_path: str) -> str:
    """
    Detect whether input is audio or image based on file extension.
    
    Args:
        file_path: Path to file
    
    Returns:
        'audio' or 'image'
    """
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in audio_extensions:
        return 'audio'
    elif ext in image_extensions:
        return 'image'
    else:
        raise ValueError(f"Unknown file type: {ext}")


def detect_input_type_from_filename(filename: str) -> str:
    """
    Detect input type from filename.
    
    Args:
        filename: Filename with extension
    
    Returns:
        'audio' or 'image'
    """
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in audio_extensions:
        return 'audio'
    elif ext in image_extensions:
        return 'image'
    else:
        raise ValueError(f"Unknown file type: {ext}")
