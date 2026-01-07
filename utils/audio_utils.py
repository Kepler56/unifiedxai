"""
Audio Processing Utilities
Handles audio file loading, spectrogram conversion, and preprocessing
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import tempfile
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class AudioProcessor:
    """
    Audio processing class for converting audio files to spectrograms.
    Used for deepfake audio detection.
    """
    
    def __init__(self, sample_rate: int = 22050, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio
            target_size: Target size for spectrogram images
        """
        self.sample_rate = sample_rate
        self.target_size = target_size
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        return y, sr
    
    def create_mel_spectrogram(self, 
                                y: np.ndarray, 
                                sr: int,
                                n_mels: int = 128,
                                hop_length: int = 512) -> np.ndarray:
        """
        Create mel spectrogram from audio data.
        
        Args:
            y: Audio time series
            sr: Sample rate
            n_mels: Number of mel bands
            hop_length: Hop length for STFT
        
        Returns:
            Mel spectrogram in dB scale
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels,
            hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def spectrogram_to_image(self, 
                              mel_spec_db: np.ndarray, 
                              sr: int,
                              save_path: Optional[str] = None) -> np.ndarray:
        """
        Convert spectrogram to RGB image.
        
        Args:
            mel_spec_db: Mel spectrogram in dB
            sr: Sample rate
            save_path: Optional path to save the image
        
        Returns:
            RGB image array
        """
        # Create figure without axes
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Display spectrogram
        librosa.display.specshow(mel_spec_db, sr=sr, ax=ax)
        ax.axis('off')
        
        # Convert to image
        fig.canvas.draw()
        
        # Get image data from figure (use buffer_rgba for newer matplotlib versions)
        data = np.asarray(fig.canvas.buffer_rgba())
        data = data[:, :, :3]  # Remove alpha channel to get RGB
        
        plt.close(fig)
        
        # Resize to target size
        from PIL import Image
        img = Image.fromarray(data)
        img = img.resize(self.target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img)
        
        if save_path:
            img.save(save_path)
        
        return img_array
    
    def process_audio_file(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full pipeline: audio file to spectrogram image.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Tuple of (spectrogram_image, mel_spectrogram_db)
        """
        y, sr = self.load_audio(audio_path)
        mel_spec_db = self.create_mel_spectrogram(y, sr)
        img = self.spectrogram_to_image(mel_spec_db, sr)
        return img, mel_spec_db
    
    def preprocess_for_model(self, img: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Preprocess spectrogram image for model input.
        
        Args:
            img: Spectrogram image
            normalize: Whether to normalize to 0-1 range
        
        Returns:
            Preprocessed image batch
        """
        if normalize and img.max() > 1:
            img = img / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        return img_batch


def create_spectrogram(audio_path: str, 
                       target_size: Tuple[int, int] = (224, 224),
                       save_path: Optional[str] = None) -> np.ndarray:
    """
    Convenience function to create spectrogram from audio file.
    
    Args:
        audio_path: Path to audio file
        target_size: Target image size
        save_path: Optional path to save the spectrogram image
    
    Returns:
        Spectrogram image as numpy array
    """
    processor = AudioProcessor(target_size=target_size)
    img, _ = processor.process_audio_file(audio_path)
    
    if save_path:
        from PIL import Image
        Image.fromarray(img).save(save_path)
    
    return img


def load_audio_file(audio_path: str, sample_rate: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return raw audio data.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    return librosa.load(audio_path, sr=sample_rate)


def save_uploaded_audio(uploaded_file, save_dir: str = 'audio_files') -> str:
    """
    Save uploaded audio file to disk.
    
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


def get_audio_waveform(audio_path: str, 
                        sample_rate: int = 22050,
                        figsize: Tuple[int, int] = (10, 3)) -> plt.Figure:
    """
    Create waveform visualization for audio file.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Sample rate
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    fig, ax = plt.subplots(figsize=figsize)
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    
    return fig


def get_audio_features(audio_path: str, sample_rate: int = 22050) -> dict:
    """
    Extract audio features for analysis.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Sample rate
    
    Returns:
        Dictionary of audio features
    """
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    features = {
        'duration': librosa.get_duration(y=y, sr=sr),
        'sample_rate': sr,
        'samples': len(y),
        'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
        'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        'rms_energy': float(np.mean(librosa.feature.rms(y=y)))
    }
    
    return features
