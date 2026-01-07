"""
Audio Classification Models for Deepfake Detection
Uses pre-trained models from TensorFlow/Keras for spectrogram-based classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50, InceptionV3


# Available audio models registry
AUDIO_MODELS = {
    'VGG16': 'VGG16 - Transfer learning model pretrained on ImageNet',
    'MobileNetV2': 'MobileNetV2 - Lightweight and efficient model',
    'ResNet50': 'ResNet50 - Deep residual network',
    'InceptionV3': 'InceptionV3 - Google\'s inception architecture',
    'CustomCNN': 'Custom CNN - Simple 3-layer convolutional network'
}


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=2):
    """
    Build a simple custom CNN for audio spectrogram classification.
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of output classes (2 for real/fake)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_transfer_model(base_model_name, input_shape=(224, 224, 3), num_classes=2, trainable=False):
    """
    Build a transfer learning model using a pre-trained base.
    
    Args:
        base_model_name: Name of the base model ('VGG16', 'MobileNetV2', 'ResNet50', 'InceptionV3')
        input_shape: Input image dimensions
        num_classes: Number of output classes
        trainable: Whether to make base model trainable
    
    Returns:
        Compiled Keras model
    """
    base_models = {
        'VGG16': VGG16,
        'MobileNetV2': MobileNetV2,
        'ResNet50': ResNet50,
        'InceptionV3': InceptionV3
    }
    
    if base_model_name not in base_models:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Load pre-trained base model
    base = base_models[base_model_name](
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base.trainable = trainable
    
    # Build full model
    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_audio_model(model_name, input_shape=(224, 224, 3), num_classes=2, weights_path=None):
    """
    Get an audio classification model by name.
    
    Args:
        model_name: Name of the model to load
        input_shape: Input image dimensions
        num_classes: Number of output classes
        weights_path: Optional path to pre-trained weights
    
    Returns:
        Keras model instance
    """
    if model_name == 'CustomCNN':
        model = build_custom_cnn(input_shape, num_classes)
    elif model_name in ['VGG16', 'MobileNetV2', 'ResNet50', 'InceptionV3']:
        model = build_transfer_model(model_name, input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AUDIO_MODELS.keys())}")
    
    # Load weights if provided
    if weights_path:
        model.load_weights(weights_path)
    
    return model


def get_last_conv_layer_name(model_name):
    """
    Get the name of the last convolutional layer for Grad-CAM.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Name of the last conv layer
    """
    conv_layer_names = {
        'VGG16': 'block5_conv3',
        'MobileNetV2': 'Conv_1',
        'ResNet50': 'conv5_block3_out',
        'InceptionV3': 'mixed10',
        'CustomCNN': 'conv2d_2'  # Third conv layer in custom CNN
    }
    
    return conv_layer_names.get(model_name, None)


def load_saved_audio_model(model_path):
    """
    Load a saved TensorFlow model from disk.
    
    Args:
        model_path: Path to the saved model directory
    
    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(model_path)
