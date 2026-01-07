"""
Audio Classification Models for Deepfake Detection
Uses pre-trained models from TensorFlow/Keras for spectrogram-based classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNet


# Available audio models registry
AUDIO_MODELS = {
    'VGG16': 'VGG16 - Transfer learning model pretrained on ImageNet',
    'MobileNet': 'MobileNet - Original MobileNet architecture (matches saved model)',
    'ResNet50': 'ResNet50 - Deep residual network',
    'InceptionV3': 'InceptionV3 - Google\'s inception architecture'
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
    Uses Functional API with flattened layer structure for Grad-CAM compatibility.
    
    Args:
        base_model_name: Name of the base model ('VGG16', 'MobileNet', 'ResNet50', 'InceptionV3')
        input_shape: Input image dimensions
        num_classes: Number of output classes
        trainable: Whether to make base model trainable
    
    Returns:
        Compiled Keras model
    """
    base_models = {
        'VGG16': VGG16,
        'MobileNet': MobileNet,
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
    
    # Build full model with flattened structure (for Grad-CAM compatibility)
    # Connect layers directly instead of wrapping base model
    inputs = base.input
    x = base.output
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs, name=f'{base_model_name}_transfer')
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_mobilenet_original(input_shape=(224, 224, 3), num_classes=2, trainable=False):
    """
    Build MobileNet model with EXACT architecture matching the saved model from 
    Deepfake-Audio-Detection-with-XAI project (Audio_classifier.ipynb).
    
    Architecture:
        - MobileNet base (frozen)
        - GlobalAveragePooling2D
        - Dense(2, sigmoid)
    
    This matches exactly:
        base_model = MobileNet(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=x)
    
    Args:
        input_shape: Input image dimensions (224, 224, 3)
        num_classes: Number of output classes (2 for real/fake)
        trainable: Whether to make base model trainable
    
    Returns:
        Compiled Keras model
    """
    # Load MobileNet v1 base (exactly as in original notebook)
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = trainable
    
    # Build model with exact same architecture as original
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
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
    if model_name in ['VGG16', 'MobileNet', 'ResNet50', 'InceptionV3']:
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
    These are the actual layer names from the base models when using Functional API.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Name of the last conv layer
    """
    conv_layer_names = {
        'VGG16': 'block5_conv3',
        'MobileNet': 'conv_pw_13_relu',  # Last pointwise conv ReLU in MobileNet v1
        'ResNet50': 'conv5_block3_out',
        'InceptionV3': 'mixed10'
    }
    
    return conv_layer_names.get(model_name, None)


def load_saved_audio_model(model_path):
    """
    Load a saved TensorFlow model from disk.
    Compatible with Keras 3 which doesn't support legacy SavedModel format.
    
    Args:
        model_path: Path to the saved model directory
    
    Returns:
        Loaded Keras model or a callable wrapper
    """
    import os
    
    try:
        # Try loading with standard load_model first (for .keras or .h5 files)
        return tf.keras.models.load_model(model_path)
    except Exception as e1:
        try:
            # Try using tf.saved_model.load for legacy SavedModel format
            loaded = tf.saved_model.load(model_path)
            
            # Create a wrapper class that acts like a Keras model
            class SavedModelWrapper:
                def __init__(self, saved_model):
                    self.saved_model = saved_model
                    # Get the inference function
                    if hasattr(saved_model, 'signatures'):
                        self.infer = saved_model.signatures['serving_default']
                    else:
                        self.infer = saved_model
                    self.name = 'MobileNet_saved'
                
                def predict(self, x, verbose=0):
                    # Ensure input is float32
                    x = tf.cast(x, tf.float32)
                    result = self.infer(x)
                    # Extract output from dict if needed
                    if isinstance(result, dict):
                        output_key = list(result.keys())[0]
                        return result[output_key].numpy()
                    return result.numpy()
                
                def __call__(self, x, training=False):
                    x = tf.cast(x, tf.float32)
                    result = self.infer(x)
                    if isinstance(result, dict):
                        output_key = list(result.keys())[0]
                        return result[output_key]
                    return result
                
                @property
                def input(self):
                    return tf.keras.Input(shape=(224, 224, 3))
                
                @property 
                def inputs(self):
                    return [tf.keras.Input(shape=(224, 224, 3))]
                
                @property
                def layers(self):
                    # Return empty list - this model doesn't expose layers
                    return []
                
                def get_layer(self, name):
                    raise ValueError(f"SavedModel wrapper doesn't expose individual layers. Layer '{name}' not accessible.")
            
            return SavedModelWrapper(loaded)
            
        except Exception as e2:
            raise ValueError(f"Could not load model from {model_path}. Error: {e2}")
