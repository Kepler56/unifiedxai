"""
Image Classification Models for Lung Cancer Detection
Simplified implementations of AlexNet and DenseNet for chest X-ray classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121


# Available image models registry
IMAGE_MODELS = {
    'AlexNet': 'AlexNet - Classic CNN architecture for image classification',
    'DenseNet': 'DenseNet121 - Dense connections for better feature propagation'
}


class AlexNet(Model):
    """
    Simplified AlexNet implementation for chest X-ray classification.
    Based on the original AlexNet architecture but adapted for medical imaging.
    """
    
    def __init__(self, num_classes=2, input_shape=(224, 224, 3)):
        super(AlexNet, self).__init__()
        
        # Conv Block 1
        self.conv1 = layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', 
                                    padding='valid', input_shape=input_shape, name='conv1')
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.pool1 = layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool1')
        
        # Conv Block 2
        self.conv2 = layers.Conv2D(256, (5, 5), activation='relu', padding='same', name='conv2')
        self.bn2 = layers.BatchNormalization(name='bn2')
        self.pool2 = layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool2')
        
        # Conv Block 3
        self.conv3 = layers.Conv2D(384, (3, 3), activation='relu', padding='same', name='conv3')
        self.bn3 = layers.BatchNormalization(name='bn3')
        
        # Conv Block 4
        self.conv4 = layers.Conv2D(384, (3, 3), activation='relu', padding='same', name='conv4')
        self.bn4 = layers.BatchNormalization(name='bn4')
        
        # Conv Block 5 (last conv layer for Grad-CAM)
        self.conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5')
        self.bn5 = layers.BatchNormalization(name='bn5')
        self.pool5 = layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool5')
        
        # Dense layers
        self.flatten = layers.Flatten(name='flatten')
        self.fc1 = layers.Dense(4096, activation='relu', name='fc1')
        self.dropout1 = layers.Dropout(0.5, name='dropout1')
        self.fc2 = layers.Dense(4096, activation='relu', name='fc2')
        self.dropout2 = layers.Dropout(0.5, name='dropout2')
        self.output_layer = layers.Dense(num_classes, activation='softmax', name='predictions')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.pool5(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        
        return self.output_layer(x)
    
    def get_last_conv_layer(self):
        """Return the last convolutional layer for Grad-CAM"""
        return self.conv5


def build_alexnet(input_shape=(224, 224, 3), num_classes=2):
    """
    Build AlexNet using Functional API for easier Grad-CAM integration.
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of output classes (2 for benign/malignant)
    
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Conv Block 1
    x = layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', 
                      padding='valid', name='conv1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Conv Block 2
    x = layers.Conv2D(256, (5, 5), activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Conv Block 3
    x = layers.Conv2D(384, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = layers.BatchNormalization()(x)
    
    # Conv Block 4
    x = layers.Conv2D(384, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = layers.BatchNormalization()(x)
    
    # Conv Block 5 (last conv layer for Grad-CAM)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs, name='AlexNet')
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class DenseNetModel:
    """
    DenseNet121-based model for chest X-ray classification.
    Uses transfer learning from ImageNet pre-trained weights.
    """
    
    def __init__(self, num_classes=2, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the DenseNet model with custom classification head"""
        # Load pre-trained DenseNet121
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build full model
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        model = Model(inputs, outputs, name='DenseNet')
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_model(self):
        return self.model


def build_densenet(input_shape=(224, 224, 3), num_classes=2):
    """
    Build DenseNet121 model using transfer learning.
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of output classes (2 for benign/malignant)
    
    Returns:
        Compiled Keras model
    """
    return DenseNetModel(num_classes, input_shape).get_model()


def get_image_model(model_name, input_shape=(224, 224, 3), num_classes=2, weights_path=None):
    """
    Get an image classification model by name.
    
    Args:
        model_name: Name of the model to load ('AlexNet' or 'DenseNet')
        input_shape: Input image dimensions
        num_classes: Number of output classes
        weights_path: Optional path to pre-trained weights
    
    Returns:
        Keras model instance
    """
    if model_name == 'AlexNet':
        model = build_alexnet(input_shape, num_classes)
    elif model_name == 'DenseNet':
        model = build_densenet(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(IMAGE_MODELS.keys())}")
    
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
        'AlexNet': 'conv5',
        'DenseNet': 'conv5_block16_concat'  # Last dense block in DenseNet121
    }
    
    return conv_layer_names.get(model_name, None)
