"""
Model Factory - Unified Interface for All Architectures
Supports: ResNet50, MobileNetV2, EfficientNet (B0, B3, B7)
"""
import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50, MobileNetV2, EfficientNetB0, EfficientNetB3, EfficientNetB7
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class ModelFactory:
    """Factory class to create different CNN architectures."""
    
    SUPPORTED_MODELS = {
        'resnet50': ResNet50,
        'mobilenetv2': MobileNetV2,
        'efficientnet-b0': EfficientNetB0,
        'efficientnet-b3': EfficientNetB3,
        'efficientnet-b7': EfficientNetB7
    }
    
    @staticmethod
    def create(model_name: str, num_classes: int = 7, img_size: tuple = (224, 224),
               trainable_base: bool = False) -> Model:
        """
        Create a model with transfer learning.
        
        Args:
            model_name: One of 'resnet50', 'mobilenetv2', 'efficientnet-b0/b3/b7'
            num_classes: Number of output classes
            img_size: Input image size
            trainable_base: If True, base model is trainable (for fine-tuning)
            
        Returns:
            Compiled Keras model
        """
        model_name = model_name.lower()
        
        if model_name not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Choose from: {list(ModelFactory.SUPPORTED_MODELS.keys())}")
        
        print(f"\n{'='*60}")
        print(f"Building Model: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Get the base model class
        BaseModel = ModelFactory.SUPPORTED_MODELS[model_name]
        
        # Load pre-trained weights from ImageNet
        base_model = BaseModel(
            weights='imagenet',
            include_top=False,  # Remove classification head
            input_shape=img_size + (3,)
        )
        
        # Freeze or unfreeze base model
        base_model.trainable = trainable_base
        
        if not trainable_base:
            print(f"✓ Base model frozen (feature extraction mode)")
        else:
            print(f"✓ Base model trainable (fine-tuning mode)")
        
        # Build classification head
        x = base_model.output
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = Dense(config.DENSE_UNITS, activation='relu', name='dense_hidden')(x)
        x = Dropout(config.DROPOUT_RATE, name='dropout')(x)
        predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
        
        # Create final model
        model = Model(inputs=base_model.input, outputs=predictions, name=model_name)
        
        # Count parameters
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"✓ Total Parameters: {total_params:,}")
        print(f"✓ Trainable Parameters: {trainable_params:,}")
        print(f"✓ Non-Trainable Parameters: {non_trainable_params:,}")
        print(f"{'='*60}\n")
        
        return model
    
    @staticmethod
    def unfreeze_layers(model: Model, num_layers: int):
        """
        Unfreeze the last N layers of the base model for fine-tuning.
        
        Args:
            model: Keras model
            num_layers: Number of layers to unfreeze from the end
        """
        # Find the base model (before the custom head)
        base_model = None
        for layer in model.layers:
            if isinstance(layer, Model):
                base_model = layer
                break
        
        if base_model is None:
            print("[WARNING] No base model found, cannot unfreeze layers")
            return
        
        # Unfreeze last N layers
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True
        
        trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"✓ Unfroze last {num_layers} layers ({trainable_count} total trainable layers in base)")
    
    @staticmethod
    def get_model_info(model_name: str) -> dict:
        """Get information about a model."""
        info = {
            'resnet50': {
                'params': '~24M',
                'depth': 50,
                'best_for': 'Baseline comparison, proven architecture',
                'speed': 'Medium',
                'accuracy': 'Good'
            },
            'mobilenetv2': {
                'params': '~3.5M',
                'depth': 53,
                'best_for': 'Edge deployment, mobile apps',
                'speed': 'Fast',
                'accuracy': 'Good'
            },
            'efficientnet-b0': {
                'params': '~5M',
                'depth': 'Variable',
                'best_for': 'Lightweight with compound scaling',
                'speed': 'Fast',
                'accuracy': 'Very Good'
            },
            'efficientnet-b3': {
                'params': '~12M',
                'depth': 'Variable',
                'best_for': 'Best accuracy/efficiency trade-off',
                'speed': 'Medium',
                'accuracy': 'Excellent'
            },
            'efficientnet-b7': {
                'params': '~66M',
                'depth': 'Variable',
                'best_for': 'Maximum accuracy (GPU required)',
                'speed': 'Slow',
                'accuracy': 'State-of-the-art'
            }
        }
        return info.get(model_name.lower(), {})

if __name__ == "__main__":
    print("=== MODEL FACTORY TEST ===\n")
    
    # Test creating different models
    for model_name in ['mobilenetv2', 'resnet50']:
        model = ModelFactory.create(model_name, num_classes=7)
        model.summary()
        print(f"\n{'-'*60}\n")
        
        # Test unfreezing
        ModelFactory.unfreeze_layers(model, num_layers=20)
        print()
    
    # Show model comparison
    print("\n=== MODEL COMPARISON ===\n")
    for name in ModelFactory.SUPPORTED_MODELS.keys():
        info = ModelFactory.get_model_info(name)
        print(f"{name.upper():20s} | Params: {info.get('params', 'N/A'):8s} | Best for: {info.get('best_for', 'N/A')}")
    
    print("\n✓ Model Factory Ready!\n")
