"""
Quick Demo Model - Creates a dummy model for app testing
Use this if you just want to see the UI without waiting for training
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from models.model_factory import ModelFactory

def create_demo_model():
    """
    Create a quick demo model (untrained) for UI testing.
    This lets you see the Streamlit app interface without waiting for training.
    
    Note: Predictions will be random since it's untrained.
    """
    print("\n" + "="*80)
    print("CREATING DEMO MODEL FOR APP TESTING")
    print("="*80 + "\n")
    
    print("Creating MobileNetV2 architecture...")
    model = ModelFactory.create('mobilenetv2', num_classes=7)
    
    # Create saved_models directory
    os.makedirs('saved_models', exist_ok=True)
    
    # Save model
    model_path = 'saved_models/mobilenetv2_best.keras'
    model.save(model_path)
    
    print(f"\n✓ Demo model saved to: {model_path}")
    print(f"\n{'='*80}")
    print("DEMO MODEL READY!")
    print("="*80)
    print("\nNow you can:")
    print("  1. Launch app: streamlit run app.py")
    print("  2. Upload an image")
    print("  3. See the UI (predictions will be random - model is untrained)")
    print("\nFor REAL predictions, train a model:")
    print("  python train_comprehensive.py --model mobilenetv2 --epochs 5")
    print("="*80 + "\n")
    
    return model_path

if __name__ == "__main__":
    create_demo_model()
