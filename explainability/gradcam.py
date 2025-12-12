"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
Step 17: Implement Explainable AI
Visualizes which parts of the image the CNN focused on for predictions
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def get_last_conv_layer_name(model):
    """
    Automatically find the last convolutional layer in the model.
    """
    for layer in reversed(model.layers):
        # Check if layer has 4D output (batch, height, width, channels)
        if len(layer.output.shape) == 4:
            return layer.name
    
    raise ValueError("Could not find a convolutional layer in the model")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for a given image.
    
    Args:
        img_array: Preprocessed image (1, H, W, 3)
        model: Trained Keras model
        last_conv_layer_name: Name of the last conv layer
        pred_index: Class index to visualize (None = use predicted class)
        
    Returns:
        heatmap: Numpy array (H, W) with values in [0, 1]
    """
    # Create a model that maps the input to the activations of the last conv layer
    # and the output predictions
    grad_model = keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for the input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        class_channel = predictions[:, pred_index]
    
    # Gradient of the output neuron (top predicted or chosen class)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        img: Original image (H, W, 3) in range [0, 255]
        heatmap: Grad-CAM heatmap (H, W) in range [0, 1]
        alpha: Transparency of heatmap
        colormap: OpenCV colormap
        
    Returns:
        superimposed_img: Result image with heatmap overlay
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap_colored * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

def visualize_gradcam(image_path, model, class_names, save_path=None, 
                      last_conv_layer_name=None):
    """
    Complete Grad-CAM visualization pipeline.
    
    Args:
        image_path: Path to input image
        model: Trained Keras model
        class_names: List of class names
        save_path: Optional path to save visualization
        last_conv_layer_name: Name of last conv layer (auto-detected if None)
        
    Returns:
        prediction, confidence, heatmap
    """
    from tensorflow.keras.preprocessing import image as keras_image
    from preprocessing.normalization import normalize_imagenet
    
    # Load and preprocess image
    img = keras_image.load_img(image_path, target_size=config.IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    original_img = img_array.copy().astype(np.uint8)
    
    # Normalize for model
    img_array_normalized = normalize_imagenet(img_array)
    img_array_normalized = np.expand_dims(img_array_normalized, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array_normalized, verbose=0)
    pred_class_idx = np.argmax(predictions[0])
    pred_class_name = class_names[pred_class_idx]
    confidence = predictions[0][pred_class_idx]
    
    # Auto-detect last conv layer if not specified
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)
        print(f"Auto-detected last conv layer: {last_conv_layer_name}")
    
    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array_normalized, model, last_conv_layer_name)
    
    # Create overlay
    superimposed_img = overlay_heatmap_on_image(original_img, heatmap)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(superimposed_img)
    axes[2].set_title(f'Prediction: {pred_class_name}\nConfidence: {confidence:.2%}',
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Verification note
    fig.text(0.5, 0.02, 
             '✓ Verification: Red regions should highlight the lesion, not skin/hair',
             ha='center', fontsize=10, style='italic', color='green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Grad-CAM visualization saved: {save_path}")
    
    plt.close()
    
    return pred_class_name, confidence, heatmap, superimposed_img

def batch_gradcam_visualization(image_paths, model, class_names, output_dir, 
                                 last_conv_layer_name=None, max_images=20):
    """
    Generate Grad-CAM for multiple images.
    
    Useful for creating examples for research paper.
    """
    print(f"\n{'='*80}")
    print(f"GENERATING GRAD-CAM FOR {min(len(image_paths), max_images)} IMAGES")
    print(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img_path in enumerate(image_paths[:max_images]):
        img_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f'gradcam_{i+1:03d}_{img_name}')
        
        print(f"[{i+1}/{min(len(image_paths), max_images)}] Processing: {img_name}")
        
        try:
            pred, conf, _, _ = visualize_gradcam(
                img_path, model, class_names, 
                save_path=save_path,
                last_conv_layer_name=last_conv_layer_name
            )
            print(f"  → Predicted: {pred} ({conf:.2%})")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n✓ Grad-CAM visualizations saved to: {output_dir}\n")

# Example usage
if __name__ == "__main__":
    import argparse
    from glob import glob
    
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image-path', type=str, default=None,
                       help='Path to single image (optional)')
    parser.add_argument('--batch', action='store_true',
                       help='Generate for multiple random images from dataset')
    parser.add_argument('--output-dir', type=str, default='./visualizations/gradcam',
                       help='Output directory for Grad-CAM images')
    parser.add_argument('--num-images', type=int, default=10,
                       help='Number of images to process in batch mode')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = keras.models.load_model(args.model_path)
    print("✓ Model loaded\n")
    
    class_names = list(config.LESION_TYPE_DICT.values())
    
    if args.image_path:
        # Single image
        print("="*80)
        print("STEP 17: GRAD-CAM EXPLAINABILITY (Single Image)")
        print("="*80 + "\n")
        
        save_path = os.path.join(args.output_dir, 'gradcam_single.png')
        os.makedirs(args.output_dir, exist_ok=True)
        
        pred, conf, _, _ = visualize_gradcam(
            args.image_path, model, class_names, save_path=save_path
        )
        
        print(f"\n✓ Prediction: {pred}")
        print(f"✓ Confidence: {conf:.2%}")
        print(f"✓ Visualization saved: {save_path}")
        
    elif args.batch:
        # Batch mode - sample random images from dataset
        print("="*80)
        print("BATCH GRAD-CAM GENERATION")
        print("="*80 + "\n")
        
        # Get random images from dataset
        image_paths = []
        for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            part_dir = os.path.join(config.DATASET_DIR, part)
            if os.path.exists(part_dir):
                image_paths.extend(glob(os.path.join(part_dir, '*.jpg')))
        
        # Randomly sample
        np.random.seed(42)
        sampled_paths = np.random.choice(image_paths, 
                                         min(args.num_images, len(image_paths)),
                                         replace=False)
        
        batch_gradcam_visualization(sampled_paths, model, class_names, 
                                     args.output_dir, max_images=args.num_images)
    else:
        print("Please specify either --image-path or --batch")
        print("\nExamples:")
        print("  Single image: python explainability/gradcam.py --model-path models/best_model.keras --image-path dataset/test.jpg")
        print("  Batch mode:   python explainability/gradcam.py --model-path models/best_model.keras --batch --num-images 20")
