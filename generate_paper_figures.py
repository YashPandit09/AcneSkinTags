"""
All-in-One Evaluation & Grad-CAM Script for Research Paper
Generates all figures needed for publication after training completes
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from glob import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from explainability.gradcam import make_gradcam_heatmap, overlay_heatmap_on_image, get_last_conv_layer_name

# ======================
# CONFIGURATION
# ======================
# UPDATE THIS AFTER TRAINING FINISHES
MODEL_PATH = 'results/mobilenetv2_*/best_model.keras'  # Use wildcard or specific path

# Output directory for paper figures
OUTPUT_DIR = 'paper_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names (8 classes)
CLASS_NAMES = list(config.LESION_TYPE_DICT.values())

print("="*80)
print("PAPER FIGURE GENERATION SCRIPT")
print("="*80)
print(f"Classes: {CLASS_NAMES}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*80 + "\n")

# ======================
# PART 1: LOAD MODEL & DATA
# ======================
def load_model_auto():
    """Auto-find the latest trained model"""
    if '*' in MODEL_PATH:
        # Find matching models
        matches = glob(MODEL_PATH)
        if not matches:
            raise FileNotFoundError(f"No model found matching: {MODEL_PATH}")
        # Use the most recent one
        model_path = max(matches, key=os.path.getmtime)
        print(f"✓ Auto-detected model: {model_path}")
    else:
        model_path = MODEL_PATH
    
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded successfully")
    return model, model_path

# ======================
# PART 2: CONFUSION MATRIX (Figure for Paper)
# ======================
def generate_confusion_matrix(y_true, y_pred, save_path):
    """Generate publication-quality confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'}, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Absolute)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Percentage'}, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()

# ======================
# PART 3: GRAD-CAM EXAMPLES (Figure for Paper)
# ======================
def generate_gradcam_examples(model, num_examples=6):
    """Generate multiple Grad-CAM examples for paper"""
    print(f"\n[INFO] Generating {num_examples} Grad-CAM examples...")
    
    # Find sample images from Dataset
    sample_images = []
    
    # Try HAM10000 first
    ham_dir = os.path.join(config.DATASET_DIR, 'HAM10000', 'HAM10000_images_part_1')
    if os.path.exists(ham_dir):
        images = glob(os.path.join(ham_dir, '*.jpg'))
        sample_images.extend(images[:num_examples-1])
    
    # Add one Acne example
    acne_dir = os.path.join(config.DATASET_DIR, 'DermNet', 'Acne and Rosacea Photos')
    if os.path.exists(acne_dir):
        acne_images = glob(os.path.join(acne_dir, '*.jpg'))
        if acne_images:
            sample_images.append(acne_images[0])
    
    if not sample_images:
        print("⚠ Warning: No sample images found. Skipping Grad-CAM generation.")
        return
    
    # Get last conv layer name
    last_conv_layer = get_last_conv_layer_name(model)
    print(f"✓ Using conv layer: {last_conv_layer}")
    
    # Generate grid of Grad-CAM visualizations
    num_samples = min(len(sample_images), num_examples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(sample_images[:num_samples]):
        # Load and preprocess image
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        img_resized = cv2.resize(img_array, config.IMG_SIZE)
        
        # CRITICAL: Use same preprocessing as training (/255)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        predictions = model.predict(img_batch, verbose=0)
        pred_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_class_idx]
        pred_class_name = CLASS_NAMES[pred_class_idx]
        
        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer)
        superimposed = overlay_heatmap_on_image(img_resized, heatmap)
        
        # Plot
        axes[idx, 0].imshow(img_resized)
        axes[idx, 0].set_title('Original Image', fontsize=12)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(heatmap, cmap='jet')
        axes[idx, 1].set_title('Grad-CAM Heatmap', fontsize=12)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(superimposed)
        axes[idx, 2].set_title(f'Overlay\n{pred_class_name} ({confidence:.1%})', fontsize=12)
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'gradcam_examples.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Grad-CAM examples saved: {save_path}")
    plt.close()

# ======================
# PART 4: CLASSIFICATION REPORT (Table for Paper)
# ======================
def generate_classification_report(y_true, y_pred):
    """Generate and save classification report"""
    report = classification_report(y_true, y_pred, 
                                   target_names=CLASS_NAMES,
                                   output_dict=True)
    
    # Save as CSV for LaTeX tables
    import pandas as pd
    df = pd.DataFrame(report).transpose()
    csv_path = os.path.join(OUTPUT_DIR, 'classification_report.csv')
    df.to_csv(csv_path)
    print(f"✓ Classification report saved: {csv_path}")
    
    # Print key metrics
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT (For Paper)")
    print("="*80)
    for class_name in CLASS_NAMES:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name:30s} | Precision: {metrics['precision']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")
    
    # Highlight critical metric
    if 'Melanoma' in report:
        melanoma_recall = report['Melanoma']['recall']
        status = "✓✓✓ EXCELLENT" if melanoma_recall > 0.80 else "⚠ NEEDS IMPROVEMENT"
        print(f"\n{'='*80}")
        print(f"CRITICAL METRIC: Melanoma Recall = {melanoma_recall:.3f} ({status})")
        print(f"{'='*80}\n")

# ======================
# MAIN EXECUTION
# ======================
def main():
    """Generate all paper figures"""
    print("\n[STEP 1] Loading model...")
    model, model_path = load_model_auto()
    
    print("\n[STEP 2] Loading test data...")
    from data_loader_enhanced import get_enhanced_data_generators
    _, _, test_gen, _, test_df = get_enhanced_data_generators()
    
    print("\n[STEP 3] Running predictions...")
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    
    print(f"✓ Predicted {len(y_pred)} test samples")
    
    print("\n[STEP 4] Generating confusion matrix...")
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_8class.png')
    generate_confusion_matrix(y_true, y_pred, cm_path)
    
    print("\n[STEP 5] Generating classification report...")
    generate_classification_report(y_true, y_pred)
    
    print("\n[STEP 6] Generating Grad-CAM examples...")
    generate_gradcam_examples(model, num_examples=6)
    
    print("\n" + "="*80)
    print("✓✓✓ ALL PAPER FIGURES GENERATED!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("\nFiles created:")
    print(f"  1. confusion_matrix_8class.png  (Figure for paper)")
    print(f"  2. gradcam_examples.png         (Figure for paper)")
    print(f"  3. classification_report.csv    (Table for paper)")
    print("\nUse these in your research paper!")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTip: Make sure training has completed and model exists!")
