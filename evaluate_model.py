"""
Comprehensive Model Evaluation Script
Generates metrics, confusion matrix, and classification report
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
import json
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader_enhanced import get_enhanced_data_generators

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model (.keras file)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save evaluation results')
    return parser.parse_args()

def load_model_and_data(model_path):
    """Load trained model and test data."""
    print(f"\n{'='*80}")
    print("LOADING MODEL AND DATA")
    print(f"{'='*80}\n")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_gen, _, test_df = get_enhanced_data_generators(
        base_dir=config.DATASET_DIR,
        img_size=config.IMG_SIZE,
        batch_size=32
    )
    print("✓ Test data loaded")
    
    return model, test_gen, test_df

def evaluate_model(model, test_gen, output_dir):
    """
    Step 15: Generate Performance Metrics
    """
    print(f"\n{'='*80}")
    print("STEP 15: GENERATING PERFORMANCE METRICS")
    print(f"{'='*80}\n")
    
    # Make predictions
    print("Running predictions on test set...")
    predictions = model.predict(test_gen, verbose=1)
    
    # Get true labels
    y_true = test_gen.classes
    y_pred = np.argmax(predictions, axis=1)
    
    # Class names
    class_names = list(config.LESION_TYPE_DICT.values())
    
    # Generate classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80 + "\n")
    
    report_dict = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    
    print(report_str)
    
    # Check Melanoma recall (Critical!)
    mel_idx = list(config.LESION_TYPE_DICT.keys()).index('mel')
    melanoma_recall = report_dict[class_names[mel_idx]]['recall']
    
    print("\n" + "="*80)
    print("CRITICAL METRIC CHECK")
    print("="*80)
    print(f"Melanoma Recall: {melanoma_recall:.4f}")
    if melanoma_recall > 0.80:
        print("✓✓✓ EXCELLENT: Melanoma recall > 0.80 (meets medical requirement)")
    elif melanoma_recall > 0.70:
        print("✓ GOOD: Melanoma recall > 0.70 (acceptable)")
    else:
        print("⚠ WARNING: Melanoma recall < 0.70 (needs improvement)")
    print("="*80 + "\n")
    
    # Save classification report
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"✓ Classification report saved: {report_path}")
    
    # Save text version
    report_txt_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_txt_path, 'w') as f:
        f.write(report_str)
    print(f"✓ Text report saved: {report_txt_path}")
    
    return y_true, y_pred, predictions, class_names

def create_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """
    Step 16: Create Confusion Matrix
    """
    print(f"\n{'='*80}")
    print("STEP 16: CREATING CONFUSION MATRIX")
    print(f"{'='*80}\n")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Absolute Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # Normalized (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Percentage'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    
    # Save
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {cm_path}")
    
    plt.close()
    
    # Analysis
    print("\nConfusion Matrix Analysis:")
    print(f"  Diagonal (Correct): {np.trace(cm)} / {cm.sum()} = {np.trace(cm)/cm.sum():.2%}")
    print(f"  Off-diagonal (Errors): {cm.sum() - np.trace(cm)} / {cm.sum()} = {(cm.sum() - np.trace(cm))/cm.sum():.2%}")
    
    return cm

def create_roc_curves(y_true, predictions, class_names, output_dir):
    """
    Bonus: ROC curves for each class
    """
    print(f"\n{'='*80}")
    print("BONUS: ROC CURVES (Multi-Class)")
    print(f"{'='*80}\n")
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_name[:20]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curves - Multi-Class (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    
    roc_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curves saved: {roc_path}")
    
    plt.close()
    
    # Print AUC scores
    print("\nAUC-ROC Scores:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:35s}: {roc_auc[i]:.4f}")
    
    return roc_auc

def create_per_class_metrics_table(y_true, y_pred, class_names, output_dir):
    """
    Create a detailed per-class metrics table
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    print(f"\n{'='*80}")
    print("PER-CLASS DETAILED METRICS")
    print(f"{'='*80}\n")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'F1-Score': f1,
        'Support (Test Samples)': support
    })
    
    # Add accuracy per class
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    metrics_df['Accuracy'] = class_accuracy
    
    # Display
    print(metrics_df.to_string(index=False))
    
    # Save
    csv_path = os.path.join(output_dir, 'per_class_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n✓ Per-class metrics saved: {csv_path}")
    
    return metrics_df

def main():
    args = parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        model_dir = os.path.dirname(args.model_path)
        output_dir = os.path.join(model_dir, 'evaluation')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Results will be saved to: {output_dir}\n")
    
    # Load model and data
    model, test_gen, test_df = load_model_and_data(args.model_path)
    
    # Step 15: Performance metrics
    y_true, y_pred, predictions, class_names = evaluate_model(model, test_gen, output_dir)
    
    # Step 16: Confusion matrix
    cm = create_confusion_matrix(y_true, y_pred, class_names, output_dir)
    
    # Bonus: ROC curves
    roc_auc = create_roc_curves(y_true, predictions, class_names, output_dir)
    
    # Bonus: Detailed metrics table
    metrics_df = create_per_class_metrics_table(y_true, y_pred, class_names, output_dir)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  ✓ classification_report.json")
    print(f"  ✓ classification_report.txt")
    print(f"  ✓ confusion_matrix.png")
    print(f"  ✓ roc_curves.png")
    print(f"  ✓ per_class_metrics.csv")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
