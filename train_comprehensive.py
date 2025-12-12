"""
Comprehensive Training Script for Derm-X Research Project
Supports multiple architectures with argument parsing
"""
import argparse
import os
import sys
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from models.model_factory import ModelFactory
from data_loader_enhanced import get_enhanced_data_generators

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train skin lesion classification model')
    parser.add_argument('--model', type=str, default='mobilenetv2',
                       choices=['resnet50', 'mobilenetv2', 'efficientnet-b0', 'efficientnet-b3', 'efficientnet-b7'],
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--hair-removal', action='store_true',
                       help='Enable hair removal preprocessing')
    parser.add_argument('--segmentation', action='store_true',
                       help='Enable lesion segmentation preprocessing')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Enable fine-tuning (unfreeze some base layers)')
    parser.add_argument('--fine-tune-epoch', type=int, default=10,
                       help='Epoch to start fine-tuning')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory for this run')
    
    return parser.parse_args()

def train_model(args):
    """Main training function."""
    
    print("\n" + "="*80)
    print(f"DERM-X TRAINING - {args.model.upper()}")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Hair Removal: {args.hair_removal}")
    print(f"  Lesion Segmentation: {args.segmentation}")
    print(f"  Fine-Tuning: {args.fine_tune}")
    print("="*80 + "\n")
    
    # --- 1. SETUP OUTPUT DIRECTORY ---
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(config.RESULTS_DIR, f"{args.model}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_file = os.path.join(output_dir, 'training_config.json')
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Output directory: {output_dir}\n")
    
    # --- 2. LOAD DATA ---
    train_gen, val_gen, test_gen, class_weights, test_df = get_enhanced_data_generators(
        base_dir=config.DATASET_DIR,
        img_size=config.IMG_SIZE,
        batch_size=args.batch_size,
        apply_hair_removal=args.hair_removal,
        apply_segmentation=args.segmentation
    )
    
    # --- 3. BUILD MODEL ---
    model = ModelFactory.create(
        model_name=args.model,
        num_classes=config.NUM_CLASSES,
        img_size=config.IMG_SIZE,
        trainable_base=False  # Start with frozen base
    )
    
    # --- 4. COMPILE MODEL ---
    learning_rate = config.LEARNING_RATES.get(args.model, 0.001)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    print(f"\n✓ Model compiled with learning rate: {learning_rate}")
    
    # --- 5. SETUP CALLBACKS ---
    callbacks = [
        ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1
        ),
        CSVLogger(
            os.path.join(output_dir, 'training_log.csv'),
            append=False
        )
    ]
    
    print("✓ Callbacks configured")
    print("  - Model checkpoint (best model)")
    print(f"  - Early stopping (patience={config.EARLY_STOPPING_PATIENCE})")
    print(f"  - Learning rate reduction (patience={config.REDUCE_LR_PATIENCE})")
    print("  - CSV logging\n")
    
    # --- 6. TRAIN MODEL ---
    print("="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    history = model.fit(
        train_gen,
        epochs=args.epochs if not args.fine_tune else args.fine_tune_epoch,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=config.VERBOSE
    )
    
    # --- 7. FINE-TUNING (Optional) ---
    if args.fine_tune:
        print("\n" + "="*80)
        print("FINE-TUNING PHASE")
        print("="*80 + "\n")
        
        # Unfreeze layers
        num_layers_to_unfreeze = config.FINE_TUNE_LAYERS.get(args.model, 30)
        ModelFactory.unfreeze_layers(model, num_layers_to_unfreeze)
        
        # Recompile with lower learning rate
        fine_tune_lr = learning_rate * 0.1
        model.compile(
            optimizer=Adam(learning_rate=fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )
        
        print(f"✓ Fine-tuning with LR: {fine_tune_lr}\n")
        
        # Continue training
        history_fine = model.fit(
            train_gen,
            initial_epoch=args.fine_tune_epoch,
            epochs=args.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=config.VERBOSE
        )
        
        # Merge histories
        for key in history.history.keys():
            history.history[key].extend(history_fine.history[key])
    
    # --- 8. SAVE FINAL MODEL ---
    final_model_path = os.path.join(output_dir, 'final_model.keras')
    model.save(final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    
    # --- 9. GENERATE TRAINING PLOTS ---
    plot_training_history(history.history, output_dir)
    
    # --- 10. EVALUATE ON TEST SET ---
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80 + "\n")
    
    test_results = model.evaluate(test_gen, verbose=1)
    test_metrics = dict(zip(model.metrics_names, test_results))
    
    print("\n--- Test Set Results ---")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name:15s}: {value:.4f}")
    
    # Save test results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")
    
    return model, history, test_metrics

def plot_training_history(history, output_dir):
    """Generate and save training plots."""
    print("\n✓ Generating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    if 'auc' in history:
        axes[1, 0].plot(history['auc'], label='Train AUC', linewidth=2)
        axes[1, 0].plot(history['val_auc'], label='Val AUC', linewidth=2)
        axes[1, 0].set_title('AUC-ROC', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Precision/Recall
    if 'precision' in history and 'recall' in history:
        axes[1, 1].plot(history['precision'], label='Train Precision', linewidth=2)
        axes[1, 1].plot(history['val_precision'], label='Val Precision', linewidth=2)
        axes[1, 1].plot(history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training plots saved: {plot_path}")

if __name__ == "__main__":
    args = parse_args()
    
    try:
        model, history, test_metrics = train_model(args)
        print("\n✓✓✓ SUCCESS ✓✓✓\n")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
