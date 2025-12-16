"""
Central Configuration for Derm-X Research Project
Contains all hyperparameters, paths, and settings
"""
import os

# ======================
# DATASET CONFIGURATION
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')
HAM_BASE_DIR = os.path.join(DATASET_DIR, 'HAM10000')  # HAM10000 subdirectory
METADATA_CSV = os.path.join(HAM_BASE_DIR, 'HAM10000_metadata.csv')

# Image directories
IMAGE_DIRS = [
    os.path.join(HAM_BASE_DIR, 'HAM10000_images_part_1'),
    os.path.join(HAM_BASE_DIR, 'HAM10000_images_part_2')
]

# Class Labels (8 skin lesion types - 7 cancers + Acne)
LESION_TYPE_DICT = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma',
    'acne': 'Acne'  # NEW: 8th class from DermNet
}

NUM_CLASSES = 8  # Updated from 7 to 8

# ======================
# PREPROCESSING
# ======================
IMG_SIZE = (224, 224)  # Standard for MobileNetV2/ResNet50
APPLY_HAIR_REMOVAL = True  # Toggle hair removal preprocessing
APPLY_LESION_SEGMENTATION = False  # Experimental: center-crop lesions
NORMALIZATION_METHOD = 'imagenet'  # 'imagenet' or 'custom'

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ======================
# DATA AUGMENTATION
# ======================
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
    'zoom_range': 0.1,
    'fill_mode': 'nearest'
}

# ======================
# TRAINING CONFIGURATION
# ======================
BATCH_SIZE = 64  # Increased from 32 for RTX 3050 (reduce to 32 if OOM)
EPOCHS = 25
VALIDATION_SPLIT = 0.10  # 10% of train+val for validation
TEST_SPLIT = 0.20  # 20% for final testing
RANDOM_SEED = 42

# Learning rates (architecture-specific)
LEARNING_RATES = {
    'mobilenetv2': 0.001,
    'resnet50': 0.0005,
    'efficientnet-b0': 0.001,
    'efficientnet-b3': 0.0008,
    'efficientnet-b7': 0.0005
}

# ======================
# MODEL ARCHITECTURE
# ======================
# Dropout rate for classification head
DROPOUT_RATE = 0.5

# Dense layer size before final classification
DENSE_UNITS = 128

# Fine-tuning strategy (number of layers to unfreeze after initial training)
FINE_TUNE_LAYERS = {
    'mobilenetv2': 30,  # Unfreeze last 30 layers
    'resnet50': 20,
    'efficientnet-b0': 40,
    'efficientnet-b3': 50, 
    'efficientnet-b7': 60
}

# ======================
# CALLBACKS
# ======================
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
MIN_LEARNING_RATE = 1e-7

# ======================
# OUTPUT PATHS
# ======================
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(BASE_DIR, 'visualizations', 'plots')
GRADCAM_DIR = os.path.join(BASE_DIR, 'visualizations', 'gradcam')

# Create directories if they don't exist
for dir_path in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, GRADCAM_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ======================
# GRAD-CAM CONFIGURATION
# ======================
# Last convolutional layer names for each architecture
GRADCAM_LAYER_NAMES = {
    'mobilenetv2': 'out_relu',  # MobileNetV2's last conv layer
    'resnet50': 'conv5_block3_out',
    'efficientnet-b0': 'top_activation',
    'efficientnet-b3': 'top_activation',
    'efficientnet-b7': 'top_activation'
}

# ======================
# CLASS IMBALANCE HANDLING
# ======================
USE_CLASS_WEIGHTS = True  # Automatically calculate class weights
USE_SMOTE = False  # SMOTE can be memory-intensive for images

# ======================
# EVALUATION METRICS
# ======================
# Which class is considered most critical (Melanoma)
CRITICAL_CLASS = 'mel'
CRITICAL_CLASS_INDEX = list(LESION_TYPE_DICT.keys()).index(CRITICAL_CLASS)

# ======================
# DEPLOYMENT
# ======================
STREAMLIT_PORT = 8501
DEFAULT_MODEL_FOR_DEMO = 'efficientnet-b3'  # Best model for demo

# ======================
# LOGGING
# ======================
VERBOSE = 1  # TensorFlow verbosity (0=silent, 1=progress bar, 2=one line per epoch)
LOG_FILE = os.path.join(RESULTS_DIR, 'training_log.txt')

# ======================
# GPU OPTIMIZATION
# ======================
USE_MIXED_PRECISION = True  # Enable AMP for 1.5-2x speedup on RTX 3050
GPU_MEMORY_GROWTH = True  # Prevent TensorFlow from allocating all GPU memory
GPU_MEMORY_LIMIT_MB = None  # Optional: Set to 3072 to limit to 3GB

# Data pipeline optimization
USE_TF_DATA_PIPELINE = True  # Use tf.data with prefetch (faster than ImageDataGenerator)
PREFETCH_BUFFER_SIZE = 'AUTOTUNE'  # Let TensorFlow auto-tune prefetch buffer
CACHE_DATASET = False  # Set to True if dataset fits in RAM (not recommended for HAM10000)

if __name__ == "__main__":
    # Test configuration
    print("=== DERM-X CONFIGURATION ===")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"\nClass Labels:")
    for key, val in LESION_TYPE_DICT.items():
        print(f"  {key}: {val}")
    print(f"\nOutput Directories:")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Plots: {PLOTS_DIR}")
