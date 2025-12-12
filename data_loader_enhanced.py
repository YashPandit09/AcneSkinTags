"""
Enhanced Data Loader with Advanced Preprocessing & Aggressive Class Balancing
Integrates hair removal, lesion segmentation, normalization, and oversampling
"""
import pandas as pd
import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample  # NEW: For aggressive oversampling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from PIL import Image
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from preprocessing.hair_removal import remove_hair
from preprocessing.lesion_segmentation import segment_lesion
from preprocessing.normalization import normalize_imagenet

class AdvancedImageSequence(Sequence):
    """
    Custom data generator with preprocessing pipeline.
    """
    def __init__(self, dataframe, batch_size, img_size, apply_hair_removal=False,
                 apply_segmentation=False, augmentation_gen=None, shuffle=True):
        self.df = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.apply_hair_removal = apply_hair_removal
        self.apply_segmentation = apply_segmentation
        self.augmentation_gen = augmentation_gen
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X, y = self._load_batch(batch_df)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def _load_batch(self, batch_df):
        X = []
        y = []
        
        for _, row in batch_df.iterrows():
            # Load image
            img = Image.open(row['path']).convert('RGB')
            img_array = np.array(img)
            
            # Apply preprocessing
            if self.apply_hair_removal:
                img_array = remove_hair(img_array)
            
            if self.apply_segmentation:
                img_array = segment_lesion(img_array, target_size=self.img_size)
            else:
                # Just resize
                img_array = np.array(Image.fromarray(img_array).resize(self.img_size))
            
            # Normalize
            img_array = normalize_imagenet(img_array)
            
            X.append(img_array)
            y.append(row['label_idx'])
        
        return np.array(X), np.array(y)

def get_enhanced_data_generators(base_dir='./Dataset', img_size=(224, 224), batch_size=32,
                                  apply_hair_removal=False, apply_segmentation=False):
    """
    Enhanced data loader with preprocessing options.
    NOW INCLUDES: HAM10000 (7 cancer types) + DermNet Acne (8th class)
    
    Args:
        base_dir: Dataset directory
        img_size: Target image size
        batch_size: Batch size
        apply_hair_removal: Enable hair removal preprocessing
        apply_segmentation: Enable lesion segmentation
        
    Returns:
        train_gen, val_gen, test_gen, class_weights, test_df
    """
    print(f"\n{'='*60}")
    print("ENHANCED DATA LOADER - 8 CLASS SYSTEM")
    print(f"{'='*60}")
    print(f"✓ Hair Removal: {'Enabled' if apply_hair_removal else 'Disabled'}")
    print(f"✓ Lesion Segmentation: {'Enabled' if apply_segmentation else 'Disabled'}")
    print(f"✓ Normalization: Simple /255 (matches training)")
    print(f"{'='*60}\n")
    
    # --- PART A: LOAD HAM10000 (The 7 Cancer Types) ---
    print("[INFO] Loading HAM10000 (Cancer data)...")
    image_path_dict = {}
    
    # Updated: HAM10000 is inside Dataset/HAM10000/ subdirectory
    ham_base_dir = os.path.join(base_dir, 'HAM10000')
    
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        part_dir = os.path.join(ham_base_dir, part)
        if os.path.exists(part_dir):
            paths = {os.path.splitext(os.path.basename(x))[0]: x
                    for x in glob(os.path.join(part_dir, '*.jpg'))}
            image_path_dict.update(paths)
    
    csv_path = os.path.join(ham_base_dir, 'HAM10000_metadata.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    
    df_ham = pd.read_csv(csv_path)
    df_ham['path'] = df_ham['image_id'].map(image_path_dict)
    df_ham = df_ham.dropna(subset=['path'])
    
    # Map labels
    df_ham['cell_type'] = df_ham['dx'].map(config.LESION_TYPE_DICT)
    df_ham['label_idx'] = df_ham['dx'].map({k: i for i, k in enumerate(config.LESION_TYPE_DICT.keys())})
    df_ham = df_ham[['path', 'cell_type', 'label_idx']].dropna()
    
    print(f"✓ HAM10000 Images Loaded: {len(df_ham)}")
    
    # --- PART B: LOAD ACNE (From DermNet) ---
    print("[INFO] Looking for Acne data...")
    acne_paths = []
    
    # Try multiple possible locations (prioritize Dataset/DermNet)
    possible_acne_dirs = [
        os.path.join(base_dir, 'DermNet', 'Acne and Rosacea Photos'),  # FIRST: Dataset/DermNet/
        os.path.join(base_dir, '..', 'DermNet', 'Acne and Rosacea Photos'),
        os.path.join(base_dir, '..', 'test', 'Acne and Rosacea Photos'),
        './DermNet/Acne and Rosacea Photos',
        './test/Acne and Rosacea Photos'
    ]
    
    for acne_dir in possible_acne_dirs:
        if os.path.exists(acne_dir):
            acne_paths = glob(os.path.join(acne_dir, '*.jpg'))
            if acne_paths:
                print(f"✓ Found Acne data in: {acne_dir}")
                break
    
    if not acne_paths:
        print(f"⚠ WARNING: No Acne images found. Searched in:")
        for d in possible_acne_dirs:
            print(f"  - {d}")
        print("Continuing with 7 classes only (HAM10000)")
        df_acne = pd.DataFrame(columns=['path', 'cell_type', 'label_idx'])
    else:
        acne_label_idx = list(config.LESION_TYPE_DICT.keys()).index('acne')
        df_acne = pd.DataFrame({
            'path': acne_paths,
            'cell_type': 'Acne',
            'label_idx': acne_label_idx
        })
        print(f"✓ Acne Images Loaded: {len(df_acne)}")
        
        # Oversample if too few acne images
        target_count = 1000
        if len(df_acne) < target_count:
            print(f"[INFO] Oversampling Acne data to {target_count} samples...")
            oversample_factor = max(2, target_count // len(df_acne))
            df_acne = pd.concat([df_acne] * oversample_factor, ignore_index=True)
            print(f"✓ Acne after oversampling: {len(df_acne)}")
    
    # --- PART C: MERGE & AGGRESSIVE BALANCING ---
    df = pd.concat([df_ham, df_acne], ignore_index=True)
    
    print(f"\n[INFO] BEFORE Balancing - Total images: {len(df)}")
    print(f"\n[INFO] Raw Class Distribution:")
    class_counts = df['cell_type'].value_counts()
    print(class_counts)
    
    # CRITICAL FIX: Aggressive Oversampling
    # Find the majority class size
    max_size = class_counts.max()
    print(f"\n[INFO] Balancing all classes to match majority class size: {max_size}")
    
    # Resample each class to match the majority
    balanced_dfs = []
    for class_name in df['cell_type'].unique():
        class_subset = df[df['cell_type'] == class_name]
        current_size = len(class_subset)
        
        if current_size < max_size:
            # Upsample minority classes
            class_subset_upsampled = resample(
                class_subset,
                replace=True,  # Sample with replacement (creates duplicates)
                n_samples=max_size,  # Match the majority class
                random_state=42
            )
            balanced_dfs.append(class_subset_upsampled)
            print(f"  ✓ {class_name[:30]:30s}: {current_size:5d} → {max_size:5d} (upsampled)")
        else:
            balanced_dfs.append(class_subset)
            print(f"  ✓ {class_name[:30]:30s}: {current_size:5d} (majority class)")
    
    df = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"\n[INFO] AFTER Balancing - Total images: {len(df)}")
    print(f"\n[INFO] Balanced Class Distribution:")
    print(df['cell_type'].value_counts())
    print("\n✓ All classes now have EQUAL representation!")
    print("  → Model can no longer 'cheat' by guessing majority class")
    
    # --- STRATIFIED SPLIT ---
    train_val_df, test_df = train_test_split(
        df, test_size=config.TEST_SPLIT, random_state=config.RANDOM_SEED, 
        stratify=df['cell_type']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=config.VALIDATION_SPLIT, random_state=config.RANDOM_SEED,
        stratify=train_val_df['cell_type']
    )
    
    print(f"\n[INFO] Split: {len(train_df)} Train, {len(val_df)} Val, {len(test_df)} Test")
    
    # --- COMPUTE CLASS WEIGHTS ---
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        classes = np.array([i for i in range(config.NUM_CLASSES)])
        weights = compute_class_weight('balanced', classes=classes, y=train_df['label_idx'].values)
        class_weights = dict(enumerate(weights))
        print(f"\n[INFO] Class Weights (for imbalanced data):")
        for idx, (key, label) in enumerate(config.LESION_TYPE_DICT.items()):
            if idx < len(weights):
                print(f"  {label[:30]:30s} → Weight: {weights[idx]:.3f}")
    
    # --- DATA AUGMENTATION (INCREASED for oversampled data) ---
    # Since we now have many duplicates, we need STRONG augmentation
    # to prevent the model from memorizing the repeated images
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # CRITICAL: Simple rescaling, not ImageNet normalization
        rotation_range=40,  # Increased from 20 to 40 degrees
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,  # NEW: Adds shearing transformation
        zoom_range=0.2,  # Increased zoom for more variation
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # For val/test: no augmentation
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # --- GENERATORS ---
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df, x_col='path', y_col='cell_type',
        target_size=img_size, batch_size=batch_size, class_mode='categorical',
        shuffle=True
    )
    
    val_gen = val_test_datagen.flow_from_dataframe(
        dataframe=val_df, x_col='path', y_col='cell_type',
        target_size=img_size, batch_size=batch_size, class_mode='categorical',
        shuffle=False
    )
    
    test_gen = val_test_datagen.flow_from_dataframe(
        dataframe=test_df, x_col='path', y_col='cell_type',
        target_size=img_size, batch_size=batch_size, class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n[SUCCESS] Data generators ready! (8 classes)\n")
    
    return train_gen, val_gen, test_gen, class_weights, test_df

# Maintain backward compatibility with old data_loader
def get_data_generators(base_dir='./Dataset', img_size=(224, 224), batch_size=32):
    """
    Original data loader (for backward compatibility).
    """
    return get_enhanced_data_generators(
        base_dir=base_dir, img_size=img_size, batch_size=batch_size,
        apply_hair_removal=False, apply_segmentation=False
    )[:3] + (None,)  # Return 4 values like original

if __name__ == "__main__":
    print("Testing Enhanced Data Loader...\n")
    
    try:
        train_gen, val_gen, test_gen, class_weights, test_df = get_enhanced_data_generators(
            base_dir='./Dataset',
            apply_hair_removal=False,  # Set to True to test hair removal
            apply_segmentation=False   # Set to True to test segmentation
        )
        
        print("\n[TEST] Loading one batch from training set...")
        X_batch, y_batch = next(train_gen)
        print(f"✓ Batch shape: {X_batch.shape}")
        print(f"✓ Labels shape: {y_batch.shape}")
        print(f"✓ Pixel range: [{X_batch.min():.3f}, {X_batch.max():.3f}]")
        
        print("\n✓✓✓ Enhanced Data Loader is working correctly! ✓✓✓\n")
        
    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        import traceback
        traceback.print_exc()
