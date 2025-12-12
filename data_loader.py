import pandas as pd
import os
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(img_size=(224, 224), batch_size=32):
    print(f"[INFO] Building Hybrid Dataset (HAM10000 + Acne from DermNet)...")

    # --- PART A: LOAD HAM10000 (The Cancers) ---
    # We look inside the folder 'HAM10000' (Make sure you renamed it!)
    base_ham = './HAM10000' 
    
    # 1. Map Image IDs to file paths
    # This finds all .jpg files inside HAM10000 (and its subfolders)
    image_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                       for x in glob(os.path.join(base_ham, '**', '*.jpg'), recursive=True)}

    # 2. Load CSV
    csv_path = os.path.join(base_ham, 'HAM10000_metadata.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV! Expected at: {csv_path}")

    df_ham = pd.read_csv(csv_path)
    
    # 3. Create 'path' column
    df_ham['path'] = df_ham['image_id'].map(image_path_dict)
    
    # 4. Map labels
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    df_ham['cell_type'] = df_ham['dx'].map(lesion_type_dict)
    df_ham = df_ham[['path', 'cell_type']].dropna()
    
    print(f"[INFO] HAM10000 Images Loaded: {len(df_ham)}")

    # --- PART B: LOAD ACNE (From DermNet) ---
    # We look inside 'DermNet/Acne and Rosacea Photos'
    # Note: We are ignoring the other 22 folders in DermNet to keep the project focused
    acne_dir = './DermNet/Acne and Rosacea Photos'
    
    acne_paths = glob(os.path.join(acne_dir, '*.jpg'))
    
    if len(acne_paths) == 0:
        print(f"[WARNING] No Acne images found in {acne_dir}. Did you rename the 'test' folder to 'DermNet'?")
    
    df_acne = pd.DataFrame({
        'path': acne_paths,
        'cell_type': 'Acne'  # Assigning the new 8th Label
    })
    
    print(f"[INFO] Acne Images Loaded: {len(df_acne)}")

    # --- PART C: MERGE & BALANCE ---
    # Combine
    final_df = pd.concat([df_ham, df_acne], ignore_index=True)
    
    # Check if we have enough Acne data compared to Cancer data
    # If Acne is < 1000 images, we repeat them to prevent the model from ignoring them
    target_count = 1000
    if len(df_acne) < target_count and len(df_acne) > 0:
        print(f"[INFO] Oversampling Acne data to match Cancer classes...")
        oversample_factor = target_count // len(df_acne)
        df_acne_oversampled = pd.concat([df_acne] * oversample_factor, ignore_index=True)
        final_df = pd.concat([df_ham, df_acne_oversampled], ignore_index=True)

    print(f"[INFO] Final Training Set Size: {len(final_df)}")
    print(final_df['cell_type'].value_counts())

    # --- PART D: GENERATORS ---
    # Stratified Split (Ensures Acne is in both Train and Test)
    train_val_df, test_df = train_test_split(
        final_df, test_size=0.20, random_state=42, stratify=final_df['cell_type']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.10, random_state=42, stratify=train_val_df['cell_type']
    )

    # Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df, x_col='path', y_col='cell_type',
        target_size=img_size, batch_size=batch_size, class_mode='categorical'
    )
    
    val_gen = val_test_datagen.flow_from_dataframe(
        dataframe=val_df, x_col='path', y_col='cell_type',
        target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    
    test_gen = val_test_datagen.flow_from_dataframe(
        dataframe=test_df, x_col='path', y_col='cell_type',
        target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    
    return train_gen, val_gen, test_gen, test_df