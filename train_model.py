import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Import the data loader function we created in the previous step
from data_loader import get_data_generators

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # You can increase this if you have a good GPU
NUM_CLASSES = 7  # HAM10000 has 7 classes

def build_model(num_classes):
    """
    Builds the MobileNetV2 model with Transfer Learning.
    """
    # 1. Load the Base Model (Pre-trained on ImageNet)
    # include_top=False removes the final classification layer (which was for 1000 things like cats/dogs)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # 2. Freeze the Base Model
    # We don't want to destroy the pre-learned features (edges, textures) yet.
    base_model.trainable = False
    
    # 3. Add Custom Head (The "Classifier")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Flattens the output
    x = Dense(128, activation='relu')(x) # Dense layer to learn patterns
    x = Dropout(0.5)(x)              # Dropout to prevent overfitting (Crucial for small datasets)
    predictions = Dense(num_classes, activation='softmax')(x) # Final prediction layer
    
    # 4. Combine them
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # 1. Get Data
    train_gen, val_gen, test_gen, _ = get_data_generators()
    
    # 2. Build Model
    model = build_model(NUM_CLASSES)
    
    # 3. Compile Model
    # We use a small learning rate (0.001) to start
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("\n[INFO] Model Built. Starting Training...")
    model.summary() # Prints a table of layers

    # 4. Define Callbacks (The "Safety Nets")
    callbacks = [
        # Save the model ONLY when validation accuracy improves (saves space)
        ModelCheckpoint('best_skin_model.keras', save_best_only=True, monitor='val_accuracy', mode='max'),
        
        # Stop training if validation loss doesn't improve for 5 epochs (saves time)
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        
        # Lower the learning rate if we get stuck on a plateau
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    # 5. Train!
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )

    # 6. Save Final Weights
    model.save('final_skin_model.h5')
    print("[SUCCESS] Training Complete. Model saved as 'best_skin_model.keras'")

    # 7. Plot Results (For your Paper)
    # This generates the "Accuracy vs Epochs" graph you need for the Results section
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("[INFO] Training plots saved as 'training_history.png'")