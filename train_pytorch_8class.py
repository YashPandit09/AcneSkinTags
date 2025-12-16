"""
Enhanced PyTorch Training Script - 8 Classes (HAM10000 + DermNet Acne)
Combines 7 classes from HAM10000 with Acne from DermNet for complete 8-class training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import config

# ============================================================
# STEP 1: GPU SETUP
# ============================================================
def setup_device():
    """Configure GPU for training"""
    print("\n" + "="*80)
    print("STEP 1: GPU CONFIGURATION")
    print("="*80)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU Available: True")
        print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"✓ TF32 enabled for faster computation")
    else:
        device = torch.device("cpu")
        print("⚠ No GPU detected - training on CPU (will be slow)")
    
    print("="*80 + "\n")
    return device

# ============================================================
# STEP 2: ENHANCED DATASET (HAM10000 + DermNet Acne)
# ============================================================
class EnhancedSkinLesionDataset(Dataset):
    """Dataset combining HAM10000 (7 classes) + DermNet Acne (8th class)"""
    
    def __init__(self, transform=None, split='train'):
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Define class mapping (0-7)
        self.class_to_idx = {
            'nv': 0,    # Melanocytic nevi
            'mel': 1,   # Melanoma
            'bkl': 2,   # Benign keratosis
            'bcc': 3,   # Basal cell carcinoma
            'akiec': 4, # Actinic keratoses
            'vasc': 5,  # Vascular lesions
            'df': 6,    # Dermatofibroma
            'acne': 7   # Acne (from DermNet)
        }
        
        self.idx_to_class = {v: config.LESION_TYPE_DICT[k] for k, v in self.class_to_idx.items()}
        
        # Load HAM10000 data (7 classes)
        self._load_ham10000()
        
        # Load DermNet Acne data (8th class)
        self._load_acne()
        
        print(f"\n[INFO] Total samples loaded: {len(self.samples)}")
        print("[INFO] Class distribution:")
        for idx in range(8):
            count = sum(1 for label in self.labels if label == idx)
            print(f"  {idx}: {self.idx_to_class[idx]:30s} - {count} images")
    
    def _load_ham10000(self):
        """Load HAM10000 dataset"""
        print("\n[INFO] Loading HAM10000 dataset...")
        
        # Read metadata
        metadata_path = config.METADATA_CSV
        df = pd.read_csv(metadata_path)
        
        # Image directories
        img_paths = {}
        for img_dir in config.IMAGE_DIRS:
            if os.path.exists(img_dir):
                for img_file in os.listdir(img_dir):
                    if img_file.endswith('.jpg'):
                        img_id = img_file.replace('.jpg', '')
                        img_paths[img_id] = os.path.join(img_dir, img_file)
        
        # Add samples
        for idx, row in df.iterrows():
            if row['image_id'] in img_paths:
                self.samples.append(img_paths[row['image_id']])
                self.labels.append(self.class_to_idx[row['dx']])
        
        print(f"✓ Loaded {len([l for l in self.labels if l < 7])} HAM10000 images")
    
    def _load_acne(self):
        """Load Acne data from DermNet"""
        print("\n[INFO] Loading DermNet Acne dataset...")
        
        acne_dir = os.path.join(config.DATASET_DIR, 'DermNet', 'Acne and Rosacea Photos')
        
        if not os.path.exists(acne_dir):
            print(f"⚠ Warning: Acne directory not found at {acne_dir}")
            return
        
        # Add all acne images
        acne_count = 0
        for img_file in os.listdir(acne_dir):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(acne_dir, img_file)
                self.samples.append(img_path)
                self.labels.append(7)  # Acne class index
                acne_count += 1
        
        print(f"✓ Loaded {acne_count} DermNet Acne images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================
# STEP 3: DATA LOADING
# ============================================================
def get_dataloaders(batch_size=64, num_workers=0):
    """Create optimized DataLoaders with 8 classes"""
    print("\n" + "="*80)
    print("STEP 2: DATA LOADING - 8 CLASSES (HAM10000 + DermNet Acne)")
    print("="*80)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    full_dataset = EnhancedSkinLesionDataset(transform=train_transform)
    
    # Split dataset (70/10/20)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Update transforms for val/test
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Pin memory: {torch.cuda.is_available()}")
    print("="*80 + "\n")
    
    return train_loader, val_loader, test_loader

# ============================================================
# STEP 4: MODEL
# ============================================================
def build_model(num_classes=8, device='cuda'):
    """Build MobileNetV2 model"""
    print("\n" + "="*80)
    print("STEP 3: BUILDING MODEL (8-Class Classification)")
    print("="*80)
    
    model = models.mobilenet_v2(weights='DEFAULT')
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, num_classes)
    )
    
    model = model.to(device)
    
    print(f"✓ Model: MobileNetV2 (pretrained)")
    print(f"✓ Output classes: {num_classes}")
    print(f"✓ Device: {device}")
    print(f"✓ Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("="*80 + "\n")
    
    return model

# ============================================================
# TRAINING FUNCTIONS (Same as before)
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# ============================================================
# MAIN
# ============================================================
def main(args):
    device = setup_device()
    
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    model = build_model(num_classes=8, device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    scaler = GradScaler() if args.use_amp else None
    
    print("\n" + "="*80)
    print("STEP 4: TRAINING CONFIGURATION (8-Class)")
    print("="*80)
    print(f"✓ Epochs: {args.epochs}")
    print(f"✓ Batch size: {args.batch_size}")
    print(f"✓ Learning rate: {args.lr}")
    print(f"✓ Mixed Precision (AMP): {args.use_amp}")
    print(f"✓ Device: {device}")
    print("="*80 + "\n")
    
    print("="*80)
    print("STEP 5: TRAINING (8-Class Model)")
    print("="*80 + "\n")
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, args.use_amp
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_8class_pytorch.pth')
            print(f"  ✓ Best model saved (val_acc: {val_acc:.2f}%)")
        
        if torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE - 8-Class Model")
    print("="*80)
    print(f"✓ Total time: {total_time/60:.1f} minutes")
    print(f"✓ Best val accuracy: {best_val_acc:.2f}%")
    print(f"✓ Average time per epoch: {total_time/args.epochs:.1f} seconds")
    print(f"✓ Model saved: best_model_8class_pytorch.pth")
    print("="*80 + "\n")
    
    # Plot history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('8-Class Training')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.title('8-Class Training')
    
    plt.tight_layout()
    plt.savefig('training_history_8class_pytorch.png', dpi=300)
    print("✓ Training plots saved: training_history_8class_pytorch.png\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='8-Class GPU-Optimized PyTorch Training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use-amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false', help='Disable AMP')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user\n")
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
