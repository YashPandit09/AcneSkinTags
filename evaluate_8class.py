"""
Grad-CAM Evaluation for 8-Class Model (Including Acne)
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import random

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = 'best_model_8class_pytorch.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CRITICAL: Alphabetical order for 8 classes (Acne is index 0)
CLASS_NAMES = [
    'Acne',
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanocytic nevi',
    'Melanoma',
    'Vascular lesions'
]

# Map to match training order (config.LESION_TYPE_DICT order)
TRAINING_CLASS_ORDER = {
    0: 'Melanocytic nevi',    # nv
    1: 'Melanoma',            # mel
    2: 'Benign keratosis-like lesions',  # bkl
    3: 'Basal cell carcinoma',  # bcc
    4: 'Actinic keratoses',   # akiec
    5: 'Vascular lesions',    # vasc
    6: 'Dermatofibroma',      # df
    7: 'Acne'                 # acne
}

# ============================================================
# GRAD-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), output

# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    print(f"\nDevice: {DEVICE}")
    print(f"\nLoading model from: {MODEL_PATH}")
    
    model = models.mobilenet_v2(weights='DEFAULT')
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 8)
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Get target layer for Grad-CAM
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    
    return model, gradcam

# ============================================================
# LOAD SAMPLE IMAGES
# ============================================================
def get_sample_images():
    """Get sample images from all 8 classes"""
    import config
    
    samples = []
    
    # HAM10000 samples (first 7 classes)
    df = pd.read_csv(config.METADATA_CSV)
    img_paths = {}
    
    for img_dir in config.IMAGE_DIRS:
        if os.path.exists(img_dir):
            for img_file in os.listdir(img_dir):
                if img_file.endswith('.jpg'):
                    img_id = img_file.replace('.jpg', '')
                    img_paths[img_id] = os.path.join(img_dir, img_file)
    
    # Get samples from each HAM class
    for dx_code in ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']:
        class_df = df[df['dx'] == dx_code]
        if len(class_df) > 0:
            sample = class_df.sample(n=min(2, len(class_df)))
            for _, row in sample.iterrows():
                if row['image_id'] in img_paths:
                    samples.append({
                        'path': img_paths[row['image_id']],
                        'label': dx_code,
                        'name': row['image_id']
                    })
    
    # Get Acne samples
    acne_dir = os.path.join(config.DATASET_DIR, 'DermNet', 'Acne and Rosacea Photos')
    if os.path.exists(acne_dir):
        acne_files = [f for f in os.listdir(acne_dir) if f.lower().endswith(('.jpg', '.png'))]
        for img_file in random.sample(acne_files, min(2, len(acne_files))):
            samples.append({
                'path': os.path.join(acne_dir, img_file),
                'label': 'acne',
                'name': img_file.replace('.jpg', '').replace('.png', '')
            })
    
    return samples

# ============================================================
# VISUALIZATION
# ============================================================
def visualize_gradcam(model, gradcam, samples, output_dir='gradcam_8class'):
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"\n{'='*80}")
    print(f"Generating Grad-CAM visualizations for {len(samples)} samples...")
    print(f"{'='*80}\n")
    
    correct = 0
    total = 0
    
    for idx, sample in enumerate(samples, 1):
        # Load image
        img_path = sample['path']
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Generate Grad-CAM
        cam, output = gradcam.generate_cam(img_tensor)
        
        # Get prediction
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(dim=1).item()
        pred_class = TRAINING_CLASS_ORDER[pred_idx]
        pred_prob = probs[pred_idx].item()
        
        # True label
        true_label_code = sample['label']
        label_map = {
            'nv': 'Melanocytic nevi', 'mel': 'Melanoma', 'bkl': 'Benign keratosis-like lesions',
            'bcc': 'Basal cell carcinoma', 'akiec': 'Actinic keratoses', 'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma', 'acne': 'Acne'
        }
        true_label = label_map[true_label_code]
        
        is_correct = (pred_class == true_label)
        if is_correct:
            correct += 1
        total += 1
        
        print(f"[{idx}/{len(samples)}] {sample['name'][:30]:30s}")
        print(f"  True: {true_label}")
        print(f"  Predicted: {pred_class} ({pred_prob*100:.1f}% confident)")
        print(f"  {'✓ CORRECT' if is_correct else '✗ WRONG'}\n")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size))
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        img_array = np.array(img)
        heatmap = plt.cm.jet(cam_resized / 255.0)[:, :, :3]
        overlay = (0.6 * img_array / 255.0 + 0.4 * heatmap * 255) / 255.0
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # Suptitle with prediction
        status = '✓ CORRECT' if is_correct else '✗ WRONG'
        fig.suptitle(f'{status} | True: {true_label} | Pred: {pred_class} ({pred_prob*100:.1f}%)',
                     fontsize=12, fontweight='bold',
                     color='green' if is_correct else 'red')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'gradcam_{idx:02d}_{sample["name"][:30]}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved visualization: {save_path}")
    
    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"✓ All visualizations saved to: {output_dir}")
    print(f"\nAccuracy on sample: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"\nInterpretation Guide:")
    print(f"  ✓ Red/Yellow hotspots on lesion → Model is learning correctly")
    print(f"  ✗ Hotspots on corners/artifacts → Model is cheating")
    print(f"  ? Scattered heatmap → Model needs more training")
    print(f"{'='*80}\n")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    model, gradcam = load_model()
    samples = get_sample_images()
    
    print(f"\n✓ Loaded {len(samples)} sample images from 8 classes")
    
    visualize_gradcam(model, gradcam, samples)
    
    print("✓ Evaluation complete!")
    print("\nCheck gradcam_8class/ folder for visualizations")
    print("**Pay special attention to Acne samples!**\n")
