"""
Image Normalization for Skin Lesion Classification
Provides ImageNet and custom normalization strategies
"""
import numpy as np
from typing import Tuple, Literal

# ImageNet statistics (used for transfer learning)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    """
    Normalize image using ImageNet mean and std.
    
    This is crucial for transfer learning since pre-trained models
    (ResNet, MobileNet, EfficientNet) were trained on ImageNet-normalized data.
    
    Args:
        image: RGB image in range [0, 255] or [0, 1]
        
    Returns:
        Normalized image
    """
    # Ensure image is in [0, 1] range first
    if image.max() > 1.0:
        image = image / 255.0
    
    # Apply ImageNet normalization
    # Formula: (x - mean) / std (per channel)
    normalized = (image - IMAGENET_MEAN) / IMAGENET_STD
    
    return normalized.astype(np.float32)

def normalize_custom(image: np.ndarray, dataset_mean: np.ndarray = None, 
                     dataset_std: np.ndarray = None) -> np.ndarray:
    """
    Normalize using custom dataset statistics.
    
    Use this if you've computed HAM10000-specific mean/std values.
    Generally not necessary for transfer learning.
    
    Args:
        image: RGB image
        dataset_mean: Custom mean values (R, G, B)
        dataset_std: Custom std values (R, G, B)
        
    Returns:
        Normalized image
    """
    if image.max() > 1.0:
        image = image / 255.0
    
    if dataset_mean is None or dataset_std is None:
        # Default to simple [0,1] rescaling
        return image.astype(np.float32)
    
    normalized = (image - dataset_mean) / dataset_std
    return normalized.astype(np.float32)

def denormalize_imagenet(image: np.ndarray) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.
    
    Args:
        image: Normalized image
        
    Returns:
        Image in [0, 1] range
    """
    denormalized = (image * IMAGENET_STD) + IMAGENET_MEAN
    denormalized = np.clip(denormalized, 0, 1)
    return denormalized

def compute_dataset_statistics(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std for a dataset of images.
    
    Useful if you want to use custom normalization instead of ImageNet.
    
    Args:
        images: Array of images (N, H, W, 3) in range [0, 255] or [0, 1]
        
    Returns:
        Tuple of (mean, std) for each channel
    """
    if images.max() > 1.0:
        images = images / 255.0
    
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    
    return mean, std

def batch_normalize(images: np.ndarray, method: Literal['imagenet', 'custom'] = 'imagenet',
                    custom_mean: np.ndarray = None, custom_std: np.ndarray = None) -> np.ndarray:
    """
    Normalize a batch of images.
    
    Args:
        images: Batch of images (N, H, W, 3)
        method: 'imagenet' or 'custom'
        custom_mean: Mean for custom normalization
        custom_std: Std for custom normalization
        
    Returns:
        Normalized batch
    """
    if method == 'imagenet':
        return np.array([normalize_imagenet(img) for img in images])
    else:
        return np.array([normalize_custom(img, custom_mean, custom_std) for img in images])

def visualize_normalization(image: np.ndarray, save_path: str = None):
    """
    Visualize the effect of normalization.
    
   Args:
        image: Original RGB image [0, 255]
        save_path: Optional save path
    """
    import matplotlib.pyplot as plt
    
    normalized = normalize_imagenet(image)
    denormalized = denormalize_imagenet(normalized)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(image.astype(np.uint8) if image.max() > 1 else image)
    axes[0].set_title('Original [0, 255]')
    axes[0].axis('off')
    
    # Normalized (needs to be denormalized for display)
    axes[1].imshow(denormalized)
    axes[1].set_title('After ImageNet Normalization\n(denormalized for display)')
    axes[1].axis('off')
    
    # Show histogram comparison
    axes[2].hist(image.flatten(), bins=50, alpha=0.5, label='Original', color='blue')
    axes[2].hist((denormalized * 255).flatten(), bins=50, alpha=0.5, label='Normalized', color='red')
    axes[2].set_title('Pixel Distribution')
    axes[2].legend()
    axes[2].set_xlabel('Pixel Value')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Normalization visualization saved to {save_path}")
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    import os
    from PIL import Image
    
    print("=== Normalization Module Test ===")
    print(f"ImageNet Mean: {IMAGENET_MEAN}")
    print(f"ImageNet Std: {IMAGENET_STD}")
    
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'HAM10000_images_part_1')
    
    if os.path.exists(dataset_dir):
        sample_images = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')][:1]
        
        if sample_images:
            sample_path = os.path.join(dataset_dir, sample_images[0])
            img = Image.open(sample_path).convert('RGB')
            img_array = np.array(img)
            
            # Test normalization
            normalized = normalize_imagenet(img_array)
            denormalized = denormalize_imagenet(normalized)
            
            print(f"\n✓ Original shape: {img_array.shape}")
            print(f"✓ Original range: [{img_array.min()}, {img_array.max()}]")
            print(f"✓ Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
            print(f"✓ Denormalized range: [{denormalized.min():.3f}, {denormalized.max():.3f}]")
            
            # Visualize
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'normalization_test.png')
            
            visualize_normalization(img_array, save_path=output_path)
            print("\n✓ Normalization module ready!")
        else:
            print("No images found")
    else:
        print("Dataset not found. Module is ready for use.")
