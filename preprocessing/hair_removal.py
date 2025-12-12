"""
Hair Removal Preprocessing for Dermoscopic Images
Uses morphological operations (Black Hat Transform) to detect and remove hair artifacts
"""
import cv2
import numpy as np
from typing import Tuple

def remove_hair(image: np.ndarray, kernel_size: int = 17) -> np.ndarray:
    """
    Remove hair from dermoscopic images using morphological black-hat transform.
    
    The Black-Hat transform detects dark structures (hair) on lighter backgrounds (skin).
    We then inpaint those detected regions to reconstruct the underlying skin texture.
    
    Args:
        image: Input RGB image as numpy array (H, W, 3)
        kernel_size: Size of morphological kernel (larger = detects thicker hair)
        
    Returns:
        Hair-removed RGB image
        
    References:
        - Lee, T., et al. "DullRazor: A software approach to hair removal from images."
        - Morphological image processing for artifact removal in medical imaging
    """
    # Convert to grayscale for hair detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. BLACK-HAT TRANSFORM
    # Creates a structuring element (cross-shaped for directional hair)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Black-hat: highlights dark structures (hair) on light background (skin)
    # Formula: BlackHat(img) = Closing(img) - img
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 2. THRESHOLD TO CREATE HAIR MASK
    # Adaptive thresholding works better than global threshold due to varying skin tones
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Optional: Dilate the mask slightly to ensure complete hair coverage
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hair_mask = cv2.dilate(hair_mask, dilate_kernel, iterations=1)
    
    # 3. INPAINTING
    # Reconstruct the hair regions using surrounding skin texture
    # INPAINT_TELEA: Fast Marching Method (better for thin structures like hair)
    result = cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return result

def batch_remove_hair(images: np.ndarray, kernel_size: int = 17, verbose: bool = False) -> np.ndarray:
    """
    Apply hair removal to a batch of images.
    
    Args:
        images: Batch of RGB images (N, H, W, 3)
        kernel_size: Morphological kernel size
        verbose: Print progress
        
    Returns:
        Batch of hair-removed images
    """
    from tqdm import tqdm
    
    results = []
    iterator = tqdm(images, desc="Removing hair") if verbose else images
    
    for img in iterator:
        processed = remove_hair(img, kernel_size)
        results.append(processed)
    
    return np.array(results)

def visualize_hair_removal(image: np.ndarray, save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Visualize before/after hair removal for quality assessment.
    
    Args:
        image: Input RGB image
        save_path: Optional path to save comparison image
        
    Returns:
        Tuple of (original, processed) images
    """
    import matplotlib.pyplot as plt
    
    processed = remove_hair(image)
    
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].imshow(image)
    axes[0].set_title('Original (With Hair)')
    axes[0].axis('off')
    
    axes[1].imshow(processed)
    axes[1].set_title('Processed (Hair Removed)')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")
    
    plt.close()
    
    return image, processed

# Test function
if __name__ == "__main__":
    import os
    from PIL import Image
    
    print("=== Hair Removal Module Test ===")
    
    # Try to load a sample image from the dataset
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'HAM10000_images_part_1')
    
    if os.path.exists(dataset_dir):
        # Get first image
        sample_images = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
        if sample_images:
            sample_path = os.path.join(dataset_dir, sample_images[0])
            print(f"Testing on: {sample_images[0]}")
            
            # Load image
            img = Image.open(sample_path).convert('RGB')
            img_array = np.array(img)
            
            # Process
            result = remove_hair(img_array)
            
            # Save comparison
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'hair_removal_test.png')
            
            visualize_hair_removal(img_array, save_path=output_path)
            
            print(f"✓ Hair removal successful!")
            print(f"✓ Input shape: {img_array.shape}")
            print(f"✓ Output shape: {result.shape}")
        else:
            print("No images found in dataset directory")
    else:
        print(f"Dataset directory not found: {dataset_dir}")
        print("This is normal if running standalone. The module is ready to use.")
