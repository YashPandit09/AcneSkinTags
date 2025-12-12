"""
Lesion Segmentation for Dermoscopic Images
Centers and crops images to focus on the lesion region
"""
import cv2
import numpy as np
from typing import Tuple

def segment_lesion(image: np.ndarray, target_size: Tuple[int, int] = (224, 224), 
                   padding_percent: float = 0.1) -> np.ndarray:
    """
    Segment and crop the lesion region from dermoscopic image.
    
    Uses Otsu's thresholding to create a binary mask, then extracts bounding box
    with padding to focus on the lesion while removing excessive background.
    
    Args:
        image: Input RGB image (H, W, 3)
        target_size: Final output size after cropping and resizing
        padding_percent: Percentage padding around lesion bounding box (0.1 = 10%)
        
    Returns:
        Cropped and resized RGB image focused on lesion
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's thresholding to separate lesion from background
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # If no lesion detected, return resized original
        return cv2.resize(image, target_size)
    
    # Get largest contour (assumed to be the lesion)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding
    pad_x = int(w * padding_percent)
    pad_y = int(h * padding_percent)
    
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(image.shape[1] - x, w + 2 * pad_x)
    h = min(image.shape[0] - y, h + 2 * pad_y)
    
    # Crop lesion region
    cropped = image[y:y+h, x:x+w]
    
    # Resize to target size
    resized = cv2.resize(cropped, target_size)
    
    return resized

def batch_segment_lesions(images: np.ndarray, target_size: Tuple[int, int] = (224, 224),
                          verbose: bool = False) -> np.ndarray:
    """
    Apply lesion segmentation to a batch of images.
    
    Args:
        images: Batch of RGB images (N, H, W, 3)
        target_size: Output size
        verbose: Show progress
        
    Returns:
        Batch of segmented images
    """
    from tqdm import tqdm
    
    results = []
    iterator = tqdm(images, desc="Segmenting lesions") if verbose else images
    
    for img in iterator:
        processed = segment_lesion(img, target_size)
        results.append(processed)
    
    return np.array(results)

def visualize_segmentation(image: np.ndarray, save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Visualize lesion segmentation process.
    
    Args:
        image: Input RGB image
        save_path: Optional path to save visualization
        
    Returns:
        Tuple of (original, segmented) images
    """
    import matplotlib.pyplot as plt
    
    segmented = segment_lesion(image)
    
    # Also show the mask for educational purposes
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title('Segmentation Mask (Otsu)')
    axes[1].axis('off')
    
    axes[2].imshow(segmented)
    axes[2].set_title('Cropped & Resized Lesion')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Segmentation visualization saved to {save_path}")
    
    plt.close()
    
    return image, segmented

if __name__ == "__main__":
    import os
    from PIL import Image
    
    print("=== Lesion Segmentation Module Test ===")
    
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'HAM10000_images_part_1')
    
    if os.path.exists(dataset_dir):
        sample_images = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')][:3]
        
        if sample_images:
            print(f"Testing on {len(sample_images)} images...")
            
            for sample_file in sample_images:
                sample_path = os.path.join(dataset_dir, sample_file)
                img = Image.open(sample_path).convert('RGB')
                img_array = np.array(img)
                
                result = segment_lesion(img_array)
                print(f"✓ {sample_file}: {img_array.shape} → {result.shape}")
            
            # Visualize first one
            img = Image.open(os.path.join(dataset_dir, sample_images[0])).convert('RGB')
            img_array = np.array(img)
            
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'lesion_segmentation_test.png')
            
            visualize_segmentation(img_array, save_path=output_path)
            print("\n✓ Lesion segmentation module ready!")
        else:
            print("No images found")
    else:
        print("Dataset not found. Module is ready for use.")
