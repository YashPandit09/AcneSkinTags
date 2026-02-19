"""
Derm-X Inference Engine
Pure Python backend module — no Streamlit dependency.
Handles model loading, image preprocessing, prediction, and Grad-CAM generation.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, Dict, List

# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES: List[str] = [
    "Melanocytic nevi",                # 0: nv
    "Melanoma",                        # 1: mel
    "Benign keratosis-like lesions",   # 2: bkl
    "Basal cell carcinoma",            # 3: bcc
    "Actinic keratoses",               # 4: akiec
    "Vascular lesions",                # 5: vasc
    "Dermatofibroma",                  # 6: df
    "Acne",                            # 7: acne
]

DISEASE_INFO: Dict[str, Dict[str, str]] = {
    "Melanoma": {
        "description": "A serious form of skin cancer that develops in melanocytes.",
        "severity": "HIGH RISK — Requires immediate medical attention",
        "risk_level": "high",
    },
    "Basal cell carcinoma": {
        "description": "The most common type of skin cancer, usually slow-growing.",
        "severity": "MODERATE RISK — Consult a dermatologist soon",
        "risk_level": "moderate",
    },
    "Melanocytic nevi": {
        "description": "Common moles that are usually benign.",
        "severity": "LOW RISK — Monitor for changes",
        "risk_level": "low",
    },
    "Benign keratosis-like lesions": {
        "description": "Non-cancerous skin growths including seborrheic keratoses.",
        "severity": "LOW RISK — Generally harmless",
        "risk_level": "low",
    },
    "Actinic keratoses": {
        "description": "Pre-cancerous patches caused by long-term sun damage.",
        "severity": "MODERATE RISK — Preventive treatment recommended",
        "risk_level": "moderate",
    },
    "Vascular lesions": {
        "description": "Blood vessel abnormalities in the skin.",
        "severity": "LOW RISK — Usually benign",
        "risk_level": "low",
    },
    "Dermatofibroma": {
        "description": "A common benign skin nodule of fibrous tissue.",
        "severity": "LOW RISK — Harmless",
        "risk_level": "low",
    },
    "Acne": {
        "description": "A common skin condition caused by clogged pores and bacteria.",
        "severity": "LOW RISK — Treatable with proper skincare",
        "risk_level": "low",
    },
}

IMG_SIZE: Tuple[int, int] = (224, 224)

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    model_path: str = "best_model_8class_pytorch.pth",
) -> Tuple[nn.Module, torch.device]:
    """
    Load the trained MobileNetV2 8-class model.

    Args:
        model_path: Path to the .pth state dict file.

    Returns:
        Tuple of (model, device).

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If the state dict is incompatible.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.mobilenet_v2(weights="DEFAULT")
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, len(CLASS_NAMES)),
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, device


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image: Image.Image,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess a PIL image for inference.

    Args:
        image: RGB PIL Image.

    Returns:
        img_tensor: Batched tensor ready for the model (1, 3, 224, 224).
        img_resized: Numpy array (224, 224, 3) uint8 for visualisation.
    """
    img_resized = cv2.resize(np.array(image), IMG_SIZE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    img_tensor = transform(Image.fromarray(img_resized))
    img_batch = img_tensor.unsqueeze(0)

    return img_batch, img_resized


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model: nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Generate a Grad-CAM heatmap for a given image tensor.

    Uses ``model.features[-1]`` (last convolutional block of MobileNetV2)
    as the target layer.

    Args:
        model: The loaded MobileNetV2 model.
        img_tensor: Preprocessed image batch (1, 3, H, W).
        device: torch.device for computation.
        target_class: Class index to explain. Defaults to the predicted class.

    Returns:
        heatmap: Numpy array (H, W) normalised to [0, 1].
        output: Raw logits tensor (1, num_classes).
    """
    model.eval()
    img_tensor = img_tensor.to(device)

    gradients: list = []
    activations: list = []

    def _backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    def _forward_hook(_module, _input, output):
        activations.append(output)

    target_layer = model.features[-1]
    handle_fwd = target_layer.register_forward_hook(_forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(_backward_hook)

    try:
        output = model(img_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        grads = gradients[0][0].cpu().data.numpy()
        acts = activations[0][0].cpu().data.numpy()

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, IMG_SIZE)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

    finally:
        handle_fwd.remove()
        handle_bwd.remove()

    return cam, output


# ============================================================
# HEATMAP OVERLAY
# ============================================================

def overlay_heatmap(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        img: Original image (H, W, 3), uint8.
        heatmap: Grad-CAM heatmap (H, W), float in [0, 1].
        alpha: Opacity of the heatmap layer.

    Returns:
        Superimposed image (H, W, 3), uint8.
    """
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = heatmap_colored * alpha + img * (1 - alpha)
    return np.clip(superimposed, 0, 255).astype(np.uint8)


# ============================================================
# HIGH-LEVEL PREDICTION API
# ============================================================

def predict(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
) -> Dict:
    """
    Run full inference + Grad-CAM on a single PIL image.

    Args:
        model: Loaded PyTorch model.
        image: RGB PIL Image.
        device: torch.device.

    Returns:
        Dictionary with keys:
            - predicted_class (str)
            - predicted_index (int)
            - confidence (float)
            - probabilities (dict[str, float])
            - heatmap (np.ndarray or None)
            - img_resized (np.ndarray)
            - disease_info (dict)
    """
    img_batch, img_resized = preprocess_image(image)

    # ---- probabilities ----
    with torch.no_grad():
        output = model(img_batch.to(device))
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[pred_idx].item()

    pred_class = CLASS_NAMES[pred_idx]
    prob_dict = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}

    # ---- Grad-CAM ----
    heatmap = None
    try:
        heatmap, _ = generate_gradcam(model, img_batch, device)
    except Exception:
        pass  # gracefully degrade if Grad-CAM fails

    info = DISEASE_INFO.get(pred_class, {
        "description": "Unknown condition.",
        "severity": "UNKNOWN",
        "risk_level": "unknown",
    })

    return {
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": prob_dict,
        "heatmap": heatmap,
        "img_resized": img_resized,
        "disease_info": info,
    }


# ============================================================
# SELF-TEST (python inference_engine.py)
# ============================================================

if __name__ == "__main__":
    import os

    print("=" * 60)
    print("Inference Engine — Self Test")
    print("=" * 60)

    # 1. Check model file exists
    model_file = "best_model_8class_pytorch.pth"
    assert os.path.exists(model_file), f"Model file not found: {model_file}"

    # 2. Load model
    model, device = load_model(model_file)
    print(f"✓ Model loaded on {device}")
    print(f"✓ Classes: {len(CLASS_NAMES)}")

    # 3. Dry run with a synthetic image
    dummy = Image.new("RGB", (300, 300), color="gray")
    result = predict(model, dummy, device)

    print(f"✓ Prediction: {result['predicted_class']}")
    print(f"✓ Confidence: {result['confidence']:.2%}")
    print(f"✓ Heatmap generated: {result['heatmap'] is not None}")
    print(f"✓ Probabilities: {len(result['probabilities'])} classes")
    print("=" * 60)
    print("All checks passed — inference_engine.py is operational.")
