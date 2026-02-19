# Derm-X: Explainable Deep Learning for Skin Lesion Detection

> A publication-ready deep learning system for 8-class skin lesion classification from dermoscopic images, with clinical explainability through Grad-CAM visualizations. Trained and optimized on consumer-grade GPU hardware.

---

## Overview

Derm-X classifies dermoscopic skin images into **8 categories** — 7 from the HAM10000 benchmark dataset and 1 (Acne) from DermNet — using a fine-tuned **MobileNetV2** architecture. The project prioritizes three goals:

1.  **Diagnostic Accuracy**: Achieving competitive accuracy on a highly imbalanced, multi-class medical imaging task.
2.  **Clinical Transparency**: Using Grad-CAM heatmaps to provide visual evidence that the model's decisions are based on pathological features, not background artifacts.
3.  **Accessibility**: Optimizing the entire training and inference pipeline to run efficiently on consumer-grade laptop GPUs.

### Supported Classes
| # | Class | Source | Clinical Significance |
|---|-------|--------|----------------------|
| 1 | Melanocytic nevi (nv) | HAM10000 | Benign |
| 2 | Melanoma (mel) | HAM10000 | **Malignant (Critical)** |
| 3 | Benign keratosis (bkl) | HAM10000 | Benign |
| 4 | Basal cell carcinoma (bcc) | HAM10000 | Malignant |
| 5 | Actinic keratoses (akiec) | HAM10000 | Pre-cancerous |
| 6 | Vascular lesions (vasc) | HAM10000 | Benign |
| 7 | Dermatofibroma (df) | HAM10000 | Benign |
| 8 | **Acne** | DermNet | Common Condition |

---

## Results & Metrics

The final 8-class model was evaluated on a held-out test set of **10,327 images**.

### Key Outcomes

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **81.19%** across 8 highly imbalanced classes |
| **Acne Detection Precision** | **99.36%** |
| **Acne Detection Recall** | **99.68%** |
| **Weighted Avg F1-Score** | **0.8004** |

- **Cross-Dataset Viability**: The Acne class, sourced entirely from DermNet, achieved near-perfect precision and recall. This validates the approach of merging external datasets with standard cancer benchmarks like HAM10000.
- **Clinical Transparency**: Grad-CAM heatmaps were generated and manually reviewed. The model consistently focuses on pathological lesion features (borders, texture, pigmentation) rather than background skin or hair artifacts.

### Per-Class Classification Report

```
                               precision    recall  f1-score   support

             Melanocytic nevi     0.8610    0.9509    0.9038      6705
                     Melanoma     0.5711    0.4223    0.4855      1113
Benign keratosis-like lesions     0.6266    0.5787    0.6017      1099
         Basal cell carcinoma     0.8065    0.5759    0.6720       514
            Actinic keratoses     0.6680    0.5291    0.5904       327
             Vascular lesions     0.9080    0.5563    0.6900       142
               Dermatofibroma     0.7414    0.3739    0.4971       115
                         Acne     0.9936    0.9968    0.9952       312

                     accuracy                         0.8119     10327
                    macro avg     0.7720    0.6230    0.6795     10327
                 weighted avg     0.7993    0.8119    0.8004     10327
```

---

## Hardware & Performance Optimization

The entire pipeline was engineered to train effectively on **consumer-grade laptop hardware**, removing the barrier of expensive cloud GPU instances.

| Parameter | Value |
|-----------|-------|
| **GPU** | NVIDIA RTX 3050 Laptop GPU |
| **Training VRAM Footprint** | ~0.7 GB |
| **Training Speedup vs. CPU** | **~15x** |
| **Mixed Precision (AMP)** | Enabled (`torch.amp`) |
| **Batch Size** | 64 |
| **TF32 Acceleration** | Enabled |

### Optimizations Applied
- **Automatic Mixed Precision (AMP)**: Leveraging `torch.amp.autocast` and `GradScaler` for FP16/FP32 mixed training, achieving significant speedup with negligible accuracy impact.
- **Optimal Batch Sizing**: Batch size of 64 was selected to maximize GPU utilization on the RTX 3050 without triggering out-of-memory errors.
- **TF32 Math**: Enabled `torch.backends.cuda.matmul.allow_tf32` for faster matrix operations on Ampere-architecture GPUs.
- **Pinned Memory**: DataLoaders use `pin_memory=True` for faster host-to-device data transfer.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | PyTorch (Primary) |
| **Architecture** | MobileNetV2 (ImageNet pre-trained) |
| **Training Strategy** | Transfer Learning + Fine-Tuning |
| **Explainability** | Grad-CAM (custom PyTorch implementation) |
| **Deployment** | Streamlit |
| **Preprocessing** | OpenCV (hair removal, segmentation) |
| **Data Handling** | Pandas, scikit-learn |

---

## Project Structure

```
Project/
├── app_pytorch.py                  # Streamlit demo app (inference + Grad-CAM)
├── train_pytorch_8class.py         # Main training script (8-class, PyTorch)
├── evaluate_8class.py              # Evaluation & Grad-CAM generation
├── config.py                       # Central configuration (paths, hyperparams)
├── data_loader_enhanced.py         # TensorFlow data loader (legacy)
├── data_loader.py                  # Basic data loader (legacy)
├── cleanup_for_deployment.py       # Utility to prune unnecessary files
│
├── best_model_8class_pytorch.pth   # Trained model weights (MobileNetV2)
├── classification_report_8class.txt
├── confusion_matrix_8class.png
├── training_history_8class_pytorch.png
├── per_class_metrics.csv
│
├── preprocessing/                  # Image preprocessing modules
│   ├── hair_removal.py             # Black-Hat transform + inpainting
│   ├── lesion_segmentation.py      # Otsu thresholding + bounding box crop
│   └── normalization.py            # ImageNet standardization
│
├── explainability/                 # Grad-CAM module
│   └── gradcam.py                  # TensorFlow Grad-CAM (legacy)
│
├── docs/                           # Research documentation
│   ├── FINAL_RESULTS_8CLASS.md
│   ├── PAPER_METHODOLOGY.md
│   ├── PYTORCH_GPU_GUIDE.md
│   └── RESEARCH_PAPER_KIT.md
│
├── gradcam_8class/                 # Generated Grad-CAM visualizations
├── Dataset/                        # Training data (HAM10000 + DermNet)
├── requirements.txt
└── README.md
```

---

## Workflow

### Primary Pipeline (PyTorch)

The finalized, primary framework for this project is **PyTorch**. All active training, evaluation, and deployment scripts use PyTorch.

#### 1. Data Loading & Class Balancing
The training script (`train_pytorch_8class.py`) defines a custom `EnhancedSkinLesionDataset` that:
- Loads HAM10000 metadata (7 cancer/lesion classes) from the CSV file.
- Loads Acne images from the DermNet directory as the 8th class.
- **Handles class imbalance dynamically** via oversampling: minority classes are upsampled using `df.sample(target_count, replace=True)` to ensure balanced representation during training.

#### 2. Training
```bash
# Default: 20 epochs, batch size 64, AMP enabled
python train_pytorch_8class.py

# Custom configuration
python train_pytorch_8class.py --epochs 30 --batch-size 32 --lr 0.0008
```

#### 3. Evaluation & Explainability
```bash
# Generate Grad-CAM visualizations and classification report
python evaluate_8class.py
```

#### 4. Deployment
```bash
# Launch the Streamlit demo application
streamlit run app_pytorch.py
```
The Streamlit app allows users to upload a skin lesion image, receive an 8-class prediction with confidence scores, and view the Grad-CAM heatmap showing the model's focus areas.

---

## Preprocessing Pipeline

The `preprocessing/` module provides framework-agnostic image processing:

| Step | Technique | Purpose | File |
|------|-----------|---------|------|
| **Hair Removal** | Morphological Black-Hat Transform + Inpainting | Remove hair artifacts that obscure lesion boundaries | `hair_removal.py` |
| **Lesion Segmentation** | Otsu Thresholding + Bounding Box Extraction | Center and crop the image on the lesion region | `lesion_segmentation.py` |
| **Normalization** | ImageNet Standardization (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]) | Match pre-trained model input expectations | `normalization.py` |

---

## Future Recommendations

- **Archive Legacy TensorFlow Scripts**: `data_loader_enhanced.py`, `data_loader.py`, and `explainability/gradcam.py` are TensorFlow-based and no longer part of the active pipeline. Archiving them will reduce technical debt and avoid confusion.
- **Improve Minority Class Recall**: Melanoma recall (42.23%) and Dermatofibroma recall (37.39%) are areas for improvement. Techniques such as focal loss, more aggressive augmentation, or curriculum learning could help.
- **K-Fold Cross-Validation**: Replace the single train/val/test split with K-Fold CV for more robust performance estimates.
- **Ensemble Methods**: Combining predictions from multiple architectures (MobileNetV2 + EfficientNet-B3) could boost overall accuracy.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA support (recommended)

### Setup
```bash
git clone https://github.com/YashPandit09/AcneSkinTags.git
cd Project

pip install -r requirements.txt

# Verify configuration
python config.py
```

---

## Citation

```bibtex
@misc{dermx2025,
  author = {Yash Pandit},
  title = {Derm-X: Explainable Deep Learning for Skin Lesion Detection},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YashPandit09/AcneSkinTags}
}
```

**Dataset Citation**:
```bibtex
@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific data},
  volume={5},
  pages={180161},
  year={2018}
}
```

---

## Contact

**Author**: Yash Pandit
**GitHub**: [YashPandit09](https://github.com/YashPandit09)
**Project**: [AcneSkinTags](https://github.com/YashPandit09/AcneSkinTags)

---

## Acknowledgments

- HAM10000 dataset — International Skin Imaging Collaboration (ISIC)
- DermNet dataset — Acne and Rosacea image collection
- PyTorch and torchvision for the deep learning framework
- MobileNetV2 pre-trained weights from ImageNet
