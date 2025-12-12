# Derm-X: Explainable Deep Learning for Skin Lesion Detection

> **Research Project**: Comparative analysis of CNN architectures (ResNet50, MobileNetV2, EfficientNet) with explainable AI (Grad-CAM) for automated skin lesion classification.

---

## 🎯 Project Overview

This project implements a publication-ready deep learning system for classifying skin lesions from dermoscopic images using the HAM10000 dataset (7 disease classes, 10,000+ images). The system compares multiple state-of-the-art architectures and provides medical explainability through Grad-CAM visualizations.

### Key Features
- ✅ **Advanced Preprocessing**: Hair removal, lesion segmentation, ImageNet normalization
- ✅ **Multi-Architecture Support**: ResNet50, MobileNetV2, EfficientNet (B0/B3/B7)
- ✅ **Class Imbalance Handling**: Automatic class weight computation
- ✅ **Transfer Learning**: Pre-trained ImageNet weights + fine-tuning
- ✅ **Comprehensive Metrics**: Accuracy, AUC-ROC, Precision, Recall (critical for medical AI)
- 🚧 **Explainability**: Grad-CAM heatmaps (in progress)
- 🚧 **Deployment**: Streamlit demo app (planned)

---

## 📊 Dataset

**HAM10000** (Human Against Machine with 10000 training images)
- **Total Images**: 10,015
- **Classes**: 7 skin lesion types
- **Source**: International Skin Imaging Collaboration (ISIC)

### Class Distribution
| Class | Count | Percentage | Medical Importance |
|-------|-------|------------|-------------------|
| Melanocytic nevi (nv) | 6,705 | 67% | Benign |
| **Melanoma (mel)** | 1,113 | 11% | **CRITICAL** (malignant) |
| Benign keratosis (bkl) | 1,099 | 11% | Benign |
| Basal cell carcinoma (bcc) | 514 | 5% | Malignant |
| Actinic keratoses (akiec) | 327 | 3% | Pre-cancerous |
| Vascular lesions (vasc) | 142 | 1% | Benign |
| Dermatofibroma (df) | 115 | 1% | Benign |

> **Note**: Class imbalance is handled via automatic class weight computation. Melanoma detection (recall) is prioritized due to medical criticality.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.13+
- CUDA-compatible GPU (recommended for EfficientNet-B7)

### Setup
```bash
# Clone repository
git clone https://github.com/YashPandit09/AcneSkinTags.git
cd Project

# Install dependencies
pip install -r requirements.txt

# Verify installation
python config.py
```

---

## 🚀 Quick Start

### 1. Train a Model (Basic)
```bash
# Train MobileNetV2 (fastest for testing)
python train_comprehensive.py --model mobilenetv2 --epochs 5

# Train EfficientNet-B3 (best accuracy/speed tradeoff)
python train_comprehensive.py --model efficientnet-b3 --epochs 25
```

### 2. Train with Preprocessing
```bash
# Enable hair removal
python train_comprehensive.py --model efficientnet-b3 --hair-removal --epochs 30

# Enable both hair removal and lesion segmentation
python train_comprehensive.py --model resnet50 --hair-removal --segmentation --epochs 25
```

### 3. Two-Phase Training (Feature Extraction + Fine-Tuning)
```bash
python train_comprehensive.py --model efficientnet-b3 \
    --fine-tune \
    --fine-tune-epoch 10 \
    --epochs 25 \
    --hair-removal
```

---

## 📁 Project Structure

```
Project/
├── Dataset/                         # HAM10000 dataset
│   ├── HAM10000_images_part_1/      # 5,000 images
│   ├── HAM10000_images_part_2/      # 5,015 images
│   └── HAM10000_metadata.csv        # Labels and metadata
│
├── preprocessing/                   # Image preprocessing modules
│   ├── hair_removal.py              # Black-hat transform + inpainting
│   ├── lesion_segmentation.py       # Otsu thresholding + cropping
│   └── normalization.py             # ImageNet normalization
│
├── models/                          # Architecture definitions
│   └── model_factory.py             # Unified interface for all models
│
├── explainability/                  # Grad-CAM (planned)
├── evaluation/                      # Metrics and analysis (planned)
├── visualizations/                  # Generated plots
├── saved_models/                    # Trained model checkpoints
└── results/                         # Training logs and metrics
│
├── config.py                        # Central configuration
├── data_loader_enhanced.py          # Advanced data pipeline
├── train_comprehensive.py           # Main training script
└── requirements.txt                 # Dependencies
```

---

## 🧪 Training Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--model` | `resnet50`, `mobilenetv2`, `efficientnet-b0/b3/b7` | `mobilenetv2` | Architecture to train |
| `--epochs` | Integer | 25 | Number of training epochs |
| `--batch-size` | Integer | 32 | Batch size |
| `--hair-removal` | Flag | False | Enable hair removal preprocessing |
| `--segmentation` | Flag | False | Enable lesion segmentation |
| `--fine-tune` | Flag | False | Enable two-phase training |
| `--fine-tune-epoch` | Integer | 10 | Epoch to start fine-tuning |

---

## 📈 Model Comparison

| Model | Parameters | Training Time* | Inference Speed* | Best For |
|-------|-----------|---------------|-----------------|----------|
| **ResNet50** | 24M | ~2 hours | 45 ms | Baseline comparison |
| **MobileNetV2** | 3.5M | ~1 hour | 12 ms | Edge deployment, mobile apps |
| **EfficientNet-B0** | 5M | ~1.5 hours | 20 ms | Lightweight efficiency |
| **EfficientNet-B3** | 12M | ~3 hours | 35 ms | **Recommended** (best tradeoff) |
| **EfficientNet-B7** | 66M | ~6 hours | 120 ms | Maximum accuracy (GPU required) |

*Estimated on T4 GPU with batch size 32

---

## 🔬 Preprocessing Pipeline

### 1. Hair Removal
**Technique**: Morphological Black-Hat Transform  
**Why**: Hair artifacts obscure lesion boundaries and confuse CNNs  
**Implementation**: `preprocessing/hair_removal.py`

### 2. Lesion Segmentation  
**Technique**: Otsu Thresholding + Bounding Box Extraction  
**Why**: Centers image on lesion, removes background noise  
**Implementation**: `preprocessing/lesion_segmentation.py`

### 3. Normalization  
**Technique**: ImageNet Standardization  
**Why**: Pre-trained models expect ImageNet-normalized inputs  
**Values**: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]

---

## 📊 Output Structure

Each training run creates a timestamped directory:

```
results/efficientnet-b3_20251212_120000/
├── best_model.keras               # Best validation model
├── final_model.keras              # Final epoch model
├── training_config.json           # Hyperparameters used
├── training_log.csv               # Epoch-by-epoch metrics
├── training_history.png           # 4-panel plot (acc, loss, AUC, precision/recall)
└── test_results.json              # Final test set evaluation
```

---

## 🎓 For Research Paper

### Recommended Experiments

#### Experiment 1: Architecture Comparison
```bash
python train_comprehensive.py --model resnet50 --epochs 25
python train_comprehensive.py --model mobilenetv2 --epochs 25
python train_comprehensive.py --model efficientnet-b3 --epochs 30
```

#### Experiment 2: Ablation Study (Preprocessing Impact)
```bash
# Baseline (no preprocessing)
python train_comprehensive.py --model efficientnet-b3 --epochs 30

# With hair removal
python train_comprehensive.py --model efficientnet-b3 --hair-removal --epochs 30

# With hair removal + segmentation
python train_comprehensive.py --model efficientnet-b3 --hair-removal --segmentation --epochs 30
```

#### Experiment 3: Fine-Tuning Impact
```bash
# Feature extraction only (frozen base)
python train_comprehensive.py --model efficientnet-b3 --epochs 25

# With fine-tuning
python train_comprehensive.py --model efficientnet-b3 --fine-tune --fine-tune-epoch 10 --epochs 25
```

### Key Metrics for Medical AI
- **Accuracy**: Overall correctness
- **AUC-ROC**: Discriminative power across all thresholds
- **Precision**: Positive predictive value (low false positives)
- **Recall (Sensitivity)**: True positive rate (**critical for melanoma detection**)
- **F1-Score**: Harmonic mean of precision and recall

---

## 🔮 Next Steps (Roadmap)

- [ ] **Grad-CAM Implementation** - Heatmap visualization for explainability
- [ ] **Confusion Matrix Analysis** - Per-class performance breakdown
- [ ] **ROC Curves** - One-vs-rest for each class
- [ ] **Model Comparison Table** - Speed, accuracy, parameter count
- [ ] **Streamlit Demo App** - Upload image → predict + explain
- [ ] **K-Fold Cross-Validation** - More robust evaluation

---

## 📝 Citation

If you use this code for your research, please cite:

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

## 📧 Contact

**Author**: Yash Pandit  
**GitHub**: [YashPandit09](https://github.com/YashPandit09)  
**Project Link**: [AcneSkinTags](https://github.com/YashPandit09/AcneSkinTags)

---

## 🙏 Acknowledgments

- HAM10000 dataset from International Skin Imaging Collaboration (ISIC)
- TensorFlow/Keras for deep learning framework
- EfficientNet, ResNet, MobileNet pre-trained models

---

**Status**: Phase 1 & 2 Complete ✅ | Ready for Training Runs 🚀
