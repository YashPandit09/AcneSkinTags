# 🚀 HOW TO USE DERM-X - QUICK START GUIDE

## ⚡ 1-Minute Quick Start

```bash
# Step 1: Test everything works
python test_local.py

# Step 2: Launch the web app
streamlit run app.py
```

That's it! The app will open in your browser at `http://localhost:8501`

---

## 📖 Complete Usage Guide (5 Minutes)

### Option A: Just Want to Use the App (No Training)

If you just want to see the interface without training:

```bash
# 1. Launch the app
streamlit run app.py

# 2. In the app:
#    - Select a model path (or it will prompt you)
#    - Upload any skin image
#    - Click "Analyze Image"
#    - See prediction (it will be random if no trained model)
```

**Note**: Without a trained model, predictions won't be accurate. See Option B to train first.

---

### Option B: Full Workflow (Train → Test → Deploy)

#### Step 1: Verify Setup (30 seconds)

```bash
python test_local.py
```

**Expected**: "✓✓✓ ALL TESTS PASSED!"

---

#### Step 2: Train a Model (30-60 minutes)

**Quick Test Training** (5 epochs, ~10 minutes):
```bash
python train_comprehensive.py --model mobilenetv2 --epochs 5
```

**Full Training** (20 epochs, ~45 minutes):
```bash
python train_comprehensive.py --model mobilenetv2 --epochs 20
```

**What happens**:
- Training progress shown in terminal
- Model saved to: `results/mobilenetv2_YYYYMMDD_HHMMSS/best_model.keras`
- Plots saved: `results/.../training_history.png`

---

#### Step 3: Evaluate the Model (2 minutes)

```bash
# Replace with your actual model path
python evaluate_model.py --model-path results/mobilenetv2_20251212_130000/best_model.keras
```

**What you get**:
- `evaluation/classification_report.json`
- `evaluation/confusion_matrix.png` ← For your report
- `evaluation/roc_curves.png` ← For your report
- Melanoma recall check (must be >80%)

---

#### Step 4: Generate Grad-CAM Examples (1 minute)

```bash
# Single image
python explainability/gradcam.py \
    --model-path results/mobilenetv2_20251212_130000/best_model.keras \
    --image-path Dataset/HAM10000_images_part_1/ISIC_0024306.jpg

# Batch mode (20 random images)
python explainability/gradcam.py \
    --model-path results/mobilenetv2_20251212_130000/best_model.keras \
    --batch --num-images 20
```

**What you get**:
- `visualizations/gradcam/gradcam_001_*.png` ← For your report

---

#### Step 5: Launch Web App (Instant)

```bash
streamlit run app.py
```

**If model not found**, update the sidebar in the app to point to your trained model:
- Edit the path in the app's sidebar text input
- Or copy your model: `cp results/.../best_model.keras saved_models/mobilenetv2_best.keras`

---

#### Step 6: Test the App

1. **Upload an Image**:
   - From dataset: `Dataset/HAM10000_images_part_1/ISIC_0024306.jpg`
   - Or download from Google: "skin lesion dermoscopy"

2. **Click "Analyze Image"**

3. **View Results**:
   - Prediction (e.g., "Melanoma")
   - Confidence (e.g., "94.2%")
   - Grad-CAM heatmap (red = AI focus areas)
   - Probability chart

4. **Take Screenshots** (for your report):
   - Main prediction screen
   - Grad-CAM visualization
   - Probability distribution

---

#### Step 7: Prepare Submission

```bash
python prepare_submission.py
```

**What it creates**:
- `Derm-X_Submission_YYYYMMDD_HHMMSS/` folder
- `Derm-X_Submission_YYYYMMDD_HHMMSS.zip` file

**Then**:
1. Convert report template to Word:
   - Open `Derm-X_Final_Descriptive_Report_Template.md`
   - Copy to Word or use: `pandoc Derm-X_Final_Descriptive_Report_Template.md -o Report.docx`

2. Fill in results (from Step 3):
   - Section 5.1: Copy metrics from `evaluation/classification_report.txt`
   - Section 5.3: Insert `evaluation/confusion_matrix.png`
   - Section 5.4: Insert `evaluation/roc_curves.png`
   - Section 5.5: Insert Grad-CAM images

3. Add report to submission folder

4. Submit the ZIP file!

---

## 🎯 Common Commands Cheat Sheet

```bash
# TEST EVERYTHING
python test_local.py

# TRAIN (pick one)
python train_comprehensive.py --model mobilenetv2 --epochs 5      # Quick
python train_comprehensive.py --model mobilenetv2 --epochs 20     # Full
python train_comprehensive.py --model efficientnet-b3 --epochs 25 # Best accuracy

# EVALUATE
python evaluate_model.py --model-path results/.../best_model.keras

# GRAD-CAM
python explainability/gradcam.py --model-path results/.../best_model.keras --batch

# RUN WEB APP
streamlit run app.py

# CREATE SUBMISSION
python prepare_submission.py
```

---

## 🔧 Troubleshooting

### "No module named 'streamlit'"
```bash
pip install streamlit
```

### "Model not found"
**Option 1**: Train a model first (see Step 2 above)

**Option 2**: Update model path in app.py sidebar

### "Can't find Dataset"
Make sure you have:
```
Project/
├── Dataset/
│   ├── HAM10000_images_part_1/  (5000 images)
│   ├── HAM10000_images_part_2/  (5015 images)
│   └── HAM10000_metadata.csv
```

### App shows error on upload
Make sure you trained a model and updated the path in the sidebar

---

## 💡 Best Workflow for Your Report

1. **Train MobileNetV2** (fastest): `python train_comprehensive.py --model mobilenetv2 --epochs 20`

2. **Evaluate**: `python evaluate_model.py --model-path results/.../best_model.keras`

3. **Generate Grad-CAM**: `python explainability/gradcam.py --model-path ... --batch`

4. **Screenshot the app**: Launch `streamlit run app.py` and capture 3-4 screenshots

5. **Fill report**: Use outputs from step 2-3

6. **Package**: `python prepare_submission.py`

**Total time**: ~1 hour (mostly training)

---

## 📊 What Files to Include in Report

From your outputs:
- ✅ `evaluation/confusion_matrix.png` → Section 5.3
- ✅ `evaluation/roc_curves.png` → Section 5.4
- ✅ `visualizations/gradcam/gradcam_*.png` → Section 5.5 (pick 3-4 best examples)
- ✅ Screenshots of Streamlit app → Appendix D

---

## 🎓 For Presentation/Demo

1. Open terminal, run: `streamlit run app.py`
2. Browser opens automatically
3. Upload a lesion image
4. Show the prediction + Grad-CAM
5. Explain: "Red areas show where AI focused to make the decision"

**Impressive feature**: Change the model dropdown and show predictions from different architectures!

---

## ⚡ Absolute Fastest Path (If Time is Short)

```bash
# 1. Quick training (10 min)
python train_comprehensive.py --model mobilenetv2 --epochs 5

# 2. Evaluate (1 min)
python evaluate_model.py --model-path results/mobilenetv2_*/best_model.keras

# 3. App (instant)
streamlit run app.py

# 4. Take screenshots, done!
```

---

**Questions? Check:**
- Full details: `USAGE_GUIDE.md`
- Project overview: `README.md`
- Phase 7 guide: Review the artifact I created

**You're all set! 🚀**
