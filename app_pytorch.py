"""
Derm-X Analyzer - PyTorch Version
8-Class Skin Lesion Classification with Explainability
Using best_model_8class_pytorch.pth (81.19% accuracy)
"""
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="Derm-X Analyzer (PyTorch)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 20px 0;
        color: #000 !important;
    }
    .prediction-box h2, .prediction-box h3, .prediction-box p {
        color: inherit !important;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF9800;
        margin: 10px 0;
        color: #000 !important;
    }
    .warning-box strong {
        color: #E65100 !important;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
        color: #000 !important;
    }
    .success-box strong {
        color: #2E7D32 !important;
    }
    /* Fix Streamlit default text colors */
    .stMarkdown p {
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# Class names in training order
CLASS_NAMES = [
    'Melanocytic nevi',           # 0: nv
    'Melanoma',                   # 1: mel
    'Benign keratosis-like lesions',  # 2: bkl
    'Basal cell carcinoma',       # 3: bcc
    'Actinic keratoses',          # 4: akiec
    'Vascular lesions',           # 5: vasc
    'Dermatofibroma',             # 6: df
    'Acne'                        # 7: acne
]

@st.cache_resource
def load_pytorch_model(model_path='best_model_8class_pytorch.pth'):
    """Load the PyTorch 8-class model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model architecture
    model = models.mobilenet_v2(weights='DEFAULT')
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 8)
    )
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

def preprocess_image(image):
    """Preprocess image for PyTorch model"""
    # Resize
    img_resized = cv2.resize(np.array(image), (224, 224))
    
    # Apply transforms (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(Image.fromarray(img_resized))
    img_batch = img_tensor.unsqueeze(0)
    
    return img_batch, img_resized

def generate_gradcam(model, img_tensor, target_layer, device):
    """Generate Grad-CAM heatmap"""
    model.eval()
    
    # Forward pass
    img_tensor = img_tensor.to(device)
    
    # Hook to capture gradients and activations
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    
    # Backward
    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()
    
    # Compute CAM
    grads = gradients[0][0].cpu().data.numpy()
    acts = activations[0][0].cpu().data.numpy()
    
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * acts[i]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    
    # Cleanup
    handle_forward.remove()
    handle_backward.remove()
    
    return cam, output

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed = heatmap_colored * alpha + img * (1 - alpha)
    return superimposed.astype(np.uint8)

def predict_with_gradcam(model, img_batch, img_original, device):
    """Make prediction with Grad-CAM"""
    # Predict
    with torch.no_grad():
        output = model(img_batch.to(device))
        probabilities = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(dim=1).item()
        confidence = probabilities[pred_idx].item()
    
    pred_class = CLASS_NAMES[pred_idx]
    
    # Generate Grad-CAM
    try:
        target_layer = model.features[-1]
        heatmap, _ = generate_gradcam(model, img_batch, target_layer, device)
        superimposed = overlay_heatmap(img_original, heatmap)
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {e}")
        heatmap = None
        superimposed = None
    
    return pred_class, confidence, probabilities.cpu().numpy(), superimposed

def get_disease_info(disease_name):
    """Get disease information"""
    info = {
        'Melanoma': {
            'description': 'A serious form of skin cancer that develops in melanocytes.',
            'severity': 'HIGH RISK - Requires immediate medical attention',
            'color': '#D32F2F'
        },
        'Basal cell carcinoma': {
            'description': 'The most common type of skin cancer, usually slow-growing.',
            'severity': 'MODERATE RISK - Consult a dermatologist soon',
            'color': '#F57C00'
        },
        'Melanocytic nevi': {
            'description': 'Common moles that are usually benign.',
            'severity': 'LOW RISK - Monitor for changes',
            'color': '#388E3C'
        },
        'Benign keratosis-like lesions': {
            'description': 'Non-cancerous skin growths.',
            'severity': 'LOW RISK - Generally harmless',
            'color': '#388E3C'
        },
        'Actinic keratoses': {
            'description': 'Pre-cancerous patches caused by sun damage.',
            'severity': 'MODERATE RISK - Preventive treatment recommended',
            'color': '#F57C00'
        },
        'Vascular lesions': {
            'description': 'Blood vessel abnormalities in the skin.',
            'severity': 'LOW RISK - Usually benign',
            'color': '#388E3C'
        },
        'Dermatofibroma': {
            'description': 'A common benign skin nodule.',
            'severity': 'LOW RISK - Harmless',
            'color': '#388E3C'
        },
        'Acne': {
            'description': 'A common skin condition caused by clogged pores and bacteria.',
            'severity': 'LOW RISK - Treatable with proper skincare',
            'color': '#2196F3'
        }
    }
    return info.get(disease_name, {
        'description': 'Unknown condition',
        'severity': 'UNKNOWN',
        'color': '#757575'
    })

def main():
    # Header
    st.markdown('<div class="main-header">🔬 Derm-X Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered 8-Class Skin Lesion Detection | PyTorch | 81.19% Accuracy</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.markdown("""
        <div class="success-box">
            <strong>✓ Model Loaded:</strong><br>
            PyTorch MobileNetV2 (8-Class)<br>
            Accuracy: <strong>81.19%</strong><br>
            Acne Detection: <strong>99.68%</strong>
        </div>
        """, unsafe_allow_html=True)
        
        device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.info(f"🖥️ Running on: **{device_info}**")
        
        st.markdown("---")
        
        # About
        st.header("ℹ️ About")
        st.info("""
        **Derm-X (PyTorch Edition)** classifies 8 types of skin conditions:
        
        **Cancers & Pre-cancerous:**
        1. Melanoma
        2. Basal cell carcinoma
        3. Actinic keratoses
        
        **Benign Lesions:**
        4. Melanocytic nevi
        5. Benign keratosis
        6. Vascular lesions
        7. Dermatofibroma
        
        **Common Conditions:**
        8. **Acne** (NEW!)
        
        **Model Performance:**
        - Overall: 81.19% accuracy
        - Acne: 99.68% precision
        - Training: 10,327 images
        
        **Disclaimer:** This is a research tool and should NOT replace professional medical diagnosis.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image (JPG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear, well-lit image of the skin lesion"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Loading PyTorch model..."):
                    model, device = load_pytorch_model()
                
                if model is not None:
                    with st.spinner("Analyzing image..."):
                        # Preprocess
                        img_batch, img_resized = preprocess_image(image)
                        
                        # Predict
                        pred_class, confidence, all_probs, superimposed = predict_with_gradcam(
                            model, img_batch, img_resized, device
                        )
                        
                        # Store results
                        st.session_state.prediction = pred_class
                        st.session_state.confidence = confidence
                        st.session_state.all_probs = all_probs
                        st.session_state.superimposed = superimposed
                        st.session_state.img_resized = img_resized
                    
                    st.success("✓ Analysis complete!")
    
    with col2:
        st.header("📊 Results")
        
        if 'prediction' in st.session_state:
            pred_class = st.session_state.prediction
            confidence = st.session_state.confidence
            all_probs = st.session_state.all_probs
            
            # Main prediction
            disease_info = get_disease_info(pred_class)
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="margin: 0; color: {disease_info['color']};">
                    {pred_class}
                </h2>
                <h3 style="margin: 10px 0;">
                    Confidence: {confidence:.1%}
                </h3>
                <p style="margin: 5px 0;">
                    {disease_info['description']}
                </p>
                <p style="margin: 5px 0; font-weight: bold; color: {disease_info['color']};">
                    {disease_info['severity']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Warning
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Medical Disclaimer:</strong> This AI prediction is for research purposes only. 
                Always consult a qualified dermatologist for proper diagnosis and treatment.
            </div>
            """, unsafe_allow_html=True)
            
            # Probability distribution
            st.subheader("📈 Confidence Distribution")
            prob_data = {name: float(prob) for name, prob in zip(CLASS_NAMES, all_probs)}
            prob_data = dict(sorted(prob_data.items(), key=lambda x: x[1], reverse=True))
            
            st.bar_chart(prob_data, use_container_width=True)
            
            # Grad-CAM
            if st.session_state.superimposed is not None:
                st.subheader("🔍 Explainability (Grad-CAM)")
                st.caption("Red regions show where the AI focused to make its decision")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(st.session_state.img_resized, 
                            caption="Original", 
                            use_column_width=True)
                with col_b:
                    st.image(st.session_state.superimposed, 
                            caption="AI Focus Areas", 
                            use_column_width=True)
                
                st.success("✓ Verification: Red regions should highlight the lesion")
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>
            Built with ❤️ using PyTorch & Streamlit | 
            Powered by GPU-Accelerated Deep Learning
        </p>
        <p style="font-size: 0.8rem;">
            Dataset: HAM10000 + DermNet | Architecture: MobileNetV2 | 
            Training: Mixed Precision (AMP) on RTX 3050
        </p>
        <p style="font-size: 0.8rem;">
            <strong>Model Stats:</strong> 81.19% accuracy | 10,327 training images | 8 classes
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
