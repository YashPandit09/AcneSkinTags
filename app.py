"""
Derm-X Analyzer - Streamlit Web Application
Steps 18-20: Frontend, Backend Connection, and Results Display
"""
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from preprocessing.normalization import normalize_imagenet
from explainability.gradcam import make_gradcam_heatmap, overlay_heatmap_on_image, get_last_conv_layer_name

# Page configuration
st.set_page_config(
    page_title="Derm-X Analyzer",
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
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 20px 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF9800;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """
    Step 19: Load the trained model (cached for performance)
    """
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """
    Step 19: Preprocess uploaded image
    - Resize to 224x224
    - Normalize by dividing by 255 (SAME AS TRAINING)
    
    CRITICAL: This must match the preprocessing used during training!
    Training used ImageDataGenerator which rescales to [0, 1] via /255
    """
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Resize to 224x224
    img_resized = cv2.resize(img_array, config.IMG_SIZE)
    
    # !!!CRITICAL FIX!!!
    # Training used simple rescaling (/255), NOT ImageNet normalization
    # We must use the SAME preprocessing
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # REMOVED: ImageNet normalization (was causing bug)
    # img_normalized = normalize_imagenet(img_resized)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_resized

def predict_with_gradcam(model, img_batch, img_original):
    """
    Make prediction and generate Grad-CAM explanation
    """
    # Make prediction
    predictions = model.predict(img_batch, verbose=0)
    pred_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_class_idx]
    
    # Get class names
    class_names = list(config.LESION_TYPE_DICT.values())
    pred_class_name = class_names[pred_class_idx]
    
    # Generate Grad-CAM
    try:
        last_conv_layer = get_last_conv_layer_name(model)
        heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer)
        superimposed = overlay_heatmap_on_image(img_original, heatmap)
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {e}")
        heatmap = None
        superimposed = None
    
    return pred_class_name, confidence, predictions[0], heatmap, superimposed

def get_disease_info(disease_name):
    """
    Provide medical information about the disease
    """
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
    # Step 18: Create the Frontend
    st.markdown('<div class="main-header">🔬 Derm-X Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Skin Lesion Detection with Explainable AI</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_options = {
            "MobileNetV2 (Fast)": "saved_models/mobilenetv2_best.keras",
            "ResNet50 (Balanced)": "saved_models/resnet50_best.keras",
            "EfficientNet-B3 (Accurate)": "saved_models/efficientnet-b3_best.keras"
        }
        
        selected_model_name = st.selectbox(
            "Select Model",
            list(model_options.keys()),
            help="Choose the AI model for analysis"
        )
        
        model_path = model_options[selected_model_name]
        
        # Check if custom model path exists
        if not os.path.exists(model_path):
            st.warning(f"Model not found at: {model_path}")
            custom_path = st.text_input("Enter custom model path:", 
                                         value="./best_model.keras")
            if os.path.exists(custom_path):
                model_path = custom_path
                st.success("Custom model loaded!")
        
        st.markdown("---")
        
        # About
        st.header("ℹ️ About")
        st.info("""
        **Derm-X** uses deep learning to classify 8 types of skin conditions:
        
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
        8. Acne
        
        **Disclaimer:** This is a research tool and should NOT replace professional medical diagnosis.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # Step 18: File uploader
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
                with st.spinner("Loading AI model..."):
                    model = load_model(model_path)
                
                if model is not None:
                    with st.spinner("Analyzing image..."):
                        # Step 19: Preprocess
                        img_batch, img_resized = preprocess_image(image)
                        
                        # Predict
                        pred_class, confidence, all_probs, heatmap, superimposed = predict_with_gradcam(
                            model, img_batch, img_resized
                        )
                        
                        # Store results in session state
                        st.session_state.prediction = pred_class
                        st.session_state.confidence = confidence
                        st.session_state.all_probs = all_probs
                        st.session_state.superimposed = superimposed
                        st.session_state.heatmap = heatmap
                        st.session_state.img_resized = img_resized
                    
                    st.success("Analysis complete!")
    
    with col2:
        st.header("📊 Results")
        
        if 'prediction' in st.session_state:
            # Step 20: Display Results
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
            
            # Warning disclaimer
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Medical Disclaimer:</strong> This AI prediction is for research purposes only. 
                Always consult a qualified dermatologist for proper diagnosis and treatment.
            </div>
            """, unsafe_allow_html=True)
            
            # Probability distribution
            st.subheader("📈 Confidence Distribution")
            class_names = list(config.LESION_TYPE_DICT.values())
            prob_data = {name: float(prob) for name, prob in zip(class_names, all_probs)}
            prob_data = dict(sorted(prob_data.items(), key=lambda x: x[1], reverse=True))
            
            # Bar chart
            st.bar_chart(prob_data, use_container_width=True)
            
            # Grad-CAM visualization
            if st.session_state.superimposed is not None:
                st.subheader("🔍 Explainability (Grad-CAM)")
                st.caption("Red regions show where the AI focused to make its decision")
                
                # Display comparison
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(st.session_state.img_resized, 
                            caption="Original", 
                            use_column_width=True)
                with col_b:
                    st.image(st.session_state.superimposed, 
                            caption="AI Focus Areas (Grad-CAM)", 
                            use_column_width=True)
                
                st.success("✓ Verification: Red regions should highlight the lesion")
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>
            Built with ❤️ using TensorFlow & Streamlit | 
            Powered by Deep Learning & Explainable AI
        </p>
        <p style="font-size: 0.8rem;">
            Dataset: HAM10000 | Architecture: MobileNetV2 / ResNet50 / EfficientNet
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
