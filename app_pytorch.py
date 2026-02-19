"""
Derm-X Analyzer — Enhanced Streamlit Frontend
Imports all inference logic from inference_engine.py.
"""

import streamlit as st
import numpy as np
import torch
from PIL import Image
import tempfile
import datetime
import os

from inference_engine import (
    load_model,
    predict,
    overlay_heatmap,
    CLASS_NAMES,
)

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Derm-X Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# NATURAL-PALETTE CSS — sage / sand / warm-stone glassmorphism
# ================================================================
#
# Palette anchors:
#   Background  — pale sage #e8efe5 → warm sand #f5f0e8
#   Glass cards — rgba(255,255,255,0.55) with subtle blur
#   Text body   — warm charcoal #3a3a38
#   Headings    — deep olive-stone #4b5548
#   Accents     — muted terracotta #b5694d, soft teal #5a8f7b
#   Risk High   — rosewood #b94a48
#   Risk Mod    — warm amber #c08b30
#   Risk Low    — sage green #5a8f7b

st.markdown("""
<style>
/* ---- Hide Streamlit chrome ---- */
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}

/* ---- Typography ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ---- Natural gradient background ---- */
.stApp {
    background: linear-gradient(165deg, #e8efe5 0%, #eef2ea 35%, #f5f0e8 100%);
}

/* ---- Glass card (organic) ---- */
.glass-card {
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(90, 143, 123, 0.12);
    border-radius: 24px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 6px 28px rgba(75, 85, 72, 0.08);
}

/* ---- Result cards (risk levels) ---- */
.result-card-high {
    background: linear-gradient(135deg, rgba(185,74,72,0.10), rgba(185,74,72,0.04));
    border: 1px solid rgba(185,74,72,0.25);
    border-radius: 20px;
    padding: 24px;
    margin: 12px 0;
}
.result-card-moderate {
    background: linear-gradient(135deg, rgba(192,139,48,0.10), rgba(192,139,48,0.04));
    border: 1px solid rgba(192,139,48,0.25);
    border-radius: 20px;
    padding: 24px;
    margin: 12px 0;
}
.result-card-low {
    background: linear-gradient(135deg, rgba(90,143,123,0.10), rgba(90,143,123,0.04));
    border: 1px solid rgba(90,143,123,0.25);
    border-radius: 20px;
    padding: 24px;
    margin: 12px 0;
}

/* ---- Risk badges ---- */
.badge-high     { color:#b94a48; background:rgba(185,74,72,0.12);  padding:5px 14px; border-radius:24px; font-size:0.75rem; font-weight:600; }
.badge-moderate { color:#96700f; background:rgba(192,139,48,0.12); padding:5px 14px; border-radius:24px; font-size:0.75rem; font-weight:600; }
.badge-low      { color:#3d6f5a; background:rgba(90,143,123,0.12); padding:5px 14px; border-radius:24px; font-size:0.75rem; font-weight:600; }

/* ---- Image containers ---- */
.image-frame {
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(75, 85, 72, 0.08);
    box-shadow: 0 4px 20px rgba(75, 85, 72, 0.10);
}
.image-frame img {
    border-radius: 20px;
}

/* ---- Section headers ---- */
.section-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #8a9184;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    margin-bottom: 12px;
}

/* ---- Hero header ---- */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4b5548, #5a8f7b, #b5694d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0px;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #8a9184;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e2eadf 0%, #f0ede6 100%);
    border-right: 1px solid rgba(75, 85, 72, 0.08);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #555a52;
    font-size: 0.88rem;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4b5548;
}

/* ---- Metrics ---- */
[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #3a3a38 !important;
}
[data-testid="stMetricLabel"] {
    color: #8a9184 !important;
}

/* ---- Button ---- */
.stButton > button {
    background: linear-gradient(135deg, #5a8f7b, #73a690) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(90,143,123,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(90,143,123,0.40) !important;
}

/* ---- File uploader ---- */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(75, 85, 72, 0.15) !important;
    border-radius: 20px !important;
    padding: 16px !important;
}

/* ---- Disclaimer box ---- */
.disclaimer {
    background: rgba(192,139,48,0.06);
    border: 1px solid rgba(192,139,48,0.18);
    border-radius: 16px;
    padding: 14px 18px;
    font-size: 0.82rem;
    color: #6b5e4b;
    margin-top: 16px;
}

/* ---- Footer ---- */
.app-footer {
    text-align: center;
    color: #aeb5a8;
    font-size: 0.78rem;
    padding: 40px 0 20px 0;
    border-top: 1px solid rgba(75, 85, 72, 0.06);
    margin-top: 48px;
}

/* ---- Ensure readable body text on light bg ---- */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
.stCaption, .stTextInput label, [data-testid="stWidgetLabel"] {
    color: #3a3a38 !important;
}

/* ---- Protect Streamlit internal icon fonts ---- */
.st-emotion-cache-1gfk4cg,
.material-symbols-rounded,
span[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
    font-size: 1.25rem !important;
}
button[data-testid="stExpanderToggleIcon"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* ---- Expander container — visible on light bg ---- */
[data-testid="stExpander"] {
    background: rgba(90, 143, 123, 0.06);
    border: 1px solid rgba(90, 143, 123, 0.15);
    border-radius: 20px;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #4b5548;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("### ⚙️ Model Info")
    st.markdown("""
    <div class="glass-card" style="padding:18px;">
        <p style="margin:0; color:#3d6f5a; font-weight:600;">✅ Model Active</p>
        <p style="margin:4px 0 0; font-size:0.85rem; color:#6b7a66;">
            PyTorch MobileNetV2 · 8 Classes<br>
            Accuracy: <strong style="color:#3a3a38;">81.19%</strong><br>
            Acne Precision: <strong style="color:#3a3a38;">99.36%</strong><br>
            <span style="color:#5a8f7b;">TTA</span> · <span style="color:#5a8f7b;">Focal Loss</span> · <span style="color:#5a8f7b;">Grad-CAM</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    device_label = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🔵 CPU"
    st.info(f"Running on: **{device_label}**")

    st.markdown("---")
    st.markdown("### 📋 Supported Classes")
    st.markdown("""
    **Cancers & Pre-cancerous**
    1. Melanoma
    2. Basal cell carcinoma
    3. Actinic keratoses

    **Benign Lesions**
    4. Melanocytic nevi
    5. Benign keratosis
    6. Vascular lesions
    7. Dermatofibroma

    **Common Conditions**
    8. Acne
    """)

    st.markdown("---")
    st.markdown("### 🩺 Patient Data (Optional)")
    st.markdown("""
    <div class="glass-card" style="padding:18px;">
        <p style="margin:0 0 8px; font-size:0.82rem; color:#8a9184;">Attach optional context for your records.</p>
    </div>
    """, unsafe_allow_html=True)
    patient_age = st.number_input(
        "Patient Age",
        min_value=0,
        max_value=120,
        value=30,
        step=1,
        help="Patient's age in years.",
    )
    patient_gender = st.selectbox(
        "Gender",
        ["Not specified", "Male", "Female", "Other"],
        help="Optional patient gender.",
    )
    patient_notes = st.text_area(
        "Clinical Notes",
        placeholder="e.g. lesion appeared 3 months ago, itchy...",
        height=80,
        help="Any relevant notes for your own reference.",
    )

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Disclaimer</strong>: This tool is for research &
        educational purposes only. It does not replace professional
        medical diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# MAIN CONTENT
# ================================================================

# ---- Hero ----
st.markdown('<p class="hero-title">🔬 Derm-X Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    'AI-Powered Skin Lesion Classification · Grad-CAM Explainability'
    '</p>',
    unsafe_allow_html=True,
)

# ---- How-to expander ----
with st.expander("How to use Derm-X Analyzer", expanded=False):
    st.markdown("**Quick Guide**")
    st.markdown("""
    1. **Upload** a clear, well-lit dermoscopic image (JPG or PNG).
    2. Click **Analyze Image** and wait a few seconds.
    3. Review the **predicted class**, confidence score, and risk level.
    4. Use the **Grad-CAM slider** to see exactly where the model focused.
    5. Check the **confidence chart** to compare probabilities across all 8 classes.
    """)
    st.caption("For best results, ensure the lesion is centered and the image is not blurry.")

# ---- Upload section ----
st.markdown('<p class="section-title">Upload & Analyze</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a dermoscopic image here (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear, well-lit photo of the skin lesion.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Show upload preview in a compact column
    col_up, _ = st.columns([1, 2])
    with col_up:
        st.image(image, caption="Uploaded image", use_container_width=True)

    # ---- Analyze button ----
    if st.button("🔍  Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Loading model & running inference …"):
            model, device = load_model()
            result = predict(model, image, device)

        st.session_state["result"] = result

# ================================================================
# RESULTS SECTION
# ================================================================
if "result" in st.session_state:
    res = st.session_state["result"]

    st.markdown("---")
    st.markdown('<p class="section-title">Diagnosis Results</p>', unsafe_allow_html=True)

    # ---- Prediction metrics row ----
    m1, m2, m3 = st.columns(3)
    risk = res["disease_info"]["risk_level"]
    with m1:
        st.metric("Predicted Class", res["predicted_class"])
    with m2:
        st.metric("Confidence", f"{res['confidence']:.1%}")
    with m3:
        risk_emoji = {"high": "🔴", "moderate": "🟠", "low": "🟢"}.get(risk, "⚪")
        st.metric("Risk Level", f"{risk_emoji} {risk.upper()}")

    # ---- Detail card ----
    card_class = f"result-card-{risk}"
    badge_class = f"badge-{risk}"
    st.markdown(f"""
    <div class="{card_class}">
        <span class="{badge_class}">{res['disease_info']['severity']}</span>
        <p style="margin-top:12px; color:#3a3a38; font-size:0.95rem;">
            {res['disease_info']['description']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Risk gauge chart ----
    try:
        import plotly.graph_objects as go

        # Map risk level to gauge value (0-100 scale)
        risk_score = res["confidence"] * 100
        risk_label = res["disease_info"]["risk_level"]
        gauge_color = {
            "high": "#b94a48",       # rosewood
            "moderate": "#c08b30",   # warm amber
            "low": "#5a8f7b",        # sage green
        }.get(risk_label, "#8a9184")

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            number={"suffix": "%", "font": {"size": 36, "color": "#3a3a38", "family": "Inter"}},
            title={"text": "Prediction Confidence", "font": {"size": 14, "color": "#8a9184", "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8a9184", "tickfont": {"color": "#8a9184"}},
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],  "color": "rgba(185,74,72,0.08)"},
                    {"range": [30, 70], "color": "rgba(192,139,48,0.08)"},
                    {"range": [70, 100], "color": "rgba(90,143,123,0.08)"},
                ],
                "threshold": {
                    "line": {"color": "#3a3a38", "width": 5},
                    "thickness": 0.85,
                    "value": risk_score,
                },
            },
        ))

        gauge_fig.update_layout(
            height=220,
            margin=dict(l=30, r=30, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
        )

        g_col1, g_col2, g_col3 = st.columns([1, 2, 1])
        with g_col2:
            st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

    except ImportError:
        pass  # Plotly not available; skip gauge

    # ---- Side-by-side images: Original | Grad-CAM ----
    st.markdown('<p class="section-title">Explainability — Grad-CAM</p>', unsafe_allow_html=True)

    if res["heatmap"] is not None:
        alpha = st.slider(
            "Heatmap Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Drag to adjust how strongly the Grad-CAM overlay is blended onto the image.",
        )

        overlay = overlay_heatmap(res["img_resized"], res["heatmap"], alpha=alpha)

        col_orig, col_cam = st.columns(2)
        with col_orig:
            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
            st.image(res["img_resized"], caption="Original (224 × 224)", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_cam:
            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
            st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.caption("🔍 Red/yellow regions show where the model focused to make its decision.")
    else:
        st.warning("Grad-CAM heatmap could not be generated for this image.")

    # ---- Confidence distribution bar chart ----
    st.markdown('<p class="section-title">Confidence Distribution (All 8 Classes)</p>', unsafe_allow_html=True)

    probs = res["probabilities"]
    sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

    # Build horizontal chart with Plotly — natural colour scheme
    try:
        import plotly.graph_objects as go

        names = list(sorted_probs.keys())
        values = [v * 100 for v in sorted_probs.values()]

        # Colour bars by risk level using organic palette
        from inference_engine import DISEASE_INFO
        bar_colors = []
        for name in names:
            rl = DISEASE_INFO.get(name, {}).get("risk_level", "low")
            if rl == "high":
                bar_colors.append("rgba(185,74,72,0.70)")    # rosewood
            elif rl == "moderate":
                bar_colors.append("rgba(192,139,48,0.70)")   # warm amber
            else:
                bar_colors.append("rgba(90,143,123,0.70)")   # sage green

        fig = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(width=0),
                cornerradius=6,
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="#4b5548", size=12),
        ))

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=50, t=10, b=10),
            height=340,
            xaxis=dict(
                title="Confidence (%)",
                range=[0, max(values) * 1.25],
                showgrid=True,
                gridcolor="rgba(75,85,72,0.07)",
                tickfont=dict(color="#8a9184"),
                title_font=dict(color="#8a9184", size=12),
            ),
            yaxis=dict(
                autorange="reversed",
                tickfont=dict(color="#4b5548", size=13),
            ),
            font=dict(family="Inter, sans-serif"),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    except ImportError:
        # Fallback to basic Streamlit bar chart if Plotly not installed
        st.bar_chart(sorted_probs)

    # ---- Downloadable PDF clinical report ----
    st.markdown('<p class="section-title">Clinical Documentation</p>', unsafe_allow_html=True)

    try:
        from fpdf import FPDF

        def _generate_pdf(result, overlay_img, age=None, gender=None, notes=None):
            """Generate a clinical PDF report."""

            def _safe(text):
                """Sanitise text for FPDF (Latin-1 only)."""
                return str(text).replace("\u2014", "-").replace("\u2013", "-").encode("latin-1", "replace").decode("latin-1")

            pdf = FPDF()
            pdf.add_page()

            # ---- Header ----
            pdf.set_font("Helvetica", style="B", size=20)
            pdf.cell(0, 15, "Derm-X: AI Dermatological Scan Report", ln=True, align="C")
            pdf.set_font("Helvetica", size=11)
            pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            pdf.ln(8)

            # ---- Prediction ----
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.cell(0, 10, _safe(f"Primary Finding: {result['predicted_class']}"), ln=True)
            pdf.set_font("Helvetica", size=12)
            pdf.cell(0, 8, f"Confidence: {result['confidence']:.1%}", ln=True)
            pdf.cell(0, 8, _safe(f"Risk Level: {result['disease_info']['severity']}"), ln=True)
            pdf.ln(4)
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, _safe(f"Description: {result['disease_info']['description']}"))
            pdf.ln(4)

            # ---- Patient data (if provided) ----
            if age or (gender and gender != "Not specified") or notes:
                pdf.set_font("Helvetica", style="B", size=12)
                pdf.cell(0, 10, "Patient Information:", ln=True)
                pdf.set_font("Helvetica", size=11)
                if age:
                    pdf.cell(0, 6, f"  Age: {age}", ln=True)
                if gender and gender != "Not specified":
                    pdf.cell(0, 6, f"  Gender: {gender}", ln=True)
                if notes:
                    pdf.multi_cell(0, 6, _safe(f"  Notes: {notes}"))
                pdf.ln(4)
            pdf.ln(2)

            # ---- Images ----
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_orig:
                Image.fromarray(result["img_resized"]).save(tmp_orig.name)
                orig_path = tmp_orig.name

            heat_path = None
            if overlay_img is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_heat:
                    Image.fromarray(overlay_img).save(tmp_heat.name)
                    heat_path = tmp_heat.name

            pdf.set_font("Helvetica", style="B", size=12)
            if heat_path:
                pdf.cell(95, 8, "Original Image:", ln=False)
                pdf.cell(95, 8, "Grad-CAM Heatmap:", ln=True)
                y_pos = pdf.get_y()
                pdf.image(orig_path, x=10, y=y_pos, w=80)
                pdf.image(heat_path, x=105, y=y_pos, w=80)
                pdf.ln(70)
            else:
                pdf.cell(0, 8, "Original Image:", ln=True)
                pdf.image(orig_path, x=10, w=80)
                pdf.ln(10)

            # ---- Confidence table ----
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.cell(0, 10, "Confidence Breakdown (All 8 Classes):", ln=True)
            pdf.set_font("Helvetica", size=10)
            for cls_name, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
                pdf.cell(120, 6, f"  {cls_name}", ln=False)
                pdf.cell(0, 6, f"{prob:.1%}", ln=True)
            pdf.ln(6)

            # ---- Disclaimer ----
            pdf.set_font("Helvetica", style="I", size=9)
            pdf.multi_cell(
                0, 5,
                "Disclaimer: This report was generated by the Derm-X experimental AI model. "
                "It is not a substitute for professional medical advice, diagnosis, or treatment. "
                "Always consult a qualified healthcare provider.",
            )

            # Cleanup temp files
            try:
                os.unlink(orig_path)
                if heat_path:
                    os.unlink(heat_path)
            except OSError:
                pass

            return bytes(pdf.output())

        # Build overlay for PDF (use default alpha=0.45)
        pdf_overlay = None
        if res["heatmap"] is not None:
            pdf_overlay = overlay_heatmap(res["img_resized"], res["heatmap"], alpha=0.45)

        pdf_bytes = _generate_pdf(
            res, pdf_overlay,
            age=patient_age,
            gender=patient_gender,
            notes=patient_notes,
        )

        st.download_button(
            label="📄 Download PDF Clinical Report",
            data=pdf_bytes,
            file_name=f"DermX_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    except ImportError:
        st.info("Install `fpdf2` to enable PDF report downloads: `pip install fpdf2`")

    # ---- Medical disclaimer ----
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer</strong>: This AI prediction is for research
        and educational purposes only. Always consult a qualified dermatologist
        for proper diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

else:
    # ---- Empty state ----
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:60px 28px;">
        <p style="font-size:3rem; margin:0;">📷</p>
        <p style="font-size:1.1rem; color:#555a52; margin-top:12px;">
            Upload a skin lesion image above to get started.
        </p>
        <p style="font-size:0.85rem; color:#8a9184;">
            The model will classify it into one of 8 categories and show you
            exactly where it looked using Grad-CAM.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# FOOTER
# ================================================================
st.markdown("""
<div class="app-footer">
    Built with PyTorch & Streamlit &nbsp;·&nbsp;
    MobileNetV2 + Focal Loss &nbsp;·&nbsp;
    Test-Time Augmentation<br>
    HAM10000 + DermNet &nbsp;·&nbsp;
    81.19% Accuracy &nbsp;·&nbsp; 10,327 Images &nbsp;·&nbsp; 8 Classes
</div>
""", unsafe_allow_html=True)
