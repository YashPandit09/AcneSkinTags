"""
Derm-X — Clinical Research Demo (Streamlit)

A presentation-focused variant of the demo for showing the project to a
supervising dermatologist. Unlike the consumer app, this one foregrounds
**safety and honesty**:

  * a persistent "research prototype — not a medical device" disclaimer
  * temperature-CALIBRATED confidence (not raw softmax)
  * out-of-distribution / non-skin rejection
  * MC-dropout UNCERTAINTY with an explicit abstain state
  * a SAFETY-FIRST triage recommendation (refer on any cancer suspicion)
  * Grad-CAM evidence for every prediction

Run:
    streamlit run app_clinical.py

Requires the trained checkpoint `best_model_8class_pytorch.pth` in the repo root.
"""
from __future__ import annotations

import os

import streamlit as st
from PIL import Image

MODEL_PATH = "best_model_8class_pytorch.pth"

st.set_page_config(page_title="Derm-X — Clinical Research Demo",
                   page_icon="🔬", layout="wide")


# --------------------------------------------------------------------------- #
@st.cache_resource
def _load():
    from inference_engine import load_model
    return load_model(MODEL_PATH)


def _disclaimer():
    st.error(
        "**RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE.** "
        "This tool is for research and educational discussion only. It is "
        "**not** validated for clinical use, must not be used to diagnose or "
        "treat any patient, and does not replace examination by a qualified "
        "dermatologist. All outputs require clinician oversight.",
        icon="⚠️",
    )


URGENCY_STYLE = {
    "urgent": ("🔴", "error"), "soon": ("🟠", "warning"),
    "review": ("🟡", "warning"), "routine": ("🟢", "success"),
    "n/a": ("⚪", "info"),
}


# --------------------------------------------------------------------------- #
def main():
    st.title("🔬 Derm-X — Clinical Research Demo")
    st.caption("8-class skin-lesion screening · calibrated · uncertainty-aware · "
               "explainable · decision-support only")
    _disclaimer()

    if not os.path.exists(MODEL_PATH):
        st.warning(f"Trained model `{MODEL_PATH}` not found. Train it first "
                   f"(`python train_pytorch_8class.py`) or copy the checkpoint "
                   f"into the repo root, then reload.")
        st.stop()

    from clinical_triage import triage
    from safe_inference import predict_with_trust
    from inference_engine import overlay_heatmap

    model, device = _load()

    up = st.file_uploader("Upload a dermoscopic / clinical skin image",
                          type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Upload an image to run the pipeline.")
        st.stop()

    image = Image.open(up).convert("RGB")
    with st.spinner("Running TTA → calibration → OOD gate → MC-dropout …"):
        res = predict_with_trust(model, image, device)
    tri = triage(res["probabilities"], abstain=res["abstain"], is_ood=res["is_ood"])

    col_img, col_cam = st.columns(2)
    with col_img:
        st.image(image, caption="Input", use_container_width=True)
    with col_cam:
        if res["heatmap"] is not None:
            st.image(overlay_heatmap(res["img_resized"], res["heatmap"]),
                     caption="Grad-CAM (model's evidence)", use_container_width=True)
        else:
            st.info("Grad-CAM skipped (input rejected as non-skin).")

    # ---- Headline state ----
    if res["is_ood"]:
        st.error(f"🚫 {res['message']}")
    elif res["abstain"]:
        st.warning(f"🤔 {res['message']}")
    else:
        st.success(f"✅ {res['message']}")

    # ---- Triage recommendation (the clinically actionable part) ----
    icon, kind = URGENCY_STYLE.get(tri["urgency"], ("⚪", "info"))
    getattr(st, kind)(f"{icon} **Triage ({tri['urgency'].upper()}):** {tri['recommendation']}")

    # ---- Metrics ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top class", res["predicted_class"])
    c2.metric("Calibrated confidence", f"{res['confidence']:.1%}")
    c3.metric("P(concerning)", f"{tri['p_concerning']:.1%}",
              help="Aggregated probability of Melanoma + BCC + Actinic keratoses")
    c4.metric("Uncertainty (entropy)", f"{res['predictive_entropy']:.2f}",
              help="MC-dropout predictive entropy; high → abstain")

    # ---- Full breakdown ----
    st.subheader("Calibrated probability breakdown")
    st.bar_chart({k: v for k, v in sorted(
        res["probabilities"].items(), key=lambda kv: -kv[1])})

    with st.expander("Technical detail (for the clinician / reviewer)"):
        st.json({
            "temperature": round(res["temperature"], 3),
            "predictive_entropy": round(res["predictive_entropy"], 3),
            "epistemic_uncertainty": round(res["epistemic"], 3),
            "is_ood": res["is_ood"], "abstain": res["abstain"],
            "p_melanoma": round(tri["p_melanoma"], 4),
            "p_concerning": round(tri["p_concerning"], 4),
        })

    st.divider()
    st.caption("Known limitation: at this operating point melanoma recall ≈ 42%. "
               "The triage policy above intentionally over-refers to compensate. "
               "See docs/clinical/ for the model card and validation protocol.")


if __name__ == "__main__":
    main()
