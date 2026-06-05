"""
Trust-Aware Inference — one entry point that wraps the existing engine.

Composes the new safety layers on top of the repo's TTA + Grad-CAM pipeline:

    raw model  ->  TTA logits  ->  temperature scaling  ->  OOD gate
               ->  MC-dropout uncertainty  ->  abstention decision

Returns a superset of ``inference_engine.predict``'s dictionary, so the
Streamlit app can switch to this with minimal changes:

    from safe_inference import predict_with_trust
    result = predict_with_trust(model, pil_image, device)
    # result adds: calibrated, temperature, predictive_entropy, epistemic,
    #              is_ood, abstain, message  (plus all original keys)

Usage
-----
    python safe_inference.py --image path/to/photo.jpg
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from calibration import load_temperature, softmax
from inference_engine import (CLASS_NAMES, DISEASE_INFO, generate_gradcam,
                              load_model, preprocess_image)
from ood_detection import load_threshold, msp_score
from uncertainty import ABSTAIN_ENTROPY, mc_predict


def predict_with_trust(model, image: Image.Image, device,
                       n_mc: int = 20, want_gradcam: bool = True) -> dict:
    img_batch, img_resized = preprocess_image(image)
    T = load_temperature()

    # ---- TTA: 3-view averaged logits (matches inference_engine.predict) ----
    model.eval()
    x = img_batch.to(device)
    with torch.no_grad():
        logits = (model(x) + model(TF.hflip(x)) + model(TF.vflip(x))) / 3.0
    logits_np = logits.cpu().numpy()
    probs = softmax(logits_np, T)[0]

    pred_idx = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # ---- OOD gate (calibrated MSP) ----
    ood = bool(msp_score(probs[None, :])[0] < load_threshold())

    # ---- MC-dropout uncertainty (original view) ----
    unc = mc_predict(model, img_batch, device, n_samples=n_mc, temperature=T)
    entropy = unc["predictive_entropy"]
    abstain = bool(ood or entropy > ABSTAIN_ENTROPY)

    if ood:
        message = "Input does not look like skin (out-of-distribution) — no diagnosis made."
    elif abstain:
        message = "Low-confidence / high-uncertainty — please consult a clinician."
    else:
        message = f"Most likely: {pred_class} ({confidence:.1%})."

    # ---- Grad-CAM (skip when OOD) ----
    heatmap = None
    if want_gradcam and not ood:
        try:
            heatmap, _ = generate_gradcam(model, img_batch, device, pred_idx)
        except Exception:
            pass

    return {
        # original-style keys
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
        "heatmap": heatmap,
        "img_resized": img_resized,
        "disease_info": DISEASE_INFO.get(pred_class, {}),
        # new trust keys
        "calibrated": True,
        "temperature": T,
        "predictive_entropy": entropy,
        "epistemic": unc["epistemic"],
        "is_ood": ood,
        "abstain": abstain,
        "message": message,
    }


def main():
    ap = argparse.ArgumentParser(description="Trust-aware inference")
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="best_model_8class_pytorch.pth")
    ap.add_argument("--mc-samples", type=int, default=20)
    args = ap.parse_args()

    model, device = load_model(args.model)
    res = predict_with_trust(model, Image.open(args.image).convert("RGB"),
                             device, n_mc=args.mc_samples)
    print(f"\n{res['message']}")
    print(f"  temperature={res['temperature']:.3f}  entropy={res['predictive_entropy']:.3f}"
          f"  epistemic={res['epistemic']:.3f}  ood={res['is_ood']}  abstain={res['abstain']}")
    print("  breakdown:")
    for name, p in sorted(res["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"    {name:<32} {p:6.1%}")


if __name__ == "__main__":
    main()
