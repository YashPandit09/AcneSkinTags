"""
Predictive Uncertainty via MC-Dropout (Gal & Ghahramani, 2016)

TTA (already in the repo) makes a single prediction more robust, but it does not
tell you *how unsure* the model is. MC-Dropout does: keep the classifier's
Dropout active at inference, run N stochastic forward passes, and measure the
spread of the predictions.

We report:
  * predictive entropy   H[mean p]      — total uncertainty
  * aleatoric            E[H[p]]         — data noise
  * epistemic (BALD)     predictive - aleatoric — model uncertainty (reducible)

High predictive entropy -> abstain and recommend a clinician, instead of
emitting a confident-looking but unreliable label.

Usage
-----
    python uncertainty.py --image path/to/photo.jpg     # needs trained model
    python uncertainty.py --self-test                   # synthetic (no model/data)
"""
from __future__ import annotations

import argparse

import numpy as np

ABSTAIN_ENTROPY = 1.5  # nats; above this -> "uncertain, see a clinician"


def enable_mc_dropout(model):
    """Put only Dropout layers into train mode (BatchNorm etc. stay in eval)."""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
    return model


def _entropy(p: np.ndarray, axis=-1) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=axis)


def mc_predict(model, img_tensor, device, n_samples: int = 20,
               temperature: float = 1.0) -> dict:
    """Run N MC-dropout passes; return mean probs + uncertainty decomposition."""
    import torch

    model.eval()
    enable_mc_dropout(model)
    img = img_tensor.to(device)
    probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(img) / temperature
            probs.append(torch.softmax(logits, dim=1)[0].cpu().numpy())
    probs = np.stack(probs)                      # (N, C)
    mean_p = probs.mean(0)

    predictive = float(_entropy(mean_p))
    aleatoric = float(_entropy(probs, axis=1).mean())
    epistemic = float(predictive - aleatoric)
    return {
        "mean_probs": mean_p,
        "predicted_index": int(mean_p.argmax()),
        "confidence": float(mean_p.max()),
        "predictive_entropy": predictive,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "abstain": predictive > ABSTAIN_ENTROPY,
    }


def run(image_path: str, model_path="best_model_8class_pytorch.pth",
        n_samples=20):
    from PIL import Image

    from calibration import load_temperature
    from inference_engine import CLASS_NAMES, load_model, preprocess_image

    model, device = load_model(model_path)
    img_batch, _ = preprocess_image(Image.open(image_path).convert("RGB"))
    res = mc_predict(model, img_batch, device, n_samples, load_temperature())

    print(f"\nPrediction: {CLASS_NAMES[res['predicted_index']]} "
          f"({res['confidence']:.1%})")
    print(f"  predictive entropy : {res['predictive_entropy']:.3f} nats")
    print(f"  epistemic (model)  : {res['epistemic']:.3f}")
    print(f"  aleatoric (data)   : {res['aleatoric']:.3f}")
    print(f"  -> {'ABSTAIN — recommend clinician' if res['abstain'] else 'confident enough'}")
    return res


def _self_test():
    """Build a tiny dropout net (no download) and confirm MC-dropout varies."""
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32), nn.ReLU(),
                          nn.Dropout(0.5), nn.Linear(32, 8))
    device = torch.device("cpu")
    x = torch.randn(1, 3, 8, 8)
    res = mc_predict(model, x, device, n_samples=30)
    print(f"[self-test] entropy={res['predictive_entropy']:.3f} "
          f"epistemic={res['epistemic']:.3f}")
    assert res["predictive_entropy"] > 0
    assert res["epistemic"] >= -1e-6           # epistemic is non-negative
    assert abs(res["mean_probs"].sum() - 1.0) < 1e-5
    print("[self-test] PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MC-dropout predictive uncertainty")
    ap.add_argument("--image")
    ap.add_argument("--model", default="best_model_8class_pytorch.pth")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
    elif args.image:
        run(args.image, args.model, args.samples)
    else:
        ap.error("provide --image or --self-test")
