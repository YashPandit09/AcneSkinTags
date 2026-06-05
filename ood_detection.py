"""
Out-of-Distribution (Non-Skin) Rejection

Users upload arbitrary photos. A medical classifier must *abstain* on inputs
that don't look like skin instead of confidently mapping a cat or a keyboard to
"Melanoma". This adds two training-free OOD signals on top of the existing
model (no retraining, no extra network):

  * MSP    — maximum softmax probability. Low MSP  -> reject.
  * Energy — -logsumexp(logits).         High energy -> reject.

``--fit`` calibrates the MSP threshold on the in-distribution test split so a
target fraction of genuine skin images still pass; the threshold is saved to
``ood_threshold.json`` and consumed by ``safe_inference.py``.

Usage
-----
    python ood_detection.py --fit            # calibrate threshold (needs model + data)
    python ood_detection.py --self-test      # synthetic check (no model)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

OOD_JSON = "ood_threshold.json"
DEFAULT_MSP_THRESHOLD = 0.45


def msp_score(probs: np.ndarray) -> np.ndarray:
    """Maximum softmax probability (higher = more in-distribution)."""
    return probs.max(axis=-1)


def energy_score(logits: np.ndarray) -> np.ndarray:
    """Free energy = -logsumexp(logits) (higher = more OOD)."""
    m = logits.max(axis=-1, keepdims=True)
    return -(m.squeeze(-1) + np.log(np.exp(logits - m).sum(axis=-1)))


def is_ood(probs: np.ndarray, threshold: float | None = None) -> np.ndarray:
    thr = load_threshold() if threshold is None else threshold
    return msp_score(probs) < thr


def fit_threshold(target_id_pass_rate: float = 0.95,
                  model_path: str = "best_model_8class_pytorch.pth",
                  batch_size: int = 64) -> dict:
    """Pick the MSP cut so ~`target_id_pass_rate` of real skin images pass."""
    from calibration import collect_logits, load_temperature, softmax
    from inference_engine import load_model
    from train_pytorch_8class import get_dataloaders

    model, device = load_model(model_path)
    _, _, test_loader = get_dataloaders(batch_size=batch_size, num_workers=0)
    logits, _ = collect_logits(model, test_loader, device)
    probs = softmax(logits, load_temperature())          # use calibrated probs
    scores = msp_score(probs)

    thr = float(np.quantile(scores, 1.0 - target_id_pass_rate))
    info = {"msp_threshold": thr, "target_id_pass_rate": target_id_pass_rate,
            "id_msp_mean": float(scores.mean()), "id_msp_min": float(scores.min())}
    with open(OOD_JSON, "w") as f:
        json.dump(info, f, indent=2)
    print(f"MSP threshold = {thr:.3f} (keeps ~{target_id_pass_rate:.0%} of skin images)")
    print(f"Saved {OOD_JSON}")
    return info


def load_threshold(default: float = DEFAULT_MSP_THRESHOLD) -> float:
    if os.path.exists(OOD_JSON):
        return float(json.load(open(OOD_JSON))["msp_threshold"])
    return default


def _self_test():
    """In-dist (peaky) vs OOD (flat) softmaxes should separate by MSP."""
    rng = np.random.default_rng(0)
    id_logits = rng.normal(0, 1, (500, 8)); id_logits[:, 0] += 6.0   # confident
    ood_logits = rng.normal(0, 0.3, (500, 8))                        # flat
    from calibration import softmax
    id_msp = msp_score(softmax(id_logits))
    ood_msp = msp_score(softmax(ood_logits))
    thr = float(np.quantile(id_msp, 0.05))
    id_pass = (id_msp >= thr).mean()
    ood_reject = (ood_msp < thr).mean()
    print(f"[self-test] thr={thr:.3f}  ID pass={id_pass:.2%}  OOD reject={ood_reject:.2%}")
    assert id_pass >= 0.90 and ood_reject >= 0.80
    print("[self-test] PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Non-skin / OOD rejection")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--id-pass-rate", type=float, default=0.95)
    ap.add_argument("--model", default="best_model_8class_pytorch.pth")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
    elif args.fit:
        fit_threshold(args.id_pass_rate, args.model)
    else:
        print(f"Current MSP threshold: {load_threshold():.3f}")
