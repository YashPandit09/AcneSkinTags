"""
Confidence Calibration — Temperature Scaling (Guo et al., 2017)

A clinical report that says "97.8%" is only trustworthy if that number is
empirically calibrated. A raw softmax from a deep net is almost always
over-confident. This module fits a single temperature T on the validation
split, reports Expected Calibration Error (ECE) before/after, and saves a
reliability diagram.

The fitted T is written to ``calibration_temperature.json`` and consumed by
``safe_inference.py`` (and can be used by the Streamlit app).

Usage
-----
    python calibration.py                 # fit T on the val split (needs model + data)
    python calibration.py --self-test     # verify the math on synthetic logits (no model)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

TEMPERATURE_JSON = "calibration_temperature.json"


# ------------------------------------------------------------------ #
# Core math (framework-light: numpy in, numpy out)
# ------------------------------------------------------------------ #
def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15):
    """ECE + per-bin (avg_confidence, accuracy, count) for a reliability plot."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            rows.append(((lo + hi) / 2, np.nan, 0))
            continue
        acc, avg_conf, weight = correct[m].mean(), conf[m].mean(), m.mean()
        ece += weight * abs(acc - avg_conf)
        rows.append((avg_conf, acc, int(m.sum())))
    return float(ece), rows


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Optimise scalar T by minimising NLL (LBFGS, the standard recipe)."""
    import torch
    import torch.nn as nn

    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    T = nn.Parameter(torch.ones(1) * 1.5)
    nll = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nll(logits_t / T.clamp_min(1e-3), labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.clamp_min(1e-3).item())


# ------------------------------------------------------------------ #
# Model/data plumbing (reuses the existing repo)
# ------------------------------------------------------------------ #
def collect_logits(model, loader, device):
    """Run `model` over a DataLoader, returning (logits, labels) as numpy."""
    import torch

    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            out = model(inputs.to(device))
            all_logits.append(out.cpu().numpy())
            all_labels.append(np.asarray(labels))
    return np.concatenate(all_logits), np.concatenate(all_labels)


def reliability_diagram(rows_before, rows_after, ece_b, ece_a, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for rows, lbl in [(rows_before, f"before (ECE={ece_b:.3f})"),
                      (rows_after, f"after T-scaling (ECE={ece_a:.3f})")]:
        xs = [c for c, a, n in rows if n > 0]
        ys = [a for c, a, n in rows if n > 0]
        ax.plot(xs, ys, "o-", label=lbl)
    ax.set(xlabel="confidence", ylabel="accuracy", title="Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(model_path="best_model_8class_pytorch.pth", batch_size=64):
    from inference_engine import load_model
    from train_pytorch_8class import get_dataloaders

    model, device = load_model(model_path)
    _, val_loader, _ = get_dataloaders(batch_size=batch_size, num_workers=0)
    logits, labels = collect_logits(model, val_loader, device)

    probs_before = softmax(logits, 1.0)
    ece_b, rows_b = expected_calibration_error(probs_before, labels)
    T = fit_temperature(logits, labels)
    probs_after = softmax(logits, T)
    ece_a, rows_a = expected_calibration_error(probs_after, labels)

    print(f"\nFitted temperature T = {T:.4f}")
    print(f"ECE before = {ece_b:.4f}  |  after = {ece_a:.4f}  "
          f"({'improved' if ece_a < ece_b else 'no improvement'})")

    with open(TEMPERATURE_JSON, "w") as f:
        json.dump({"temperature": T, "ece_before": ece_b, "ece_after": ece_a}, f, indent=2)
    reliability_diagram(rows_b, rows_a, ece_b, ece_a,
                        os.path.join("evaluation", "reliability_diagram.png"))
    print(f"Saved {TEMPERATURE_JSON} and evaluation/reliability_diagram.png")


def load_temperature(default: float = 1.0) -> float:
    if os.path.exists(TEMPERATURE_JSON):
        return float(json.load(open(TEMPERATURE_JSON))["temperature"])
    return default


# ------------------------------------------------------------------ #
def _self_test():
    """Verify the math on synthetic over-confident logits (no model needed)."""
    rng = np.random.default_rng(0)
    n, c = 2000, 8
    labels = rng.integers(0, c, n)
    logits = rng.normal(0, 1, (n, c))
    logits[np.arange(n), labels] += 2.0          # correct class favoured
    logits *= 3.0                                # inflate -> over-confident
    ece_b, _ = expected_calibration_error(softmax(logits, 1.0), labels)
    T = fit_temperature(logits, labels)
    ece_a, _ = expected_calibration_error(softmax(logits, T), labels)
    print(f"[self-test] T={T:.3f}  ECE {ece_b:.4f} -> {ece_a:.4f}")
    assert T > 1.0, "over-confident logits should need T>1"
    assert ece_a <= ece_b + 1e-6, "calibration should not worsen ECE"
    print("[self-test] PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Temperature-scaling calibration")
    ap.add_argument("--model", default="best_model_8class_pytorch.pth")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
    else:
        run(args.model, args.batch_size)
