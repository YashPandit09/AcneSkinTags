"""
Self-contained tests for the trust & safety additions.
No trained model or dataset required — they validate the math.

    pytest tests/test_trust.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_calibration_reduces_overconfidence():
    from calibration import (expected_calibration_error, fit_temperature,
                             softmax)
    rng = np.random.default_rng(0)
    n, c = 2000, 8
    labels = rng.integers(0, c, n)
    logits = rng.normal(0, 1, (n, c))
    logits[np.arange(n), labels] += 2.0
    logits *= 3.0                                  # over-confident
    ece_b, _ = expected_calibration_error(softmax(logits), labels)
    T = fit_temperature(logits, labels)
    ece_a, _ = expected_calibration_error(softmax(logits, T), labels)
    assert T > 1.0
    assert ece_a <= ece_b + 1e-6


def test_ood_separates_flat_from_peaky():
    from calibration import softmax
    from ood_detection import msp_score
    rng = np.random.default_rng(1)
    id_logits = rng.normal(0, 1, (500, 8)); id_logits[:, 0] += 6.0
    ood_logits = rng.normal(0, 0.3, (500, 8))
    thr = float(np.quantile(msp_score(softmax(id_logits)), 0.05))
    assert (msp_score(softmax(id_logits)) >= thr).mean() >= 0.90
    assert (msp_score(softmax(ood_logits)) < thr).mean() >= 0.80


def test_energy_score_higher_for_ood():
    from ood_detection import energy_score
    rng = np.random.default_rng(2)
    confident = rng.normal(0, 1, (200, 8)); confident[:, 0] += 6.0
    flat = rng.normal(0, 0.2, (200, 8))
    assert energy_score(confident).mean() < energy_score(flat).mean()


def test_mc_dropout_uncertainty_nonnegative():
    import torch
    import torch.nn as nn
    from uncertainty import mc_predict
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32), nn.ReLU(),
                          nn.Dropout(0.5), nn.Linear(32, 8))
    res = mc_predict(model, torch.randn(1, 3, 8, 8), torch.device("cpu"), n_samples=30)
    assert res["predictive_entropy"] > 0
    assert res["epistemic"] >= -1e-6
    assert abs(res["mean_probs"].sum() - 1.0) < 1e-5
