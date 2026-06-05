"""
Safety-First Clinical Triage Policy

Top-1 accuracy is the wrong objective for a *screening* tool. Missing a melanoma
(false negative) is far more costly than an unnecessary referral (false positive).
This module converts the 8-class probability vector into a **referral decision**
by aggregating the probability mass on concerning classes and erring toward
"see a dermatologist" whenever there is any meaningful suspicion.

Risk tiers (from inference_engine.DISEASE_INFO):
  * HIGH      — Melanoma                          (rule out urgently)
  * MODERATE  — Basal cell carcinoma, Actinic keratoses
  * LOW       — Melanocytic nevi, Benign keratosis, Vascular lesions,
                Dermatofibroma, Acne

Rationale for clinicians: at the operating point in this repo, per-class
melanoma *recall* is ~42%. Aggregating P(concerning) and referring on a low
threshold trades specificity for **sensitivity** — the correct direction for
cancer screening. The threshold is a tunable safety parameter the supervising
dermatologist should set.

    python clinical_triage.py --self-test
"""
from __future__ import annotations

import argparse

HIGH_RISK = {"Melanoma"}
MODERATE_RISK = {"Basal cell carcinoma", "Actinic keratoses"}
CONCERNING = HIGH_RISK | MODERATE_RISK

# Defaults — tune WITH the supervising clinician (see docs/clinical/VALIDATION_PROTOCOL.md)
DEFAULT_REFER_THRESHOLD = 0.10      # P(concerning) at/above this -> refer
DEFAULT_URGENT_THRESHOLD = 0.05     # P(melanoma) at/above this -> urgent


def triage(prob_dict: dict[str, float],
           refer_threshold: float = DEFAULT_REFER_THRESHOLD,
           urgent_threshold: float = DEFAULT_URGENT_THRESHOLD,
           abstain: bool = False, is_ood: bool = False) -> dict:
    """Map class probabilities -> a referral recommendation.

    Returns dict: p_melanoma, p_concerning, refer (bool), urgency, recommendation.
    """
    p_mel = float(prob_dict.get("Melanoma", 0.0))
    p_concerning = float(sum(prob_dict.get(c, 0.0) for c in CONCERNING))
    top_class = max(prob_dict, key=prob_dict.get) if prob_dict else None

    refer = bool(
        is_ood is False and (
            p_concerning >= refer_threshold or top_class in CONCERNING
        )
    )

    if is_ood:
        urgency = "n/a"
        rec = ("Image not recognised as skin — no triage made. "
               "Capture a clear, well-lit photo of the lesion.")
    elif abstain:
        urgency = "review"
        rec = ("Model is uncertain about this image. Recommend in-person "
               "dermatologist review and do not rely on the automated label.")
    elif p_mel >= urgent_threshold or top_class == "Melanoma":
        urgency = "urgent"
        rec = ("Findings cannot exclude melanoma. Recommend PROMPT in-person "
               "dermatologist evaluation (rule out melanoma).")
    elif refer:
        urgency = "soon"
        rec = ("Features may be consistent with a malignant/pre-malignant lesion. "
               "Recommend dermatologist evaluation.")
    else:
        urgency = "routine"
        rec = ("No high-risk features flagged by the model. Routine monitoring; "
               "seek review if the lesion changes (asymmetry, border, colour, "
               "diameter, evolution).")

    return {
        "p_melanoma": p_mel,
        "p_concerning": p_concerning,
        "top_class": top_class,
        "refer": refer,
        "urgency": urgency,                 # urgent | soon | routine | review | n/a
        "recommendation": rec,
    }


def _self_test():
    benign = {"Melanocytic nevi": 0.95, "Melanoma": 0.01, "Acne": 0.04}
    r = triage(benign)
    assert r["refer"] is False and r["urgency"] == "routine", r

    suspicious = {"Melanocytic nevi": 0.55, "Melanoma": 0.30,
                  "Basal cell carcinoma": 0.15}
    r = triage(suspicious)
    assert r["refer"] is True and r["urgency"] == "urgent", r

    low_mel_but_concerning = {"Melanocytic nevi": 0.80, "Melanoma": 0.02,
                              "Basal cell carcinoma": 0.12, "Acne": 0.06}
    r = triage(low_mel_but_concerning)
    assert r["refer"] is True, r            # aggregate concerning >= 0.10

    ood = {"Melanocytic nevi": 0.2, "Melanoma": 0.2}
    r = triage(ood, is_ood=True)
    assert r["refer"] is False and r["urgency"] == "n/a", r
    print("[self-test] PASSED — triage policy behaves as designed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Safety-first triage policy")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
    else:
        # demo on a sample vector
        demo = {"Melanocytic nevi": 0.6, "Melanoma": 0.25,
                "Basal cell carcinoma": 0.1, "Acne": 0.05}
        import json
        print(json.dumps(triage(demo), indent=2))
