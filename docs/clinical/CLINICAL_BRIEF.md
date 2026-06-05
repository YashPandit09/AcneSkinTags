# Derm-X — One-Page Clinical Brief

> **Research prototype · not a medical device · decision-support only.**
> Prepared for a supervising dermatologist's review.

## What it is
An explainable deep-learning **screening aid** that classifies a skin-lesion photo
into 8 categories (7 HAM10000 lesion types + Acne) using MobileNetV2, and returns a
**calibrated** probability, a **Grad-CAM** evidence map, an **uncertainty** estimate,
and a **safety-first referral recommendation**. It runs on a consumer laptop GPU
(RTX 3050).

## Why it might be useful
- Triage support in low-resource / pre-screening settings.
- A teaching and research platform for trustworthy medical AI (calibration, OOD,
  uncertainty, explainability).
- A reproducible base for a dermatologist-supervised reader study.

## Honest performance (test set, n = 10,327)
| Metric | Value |
|---|---|
| Overall accuracy | **81.2%** |
| Acne precision / recall | 99.4% / 99.7% |
| **Melanoma sensitivity** | **42.2%** ⚠️ (known limitation) |
| Basal cell carcinoma sensitivity | 57.6% |
| Macro-avg F1 | 0.68 |

**The melanoma number is the headline risk and we are not hiding it.** As an
autonomous detector it is unsafe. The design therefore (1) only ever *supports* a
clinician, (2) **over-refers** on any malignant suspicion via an aggregated
P(concerning) threshold, and (3) **abstains** on uncertain or non-skin images.

## What makes it safer than a plain classifier
- **Calibrated confidence** (temperature scaling + ECE) — "80%" actually means 80%.
- **Out-of-distribution rejection** — refuses non-skin photos instead of guessing.
- **Predictive uncertainty** (MC-dropout) — flags "I'm not sure → see a clinician."
- **Safety-first triage** — refers on aggregated cancer probability, not top-1.
- **Grad-CAM** — every prediction shows the pixels it used.

## What I'm asking you for
1. **Clinical face-validity review** — are the classes, triage logic, and disclaimers
   sound?
2. **Set the safety thresholds** — referral and abstention cut-offs are clinical
   decisions, not engineering ones.
3. **Advise/supervise a small reader study** to measure real triage value
   (melanoma-sensitivity focus). Protocol drafted: `VALIDATION_PROTOCOL.md`.
4. **Guidance on data & ethics** for any move beyond public datasets.

## Key limitations (full list in the model card)
Trained on mostly lighter skin tones; image-level (not lesion-level) split;
single-source datasets; retrospective only; melanoma recall low; **not validated**.

*Full detail: [CLINICAL_MODEL_CARD.md](CLINICAL_MODEL_CARD.md) ·
Study design: [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md) ·
Live demo: `streamlit run app_clinical.py`.*
