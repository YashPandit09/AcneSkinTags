---
marp: true
paginate: true
title: "Derm-X — Clinical Research Pitch"
description: "Explainable, calibrated skin-lesion screening aid"
---

<!--
Render to PDF/PPTX with Marp:
  npx @marp-team/marp-cli docs/clinical/PITCH_DECK.md --pdf
  npx @marp-team/marp-cli docs/clinical/PITCH_DECK.md --pptx
Or use the "Marp for VS Code" extension (preview + export).
Speaker notes are in HTML comments under each slide.
-->

# Derm-X
### An explainable, calibrated skin-lesion **screening aid**

Yash Pandit · Research prototype

**Not a medical device · decision-support only**

<!-- Open by stating plainly this is research, not a product, and you want their guidance. -->

---

## The problem

- Skin cancer is common and **melanoma is deadly when caught late**.
- Specialist access is limited; many lesions are seen first by non-experts.
- AI triage *could* help — but most demos are **over-confident black boxes**
  that are unsafe in a clinic.

**Goal:** a transparent, calibrated, uncertainty-aware aid that knows its limits
and supports — never replaces — a clinician.

<!-- Frame the clinical need, then pivot to "trustworthy" as the differentiator. -->

---

## What it does

Upload a lesion image → the system returns:

1. **8-class** probabilities (7 HAM10000 lesion types + Acne), **calibrated**
2. A **Grad-CAM** map — the pixels it used (evidence, not a verdict)
3. An **uncertainty** estimate — and it **abstains** when unsure
4. A **safety-first triage** recommendation (refer / routine)

Runs on a **laptop GPU (RTX 3050)** — no cloud required.

---

## Honest performance (test n = 10,327)

| Class | Sensitivity (recall) |
|---|---|
| Acne | 99.7% |
| Melanocytic nevi | 95.1% |
| Basal cell carcinoma | 57.6% |
| Actinic keratoses | 52.9% |
| **Melanoma** | **42.2%** ⚠️ |

Overall accuracy **81.2%**, macro-F1 **0.68**.

<!-- Put the melanoma number on screen yourself. Credibility comes from owning it. -->

---

## The melanoma problem — and how we handle it

A detector that misses **~58% of melanomas** is **unsafe to trust alone.**
We do **not** hide this. Three design responses:

1. **Decision-support only** — the clinician decides, always.
2. **Safety-first triage** — refer on *aggregated* cancer probability, not top-1
   → trades specificity for **sensitivity** (right direction for screening).
3. **Abstention** — refuse uncertain / non-skin inputs instead of guessing.

> The ask: help us **set the thresholds** and **measure** the real safety gain.

---

## What makes it *trustworthy*, not just accurate

- **Calibration** (temperature scaling + ECE) — "80%" means 80%.
- **Out-of-distribution rejection** — a keyboard photo is refused, not diagnosed.
- **Uncertainty** (MC-dropout) — explicit "I'm not sure → see a clinician."
- **Explainability** (Grad-CAM) — every call is inspectable.
- **Reproducible** — open code, fixed config, automated tests.

<!-- This slide is the heart of the pitch: trustworthy ML, not a leaderboard score. -->

---

## Live demo

`streamlit run app_clinical.py`

- Persistent "research prototype" disclaimer
- Calibrated breakdown + Grad-CAM side-by-side
- Triage banner (URGENT / SOON / ROUTINE / REVIEW)
- Uncertainty + P(concerning) metrics

*(Try a clear lesion, a blurry one, and a non-skin image to show abstention.)*

---

## What I'm asking you for

1. **Clinical face-validity** — classes, triage logic, disclaimers: sound?
2. **Set the safety thresholds** — referral & abstention cut-offs are *clinical*.
3. **Supervise a small reader study** — melanoma-sensitivity focus.
4. **Ethics & data guidance** beyond public datasets.
5. (Hoped) **co-supervise / co-author** the resulting report.

Protocol drafted → `docs/clinical/VALIDATION_PROTOCOL.md`

---

## Limitations (full list in the model card)

- Melanoma sensitivity low at this operating point.
- Trained mostly on **lighter skin tones** (Fitzpatrick I–III).
- **Image-level** (not lesion-level) split → possible optimism.
- Single-source datasets; retrospective only; **not validated**.

**Nothing here is fit for patient care today — that's exactly what your input is for.**

---

## Thank you

**Derm-X** — explainable, calibrated, honest skin-lesion screening research.

Brief · Model card · Protocol → `docs/clinical/`
Demo → `streamlit run app_clinical.py`

*Research prototype · not a medical device · decision-support only.*
