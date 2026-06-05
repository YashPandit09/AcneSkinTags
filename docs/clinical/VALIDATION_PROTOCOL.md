# Derm-X — Clinical Validation & Reader-Study Protocol (Draft v0.1)

> **Draft for discussion with the supervising dermatologist.** Not yet an approved
> study. No patient recruitment or prospective data collection is proposed at this
> stage — only retrospective evaluation on public/de-identified images.

| Field | Value |
|---|---|
| Title | Retrospective evaluation of an explainable 8-class skin-lesion screening aid, with a dermatologist reader sub-study |
| Sponsor / investigator | Yash Pandit (student researcher) |
| Clinical supervisor | *(dermatologist — to be confirmed)* |
| Version / date | 0.1 — to be dated on sign-off |
| Status | Draft pending clinical advisor input |

---

## 1. Background & rationale
Derm-X is a MobileNetV2 classifier (7 HAM10000 lesion classes + Acne) augmented with
calibration, out-of-distribution rejection, MC-dropout uncertainty, and a
safety-first triage policy. Standalone test accuracy is 81.2%, but **melanoma
sensitivity is only 42.2%** — unacceptable for autonomous use. This study evaluates
whether the system, used **as a decision-support aid**, is safe and potentially
useful, and quantifies the gap to be closed before any prospective use.

## 2. Objectives
**Primary**
- Estimate the **melanoma sensitivity** (and 95% CI) of the *AI-assisted triage
  policy* (aggregated P(concerning) referral) on an enriched test set.

**Secondary**
- Standalone per-class sensitivity/specificity and confusion structure.
- **Calibration** quality (ECE, reliability diagram) before/after temperature scaling.
- **Reader sub-study:** dermatologist diagnostic performance and confidence
  *unaided* vs *AI-assisted* (paired), plus inter-reader agreement (Cohen's/Fleiss' κ).
- **Explainability face-validity:** clinician rating of Grad-CAM localisation.
- **Robustness:** OOD rejection rate on non-skin images; abstention behaviour.
- **Subgroup:** performance by Fitzpatrick skin type where labels permit.

## 3. Design
Retrospective, reader-blinded, paired-comparison study on archived de-identified
images. No change to patient care; no prospective recruitment.

## 4. Data
- **Primary test data:** held-out HAM10000 + DermNet split (`classification_report_8class.txt`).
- **Enriched melanoma set:** assemble ≥120 biopsy-confirmed melanoma images plus a
  matched benign set from public archives (e.g., ISIC) to power the sensitivity
  estimate.
- **OOD probe set:** ~200 non-skin images (objects, other body regions) to measure
  rejection.
- **Reference standard:** dataset ground-truth labels (histopathology where
  available); the supervising dermatologist adjudicates ambiguous cases.
- **Governance:** public, de-identified data only; no PHI; storage and access per
  the institution's data policy. Any move to local/clinical images requires separate
  IRB/ethics approval.

## 5. Index test (the AI system)
Frozen model `best_model_8class_pytorch.pth` + the pipeline in this PR series:
TTA → temperature calibration → OOD gate → MC-dropout uncertainty → triage
(`safe_inference.py`, `clinical_triage.py`). All thresholds fixed **before** analysis
and recorded.

## 6. Reader sub-study procedure
- **Readers:** ≥3 dermatologists (or dermatology trainees + 1 attending).
- **Cases:** a balanced subset (e.g., 100–150 images) enriched for melanoma.
- **Two sessions ≥4 weeks apart** (wash-out) in counter-balanced order:
  *unaided* (image only) and *AI-assisted* (image + calibrated probabilities +
  Grad-CAM + triage). Readers record diagnosis, management (refer/routine), and
  confidence (0–100).
- Demo used for the assisted arm: `streamlit run app_clinical.py`.

## 7. Outcome measures
Sensitivity, specificity, PPV/NPV (with 95% CIs) for the malignant-vs-benign
decision; per-class metrics; ECE; reader sensitivity/specificity unaided vs assisted
(McNemar / paired); κ agreement; Grad-CAM usefulness (Likert); OOD rejection rate.

## 8. Sample size (indicative)
To estimate melanoma sensitivity of ~0.80 with a 95% CI half-width of ±0.07:
n ≈ 1.96²·0.8·0.2 / 0.07² ≈ **125 confirmed melanoma cases** (plus a comparably
sized benign group). The reader sub-study (~120 cases, 3 readers, paired) detects a
~10–15 percentage-point sensitivity change at conventional power; the supervisor
will finalise.

## 9. Statistical analysis
Proportions with Wilson 95% CIs; paired reader comparisons via McNemar; agreement via
κ; calibration via ECE and reliability diagrams. Pre-specified; analysis code in the
repo (`evaluate_8class.py`, `calibration.py`).

## 10. Success / stop criteria
- **Go (toward further research):** AI-assisted reader sensitivity ≥ unaided **without**
  a clinically meaningful specificity loss, and calibration ECE materially improved.
- **No-go / redesign:** AI assistance reduces reader sensitivity, induces automation
  bias (readers follow wrong AI calls), or melanoma sensitivity stays unsafe with no
  mitigation path.

## 11. Risks & ethics
Primary risk is **automation bias** leading a reader to miss a melanoma. Mitigations:
explicit "decision-support, not diagnosis" framing, displayed uncertainty,
safety-first over-referral, and clinician final say. Retrospective public data only;
no patient contact; no treatment influence.

## 12. Known limitations to disclose
Image-level (not lesion-level) split in current results; single-source training;
skin-tone imbalance; retrospective design; small reader panel; dataset-label (not
always histopathology) reference standard.

## 13. Roles & the ask of the supervisor
- Confirm clinical face-validity of classes, triage logic, and disclaimers.
- **Set referral / abstention thresholds** (clinical decisions).
- Adjudicate ambiguous reference labels.
- Advise sample size, recruitment of readers, and institutional ethics route.
- (Hoped) co-supervise / co-author the resulting report.

## 14. Timeline (indicative)
Wk 1–2 finalise protocol + thresholds · Wk 3–4 assemble enriched + OOD sets ·
Wk 5 standalone evaluation + calibration · Wk 6–9 reader sessions (wash-out) ·
Wk 10–12 analysis + write-up.

## References
Tschandl 2018 (HAM10000); Guo 2017 (calibration); Gal & Ghahramani 2016 (MC-dropout);
Hendrycks & Gimpel 2017 (MSP OOD); STARD 2015 (diagnostic-accuracy reporting).
