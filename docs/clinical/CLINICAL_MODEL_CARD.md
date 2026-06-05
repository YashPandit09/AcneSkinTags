# Clinical Model Card — Derm-X (8-class skin-lesion screener)

> **Status: research prototype. NOT a medical device. NOT clinically validated.**
> For research discussion and decision-support experimentation only. Every output
> requires review by a qualified dermatologist.

*Format follows Mitchell et al., "Model Cards for Model Reporting" (2019).*

---

## 1. Model details
| | |
|---|---|
| Architecture | MobileNetV2 (ImageNet-pretrained) + dropout + linear head |
| Task | Single-label classification into 8 classes |
| Classes (idx 0–7) | Melanocytic nevi, Melanoma, Benign keratosis-like lesions, Basal cell carcinoma, Actinic keratoses, Vascular lesions, Dermatofibroma, Acne |
| Input | RGB image, 224×224, ImageNet-normalised |
| Output | Calibrated probability vector over 8 classes |
| Training | Transfer learning, Focal Loss (γ=2), AMP on an RTX 3050 laptop GPU |
| Safety add-ons | Temperature calibration, OOD rejection, MC-dropout uncertainty, safety-first triage (this PR series) |
| Version | v0.1 research prototype |

## 2. Intended use
- **Intended:** a research aid to (a) explore automated skin-lesion triage, (b) study
  calibration/uncertainty/explainability in dermatology AI, and (c) support a
  dermatologist-supervised reader study.
- **Intended users:** the student researcher and supervising clinician(s).
- **NOT intended:** autonomous diagnosis, patient-facing triage, treatment
  decisions, or any use without clinician oversight. Not for use on minors'
  clinical images outside an approved protocol.

## 3. Factors
Performance varies by **skin phototype, lesion type, body site, image quality,
lighting, and capture device**. Training data (HAM10000) is predominantly lighter
skin tones (Fitzpatrick I–III); performance on darker skin is expected to be
**worse** and is currently **unmeasured**.

## 4. Metrics & results
Held-out test set, **n = 10,327** images (source: `classification_report_8class.txt`).

| Class | Precision | **Recall (sensitivity)** | F1 | Support |
|---|---|---|---|---|
| Melanocytic nevi | 0.861 | 0.951 | 0.904 | 6,705 |
| **Melanoma** | 0.571 | **0.422** ⚠️ | 0.486 | 1,113 |
| Benign keratosis-like | 0.627 | 0.579 | 0.602 | 1,099 |
| Basal cell carcinoma | 0.807 | 0.576 | 0.672 | 514 |
| Actinic keratoses | 0.668 | 0.529 | 0.590 | 327 |
| Vascular lesions | 0.908 | 0.556 | 0.690 | 142 |
| Dermatofibroma | 0.741 | 0.374 | 0.497 | 115 |
| Acne | 0.994 | 0.997 | 0.995 | 312 |
| **Overall accuracy** | | **0.812** | | 10,327 |
| Macro avg | 0.772 | 0.623 | 0.680 | |

- **Confidence calibration:** reported probabilities are temperature-scaled; ECE is
  reported before/after by `calibration.py`. (Run after training to populate.)
- **Decision threshold:** the demo uses a safety-first triage policy
  (`clinical_triage.py`) that refers on aggregated P(concerning) ≥ 0.10 rather than
  top-1, to raise effective sensitivity.

> ⚠️ `per_class_metrics.csv` in the repo is **inconsistent/corrupted** (mislabeled
> supports, near-zero recalls) and must not be cited; regenerate or delete it.
> Use `classification_report_8class.txt`.

## 5. The melanoma-sensitivity caveat (read this first)
At this operating point the model **misses ~58% of melanomas** (recall 42.2%). As an
autonomous detector this is **unsafe**. The project mitigates this three ways, all of
which are exactly what we want the supervising dermatologist to scrutinise:
1. **Decision-support framing** — the tool never gives a verdict; a clinician does.
2. **Safety-first triage** — over-refers on any malignant suspicion (trades
   specificity for sensitivity).
3. **Abstention** — high-uncertainty / non-skin inputs are refused, not guessed.

## 6. Training & evaluation data
- **HAM10000** (Tschandl et al., 2018) — 7 lesion classes, dermoscopic images.
- **DermNet** "Acne and Rosacea" — the 8th (Acne) class.
- Split 70/10/20 train/val/test. *Limitation:* current split is image-level random;
  lesion/patient-level grouping is not enforced, so some optimism from near-duplicate
  leakage is possible. Documented as a validation-study item.

## 7. Ethical considerations, risks & mitigations
| Risk | Mitigation |
|---|---|
| **False-negative melanoma** (patient harm) | decision-support only; safety-first triage; clinician sign-off mandatory |
| **Over-confidence** | temperature calibration + reliability reporting |
| **Non-skin / garbage inputs** | OOD rejection + abstention |
| **Skin-tone bias** | disclosed; planned subgroup evaluation in the protocol |
| **Automation bias** (clinician over-trusts AI) | uncertainty shown; "evidence, not verdict" framing; Grad-CAM for scrutiny |
| **Data governance** | public, de-identified datasets only; no patient PHI stored |

## 8. Caveats and recommendations
This is a coursework/research prototype demonstrating a reproducible, calibrated,
explainable, uncertainty-aware pipeline — **not** a validated diagnostic tool. Before
any prospective use it requires: lesion-level split re-evaluation, prospective and
multi-site validation, subgroup (skin-tone) analysis, and a dermatologist-led reader
study (see [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md)).
