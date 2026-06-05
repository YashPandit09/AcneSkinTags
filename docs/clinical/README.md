# Clinical Presentation Package

Materials for presenting Derm-X to a supervising dermatologist for **research
advisor buy-in**. Everything here is framed as a **research prototype and
decision-support aid — not a medical device.**

| Document | Use it for |
|---|---|
| [CLINICAL_BRIEF.md](CLINICAL_BRIEF.md) | The 2-minute read to hand a busy clinician first. |
| [PITCH_DECK.md](PITCH_DECK.md) | The meeting slide deck (Marp → PDF/PPTX). |
| [CLINICAL_MODEL_CARD.md](CLINICAL_MODEL_CARD.md) | Full intended-use, honest metrics, limitations & risks. |
| [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md) | The reader-study design that turns "interesting" into "approved research". |

**Live demo:** `streamlit run app_clinical.py` (needs the trained
`best_model_8class_pytorch.pth`).

## Suggested flow for the meeting
1. Hand over the **brief**. 2. Walk the **deck**. 3. Run the **demo** (clear lesion →
blurry → non-skin to show abstention). 4. Open the **protocol** and agree the
**thresholds** + reader-study scope. 5. Point to the **model card** for the full
limitations.

## Render the deck
```bash
npx @marp-team/marp-cli docs/clinical/PITCH_DECK.md --pdf
# or use the "Marp for VS Code" extension
```

## Honesty guardrails (deliberate)
- The **melanoma sensitivity (42.2%)** is stated up front everywhere — owning it is
  what earns clinical trust.
- No claim of validation, safety, or regulatory clearance is made anywhere.
- `per_class_metrics.csv` (previously corrupted/mislabeled) has been
  **regenerated** to match `classification_report_8class.txt`.
