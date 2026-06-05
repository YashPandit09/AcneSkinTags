"""
Longitudinal Acne Treatment Analysis

The Derm-X image classifier answers "what is this?" for a single photo. This
companion analysis answers "is the treatment working?" over time — the question
a real acne patient actually cares about.

Using `data/sim_acne.csv` (10 patients tracked daily under three regimens —
Baseline / Antibiotics / Cream), we:
  1. plot mean severity trajectories per treatment,
  2. fit a linear mixed-effects model (random intercept per patient) to estimate
     each treatment's effect vs Baseline,
  3. plot the per-treatment severity distributions.

Outputs -> evaluation/acne_*.png  and a printed effect table.

    python analysis/acne_longitudinal.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CSV = os.path.join(REPO, "data", "sim_acne.csv")
OUT = os.path.join(REPO, "evaluation")


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df.rename(columns={df.columns[0]: "row"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def trajectories(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for treat, g in df.groupby("treatment"):
        m = g.groupby("day")["AcneSeverity"].mean()
        ax.plot(m.index, m.values, marker="o", ms=3, label=treat)
    ax.set(xlabel="day in block", ylabel="mean acne severity (0-3)",
           title="Mean acne-severity trajectory by treatment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "acne_trajectories.png"), dpi=150)
    plt.close(fig)


def distributions(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    order = [t for t in ["Baseline", "Antibiotics", "Cream"] if t in df.treatment.values]
    ax.boxplot([df[df.treatment == t]["AcneSeverity"] for t in order])
    ax.set_xticklabels(order)
    ax.set(ylabel="acne severity (0-3)", title="Severity distribution by treatment")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "acne_distributions.png"), dpi=150)
    plt.close(fig)


def mixed_effects(df: pd.DataFrame):
    import statsmodels.formula.api as smf

    df = df.copy()
    df["treatment"] = pd.Categorical(
        df["treatment"], categories=["Baseline", "Antibiotics", "Cream"])
    res = smf.mixedlm("AcneSeverity ~ C(treatment)", df,
                      groups=df["patient_id"]).fit(method="lbfgs")
    print(res.summary())
    print("\nTreatment effect vs Baseline (negative = improvement):")
    for name in res.params.index:
        if name.startswith("C(treatment)"):
            t = name.split("T.")[-1].rstrip("]")
            print(f"  {t:<12} delta_severity = {res.params[name]:+.3f}  "
                  f"(p = {res.pvalues[name]:.3g})")
    return res


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = load()
    print(f"Loaded {len(df)} rows | {df['patient_id'].nunique()} patients | "
          f"treatments={sorted(df['treatment'].unique())}")
    trajectories(df)
    distributions(df)
    try:
        mixed_effects(df)
    except Exception as e:
        print(f"[mixed-effects skipped: {e}]")
    print(f"\nFigures -> {OUT}")


if __name__ == "__main__":
    main()
