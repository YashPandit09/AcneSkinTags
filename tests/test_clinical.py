"""
Tests for the safety-first clinical triage policy (no model/data needed).

    pytest tests/test_clinical.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clinical_triage import triage


def test_benign_is_routine():
    r = triage({"Melanocytic nevi": 0.95, "Melanoma": 0.01, "Acne": 0.04})
    assert r["refer"] is False
    assert r["urgency"] == "routine"


def test_high_melanoma_is_urgent():
    r = triage({"Melanocytic nevi": 0.55, "Melanoma": 0.30,
                "Basal cell carcinoma": 0.15})
    assert r["refer"] is True
    assert r["urgency"] == "urgent"


def test_aggregated_concerning_triggers_referral():
    # melanoma low, but BCC+AK push aggregate over the 0.10 threshold
    r = triage({"Melanocytic nevi": 0.80, "Melanoma": 0.02,
                "Basal cell carcinoma": 0.10, "Actinic keratoses": 0.03,
                "Acne": 0.05})
    assert r["refer"] is True


def test_ood_makes_no_recommendation():
    r = triage({"Melanocytic nevi": 0.2, "Melanoma": 0.2}, is_ood=True)
    assert r["refer"] is False
    assert r["urgency"] == "n/a"


def test_abstain_requests_review():
    r = triage({"Melanocytic nevi": 0.4, "Melanoma": 0.3,
                "Acne": 0.3}, abstain=True)
    assert r["urgency"] == "review"
