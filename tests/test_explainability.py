"""
test_explainability.py
======================
Unit tests for LIME, SHAP, and Uncertainty assessment.
"""

import numpy as np
import pytest

from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.uncertainty import compute_uncertainty
from src.ml.predictor import EnsemblePredictor


class TestUncertainty:
    def test_low_uncertainty_strong_consensus(self):
        assessment = compute_uncertainty(
            confidence=0.95,
            fake_probability=0.95,
            model_scores={"svm": 0.96, "rf": 0.94, "logreg": 0.95},
            article_text=" ".join(["word"] * 100),
            fact_check_verdict="VERIFIED_FALSE",
        )
        assert assessment.level == "LOW"
        assert assessment.uncertainty_score < 0.35
        assert not assessment.is_insufficient

    def test_high_uncertainty_short_text(self):
        assessment = compute_uncertainty(
            confidence=0.55,
            fake_probability=0.55,
            model_scores={"svm": 0.70, "rf": 0.35},
            article_text="Too short text",
        )
        assert assessment.level == "HIGH"
        assert assessment.is_insufficient
        assert any("Short text" in r for r in assessment.reasons)

    def test_model_disagreement_penalty(self):
        assessment = compute_uncertainty(
            confidence=0.60,
            fake_probability=0.60,
            model_scores={"svm": 0.85, "rf": 0.20},
            article_text=" ".join(["valid"] * 80),
        )
        assert any("disagreement" in r.lower() for r in assessment.reasons)


class TestLIMEExplainer:
    def test_lime_explain_structured(self):
        predictor = EnsemblePredictor.load_default()
        explainer = LIMEExplainer(num_features=5, num_samples=100)
        sample = "Miracle lemon cure completely obliterates cancer cells according to reports."
        structured = explainer.explain_structured(sample, predictor.predict_proba)

        assert isinstance(structured, list)
        assert len(structured) > 0
        first = structured[0]
        assert "word" in first
        assert "weight" in first
        assert first["influence"] in ("FAKE", "REAL")
