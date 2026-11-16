"""
test_ml.py
==========
Unit tests for ML trainer, evaluator, and ensemble predictor.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.ml.trainer import train_classical_model
from src.ml.evaluator import evaluate_model
from src.ml.predictor import EnsemblePredictor, PredictionResult
from src.preprocessing.feature_extractor import TFIDFFeatureExtractor


class TestTrainerAndEvaluator:
    @pytest.fixture
    def dummy_data(self):
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            random_state=42,
        )
        return X, y

    def test_train_svm(self, dummy_data):
        X, y = dummy_data
        clf = train_classical_model(X, y, model_type="svm")
        assert hasattr(clf, "predict")
        assert hasattr(clf, "predict_proba")
        report = evaluate_model(clf, X, y)
        assert report.accuracy > 0.6
        assert report.f1 > 0.6

    def test_train_random_forest(self, dummy_data):
        X, y = dummy_data
        clf = train_classical_model(X, y, model_type="random_forest", model_params={"n_estimators": 10})
        assert hasattr(clf, "predict")
        report = evaluate_model(clf, X, y)
        assert report.accuracy > 0.6

    def test_train_logistic(self, dummy_data):
        X, y = dummy_data
        clf = train_classical_model(X, y, model_type="logistic")
        assert hasattr(clf, "predict")
        report = evaluate_model(clf, X, y)
        assert report.accuracy > 0.6

    def test_invalid_model_type(self, dummy_data):
        X, y = dummy_data
        with pytest.raises(ValueError):
            train_classical_model(X, y, model_type="invalid_type")


class TestEnsemblePredictor:
    def test_load_default_and_predict(self):
        predictor = EnsemblePredictor.load_default()
        assert len(predictor.model_names) > 0

        text = "Government officials announced new bilateral trade agreements this afternoon."
        result = predictor.predict(text)
        assert isinstance(result, PredictionResult)
        assert result.label in ("REAL", "FAKE")
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.fake_probability <= 1.0

    def test_predict_proba_shape(self):
        predictor = EnsemblePredictor.load_default()
        texts = [
            "Breaking news alert on shocking scandal",
            "Scientific researchers published peer-reviewed findings today",
        ]
        probs = predictor.predict_proba(texts)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (2, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
