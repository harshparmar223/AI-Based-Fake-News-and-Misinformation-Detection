"""
predictor.py
============
Unified inference interface: ensemble of classical ML models (SVM, Random Forest, Logistic).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.feature_extractor import TFIDFFeatureExtractor
from src.ml.trainer import MODELS_DIR, load_model


@dataclass
class PredictionResult:
    label: str                  # "REAL" or "FAKE"
    confidence: float           # 0.0 – 1.0 confidence of predicted class
    fake_probability: float     # 0.0 – 1.0 raw probability of being FAKE
    model_scores: Dict[str, float]


class EnsemblePredictor:
    """
    Wraps one or more fitted classifiers into a single ensemble prediction endpoint.

    Parameters
    ----------
    model_names:
        Names of serialised models to load from the ``models/`` directory.
    feature_extractor:
        A fitted TFIDFFeatureExtractor instance.
    weights:
        Optional list of weights matching ``model_names`` order.
        Defaults to equal weighting.
    threshold:
        Probability threshold above which an article is classified as FAKE
        (default 0.5).
    """

    def __init__(
        self,
        model_names: List[str],
        feature_extractor: TFIDFFeatureExtractor,
        weights: Optional[List[float]] = None,
        threshold: float = 0.5,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.model_names = model_names
        self._models: Dict[str, Any] = {name: load_model(name) for name in model_names}
        self._weights = weights or [1.0 / len(model_names)] * len(model_names)

    @classmethod
    def load_default(
        cls,
        model_names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ) -> "EnsemblePredictor":
        """Load default trained models and TF-IDF feature extractor from models/ dir."""
        extractor = load_model("tfidf_vectorizer")
        available_models = []
        candidates = model_names or ["svm_model", "random_forest_model", "logistic_model"]
        for cand in candidates:
            if (MODELS_DIR / f"{cand}.pkl").exists():
                available_models.append(cand)

        if not available_models:
            raise FileNotFoundError("No trained models found in models/ directory. Run training first.")

        return cls(model_names=available_models, feature_extractor=extractor, weights=weights)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Compute class probabilities for a list of raw text inputs.
        Returns a 2D array of shape (n_samples, 2), where column 0 = Real, column 1 = Fake.
        Compatible with LIME text explainers.
        """
        cleaned_list = [clean_text(t) for t in texts]
        features = self.feature_extractor.transform(cleaned_list)

        all_probs: List[np.ndarray] = []
        for name, model in self._models.items():
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)
            else:
                preds = model.predict(features)
                fake_prob = preds.astype(float)
                real_prob = 1.0 - fake_prob
                probs = np.column_stack([real_prob, fake_prob])
            all_probs.append(probs)

        # Weighted average
        weighted = np.zeros_like(all_probs[0], dtype=float)
        for prob_matrix, w in zip(all_probs, self._weights):
            weighted += prob_matrix * w

        return weighted

    def predict(self, text: str) -> PredictionResult:
        """
        Run the ensemble on a single article text.

        Parameters
        ----------
        text:
            Raw (or pre-cleaned) article string.

        Returns
        -------
        PredictionResult
        """
        cleaned = clean_text(text)
        features = self.feature_extractor.transform([cleaned])

        scores: Dict[str, float] = {}
        for (name, model), w in zip(self._models.items(), self._weights):
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(features)[0, 1])
            else:
                prob = float(model.predict(features)[0])
            scores[name] = prob

        weighted_fake_prob = sum(
            scores[name] * w
            for (name, _), w in zip(self._models.items(), self._weights)
        )

        label = "FAKE" if weighted_fake_prob >= self.threshold else "REAL"
        confidence = weighted_fake_prob if label == "FAKE" else (1.0 - weighted_fake_prob)

        logger.debug(f"Ensemble result: {label} (confidence={confidence:.3f}, fake_p={weighted_fake_prob:.3f})")

        return PredictionResult(
            label=label,
            confidence=round(confidence, 4),
            fake_probability=round(weighted_fake_prob, 4),
            model_scores={k: round(v, 4) for k, v in scores.items()},
        )
