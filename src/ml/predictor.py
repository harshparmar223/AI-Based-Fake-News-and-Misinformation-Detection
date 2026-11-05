"""
predictor.py
============
Unified inference interface: ensemble of classical ML + transformer models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.feature_extractor import TFIDFFeatureExtractor
from src.ml.trainer import load_model


@dataclass
class PredictionResult:
    label: str            # "REAL" or "FAKE"
    confidence: float     # 0.0 – 1.0 probability of being FAKE
    model_scores: Dict[str, float]


class EnsemblePredictor:
    """
    Wraps one or more fitted classifiers into a single prediction endpoint.

    Parameters
    ----------
    model_names:
        Names of serialised models to load from the ``models/`` directory.
    feature_extractor:
        A *fitted* TFIDFFeatureExtractor instance.
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
        self._models: Dict[str, Any] = {name: load_model(name) for name in model_names}
        self._weights = weights or [1.0 / len(model_names)] * len(model_names)

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

        weighted_prob = sum(
            scores[name] * w
            for (name, _), w in zip(self._models.items(), self._weights)
        )

        label = "FAKE" if weighted_prob >= self.threshold else "REAL"
        logger.debug(f"Ensemble result: {label} ({weighted_prob:.3f})")

        return PredictionResult(
            label=label,
            confidence=weighted_prob,
            model_scores=scores,
        )

