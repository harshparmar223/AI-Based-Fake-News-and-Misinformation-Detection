"""
trainer.py
==========
Training loops for classical ML and transformer-based models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing.feature_extractor import TFIDFFeatureExtractor
from src.ml.evaluator import evaluate_model


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_classical_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "xgboost",
    model_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Train a classical ML classifier.

    Parameters
    ----------
    X_train:
        Feature matrix (TF-IDF + linguistic).
    y_train:
        Binary labels (1 = fake, 0 = real).
    model_type:
        One of ``"xgboost"`` or ``"logistic"``.
    model_params:
        Optional hyperparameter overrides.

    Returns
    -------
    Fitted sklearn-compatible estimator.
    """
    params = model_params or {}

    if model_type == "xgboost":
        default = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        default.update(params)
        clf = XGBClassifier(**default)
    elif model_type == "logistic":
        default = dict(max_iter=1000, C=1.0, solver="lbfgs", random_state=42, n_jobs=-1)
        default.update(params)
        clf = LogisticRegression(**default)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    logger.info(f"Training {model_type} model on {len(y_train)} samples…")
    clf.fit(X_train, y_train)
    logger.success(f"{model_type} training complete.")
    return clf


def save_model(model: Any, name: str) -> Path:
    """Serialise a fitted model to disk with joblib."""
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path


def load_model(name: str) -> Any:
    """Deserialise a model from disk."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    return joblib.load(path)

