"""
trainer.py
==========
Training loops for classical ML models: SVM, Random Forest, Logistic Regression, XGBoost.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

from src.preprocessing.feature_extractor import TFIDFFeatureExtractor
from src.ml.evaluator import EvaluationReport, evaluate_model


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_classical_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "svm",
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
        One of ``"svm"``, ``"random_forest"``, ``"logistic"``, ``"xgboost"``.
    model_params:
        Optional hyperparameter overrides.

    Returns
    -------
    Fitted sklearn-compatible estimator.
    """
    params = model_params or {}
    model_key = model_type.lower()

    if model_key in ("svm", "linear_svc"):
        base_svm = LinearSVC(
            C=params.get("C", 1.0),
            dual="auto",
            max_iter=params.get("max_iter", 3000),
            random_state=params.get("random_state", 42),
        )
        # Wrap with CalibratedClassifierCV to support predict_proba for confidence and LIME
        clf = CalibratedClassifierCV(estimator=base_svm, cv=3)

    elif model_key in ("random_forest", "rf"):
        default = dict(
            n_estimators=100,
            max_depth=25,
            random_state=42,
            n_jobs=-1,
        )
        default.update(params)
        clf = RandomForestClassifier(**default)

    elif model_key in ("logistic", "logreg"):
        default = dict(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )
        default.update(params)
        clf = LogisticRegression(**default)

    elif model_key == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed.")
        default = dict(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        default.update(params)
        clf = XGBClassifier(**default)

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    logger.info(f"Training {model_type} model on {len(y_train)} samples…")
    clf.fit(X_train, y_train)
    logger.success(f"{model_type} training complete.")
    return clf


def save_model(model: Any, name: str) -> Path:
    """Serialise a fitted model or vectorizer to disk with joblib."""
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


def train_and_evaluate_all(
    train_path: Path | str = "data/processed/train.csv",
    test_path: Path | str = "data/processed/test.csv",
    models_to_train: Optional[List[str]] = None,
) -> Dict[str, EvaluationReport]:
    """
    Full pipeline to fit TF-IDF vectorizer, train specified models, evaluate on test set,
    and persist models and evaluation reports.
    """
    models_to_train = models_to_train or ["svm", "random_forest", "logistic"]

    logger.info(f"Loading processed training and test data…")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_text = train_df["clean_text"].fillna("").tolist()
    y_train = train_df["target"].values
    X_test_text = test_df["clean_text"].fillna("").tolist()
    y_test = test_df["target"].values

    # Fit TF-IDF Extractor
    logger.info("Fitting TFIDFFeatureExtractor…")
    extractor = TFIDFFeatureExtractor(max_features=10_000, ngram_range=(1, 2), add_linguistic=True)
    X_train_feat = extractor.fit_transform(X_train_text)
    X_test_feat = extractor.transform(X_test_text)

    # Save feature extractor
    save_model(extractor, "tfidf_vectorizer")

    results: Dict[str, EvaluationReport] = {}
    metrics_summary: Dict[str, Dict[str, float]] = {}

    for m_type in models_to_train:
        clf = train_classical_model(X_train_feat, y_train, model_type=m_type)
        save_model(clf, f"{m_type}_model")

        report = evaluate_model(clf, X_test_feat, y_test)
        results[m_type] = report
        metrics_summary[m_type] = {
            "accuracy": round(float(report.accuracy), 4),
            "precision": round(float(report.precision), 4),
            "recall": round(float(report.recall), 4),
            "f1": round(float(report.f1), 4),
            "roc_auc": round(float(report.roc_auc), 4),
        }

    # Save metrics summary to json
    summary_path = MODELS_DIR / "evaluation_results.json"
    with open(summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.success(f"Saved evaluation metrics to {summary_path}")

    return results


if __name__ == "__main__":
    train_and_evaluate_all()
