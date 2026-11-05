"""
evaluator.py
============
Classification metrics: accuracy, F1, precision, recall, ROC-AUC, confusion matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: str

    def __str__(self) -> str:  # noqa: D105
        return (
            f"Accuracy : {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall   : {self.recall:.4f}\n"
            f"F1-Score : {self.f1:.4f}\n"
            f"ROC-AUC  : {self.roc_auc:.4f}\n\n"
            f"Confusion Matrix:\n{self.confusion_matrix}\n\n"
            f"Classification Report:\n{self.classification_report}"
        )


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> EvaluationReport:
    """
    Generate a full evaluation report for a fitted classifier.

    Parameters
    ----------
    model:
        A fitted sklearn-compatible estimator.
    X_test:
        Test feature matrix.
    y_test:
        True binary labels.

    Returns
    -------
    EvaluationReport
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    report = EvaluationReport(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_prob),
        confusion_matrix=confusion_matrix(y_test, y_pred),
        classification_report=classification_report(
            y_test, y_pred, target_names=["Real", "Fake"]
        ),
    )

    logger.info(f"\n{report}")
    return report

