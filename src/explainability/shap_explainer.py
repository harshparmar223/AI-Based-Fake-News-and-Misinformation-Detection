"""
shap_explainer.py
=================
SHAP wrapper for tree-based and linear classifiers.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from loguru import logger

try:
    import shap
    HAS_SHAP = True
except ImportError:  # pragma: no cover
    HAS_SHAP = False
    logger.warning("shap not installed – SHAP explanations disabled.")


class SHAPExplainer:
    """
    Wraps SHAP's TreeExplainer (or LinearExplainer) for a fitted model.

    Parameters
    ----------
    model:
        A fitted sklearn-compatible tree/linear estimator.
    feature_names:
        Optional list of feature names for display.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None) -> None:
        if not HAS_SHAP:
            raise ImportError("Install shap: pip install shap")
        self.model = model
        self.feature_names = feature_names
        self._explainer: Optional[Any] = None

    def build(self, background_data: Optional[np.ndarray] = None) -> "SHAPExplainer":
        """Initialise the underlying SHAP explainer (call once after training)."""
        try:
            self._explainer = shap.TreeExplainer(self.model)
        except Exception:
            if background_data is None:
                raise ValueError(
                    "background_data required for non-tree models (LinearExplainer)."
                )
            self._explainer = shap.LinearExplainer(self.model, background_data)
        logger.info("SHAP explainer built successfully.")
        return self

    def explain(self, features: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features).
        """
        if self._explainer is None:
            raise RuntimeError("Call build() before explain().")
        shap_values = self._explainer.shap_values(features)
        # For binary classifiers, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            return shap_values[1]
        return shap_values

    def top_features(
        self, features: np.ndarray, n: int = 10
    ) -> List[tuple[str, float]]:
        """Return the *n* features with the largest mean |SHAP value|."""
        values = self.explain(features)
        mean_abs = np.abs(values).mean(axis=0)
        indices = np.argsort(mean_abs)[::-1][:n]
        names = self.feature_names or [str(i) for i in range(len(mean_abs))]
        return [(names[i], float(mean_abs[i])) for i in indices]

