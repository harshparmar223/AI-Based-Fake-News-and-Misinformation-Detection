# Model Evaluation Report

## Executive Summary

Three distinct classical machine learning architectures were trained on 4,833 news articles and rigorously evaluated on an unseen held-out test partition of 1,209 articles. Features consisted of sublinear TF-IDF (1, 2 n-grams) capped at 10,000 vocabulary combined with 8 hand-crafted linguistic indicators scaled via MaxAbsScaler.

## Comparative Performance Metrics

| Classifier | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Support Vector Machine (LinearSVC + Platt Calibration)** | **93.71%** | **93.37%** | 94.28% | **93.82%** | **0.9856** |
| **Logistic Regression (L2 Regularized)** | 92.64% | 91.18% | **94.61%** | 92.86% | 0.9802 |
| **Random Forest Classifier (100 Trees)** | 89.33% | 87.56% | 91.99% | 0.8972 | 0.9570 |

---

## Detailed Model Analysis

### 1. Support Vector Machine (LinearSVC + CalibratedClassifierCV)
- **Strengths**: Highest overall accuracy (**93.71%**) and strongest discrimination (**0.9856 ROC-AUC**). Platt scaling via 3-fold cross validation provides well-calibrated probabilities between 0.0 and 1.0.
- **Confusion Matrix**:
  - True Real: 556
  - False Fake: 41
  - False Real: 35
  - True Fake: 577

### 2. Logistic Regression
- **Strengths**: Highest recall on Fake news (**94.61%**), minimizing false negatives. Highly efficient training and inference latency (< 5ms per sample).
- **Confusion Matrix**:
  - True Real: 541
  - False Fake: 56
  - False Real: 33
  - True Fake: 579

### 3. Random Forest Classifier
- **Strengths**: Robust to outliers and captures non-linear feature interactions without risk of gradient explosion.
- **Confusion Matrix**:
  - True Real: 517
  - False Fake: 80
  - False Real: 49
  - True Fake: 563

---

## Multi-Model Ensemble

The production inference pipeline ensembles the predictions of all three models using weighted soft voting:
$$\hat{P}(\text{Fake}) = w_{\text{svm}} P_{\text{svm}} + w_{\text{rf}} P_{\text{rf}} + w_{\text{lr}} P_{\text{lr}}$$

This ensures that edge-case predictions where one model exhibits high variance are moderated by the consensus of the remaining classifiers, dramatically reducing false alarm rates.
