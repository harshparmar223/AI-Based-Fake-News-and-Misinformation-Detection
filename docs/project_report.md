# Comprehensive Project Report
## AI-Based Fake News & Misinformation Detection (TruthLens)

---

### Abstract
In the modern digital information ecosystem, the rapid dissemination of unverified claims, fabricated narratives, and disinformation campaigns poses significant risks to public health, democratic institutions, and societal cohesion. This report presents **TruthLens**, an end-to-end Machine Learning and Natural Language Processing (NLP) framework for automated fake news detection, source credibility assessment, external fact-check cross-referencing, and model explainability. Trained on a balanced benchmark of 6,042 real and fake news articles, our multi-model ensemble (Support Vector Machine with Platt calibration, Random Forest, and Logistic Regression) achieves a peak test accuracy of **93.71%** and a **0.9856 ROC-AUC**. Furthermore, we integrate Local Interpretable Model-agnostic Explanations (LIME) to provide transparent, token-level attributions and introduce a multi-dimensional uncertainty scoring engine to flag inconclusive or low-substance submissions.

---

### 1. Introduction & Objectives
The goal of this project is to develop an interpretable, transparent, and multi-layered web platform that helps users assess news credibility. 
Key objectives achieved across project phases include:
1. **Dataset Pipeline**: Collecting, organizing, and preprocessing real and fake news datasets.
2. **Feature Engineering**: Extracting sublinear TF-IDF n-grams augmented with hand-crafted linguistic metrics (sensationalist punctuation, lexical diversity, uppercase letter ratio).
3. **Machine Learning Development**: Implementing, training, evaluating, and ensembling classical ML classifiers (SVM, Random Forest, Logistic Regression).
4. **Verification Layer**: Evaluating domain credibility and searching external fact-check records.
5. **Explainability & Uncertainty**: Visualizing influential words with LIME and calculating an uncertainty metric.
6. **Web Dashboard & Persistence**: Building a modern Streamlit user interface connected to a SQLite database for tracking analysis history and user feedback.

---

### 2. Methodology & Implementation

#### 2.1 NLP Preprocessing
Raw news text undergoes systematic cleaning:
- Stripping HTML artifacts via dedicated parser logic.
- Unicode NFC normalization.
- Hyperlink scrubbing via regular expressions.
- Contraction preservation and punctuation removal.
- Tokenization, stop-word removal, and WordNet lemmatization.

#### 2.2 Feature Representation
- **TF-IDF Vectorization**: Sublinear term frequency scaling with $(1, 2)$ n-grams capped at the top 10,000 most informative vocabulary items.
- **Linguistic Markers**: 8 hand-crafted indicators (character count, word count, sentence count, average word length, exclamation frequency, question frequency, capital letter ratio, unique word ratio) normalized using `MaxAbsScaler`.

#### 2.3 Model Training & Calibration
- **Linear Support Vector Classifier (LinearSVC)**: Trained using convex optimization and calibrated via 3-fold Platt scaling (`CalibratedClassifierCV`) to output well-calibrated class probabilities.
- **Random Forest**: 100 estimator trees with constrained maximum depth to prevent overfitting while capturing non-linear token interactions.
- **Logistic Regression**: High-efficiency linear benchmark with L2 regularization.
- **Ensemble Predictor**: Weighted soft-voting combining individual model probabilities.

#### 2.4 Explainability & Uncertainty Quantification
- **LIME Local Surrogate**: Generates locally perturbed text instances to determine exact signed feature attributions for words swaying predictions towards Real or Fake.
- **Uncertainty Score**: Derived from:
  $$\text{Uncertainty} = f(\text{Margin from 0.5}, \text{Model Disagreement}, \text{Text Brevity}, \text{Fact-Check Concordance})$$

---

### 3. Experimental Results

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Linear SVM (Calibrated)** | **93.71%** | **93.37%** | 94.28% | **93.82%** | **0.9856** |
| **Logistic Regression** | 92.64% | 91.18% | **94.61%** | 92.86% | 0.9802 |
| **Random Forest** | 89.33% | 87.56% | 91.99% | 0.8972 | 0.9570 |

---

### 4. System Validation & Verification
All 37 test specifications in the test suite pass with 100% success rate:
- Preprocessing unit tests (cleaning, tokenization, lemmatization).
- Machine learning tests (training, evaluation, ensemble prediction).
- Verification tests (domain credibility heuristics, claim extraction).
- Explainability tests (LIME attributions, uncertainty penalties).
- End-to-end integration tests (raw input to database record and feedback update).

---

### 5. Ethical Considerations & Limitations
- **Automated Advisory Only**: The platform is an advisory tool rather than an infallible arbiter of truth.
- **Evolving Language**: Misinformation phrasing changes over time; user feedback logged in the database allows continuous retraining.
