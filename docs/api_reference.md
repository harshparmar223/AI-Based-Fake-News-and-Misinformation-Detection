# API Reference

## Preprocessing (`src/preprocessing/`)

### `text_cleaner.py`
- `clean_text(text: str, lowercase: bool = True) -> str`: Full cleaning pipeline stripping HTML, normalizing Unicode, removing URLs, stripping punctuation, and collapsing whitespaces.
- `strip_html(text: str) -> str`: Remove HTML tags using HTMLParser.
- `remove_urls(text: str) -> str`: Strip web URLs and domains.
- `remove_punctuation(text: str, keep_apostrophes: bool = True) -> str`: Remove punctuation while preserving contractions.

### `tokenizer.py`
- `tokenize(text: str) -> List[str]`: Lowercase alphabetical word tokenization.
- `remove_stopwords(tokens: List[str], language: str = 'english') -> List[str]`: Stop-word filtering.
- `lemmatize(tokens: List[str]) -> List[str]`: WordNet lemmatizer.
- `preprocess_tokens(text: str) -> List[str]`: Full tokenizer pipeline.

### `feature_extractor.py`
- `extract_linguistic_features(text: str) -> LinguisticFeatures`: Computes character count, word count, sentence count, average word length, punctuation counts, capital letter ratio, unique word ratio.
- `TFIDFFeatureExtractor(max_features=10_000, ngram_range=(1,2), add_linguistic=True)`: Scikit-learn compatible vectorizer combining TF-IDF with scaled linguistic features.
  - `fit(texts)`
  - `transform(texts)`
  - `get_feature_names_out()`

---

## Machine Learning (`src/ml/`)

### `trainer.py`
- `train_classical_model(X_train, y_train, model_type="svm", model_params=None)`: Train SVM, Random Forest, Logistic Regression, or XGBoost.
- `save_model(model, name: str) -> Path`: Serialise estimator to `models/<name>.pkl`.
- `load_model(name: str) -> Any`: Deserialise estimator.
- `train_and_evaluate_all()`: Full end-to-end training and evaluation script.

### `evaluator.py`
- `evaluate_model(model, X_test, y_test) -> EvaluationReport`: Computes accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report.

### `predictor.py`
- `EnsemblePredictor(model_names, feature_extractor, weights=None, threshold=0.5)`:
  - `load_default()`: Factory method loading all active saved models and vectorizer.
  - `predict(text: str) -> PredictionResult`: Returns label, confidence, fake probability, individual model scores.
  - `predict_proba(texts: List[str]) -> np.ndarray`: Returns 2D class probabilities for LIME compatibility.

---

## Verification (`src/verification/`)

### `source_credibility.py`
- `score_source(url: str) -> CredibilityScore`: Evaluates URL domain credibility returning score (0.0–1.0), category (HIGH, MEDIUM, LOW, SATIRE), and rationale.

### `claim_extractor.py`
- `extract_claims(text: str, title: str = "", max_claims: int = 3) -> ExtractedClaims`: Extracts primary claim, supporting assertions, and search query keywords.

### `fact_checker.py`
- `GoogleFactChecker(api_key=None)`:
  - `search(query: str) -> FactCheckResult`: Queries Google Fact Check Tools API with fallback to curated debunking database.

### `news_fetcher.py`
- `fetch_article(url: str) -> FetchedArticle`: Scrapes article title, author, date, and body using newspaper3k with BeautifulSoup fallback.

---

## Explainability (`src/explainability/`)

### `lime_explainer.py`
- `LIMEExplainer(class_names=None, num_features=15, num_samples=300)`:
  - `explain(text, predict_fn) -> List[Tuple[str, float]]`: Token-level signed influence weights.
  - `explain_structured(text, predict_fn) -> List[dict]`: Structured tokens with color codes and labels.
  - `explain_html(text, predict_fn) -> str`: Interactive HTML visualization.

### `uncertainty.py`
- `compute_uncertainty(...) -> UncertaintyAssessment`: Combines model margins, inter-model discordance, text length sufficiency, and fact-check concordance.

---

## Database (`src/database/`)

### `models.py`
- `AnalysisRecord`: SQLAlchemy ORM model for storing article analyses.
- `FeedbackRecord`: SQLAlchemy ORM model for storing user corrections.

### `crud.py`
- `create_analysis(db, **kwargs) -> AnalysisRecord`
- `get_analysis(db, record_id: int) -> Optional[AnalysisRecord]`
- `list_analyses(db, limit=100, offset=0) -> List[AnalysisRecord]`
- `update_analysis_feedback(db, record_id: int, feedback: str)`
- `create_feedback(db, analysis_id: int, correct_label: str, comment=None) -> FeedbackRecord`
- `list_feedback(db, limit=500) -> List[FeedbackRecord]`
