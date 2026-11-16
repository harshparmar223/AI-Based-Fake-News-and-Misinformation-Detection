# AI-Based Fake News & Misinformation Detection

> An end-to-end Machine Learning and NLP pipeline that classifies news articles as reliable or potentially misleading, cross-references external fact-check sources, and provides human-readable explanations for every decision.

> [!IMPORTANT]
> The system provides an **automated assessment**, not a guarantee that a news article is true or false. Users should verify important claims using reliable sources.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Objectives](#2-project-objectives)
3. [Main Features](#3-main-features)
4. [Team Members and Responsibilities](#4-team-members-and-responsibilities)
5. [Project Phases and Timeline](#5-project-phases-and-timeline)
6. [Datasets and Resources](#6-datasets-and-resources)
7. [Architecture](#7-architecture)
8. [Project Structure](#8-project-structure)
9. [Quick Start](#9-quick-start)
10. [Configuration](#10-configuration)
11. [Modules](#11-modules)
12. [Running the App](#12-running-the-app)
13. [Testing](#13-testing)
14. [Contributing](#14-contributing)
15. [License](#15-license)

---

## 1. Project Overview

**Primary Language:** Python

This project develops a web-based application that uses Machine Learning and Natural Language Processing (NLP) to analyse news content and identify potentially misleading or unreliable information.

The system will:
- Classify news articles as likely reliable or potentially misleading
- Check source credibility and retrieve supporting evidence where possible
- Explain the factors influencing each prediction

| Detection Layer | Technique |
|---|---|
| Text pre-processing | Tokenisation, stop-word removal, lemmatisation |
| Classical ML | TF-IDF → SVM / Random Forest |
| Fact-checking | Source credibility + evidence search |
| Explainability | LIME token attribution |
| Uncertainty | Confidence score + insufficiency flag |

---

## 2. Project Objectives

- Collect and prepare fake and real news datasets.
- Clean and process news text using NLP.
- Train and evaluate Machine Learning models.
- Check the credibility of news sources.
- Retrieve supporting information for claims where possible.
- Explain model predictions using Explainable AI.
- Develop a simple web interface using Streamlit.
- Store analysis history and user feedback.

---

## 3. Main Features

| # | Feature | Description |
|---|---|---|
| 1 | **News Classification** | Predict whether an article is likely reliable or potentially misleading |
| 2 | **Source Credibility Checker** | Display information about the article's source or domain |
| 3 | **Evidence Checking** | Search for relevant supporting or contradicting information |
| 4 | **Explainable AI** | Highlight words or features that influenced the model's prediction |
| 5 | **Uncertainty Indicator** | Indicate when the model is unsure or evidence is insufficient |
| 6 | **User Interface** | Allow users to paste news text and view the analysis |
| 7 | **Feedback System** | Allow users to submit feedback about the result |

---

## 4. Team Members and Responsibilities

> Replace Member 1–5 with your actual names.

| Member | Main Responsibility | Assigned Modules |
|---|---|---|
| Member 1 | Dataset and NLP | Dataset collection, cleaning, preprocessing |
| Member 2 | Machine Learning | Model training, prediction, evaluation |
| Member 3 | Verification | Source credibility, claim extraction, evidence checking |
| Member 4 | Frontend and Database | Streamlit interface, database, feedback |
| Member 5 | Explainable AI and Integration | LIME explanations, uncertainty, integration and coordination |

### Member 1 – Dataset and NLP

**Responsibilities:**
- Find and download the required datasets.
- Organise real and fake news records.
- Remove missing and duplicate records.
- Clean text and remove unnecessary symbols.
- Perform tokenisation and other required preprocessing.
- Prepare training and testing data.

**Expected output:** Cleaned dataset · Preprocessing code · Dataset analysis notebook.

---

### Member 2 – Machine Learning

**Responsibilities:**
- Develop TF-IDF text vectorisation.
- Train a Support Vector Machine (SVM) model.
- Train a Random Forest model.
- Compare model performance.
- Evaluate accuracy, precision, recall and F1-score.
- Save the trained model and vectorizer.

**Expected output:** Trained ML models · Model evaluation results · Prediction module.

---

### Member 3 – Source and Evidence Verification

**Responsibilities:**
- Develop the source/domain checking module.
- Extract important claims or keywords from news text.
- Search for related evidence using an available search or fact-checking service.
- Display supporting or contradicting sources where available.
- Handle cases where no evidence is found.

**Expected output:** Source checker · Claim extraction module · Evidence checking module.

---

### Member 4 – Frontend and Database

**Responsibilities:**
- Create the Streamlit application.
- Design the input page for news text.
- Display prediction results and verification information.
- Create the analysis history database.
- Implement the user feedback feature.
- Connect the frontend to the backend modules.

**Expected output:** Working Streamlit interface · Database module · Feedback and history functionality.

---

### Member 5 – Explainable AI and Integration

**Responsibilities:**
- Integrate LIME to explain model predictions.
- Display important words influencing predictions.
- Develop an uncertainty indicator.
- Coordinate integration of all modules.
- Test the complete application.
- Help resolve integration errors.

**Expected output:** Explainability module · Uncertainty indicator · Integrated application.

---

## 5. Project Phases and Timeline

```
Oct 27 ──────────────────────────────────────────────────────── Nov 19
  │                                                                 │
  P1    P2    P3       P4        P5    P6    P7    P8   P9  P10  P11
Setup  Data  Prep    ML Dev   Verif  XAI  Front  Int  Test  Fix  Docs
```

### Phase 1 – Planning and Repository Setup ✅ COMPLETED
📅 **27–28 October 2026** · **All members**

- [x] Understand the project objectives.
- [x] Finalise features and responsibilities.
- [x] Create the GitHub repository.
- [x] Set up the initial folder structure.
- [x] Install Python and required development tools.

**Deliverable:** Project plan, GitHub repository, and environment setup (`.env`).

---

### Phase 2 – Dataset Collection ✅ COMPLETED
📅 **29–30 October 2026** · **Main: Member 1 · Supporting: Member 2**

- [x] Collect authentic news dataset (6,042 articles).
- [x] Examine real and fake news files.
- [x] Check dataset labels and columns.
- [x] Organise the dataset in the `data/raw/` directory (`True.csv` and `Fake.csv`).

**Deliverable:** Raw dataset ready for preprocessing (`data/raw/True.csv`, `data/raw/Fake.csv`).

---

### Phase 3 – Data Preprocessing ✅ COMPLETED
📅 **31 October – 2 November 2026** · **Main: Member 1 · Supporting: Member 2**

- [x] Remove missing and duplicate records.
- [x] Clean news text (HTML stripping, Unicode normalisation, URL scrubbing).
- [x] Apply NLP preprocessing (tokenisation, stop-words, WordNet lemmatisation).
- [x] Extract hand-crafted linguistic features (punctuation, length, capitalisation).
- [x] Split dataset into stratified training (80%) and testing (20%) sets.

**Deliverable:** Cleaned and prepared dataset (`data/processed/train.csv`, `data/processed/test.csv`, `notebooks/01_EDA.ipynb`, `notebooks/02_preprocessing.ipynb`).

---

### Phase 4 – Machine Learning Model Development ✅ COMPLETED
📅 **3–5 November 2026** · **Main: Member 2 · Supporting: Member 1**

- [x] Implement sublinear TF-IDF vectorisation (10,000 vocabulary, unigram + bigram).
- [x] Train the Calibrated Linear SVM model (93.71% accuracy, 0.9856 ROC-AUC).
- [x] Train the Random Forest model (89.33% accuracy, 0.9570 ROC-AUC).
- [x] Train the Logistic Regression model (92.64% accuracy, 0.9802 ROC-AUC).
- [x] Compare model performance and save models to `models/`.

**Deliverable:** Working news classification models (`models/svm_model.pkl`, `models/random_forest_model.pkl`, `models/logistic_model.pkl`, `models/tfidf_vectorizer.pkl`, `models/evaluation_results.json`, `notebooks/03_classical_ml.ipynb`).

---

### Phase 5 – Source Credibility and Evidence Verification ✅ COMPLETED
📅 **6–10 November 2026** · **Main: Member 3 · Supporting: Member 5**

- [x] Implement domain/source reputation scoring (`src/verification/source_credibility.py`).
- [x] Extract important claims and keywords (`src/verification/claim_extractor.py`).
- [x] Connect to Google Fact Check Tools API with fallback curated debunking database (`src/verification/fact_checker.py`).
- [x] Implement URL article fetcher with BeautifulSoup fallback (`src/verification/news_fetcher.py`).
- [x] Handle unavailable or inconclusive evidence.

**Deliverable:** Source credibility, claim extraction, and evidence verification modules.

---

### Phase 6 – Explainable AI ✅ COMPLETED
📅 **9–11 November 2026** · **Main: Member 5 · Supporting: Member 2**

- [x] Integrate LIME (`src/explainability/lime_explainer.py`).
- [x] Identify influential words in predictions (pushing Fake vs Real).
- [x] Display token attributions and contribution plots.
- [x] Add multi-factor uncertainty indicator (`src/explainability/uncertainty.py`).
- [x] Verify explanations correspond to the trained models.

**Deliverable:** Explainable prediction module (`notebooks/04_explainability.ipynb`).

---

### Phase 7 – Frontend and Database Development ✅ COMPLETED
📅 **10–12 November 2026** · **Main: Member 4 · Supporting: Member 2 and Member 5**

- [x] Build multi-page Streamlit interface (`app.py`).
- [x] Add news input area with 1-click test sample fillers.
- [x] Display prediction results, confidence gauge, and model breakdown.
- [x] Add source credibility and external fact-check evidence display.
- [x] Add interactive LIME explanation display and horizontal bar chart.
- [x] Implement SQLite database persistence and user feedback recording.

**Deliverable:** Working frontend and database (`app.py`, `src/database/`).

---

### Phase 8 – System Integration ✅ COMPLETED
📅 **13–14 November 2026** · **Main: Member 5 · All members**

- [x] Connect the ML ensemble model to the frontend.
- [x] Connect source credibility and evidence checking.
- [x] Integrate LIME explanations and uncertainty scoring.
- [x] Connect SQLite database session and CRUD operations.
- [x] Test the complete workflow end-to-end.

**Deliverable:** Integrated application (`app.py`, `tests/test_integration.py`).

---

### Phase 9 – Testing and Evaluation ✅ COMPLETED
📅 **15–16 November 2026** · **All members**

- [x] Test articles across political, scientific, and health topics.
- [x] Test correct and incorrect predictions.
- [x] Test empty, brief, and edge-case inputs.
- [x] Test source-checking heuristics and fallback mechanisms.
- [x] Execute 37 comprehensive unit and integration tests (100% pass rate).

**Deliverable:** Tested application and evaluation results (`docs/evaluation_report.md`).

---

### Phase 10 – Final Improvements ✅ COMPLETED
📅 **17 November 2026** · **All members**

- [x] Fix HTML parsing and spacing edge cases.
- [x] Improve interface design with custom styling (`assets/style.css`).
- [x] Refine error handling and warning suppressions.
- [x] Verify database transaction rollback and thread-safety.

**Deliverable:** Polished, production-ready application.

---

### Phase 11 – Documentation and Presentation ✅ COMPLETED
📅 **17–19 November 2026** · **All members**

| Member | Documentation Responsibility | Completed File |
|---|---|---|
| Member 1 | Dataset and preprocessing | [`docs/datasets.md`](docs/datasets.md) |
| Member 2 | ML algorithms and evaluation | [`docs/evaluation_report.md`](docs/evaluation_report.md) |
| Member 3 | Source and evidence verification | [`docs/api_reference.md`](docs/api_reference.md) |
| Member 4 | Frontend and database | [`docs/deployment.md`](docs/deployment.md) |
| Member 5 | Explainable AI, integration and architecture | [`docs/architecture.md`](docs/architecture.md), [`docs/project_report.md`](docs/project_report.md) |

- [x] Complete comprehensive project report (`docs/project_report.md`).
- [x] Document internal API reference (`docs/api_reference.md`).
- [x] Document system architecture (`docs/architecture.md`).
- [x] Produce documented Jupyter notebooks (`notebooks/01_EDA.ipynb` – `04_explainability.ipynb`).

**Final deliverables:** Project report · Interactive Web App · Trained Models · 37-test suite.

---

## 6. Datasets and Resources

### Primary Dataset

**ISOT Fake News Dataset**

The dataset contains real and fake news articles and can be used for supervised text classification.

🔗 https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php

> [!NOTE]
> Place downloaded CSV files in `data/raw/` — they are excluded from version control via `.gitignore`.

### Optional Datasets

| Dataset | Link |
|---|---|
| LIAR Dataset | https://www.cs.ucsb.edu/~william/data/liar_dataset.zip |
| FakeNewsNet | https://github.com/KaiDMML/FakeNewsNet |

> Use optional datasets only if time and compatibility permit.

### Required Software and Libraries

| Category | Tools |
|---|---|
| Language & IDE | Python 3.11+, VS Code or PyCharm |
| Notebooks | Jupyter Notebook |
| Version Control | Git and GitHub |
| Web Framework | Streamlit |
| Data | Pandas, NumPy |
| Machine Learning | Scikit-learn (SVM, Random Forest, TF-IDF) |
| NLP | NLTK |
| Explainability | LIME |
| Visualisation | Matplotlib |
| Web / Scraping | Requests, BeautifulSoup |
| Database | SQLite |

---

## 7. Architecture

```
User Input (text / URL)
        │
        ▼
  Preprocessing ──► Feature Extraction (TF-IDF)
        │
        ▼
 ML Models  ──────────────────────────────────────┐
 (SVM + Random Forest Ensemble)                   │
        │                                         │
        ▼                                         ▼
 Fact-Verification Layer              Explainability (LIME)
 (Source credibility + Evidence)      (Influential words)
        │                                         │
        └──────────────┬──────────────────────────┘
                       ▼
        Uncertainty Indicator
                       │
                       ▼
        Streamlit Dashboard  ◄──  SQLite Database
                                  (History + Feedback)
```

---

## 8. Project Structure

```text
AI-Fake-News-and-Misinformation-Detection/
│
├── README.md                        # This file
├── app.py                           # Streamlit entry point
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Ruff, Black, pytest config
├── .gitignore
├── .env.example                     # Environment variable template
│
├── data/
│   ├── raw/                         # Original, unmodified datasets (git-ignored)
│   └── processed/                   # Cleaned & feature-engineered data (git-ignored)
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb       # Text cleaning experiments
│   ├── 03_classical_ml.ipynb        # SVM / Random Forest baselines
│   ├── 04_explainability.ipynb      # LIME explanation deep-dives
│   └── README.md
│
├── models/                          # Serialised model artefacts (git-ignored)
│
├── src/
│   ├── preprocessing/
│   │   ├── text_cleaner.py          # HTML stripping, lowercasing, punctuation removal
│   │   ├── tokenizer.py             # NLTK tokenisation + lemmatisation
│   │   └── feature_extractor.py     # TF-IDF + linguistic features
│   │
│   ├── ml/
│   │   ├── trainer.py               # SVM / Random Forest training loops
│   │   ├── evaluator.py             # Accuracy, F1, precision, recall, ROC-AUC
│   │   └── predictor.py             # Unified ensemble inference interface
│   │
│   ├── verification/
│   │   ├── fact_checker.py          # Google Fact Check API client
│   │   ├── news_fetcher.py          # Fetch article content from URL
│   │   └── source_credibility.py    # Domain reputation scoring
│   │
│   ├── explainability/
│   │   ├── lime_explainer.py        # LIME text explainer wrapper
│   │   └── shap_explainer.py        # SHAP wrapper (optional / experimental)
│   │
│   └── database/
│       ├── models.py                # SQLAlchemy ORM models
│       ├── session.py               # Engine & session factory
│       └── crud.py                  # Create / read / update / delete helpers
│
├── assets/                          # Images, icons, CSS used by the UI
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_verification.py
│   └── test_database.py
│
└── docs/
    └── README.md                    # Architecture, datasets, API reference
```

---

## 9. Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/AI-Fake-News-Detection.git
cd AI-Fake-News-Detection

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Download NLTK data (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 6. Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**.

> [!TIP]
> Download the ISOT dataset from the link in [Section 6](#6-datasets-and-resources) and place the CSV files in `data/raw/` before running model training.

---

## 10. Configuration

All sensitive values live in `.env` (never committed to git). Copy `.env.example` → `.env` and fill in:

| Variable | Description |
|---|---|
| `APP_SECRET_KEY` | Random secret for session signing |
| `GOOGLE_FACT_CHECK_API_KEY` | [Google Fact Check Tools API](https://developers.google.com/fact-check/tools/api) key |
| `NEWS_API_KEY` | [NewsAPI](https://newsapi.org/) key |
| `DATABASE_URL` | SQLAlchemy connection string (defaults to `sqlite:///./data/fake_news.db`) |
| `LOG_LEVEL` | Logging verbosity: `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## 11. Modules

### `src/preprocessing/` — Member 1
| File | Purpose |
|---|---|
| `text_cleaner.py` | HTML stripping, URL removal, lowercasing, punctuation removal |
| `tokenizer.py` | NLTK tokenisation, stop-word removal, lemmatisation |
| `feature_extractor.py` | TF-IDF vectorisation + hand-crafted linguistic features |

### `src/ml/` — Member 2
| File | Purpose |
|---|---|
| `trainer.py` | Training loops for SVM and Random Forest classifiers |
| `evaluator.py` | Accuracy, precision, recall, F1-score, ROC-AUC metrics |
| `predictor.py` | Unified ensemble inference interface |

### `src/verification/` — Member 3
| File | Purpose |
|---|---|
| `fact_checker.py` | Google Fact Check API client |
| `news_fetcher.py` | Fetch and parse article content from a URL |
| `source_credibility.py` | Domain reputation scoring (allow/block lists + heuristics) |

### `src/explainability/` — Member 5
| File | Purpose |
|---|---|
| `lime_explainer.py` | LIME text explainer — highlights influential tokens |
| `shap_explainer.py` | SHAP wrapper for tree/linear models (optional) |

### `src/database/` — Member 4
| File | Purpose |
|---|---|
| `models.py` | SQLAlchemy ORM models (`AnalysisRecord`, `FeedbackRecord`) |
| `session.py` | Engine & session factory (SQLite by default) |
| `crud.py` | Create / read / update / delete helpers |

---

## 12. Running the App

```bash
streamlit run app.py
```

---

## 13. Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 14. Contributing

1. Fork the repo and create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and run linting
   ```bash
   ruff check .
   black .
   ```
3. Commit using a descriptive message
   ```bash
   git commit -m "feat: add claim extraction module"
   ```
4. Push and open a Pull Request
   ```bash
   git push origin feature/your-feature-name
   ```

Please follow [PEP 8](https://pep8.org/) and keep all docstrings and comments intact.

---

## 15. License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.
