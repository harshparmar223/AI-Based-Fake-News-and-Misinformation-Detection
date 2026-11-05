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

### Phase 1 – Planning and Repository Setup
📅 **27–28 October 2026** · **All members**

- Understand the project objectives.
- Finalise features and responsibilities.
- Create the GitHub repository.
- Set up the initial folder structure.
- Install Python and required development tools.

**Deliverable:** Project plan and GitHub repository.

---

### Phase 2 – Dataset Collection
📅 **29–30 October 2026** · **Main: Member 1 · Supporting: Member 2**

- Collect the ISOT Fake News Dataset.
- Examine the real and fake news files.
- Check dataset labels and columns.
- Organise the dataset in the `data/raw/` directory.

**Deliverable:** Raw dataset ready for preprocessing.

---

### Phase 3 – Data Preprocessing
📅 **31 October – 2 November 2026** · **Main: Member 1 · Supporting: Member 2**

- Remove missing and duplicate records.
- Clean news text.
- Apply NLP preprocessing.
- Convert text into a suitable format.
- Split the dataset into training and testing sets.

**Deliverable:** Cleaned and prepared dataset.

---

### Phase 4 – Machine Learning Model Development
📅 **3–5 November 2026** · **Main: Member 2 · Supporting: Member 1**

- Implement TF-IDF vectorisation.
- Train the SVM model.
- Train the Random Forest model.
- Compare model performance.
- Evaluate the models.
- Save the selected model and vectorizer.

**Deliverable:** Working news classification model.

---

### Phase 5 – Source Credibility and Evidence Verification
📅 **6–10 November 2026** · **Main: Member 3 · Supporting: Member 5**

- Implement domain/source checking.
- Extract important claims or keywords.
- Connect to an available evidence-search service.
- Display relevant evidence and source links.
- Handle unavailable or inconclusive evidence.

**Deliverable:** Source and evidence verification modules.

---

### Phase 6 – Explainable AI
📅 **9–11 November 2026** · **Main: Member 5 · Supporting: Member 2**

- Integrate LIME.
- Identify influential words in predictions.
- Display explanations.
- Add an uncertainty indicator.
- Check that explanations correspond to the trained model.

**Deliverable:** Explainable prediction module.

---

### Phase 7 – Frontend and Database Development
📅 **10–12 November 2026** · **Main: Member 4 · Supporting: Member 2 and Member 5**

- Build the Streamlit interface.
- Add the news input area.
- Display prediction results.
- Add source and evidence information.
- Add explanation display.
- Implement database and feedback functionality.

**Deliverable:** Working frontend and database.

---

### Phase 8 – System Integration
📅 **13–14 November 2026** · **Main: Member 5 · All members**

- Connect the ML model to the frontend.
- Connect source and evidence checking.
- Integrate LIME explanations.
- Connect the database.
- Test the complete workflow.

**Deliverable:** Integrated application.

---

### Phase 9 – Testing and Evaluation
📅 **15–16 November 2026** · **All members**

- Test different news articles.
- Check correct and incorrect predictions.
- Test empty and invalid inputs.
- Test source-checking failures.
- Check application performance.
- Fix integration problems.

**Deliverable:** Tested application and evaluation results.

---

### Phase 10 – Final Improvements
📅 **17 November 2026** · **All members**

- Fix remaining bugs.
- Improve interface design.
- Verify result messages.
- Check database functionality.
- Prepare final screenshots.

**Deliverable:** Final working version.

---

### Phase 11 – Documentation and Presentation
📅 **17–19 November 2026** · **All members**

| Member | Documentation Responsibility |
|---|---|
| Member 1 | Dataset and preprocessing |
| Member 2 | ML algorithms and evaluation |
| Member 3 | Source and evidence verification |
| Member 4 | Frontend and database |
| Member 5 | Explainable AI, integration and architecture |

- Complete the project report.
- Prepare presentation slides.
- Add system screenshots.
- Prepare the project demonstration.
- Practise explaining individual modules.

**Final deliverables:** Project report · Presentation · GitHub repository · Working application.

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
