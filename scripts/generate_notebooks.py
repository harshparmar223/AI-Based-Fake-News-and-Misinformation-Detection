"""
generate_notebooks.py
=====================
Generates documented, executable Jupyter notebooks for EDA, Preprocessing,
Classical ML, and Explainability.
"""

import json
from pathlib import Path


def create_nb(cells, path):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.13.0"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Created", path)


# 1. 01_EDA.ipynb
cells_eda = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 01 - Exploratory Data Analysis (EDA)\n",
            "**AI-Based Fake News & Misinformation Detection**\n",
            "\n",
            "This notebook explores the dataset used for training the detection pipeline:\n",
            "- Class balance (Real vs Fake)\n",
            "- Article text length distributions\n",
            "- Distinctive linguistic characteristics\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from pathlib import Path\n",
            "\n",
            "# Load processed datasets\n",
            "train_df = pd.read_csv('../data/processed/train.csv')\n",
            "test_df = pd.read_csv('../data/processed/test.csv')\n",
            "print(f'Training samples: {len(train_df)}, Testing samples: {len(test_df)}')\n",
            "train_df.head(3)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Class distribution\n",
            "print('Class distribution in training set:')\n",
            "print(train_df['label'].value_counts())\n",
            "\n",
            "train_df['label'].value_counts().plot(kind='bar', color=['#3b82f6', '#ef4444'], title='Class Distribution')\n",
            "plt.ylabel('Count')\n",
            "plt.show()\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Length statistics\n",
            "print('Summary of word counts:')\n",
            "print(train_df.groupby('label')['word_count'].describe())\n",
        ],
    },
]

# 2. 02_preprocessing.ipynb
cells_prep = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 02 - Text Preprocessing Pipeline\n",
            "Demonstration of text cleaning, tokenisation, stop-word filtering, and lemmatisation.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.append('..')\n",
            "from src.preprocessing.text_cleaner import clean_text, strip_html, remove_urls\n",
            "from src.preprocessing.tokenizer import tokenize, remove_stopwords, lemmatize, preprocess_tokens\n",
            "from src.preprocessing.feature_extractor import extract_linguistic_features\n",
            "\n",
            "sample = '<p>BREAKING: Check out https://fakenews.com for shocking cures!! Doctors STUNNED!</p>'\n",
            "print('Original:', sample)\n",
            "cleaned = clean_text(sample)\n",
            "print('Cleaned:', cleaned)\n",
            "tokens = preprocess_tokens(cleaned)\n",
            "print('Processed Tokens:', tokens)\n",
            "ling = extract_linguistic_features(sample)\n",
            "print('Linguistic Metrics:', ling)\n",
        ],
    },
]

# 3. 03_classical_ml.ipynb
cells_ml = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 03 - Machine Learning Model Baselines & Evaluation\n",
            "Comparative evaluation of SVM, Random Forest, and Logistic Regression models.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "import pandas as pd\n",
            "\n",
            "with open('../models/evaluation_results.json') as f:\n",
            "    metrics = json.load(f)\n",
            "\n",
            "df = pd.DataFrame(metrics).T\n",
            "print('Model Performance Comparison:')\n",
            "print(df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']])\n",
            "\n",
            "df[['accuracy', 'f1', 'roc_auc']].plot(kind='bar', figsize=(8, 4), title='Model Performance Comparison')\n",
        ],
    },
]

# 4. 04_explainability.ipynb
cells_xai = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 04 - Explainable AI (LIME) and Uncertainty\n",
            "Interpreting predictions at token level and quantifying uncertainty.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "sys.path.append('..')\n",
            "from src.ml.predictor import EnsemblePredictor\n",
            "from src.explainability.lime_explainer import LIMEExplainer\n",
            "from src.explainability.uncertainty import compute_uncertainty\n",
            "\n",
            "predictor = EnsemblePredictor.load_default()\n",
            "explainer = LIMEExplainer(num_features=6, num_samples=150)\n",
            "\n",
            "sample = 'Government secret project involving artificial weather control revealed by whistleblowers.'\n",
            "pred = predictor.predict(sample)\n",
            "print(f'Prediction: {pred.label} (Confidence: {pred.confidence:.1%})')\n",
            "\n",
            "unc = compute_uncertainty(pred.confidence, pred.fake_probability, pred.model_scores, sample)\n",
            "print(f'Uncertainty: {unc.level} (Score: {unc.uncertainty_score})')\n",
            "\n",
            "words = explainer.explain_structured(sample, predictor.predict_proba)\n",
            "print('LIME Word Attributions:')\n",
            "for w in words:\n",
            "    print(f\"  {w['word']:<15} {w['weight']:>+7.4f} ({w['influence']})\")\n",
        ],
    },
]


if __name__ == "__main__":
    out_dir = Path("notebooks")
    create_nb(cells_eda, out_dir / "01_EDA.ipynb")
    create_nb(cells_prep, out_dir / "02_preprocessing.ipynb")
    create_nb(cells_ml, out_dir / "03_classical_ml.ipynb")
    create_nb(cells_xai, out_dir / "04_explainability.ipynb")
