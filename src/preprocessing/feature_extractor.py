"""
feature_extractor.py
====================
TF-IDF vectorisation, linguistic features, and readability scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.tokenizer import preprocess_tokens


@dataclass
class LinguisticFeatures:
    """Container for hand-crafted linguistic features."""

    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    exclamation_count: int = 0
    question_count: int = 0
    capital_ratio: float = 0.0
    unique_word_ratio: float = 0.0


def extract_linguistic_features(text: str) -> LinguisticFeatures:
    """Extract hand-crafted linguistic features from raw text."""
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]

    char_count = len(text)
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0
    exclamation_count = text.count("!")
    question_count = text.count("?")
    capital_letters = sum(1 for c in text if c.isupper())
    capital_ratio = capital_letters / char_count if char_count else 0.0
    unique_words = set(w.lower() for w in words)
    unique_word_ratio = len(unique_words) / word_count if word_count else 0.0

    return LinguisticFeatures(
        char_count=char_count,
        word_count=word_count,
        sentence_count=sentence_count,
        avg_word_length=float(avg_word_length),
        exclamation_count=exclamation_count,
        question_count=question_count,
        capital_ratio=capital_ratio,
        unique_word_ratio=unique_word_ratio,
    )


class TFIDFFeatureExtractor:
    """
    Wrapper around sklearn's TfidfVectorizer with optional linguistic features.

    Parameters
    ----------
    max_features:
        Vocabulary size cap (default 50 000).
    ngram_range:
        N-gram range (default ``(1, 2)``).
    add_linguistic:
        Whether to append hand-crafted linguistic features (default True).
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        add_linguistic: bool = True,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.add_linguistic = add_linguistic
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            tokenizer=preprocess_tokens,
            token_pattern=None,
        )
        self._fitted = False

    def fit(self, texts: List[str]) -> "TFIDFFeatureExtractor":
        self._vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        tfidf_matrix = self._vectorizer.transform(texts).toarray()
        if self.add_linguistic:
            ling_features = np.array(
                [list(vars(extract_linguistic_features(t)).values()) for t in texts],
                dtype=float,
            )
            return np.hstack([tfidf_matrix, ling_features])
        return tfidf_matrix

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

