"""
tokenizer.py
============
NLTK / spaCy tokenisation, stop-word removal, and lemmatisation.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK assets on first import (silent if already present)
for _pkg in ("punkt", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)


@lru_cache(maxsize=1)
def _get_stop_words(language: str = "english") -> frozenset[str]:
    return frozenset(stopwords.words(language))


@lru_cache(maxsize=1)
def _get_lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


def tokenize(text: str) -> List[str]:
    """Return a list of lowercase word tokens (no punctuation tokens)."""
    return [t for t in word_tokenize(text.lower()) if t.isalpha()]


def remove_stopwords(tokens: List[str], language: str = "english") -> List[str]:
    """Filter out stop words from a token list."""
    sw = _get_stop_words(language)
    return [t for t in tokens if t not in sw]


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatise a list of tokens using WordNetLemmatizer."""
    lemmatizer = _get_lemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_tokens(
    text: str,
    remove_stops: bool = True,
    do_lemmatize: bool = True,
    language: str = "english",
) -> List[str]:
    """
    Full tokenisation pipeline: tokenise → (stop-word removal) → (lemmatise).

    Parameters
    ----------
    text:
        Pre-cleaned input string.
    remove_stops:
        Whether to remove stop words (default True).
    do_lemmatize:
        Whether to lemmatise tokens (default True).
    language:
        NLTK stop-word language (default ``"english"``).

    Returns
    -------
    List[str]
        Processed token list.
    """
    tokens = tokenize(text)
    if remove_stops:
        tokens = remove_stopwords(tokens, language)
    if do_lemmatize:
        tokens = lemmatize(tokens)
    return tokens

