"""
test_preprocessing.py
=====================
Unit tests for text cleaning, tokenisation, and feature extraction.
"""

import pytest

from src.preprocessing.text_cleaner import (
    clean_text,
    remove_urls,
    strip_html,
)
from src.preprocessing.tokenizer import (
    lemmatize,
    preprocess_tokens,
    remove_stopwords,
    tokenize,
)


class TestTextCleaner:
    def test_strip_html_removes_tags(self):
        assert strip_html("<p>Hello <b>World</b></p>") == "Hello World"

    def test_remove_urls(self):
        text = "Check https://example.com for details"
        assert "https://" not in remove_urls(text)

    def test_clean_text_lowercase(self):
        result = clean_text("HELLO WORLD", lowercase=True)
        assert result == result.lower()

    def test_clean_text_no_html(self):
        result = clean_text("<div>Some text</div>")
        assert "<" not in result and ">" not in result

    def test_clean_text_empty_string(self):
        assert clean_text("") == ""


class TestTokenizer:
    def test_tokenize_returns_list(self):
        tokens = tokenize("The quick brown fox")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_tokenize_no_punctuation(self):
        tokens = tokenize("Hello, world! How are you?")
        assert all(t.isalpha() for t in tokens)

    def test_remove_stopwords(self):
        tokens = ["the", "quick", "brown", "fox"]
        filtered = remove_stopwords(tokens)
        assert "the" not in filtered
        assert "quick" in filtered

    def test_lemmatize(self):
        tokens = ["running", "dogs", "better"]
        lemmas = lemmatize(tokens)
        assert isinstance(lemmas, list)
        assert len(lemmas) == len(tokens)

    def test_preprocess_tokens_pipeline(self):
        tokens = preprocess_tokens("The cats are running quickly over the hills")
        # Stop words removed → "the" should not appear
        assert "the" not in tokens

