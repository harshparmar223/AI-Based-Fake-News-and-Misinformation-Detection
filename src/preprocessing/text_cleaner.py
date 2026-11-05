"""
text_cleaner.py
===============
HTML stripping, Unicode normalisation, lowercasing, and punctuation removal.
"""

import re
import unicodedata
from html.parser import HTMLParser


class _HTMLStripper(HTMLParser):
    """Minimal HTMLParser subclass that discards tags and collects data."""

    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    @property
    def text(self) -> str:
        return " ".join(self._chunks)


def strip_html(text: str) -> str:
    """Remove HTML tags and return plain text."""
    parser = _HTMLStripper()
    parser.feed(text)
    return parser.text


def normalize_unicode(text: str) -> str:
    """Normalise Unicode to NFC and strip non-ASCII characters."""
    return unicodedata.normalize("NFC", text)


def remove_urls(text: str) -> str:
    """Remove http/https URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_punctuation(text: str, keep_apostrophes: bool = True) -> str:
    """Remove punctuation, optionally preserving apostrophes for contractions."""
    if keep_apostrophes:
        return re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"[^\w\s]", " ", text)


def collapse_whitespace(text: str) -> str:
    """Replace runs of whitespace (including newlines) with a single space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Full cleaning pipeline:
    HTML → Unicode normalise → remove URLs → remove punctuation →
    lowercase → collapse whitespace.

    Parameters
    ----------
    text:
        Raw input string.
    lowercase:
        Whether to convert text to lower case (default True).

    Returns
    -------
    str
        Cleaned text ready for tokenisation.
    """
    text = strip_html(text)
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    if lowercase:
        text = text.lower()
    return collapse_whitespace(text)

