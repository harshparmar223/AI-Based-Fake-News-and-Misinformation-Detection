"""
claim_extractor.py
==================
Extract important claims, factual assertions, and search keywords from article text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExtractedClaims:
    primary_claim: str
    claims: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    search_query: str = ""


# Common stop words to exclude when selecting key search terms
COMMON_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "in", "on", "at",
    "to", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "from", "up", "down", "in", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "says", "said", "saying", "according", "also", "reported"
})


def split_sentences(text: str) -> List[str]:
    """Split text into individual clean sentences."""
    # Split on sentence ending punctuation
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for s in raw_sentences:
        s_clean = re.sub(r"\s+", " ", s).strip()
        # Keep non-empty sentences of reasonable length
        if len(s_clean) >= 20 and len(s_clean.split()) >= 4:
            sentences.append(s_clean)
    return sentences


def extract_keywords(text: str, top_k: int = 6) -> List[str]:
    """Extract informative keywords from text using frequency and capitalisation cues."""
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b", text)
    freq: dict[str, int] = {}
    capitalized: dict[str, int] = {}

    for w in words:
        lower = w.lower()
        if lower in COMMON_STOPWORDS:
            continue
        freq[lower] = freq.get(lower, 0) + 1
        if w[0].isupper():
            capitalized[lower] = capitalized.get(lower, 0) + 1

    # Boost words that are capitalized (often proper nouns / entities)
    scored = []
    for w, count in freq.items():
        score = count + (capitalized.get(w, 0) * 1.5)
        scored.append((w, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scored[:top_k]]


def extract_claims(text: str, title: str = "", max_claims: int = 3) -> ExtractedClaims:
    """
    Identify key claims from text and generate search queries for verification.

    Parameters
    ----------
    text:
        Full body of the news article.
    title:
        Article title or headline (if available).
    max_claims:
        Maximum number of distinct claims to extract.

    Returns
    -------
    ExtractedClaims
    """
    sentences = split_sentences(text)

    # If title is provided, it is often the most direct summary of the main claim
    primary_claim = title.strip() if title.strip() else ""

    claim_candidates = []
    # Rank sentences by presence of assertion markers (numbers, quotes, entity names)
    for sent in sentences:
        score = 0
        # Numbers / statistics often indicate factual claims
        if re.search(r"\b\d+(?:\.\d+)?%?\b", sent):
            score += 2
        # Capitalized words (proper nouns)
        cap_count = sum(1 for w in sent.split() if w and w[0].isupper())
        score += min(cap_count, 5)
        # Sentence length sweet spot for claims (10 to 30 words)
        word_count = len(sent.split())
        if 10 <= word_count <= 30:
            score += 2

        claim_candidates.append((sent, score))

    claim_candidates.sort(key=lambda x: x[1], reverse=True)
    selected_claims = [s for s, _ in claim_candidates[:max_claims]]

    if not primary_claim:
        primary_claim = selected_claims[0] if selected_claims else text[:120].strip()

    keywords = extract_keywords(title + " " + (selected_claims[0] if selected_claims else text[:300]))
    search_query = " ".join(keywords[:5]) if keywords else primary_claim[:80]

    return ExtractedClaims(
        primary_claim=primary_claim,
        claims=selected_claims,
        keywords=keywords,
        search_query=search_query,
    )
