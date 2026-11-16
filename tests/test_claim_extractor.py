"""
test_claim_extractor.py
=======================
Unit tests for claim and keyword extraction.
"""

from src.verification.claim_extractor import (
    extract_claims,
    extract_keywords,
    split_sentences,
)


def test_split_sentences():
    text = (
        "The President signed the landmark legislation on Thursday afternoon. "
        "The bill provides over fifty billion dollars for renewable energy initiatives. "
        "Critics however voiced deep concerns regarding inflation."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert all(len(s) > 15 for s in sentences)


def test_extract_keywords():
    text = "NASA scientists launched the James Webb telescope to explore distant galaxies."
    kw = extract_keywords(text, top_k=4)
    assert isinstance(kw, list)
    assert len(kw) <= 4
    assert any(k.lower() in ("nasa", "webb", "telescope", "galaxies") for k in kw)


def test_extract_claims():
    text = (
        "A massive storm system approached the eastern seaboard on Friday. "
        "Officials recorded winds exceeding 85 miles per hour across coastal counties. "
        "Over 200,000 households lost electrical power during the initial peak."
    )
    result = extract_claims(text, title="Severe Coastal Storm")
    assert result.primary_claim == "Severe Coastal Storm"
    assert len(result.claims) >= 1
    assert len(result.keywords) > 0
    assert len(result.search_query) > 0
