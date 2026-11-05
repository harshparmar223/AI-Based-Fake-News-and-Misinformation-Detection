"""
test_verification.py
====================
Unit tests for source credibility scoring and URL parsing.
"""

import pytest

from src.verification.source_credibility import (
    CredibilityScore,
    _extract_domain,
    score_source,
)


class TestExtractDomain:
    def test_basic_url(self):
        assert _extract_domain("https://www.reuters.com/article/123") == "reuters.com"

    def test_no_scheme(self):
        assert _extract_domain("bbc.co.uk") == "bbc.co.uk"

    def test_strips_www(self):
        assert _extract_domain("https://www.example.com") == "example.com"


class TestScoreSource:
    def test_high_credibility_domain(self):
        result = score_source("https://www.reuters.com/article/xyz")
        assert result.category == "HIGH"
        assert result.score >= 0.9

    def test_satirical_domain(self):
        result = score_source("https://www.theonion.com/article")
        assert result.category == "SATIRE"

    def test_low_credibility_domain(self):
        result = score_source("https://worldnewsdailyreport.com/story")
        assert result.category == "LOW"

    def test_unknown_domain_returns_credibility_score(self):
        result = score_source("https://some-unknown-news-site.com/article")
        assert isinstance(result, CredibilityScore)
        assert 0.0 <= result.score <= 1.0

    def test_edu_domain_boosts_score(self):
        result = score_source("https://news.mit.edu/article")
        assert result.score > 0.5

