"""
source_credibility.py
=====================
Domain reputation scoring based on a curated allow/block list and heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Curated credibility lists (extend as needed)
# ---------------------------------------------------------------------------
HIGH_CREDIBILITY_DOMAINS: frozenset[str] = frozenset(
    {
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "bbc.co.uk",
        "theguardian.com",
        "nytimes.com",
        "washingtonpost.com",
        "wsj.com",
        "economist.com",
        "ft.com",
        "npr.org",
        "pbs.org",
        "theatlantic.com",
        "nature.com",
        "science.org",
    }
)

LOW_CREDIBILITY_DOMAINS: frozenset[str] = frozenset(
    {
        "worldnewsdailyreport.com",
        "empirenews.net",
        "nationalreport.net",
        "huzlers.com",
        "nowtheendbegins.com",
    }
)

SATIRICAL_DOMAINS: frozenset[str] = frozenset(
    {"theonion.com", "babylonbee.com", "clickhole.com", "waterfordwhispersnews.com"}
)


@dataclass
class CredibilityScore:
    domain: str
    score: float            # 0.0 (not credible) – 1.0 (highly credible)
    category: str           # HIGH | MEDIUM | LOW | SATIRE | UNKNOWN
    rationale: str


def _extract_domain(url: str) -> str:
    """Return the registered domain (without subdomains or www prefix)."""
    parsed = urlparse(url if url.startswith("http") else f"http://{url}")
    hostname = parsed.hostname or ""
    # Strip leading www.
    return re.sub(r"^www\.", "", hostname).lower()


def score_source(url: str) -> CredibilityScore:
    """
    Assign a credibility score to a news source URL.

    Parameters
    ----------
    url:
        Full article URL or bare domain.

    Returns
    -------
    CredibilityScore
    """
    domain = _extract_domain(url)

    if domain in HIGH_CREDIBILITY_DOMAINS:
        return CredibilityScore(domain, 0.95, "HIGH", "Known reputable news outlet.")

    if domain in SATIRICAL_DOMAINS:
        return CredibilityScore(domain, 0.1, "SATIRE", "Known satire publication.")

    if domain in LOW_CREDIBILITY_DOMAINS:
        return CredibilityScore(domain, 0.05, "LOW", "Known misinformation source.")

    # Heuristics for unknown domains
    score = 0.5
    rationale_parts = []

    if re.search(r"\d{4,}", domain):
        score -= 0.1
        rationale_parts.append("Domain contains unusual numbers.")

    if len(domain.split(".")) > 3:
        score -= 0.1
        rationale_parts.append("Deep subdomain structure.")

    tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
    if tld in ("gov", "edu", "ac"):
        score = min(score + 0.3, 1.0)
        rationale_parts.append("Government or academic TLD.")

    category = "HIGH" if score >= 0.75 else "MEDIUM" if score >= 0.45 else "LOW"
    rationale = " ".join(rationale_parts) or "No strong credibility signals found."

    return CredibilityScore(domain, round(score, 2), category, rationale)

