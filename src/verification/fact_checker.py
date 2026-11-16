"""
fact_checker.py
===============
Google Fact Check Tools API client with offline/fallback knowledge base.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
from loguru import logger


FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


@dataclass
class FactCheckClaim:
    text: str
    claimant: str
    claim_date: Optional[str]
    rating: str
    source: str
    url: str


@dataclass
class FactCheckResult:
    query: str
    claims: List[FactCheckClaim] = field(default_factory=list)
    verdict: str = "UNVERIFIED"  # VERIFIED_TRUE | VERIFIED_FALSE | UNVERIFIED
    source_type: str = "offline_db"  # "api" | "offline_db" | "none"


# ---------------------------------------------------------------------------
# Offline Fact-Check Knowledge Base (Curated Debunked & Verified Claims)
# ---------------------------------------------------------------------------
CURATED_FACT_CHECKS: List[dict] = [
    {
        "keywords": ["5g", "covid", "virus", "radiation"],
        "text": "5G mobile networks spread COVID-19 or coronavirus.",
        "claimant": "Social Media Posts",
        "rating": "False",
        "source": "World Health Organization / Reuters Fact Check",
        "url": "https://www.reuters.com/article/uk-factcheck-5g-coronavirus/false-5g-technology-spreads-covid-19-idUSKBN21P1ER",
        "verdict": "VERIFIED_FALSE",
    },
    {
        "keywords": ["microchip", "vaccine", "bill gates", "tracking"],
        "text": "COVID-19 vaccines contain microchips or tracking devices sponsored by Bill Gates.",
        "claimant": "Online Conspiracy Theories",
        "rating": "Pants on Fire",
        "source": "PolitiFact",
        "url": "https://www.politifact.com/factchecks/2020/may/05/facebook-posts/no-bill-gates-not-planning-microchip-vaccines/",
        "verdict": "VERIFIED_FALSE",
    },
    {
        "keywords": ["pope", "francis", "trump", "endorse"],
        "text": "Pope Francis endorsed Donald Trump for President.",
        "claimant": "WTOE 5 News",
        "rating": "False",
        "source": "Snopes",
        "url": "https://www.snopes.com/fact-check/pope-francis-donald-trump-endorsement/",
        "verdict": "VERIFIED_FALSE",
    },
    {
        "keywords": ["lemon", "hot water", "cancer", "cure"],
        "text": "Drinking hot water with lemon slices cures cancer and kills 10,000 times more cancer cells than chemotherapy.",
        "claimant": "Viral Chain Emails",
        "rating": "False",
        "source": "AFP Fact Check",
        "url": "https://factcheck.afp.com/lemon-water-cancer-cure-hoax",
        "verdict": "VERIFIED_FALSE",
    },
    {
        "keywords": ["earth", "flat", "nasa", "ice wall"],
        "text": "Earth is flat and NASA guards an ice wall around the perimeter.",
        "claimant": "Flat Earth Society",
        "rating": "False",
        "source": "NASA Science",
        "url": "https://science.nasa.gov/earth/",
        "verdict": "VERIFIED_FALSE",
    },
    {
        "keywords": ["who", "monkeypox", "pandemic", "lockdown"],
        "text": "WHO ordered mega-lockdowns for monkeypox / mpox in 2024.",
        "claimant": "Social Media Posts",
        "rating": "False",
        "source": "AP News Fact Check",
        "url": "https://apnews.com/article/fact-check-mpox-who-lockdowns-false",
        "verdict": "VERIFIED_FALSE",
    },
    {
        "keywords": ["nasa", "james webb", "big bang", "disproved"],
        "text": "James Webb Space Telescope images completely disproved the Big Bang theory.",
        "claimant": "Viral Blog Posts",
        "rating": "False",
        "source": "Space.com / NASA",
        "url": "https://www.space.com/james-webb-space-telescope-didnt-break-big-bang",
        "verdict": "VERIFIED_FALSE",
    },
    {
        "keywords": ["climate", "change", "global", "warming", "hoax"],
        "text": "Global scientific consensus concludes that human activities have unequivocally warmed the atmosphere and oceans.",
        "claimant": "IPCC Assessment Report",
        "rating": "True",
        "source": "IPCC / National Academy of Sciences",
        "url": "https://www.ipcc.ch/report/ar6/syr/",
        "verdict": "VERIFIED_TRUE",
    },
]


class GoogleFactChecker:
    """
    Wrapper around the Google Fact Check Tools API with built-in curated knowledge base.

    Parameters
    ----------
    api_key:
        Google API key with the Fact Check Tools API enabled.
        Falls back to the ``GOOGLE_FACT_CHECK_API_KEY`` env variable.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
        if not self.api_key:
            logger.info("No Google Fact Check API key provided – using curated fact-check database.")

    def _search_curated_db(self, query: str) -> List[FactCheckClaim]:
        """Match query against curated fact-checks using keyword overlap."""
        tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))
        matched = []

        for item in CURATED_FACT_CHECKS:
            kw = set(item["keywords"])
            overlap = tokens.intersection(kw)
            # Require at least 2 keyword matches or 50% match
            if len(overlap) >= 2 or (len(kw) <= 2 and len(overlap) >= 1):
                matched.append(
                    FactCheckClaim(
                        text=item["text"],
                        claimant=item["claimant"],
                        claim_date="2024-2026",
                        rating=item["rating"],
                        source=item["source"],
                        url=item["url"],
                    )
                )
        return matched

    def search(self, query: str, language_code: str = "en") -> FactCheckResult:
        """
        Search for claims related to *query*.
        Uses Google Fact Check API if key is present; otherwise searches curated DB.
        """
        result = FactCheckResult(query=query)

        # 1. Try Google Fact Check API if key is present
        if self.api_key:
            params = {
                "key": self.api_key,
                "query": query,
                "languageCode": language_code,
            }
            try:
                with httpx.Client(timeout=8.0) as client:
                    response = client.get(FACT_CHECK_API_URL, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        for item in data.get("claims", []):
                            for review in item.get("claimReview", []):
                                result.claims.append(
                                    FactCheckClaim(
                                        text=item.get("text", ""),
                                        claimant=item.get("claimant", "Unknown"),
                                        claim_date=item.get("claimDate"),
                                        rating=review.get("textualRating", ""),
                                        source=review.get("publisher", {}).get("name", ""),
                                        url=review.get("url", ""),
                                    )
                                )
                        result.source_type = "api"
            except Exception as exc:
                logger.warning(f"Fact-check API query error: {exc}")

        # 2. If no claims from API, search curated database
        if not result.claims:
            offline_claims = self._search_curated_db(query)
            if offline_claims:
                result.claims.extend(offline_claims)
                result.source_type = "offline_db"

        # 3. Derive overall verdict
        if result.claims:
            ratings = [c.rating.lower() for c in result.claims]
            if any(r in ("false", "mostly false", "pants on fire", "fake") for r in ratings):
                result.verdict = "VERIFIED_FALSE"
            elif any(r in ("true", "mostly true", "correct") for r in ratings):
                result.verdict = "VERIFIED_TRUE"

        return result
