"""
fact_checker.py
===============
Google Fact Check Tools API client.
"""

from __future__ import annotations

import os
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


class GoogleFactChecker:
    """
    Thin wrapper around the Google Fact Check Tools API.

    Parameters
    ----------
    api_key:
        Google API key with the Fact Check Tools API enabled.
        Falls back to the ``GOOGLE_FACT_CHECK_API_KEY`` env variable.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
        if not self.api_key:
            logger.warning("No Google Fact Check API key provided – verification disabled.")

    def search(self, query: str, language_code: str = "en") -> FactCheckResult:
        """
        Search the Google Fact Check API for claims related to *query*.

        Parameters
        ----------
        query:
            Article headline or key claim to search.
        language_code:
            BCP-47 language code (default ``"en"``).

        Returns
        -------
        FactCheckResult
        """
        result = FactCheckResult(query=query)

        if not self.api_key:
            logger.debug("Skipping fact-check: no API key.")
            return result

        params = {
            "key": self.api_key,
            "query": query,
            "languageCode": language_code,
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(FACT_CHECK_API_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            logger.error(f"Fact-check API error: {exc}")
            return result

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

        if result.claims:
            ratings = [c.rating.lower() for c in result.claims]
            if any(r in ("false", "mostly false", "pants on fire") for r in ratings):
                result.verdict = "VERIFIED_FALSE"
            elif any(r in ("true", "mostly true", "correct") for r in ratings):
                result.verdict = "VERIFIED_TRUE"

        return result

