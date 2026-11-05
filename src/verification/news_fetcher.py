"""
news_fetcher.py
===============
Fetch and parse article content from a URL using newspaper3k.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

try:
    from newspaper import Article
    HAS_NEWSPAPER = True
except ImportError:  # pragma: no cover
    HAS_NEWSPAPER = False
    logger.warning("newspaper3k not installed – URL fetching disabled.")


@dataclass
class FetchedArticle:
    url: str
    title: str = ""
    text: str = ""
    authors: List[str] = None  # type: ignore[assignment]
    publish_date: Optional[str] = None
    top_image: str = ""
    success: bool = False

    def __post_init__(self):
        if self.authors is None:
            self.authors = []


def fetch_article(url: str, language: str = "en") -> FetchedArticle:
    """
    Download and parse an article from *url*.

    Parameters
    ----------
    url:
        Full HTTP/HTTPS URL of the article.
    language:
        ISO 639-1 language code (default ``"en"``).

    Returns
    -------
    FetchedArticle
    """
    result = FetchedArticle(url=url)

    if not HAS_NEWSPAPER:
        logger.error("newspaper3k is required to fetch articles.")
        return result

    try:
        article = Article(url, language=language)
        article.download()
        article.parse()

        result.title = article.title or ""
        result.text = article.text or ""
        result.authors = article.authors or []
        result.publish_date = str(article.publish_date) if article.publish_date else None
        result.top_image = article.top_image or ""
        result.success = True

        logger.info(f"Fetched article: {result.title!r} ({len(result.text)} chars)")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to fetch {url}: {exc}")

    return result

