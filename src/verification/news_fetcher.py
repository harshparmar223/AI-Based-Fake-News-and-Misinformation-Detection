"""
news_fetcher.py
===============
Fetch and parse article content from a URL using newspaper3k or BeautifulSoup fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re
import httpx
from loguru import logger

try:
    from newspaper import Article
    HAS_NEWSPAPER = True
except ImportError:  # pragma: no cover
    HAS_NEWSPAPER = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:  # pragma: no cover
    HAS_BS4 = False


@dataclass
class FetchedArticle:
    url: str
    title: str = ""
    text: str = ""
    authors: List[str] = None  # type: ignore[assignment]
    publish_date: Optional[str] = None
    top_image: str = ""
    success: bool = False
    error: Optional[str] = None

    def __post_init__(self):
        if self.authors is None:
            self.authors = []


def _fetch_with_bs4(url: str) -> FetchedArticle:
    """Fallback fetcher using httpx and BeautifulSoup."""
    result = FetchedArticle(url=url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 20]
        text = "\n\n".join(paragraphs)

        result.title = title
        result.text = text
        result.success = bool(text)
        return result
    except Exception as exc:
        result.error = str(exc)
        return result


def fetch_article(url: str, language: str = "en") -> FetchedArticle:
    """
    Download and parse an article from url using newspaper3k or bs4.
    """
    result = FetchedArticle(url=url)

    if HAS_NEWSPAPER:
        try:
            article = Article(url, language=language)
            article.download()
            article.parse()

            result.title = article.title or ""
            result.text = article.text or ""
            result.authors = article.authors or []
            result.publish_date = str(article.publish_date) if article.publish_date else None
            result.top_image = article.top_image or ""
            result.success = bool(result.text)

            if result.success:
                logger.info(f"Fetched article via newspaper: {result.title!r} ({len(result.text)} chars)")
                return result
        except Exception as exc:
            logger.warning(f"newspaper3k failed for {url} ({exc}); attempting BeautifulSoup fallback…")

    if HAS_BS4:
        bs_res = _fetch_with_bs4(url)
        if bs_res.success:
            logger.info(f"Fetched article via BeautifulSoup: {bs_res.title!r} ({len(bs_res.text)} chars)")
            return bs_res

    result.error = "Could not parse article from URL."
    return result
