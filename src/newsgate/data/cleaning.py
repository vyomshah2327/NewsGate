"""Cleaning: normalise headlines and remove repeated or syndicated copies."""

from __future__ import annotations

import re

import pandas as pd

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")


def normalize_headline(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def deduplicate_news(news: pd.DataFrame) -> pd.DataFrame:
    """Drop empty headlines and exact repeats.

    Two rows are repeats when they share a ticker and a normalised headline (this removes
    syndicated copies from other publishers) or a ticker and a URL. The earliest copy is kept,
    so deduplication never moves an article later in time than its first appearance.
    """
    frame = news.copy()
    frame["_norm"] = frame["headline"].map(normalize_headline)
    frame = frame[frame["_norm"].str.len() > 0]
    frame = frame.sort_values(["published_at", "article_id"], kind="stable")
    frame = frame.drop_duplicates(subset=["ticker", "_norm"], keep="first")
    has_url = frame["url"].notna() & (frame["url"].astype(str).str.len() > 0)
    with_url = frame[has_url].drop_duplicates(subset=["ticker", "url"], keep="first")
    frame = pd.concat([with_url, frame[~has_url]]).sort_values(["published_at", "article_id"], kind="stable")
    return frame.drop(columns="_norm").reset_index(drop=True)
