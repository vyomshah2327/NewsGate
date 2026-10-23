"""Plain-language evidence label shown next to a forecast."""

from __future__ import annotations


def evidence_label(n_articles: float, novel_share: float, has_history: bool = True) -> str:
    """Summarise how much fresh news stands behind a news-enhanced forecast.

    * ``none``     no article reached the cutoff, so the forecast rests on prices alone
    * ``weak``     a single article, or mostly repeated coverage
    * ``moderate`` several articles with some new information
    * ``strong``   several articles, mostly new information
    """
    if not has_history or n_articles <= 0:
        return "none"
    if n_articles < 2 or novel_share < 0.34:
        return "weak"
    return "strong" if novel_share >= 0.67 else "moderate"
