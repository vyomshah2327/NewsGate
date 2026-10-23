"""VADER: a fast rule-based baseline scorer."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from newsgate.config import SentimentConfig
from newsgate.sentiment.base import SentimentScorer, register_scorer


class VaderScorer(SentimentScorer):
    name = "vader"

    def __init__(self, analyzer=None):
        if analyzer is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # optional dependency

            analyzer = SentimentIntensityAnalyzer()
        self._analyzer = analyzer

    def score(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self._analyzer.polarity_scores(str(t))["compound"] for t in texts], dtype="float64")


@register_scorer("vader")
def _make_vader(cfg: SentimentConfig) -> SentimentScorer:
    return VaderScorer()
