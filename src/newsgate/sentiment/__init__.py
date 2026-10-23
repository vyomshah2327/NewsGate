"""Sentiment scorers (strategy pattern) and the scoring orchestrator."""

from newsgate.sentiment.base import SentimentScorer, available_scorers, create_scorer, register_scorer, score_news
from newsgate.sentiment import finbert, lexicon, vader  # noqa: F401  (registers the built-in scorers)

__all__ = ["SentimentScorer", "available_scorers", "create_scorer", "register_scorer", "score_news"]
