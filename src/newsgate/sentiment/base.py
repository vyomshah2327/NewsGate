"""Scorer interface, registry (factory) and the orchestrator that scores a news frame."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from newsgate.config import SentimentConfig
from newsgate.data.cleaning import normalize_headline
from newsgate.logging_utils import get_logger

log = get_logger(__name__)


class SentimentScorer(ABC):
    """Maps headlines to a polarity in [-1, 1] (negative to positive)."""

    name: str = "scorer"

    @abstractmethod
    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Return one polarity per text, in the same order."""


_REGISTRY: dict[str, Callable[[SentimentConfig], SentimentScorer]] = {}


def register_scorer(name: str):
    """Decorator that registers a factory ``(SentimentConfig) -> SentimentScorer``."""

    def decorator(factory: Callable[[SentimentConfig], SentimentScorer]):
        _REGISTRY[name] = factory
        return factory

    return decorator


def available_scorers() -> list[str]:
    return sorted(_REGISTRY)


def create_scorer(name: str, cfg: SentimentConfig) -> SentimentScorer:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown sentiment scorer '{name}'. Available: {available_scorers()}")
    return _REGISTRY[name](cfg)


def score_news(
    news: pd.DataFrame,
    scorers: Sequence[SentimentScorer],
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Add one ``sent_<name>`` column per scorer.

    Scoring is done once per distinct normalised headline (syndicated copies share a score)
    and, when ``cache_dir`` is given, stored on disk per scorer so reruns are cheap.
    """
    out = news.copy()
    norm = out["headline"].map(normalize_headline)
    unique = pd.Series(norm.unique(), name="norm")
    first_text = out.assign(_norm=norm).drop_duplicates("_norm").set_index("_norm")["headline"]

    for scorer in scorers:
        cache_file = Path(cache_dir) / f"scores_{scorer.name}.parquet" if cache_dir else None
        known = pd.Series(dtype="float64")
        if cache_file is not None and cache_file.exists():
            cached = pd.read_parquet(cache_file)
            known = cached.set_index("norm")["score"]
        todo = unique[~unique.isin(known.index)]
        if len(todo):
            log.info("Scoring %d headlines with %s", len(todo), scorer.name)
            values = np.asarray(scorer.score([first_text[k] for k in todo]), dtype="float64")
            new = pd.Series(values, index=todo.to_numpy())
            known = pd.concat([known, new])
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                known.rename("score").rename_axis("norm").reset_index().to_parquet(cache_file)
        out[f"sent_{scorer.name}"] = norm.map(known).astype("float64")
    return out
