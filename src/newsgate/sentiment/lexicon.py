"""Loughran and McDonald finance dictionary scorer.

The master dictionary is distributed by its authors and cannot be bundled here. Download
``Loughran-McDonald_MasterDictionary_*.csv`` and point ``sentiment.lm_dictionary_path`` at it. The
score is (positive - negative) / (positive + negative) over dictionary words, 0 when none match.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from newsgate.config import SentimentConfig
from newsgate.sentiment.base import SentimentScorer, register_scorer

_TOKEN = re.compile(r"[a-z]+")


class LoughranMcDonaldScorer(SentimentScorer):
    name = "lm"

    def __init__(self, positive: Iterable[str], negative: Iterable[str]):
        self.positive = {w.lower() for w in positive}
        self.negative = {w.lower() for w in negative}
        if not self.positive or not self.negative:
            raise ValueError("Both positive and negative word lists must be non-empty")

    @classmethod
    def from_dictionary_csv(cls, path: str | Path) -> "LoughranMcDonaldScorer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Loughran-McDonald dictionary not found at {path}. Download the master dictionary "
                "CSV from the authors and set sentiment.lm_dictionary_path."
            )
        table = pd.read_csv(path)
        table.columns = [c.strip().lower() for c in table.columns]
        for col in ("word", "negative", "positive"):
            if col not in table:
                raise ValueError(f"Dictionary file is missing the '{col}' column")
        words = table["word"].astype(str).str.lower()
        return cls(words[table["positive"] > 0], words[table["negative"] > 0])

    def score(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(texts))
        for i, text in enumerate(texts):
            tokens = _TOKEN.findall(str(text).lower())
            pos = sum(t in self.positive for t in tokens)
            neg = sum(t in self.negative for t in tokens)
            out[i] = 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)
        return out


@register_scorer("lm")
def _make_lm(cfg: SentimentConfig) -> SentimentScorer:
    return LoughranMcDonaldScorer.from_dictionary_csv(cfg.lm_dictionary_path)
