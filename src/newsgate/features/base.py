"""Shared types for feature blocks.

Every block returns a frame indexed by ``(ticker, date)`` whose columns all start with the
block's prefix. All features for date ``d`` are computed from information available at the
16:00 ET close of ``d``. Rolling windows only look backwards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from newsgate.config import FeatureConfig


@dataclass
class FeatureContext:
    prices: pd.DataFrame  # standard price schema, stock tickers only
    market: pd.DataFrame  # columns: date, mkt_logret
    news: pd.DataFrame | None  # aligned, scored and clustered news (see pipeline.prepare_news)
    scorer_names: list[str]
    cfg: FeatureConfig

    @property
    def index(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_frame(self.prices[["ticker", "date"]]).sort_values()


class FeatureBlock(ABC):
    """Template method: ``build`` validates the output of ``_compute``."""

    prefix: str = ""

    def build(self, ctx: FeatureContext) -> pd.DataFrame:
        frame = self._compute(ctx)
        bad = [c for c in frame.columns if not c.startswith(self.prefix + "_")]
        if bad:
            raise ValueError(f"{type(self).__name__} produced columns without the '{self.prefix}_' prefix: {bad}")
        if list(frame.index.names) != ["ticker", "date"]:
            raise ValueError(f"{type(self).__name__} must return a (ticker, date) index")
        return frame

    @abstractmethod
    def _compute(self, ctx: FeatureContext) -> pd.DataFrame:
        ...


def per_ticker(frame: pd.DataFrame, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    """Apply ``fn`` to each ticker's date-sorted rows and stack the results."""
    pieces = []
    for ticker, group in frame.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        piece = fn(group.reset_index(drop=True))
        piece.insert(0, "date", group["date"].to_numpy())
        piece.insert(0, "ticker", ticker)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True).set_index(["ticker", "date"])
