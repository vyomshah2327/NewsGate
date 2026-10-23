"""Trading calendar built from observed price dates.

Using the dates that actually appear in the price data means holidays and market closures
are handled without an external calendar library. Early-close days (13:00 ET) are not
modelled; the cutoff is always 16:00 ET, which is slightly conservative for those days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TradingCalendar:
    def __init__(self, dates: pd.DatetimeIndex | np.ndarray | list):
        idx = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
        if len(idx) == 0:
            raise ValueError("TradingCalendar needs at least one date")
        self._dates = idx.astype("datetime64[ns]")
        self._values = self._dates.values

    @classmethod
    def from_prices(cls, prices: pd.DataFrame) -> "TradingCalendar":
        return cls(prices["date"].unique())

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._dates

    def __len__(self) -> int:
        return len(self._dates)

    def contains(self, days: pd.Series | pd.DatetimeIndex | np.ndarray) -> np.ndarray:
        values = np.asarray(pd.to_datetime(days).astype("datetime64[ns]"))
        pos = np.searchsorted(self._values, values, side="left")
        pos_clipped = np.minimum(pos, len(self._values) - 1)
        return (pos < len(self._values)) & (self._values[pos_clipped] == values)

    def next_trading_day(self, days: pd.Series | pd.DatetimeIndex | np.ndarray, strict: bool) -> pd.DatetimeIndex:
        """First trading day after ``days`` (strict) or on/after ``days`` (not strict).

        Days beyond the end of the calendar map to NaT.
        """
        values = np.asarray(pd.to_datetime(days).astype("datetime64[ns]"))
        pos = np.searchsorted(self._values, values, side="right" if strict else "left")
        out = np.full(len(values), np.datetime64("NaT"), dtype="datetime64[ns]")
        ok = pos < len(self._values)
        out[ok] = self._values[pos[ok]]
        return pd.DatetimeIndex(out)

    def shift(self, date: pd.Timestamp, n: int) -> pd.Timestamp:
        """The trading day ``n`` sessions after ``date`` (n may be negative)."""
        pos = int(np.searchsorted(self._values, np.datetime64(pd.Timestamp(date).normalize()), side="left"))
        new = pos + n
        if new < 0 or new >= len(self._values):
            return pd.NaT
        return self._dates[new]
