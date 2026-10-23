"""Timestamp discipline: decide on which prediction date each article may be used.

The prediction cutoff is the 16:00 US/Eastern close of a trading day ``d``. Features for
date ``d`` may use only articles published strictly before that cutoff.

* Article with a time of day, published on a trading day before 16:00 ET -> date ``d`` itself.
* Article published at or after 16:00 ET, or on a non-trading day -> next trading day.
* Article with a date but no time of day -> next trading day strictly after that date,
  because it might have been published after the close. This is the conservative rule.

Input ``published_at`` values are naive UTC timestamps (the sources normalise to that).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsgate.data.calendar import TradingCalendar
from newsgate.data.schemas import check_columns

MARKET_TZ = "America/New_York"
CLOSE_MINUTES = 16 * 60


def assign_feature_dates(news: pd.DataFrame, calendar: TradingCalendar) -> pd.DataFrame:
    """Return ``news`` with ``feature_date`` (the first date the article may be used).

    Rows that map beyond the end of the calendar are dropped.
    """
    check_columns(news, ["published_at", "has_time"], "news")
    if news.empty:
        return news.assign(feature_date=pd.DatetimeIndex([]))

    published = pd.to_datetime(news["published_at"])
    has_time = news["has_time"].to_numpy(dtype=bool)

    utc = published.dt.tz_localize("UTC")
    local = utc.dt.tz_convert(MARKET_TZ)
    local_naive = local.dt.tz_localize(None)

    # Date-only rows keep their calendar date; timed rows use the New York calendar date.
    local_date = np.where(has_time, local_naive.dt.normalize(), published.dt.normalize())
    local_date = pd.DatetimeIndex(local_date)
    minutes = (local_naive.dt.hour * 60 + local_naive.dt.minute).to_numpy()

    before_close = has_time & (minutes < CLOSE_MINUTES)
    on_trading_day = calendar.contains(local_date)
    same_day = before_close & on_trading_day

    next_day = calendar.next_trading_day(local_date, strict=True)
    feature_date = pd.DatetimeIndex(np.where(same_day, local_date, next_day))

    out = news.copy()
    out["feature_date"] = feature_date
    return out.dropna(subset=["feature_date"]).reset_index(drop=True)
