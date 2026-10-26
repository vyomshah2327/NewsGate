"""Targets and the large-move label.

Targets look *forward* from the prediction date ``t`` (the close of day ``t``) to the close
``horizon`` trading days later. The binary large-move label is made from a threshold that is
learned on the training window only (``LargeMoveThreshold.fit``) and then applied unchanged to
the test window, so no information about the test period enters the label definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["tgt_fwd_ret", "tgt_abs", "tgt_abn", "tgt_end_date"]


def market_returns(all_prices: pd.DataFrame, symbol: str, stock_tickers: list[str]) -> pd.DataFrame:
    """Daily market log return: the benchmark symbol if present, else the equal-weight stock mean."""
    if symbol in set(all_prices["ticker"]):
        bench = all_prices[all_prices["ticker"] == symbol].sort_values("date")
        series = np.log(bench["adj_close"]).diff().set_axis(bench["date"].to_numpy())
    else:
        stocks = all_prices[all_prices["ticker"].isin(stock_tickers)].sort_values(["ticker", "date"])
        stocks = stocks.assign(lr=stocks.groupby("ticker")["adj_close"].transform(lambda s: np.log(s).diff()))
        series = stocks.groupby("date")["lr"].mean()
    series.index.name = "date"
    return series.rename("mkt_logret").reset_index()


# Longest plausible calendar span of a window of ``horizon`` sessions: 4 + 7 * horizon days.
MAX_SPAN_BASE_DAYS = 4
MAX_SPAN_PER_SESSION_DAYS = 7


def add_targets(prices: pd.DataFrame, market: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward log return, its absolute value, the market-adjusted ("abnormal") return and end date.

    A target whose window spans an abnormally long stretch of calendar time (a trading halt,
    delisting and relisting, or a hole in the price file) is not a ``horizon``-session return, so
    it is set to missing instead of being treated as one.
    """
    max_span = pd.Timedelta(days=MAX_SPAN_BASE_DAYS + MAX_SPAN_PER_SESSION_DAYS * horizon)
    mkt = market.sort_values("date").set_index("date")["mkt_logret"].fillna(0.0).cumsum()
    pieces = []
    for ticker, g in prices.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        logp = np.log(g["adj_close"].to_numpy())
        fwd = pd.Series(logp).shift(-horizon).to_numpy() - logp
        end = g["date"].shift(-horizon).reset_index(drop=True)
        too_long = ((end - g["date"].reset_index(drop=True)) > max_span).to_numpy()
        fwd = np.where(too_long, np.nan, fwd)
        end = end.where(~pd.Series(too_long))
        mkt_start = mkt.reindex(g["date"]).to_numpy()
        mkt_end = mkt.reindex(end).to_numpy()
        pieces.append(pd.DataFrame({
            "ticker": ticker,
            "date": g["date"].to_numpy(),
            "tgt_fwd_ret": fwd,
            "tgt_abs": np.abs(fwd),
            "tgt_abn": fwd - (mkt_end - mkt_start),
            "tgt_end_date": end.to_numpy(),
        }))
    return pd.concat(pieces, ignore_index=True).set_index(["ticker", "date"])


class LargeMoveThreshold:
    """Quantile threshold for ``tgt_abs`` fit on training rows only."""

    MIN_TICKER_OBS = 50

    def __init__(self, quantile: float = 0.9, scope: str = "ticker"):
        self.quantile, self.scope = quantile, scope
        self.global_: float | None = None
        self.by_ticker: dict[str, float] = {}

    def fit(self, train: pd.DataFrame) -> "LargeMoveThreshold":
        values = train["tgt_abs"].dropna()
        if values.empty:
            raise ValueError("Cannot fit a large-move threshold on an empty training set")
        self.global_ = float(values.quantile(self.quantile))
        if self.scope == "ticker":
            for ticker, g in train.dropna(subset=["tgt_abs"]).groupby("ticker"):
                if len(g) >= self.MIN_TICKER_OBS:
                    self.by_ticker[ticker] = float(g["tgt_abs"].quantile(self.quantile))
        return self

    def thresholds(self, frame: pd.DataFrame) -> np.ndarray:
        if self.global_ is None:
            raise RuntimeError("Threshold has not been fit")
        return frame["ticker"].map(self.by_ticker).fillna(self.global_).to_numpy(dtype="float64")

    def label(self, frame: pd.DataFrame) -> np.ndarray:
        return (frame["tgt_abs"].to_numpy() > self.thresholds(frame)).astype("int8")
