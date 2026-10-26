"""Assemble the modelling panel: one row per (ticker, date) with features and forward targets."""

from __future__ import annotations

import pandas as pd

from newsgate.config import Config
from newsgate.data.alignment import assign_feature_dates
from newsgate.data.calendar import TradingCalendar
from newsgate.data.cleaning import deduplicate_news
from newsgate.events import cluster_headlines
from newsgate.features.base import FeatureContext
from newsgate.features.pipeline import default_pipeline
from newsgate.labels import add_targets, market_returns
from newsgate.logging_utils import get_logger
from newsgate.sentiment import create_scorer, score_news

log = get_logger(__name__)


def prepare_news(
    news: pd.DataFrame, calendar: TradingCalendar, cfg: Config, cache_dir: str | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Deduplicate, align to prediction dates, score sentiment and cluster into events."""
    cleaned = deduplicate_news(news)
    aligned = assign_feature_dates(cleaned, calendar)
    scorers = [create_scorer(name, cfg.sentiment) for name in cfg.sentiment.scorers]
    scored = score_news(aligned, scorers, cache_dir)
    clustered = cluster_headlines(scored, cfg.events)
    log.info("News: %d raw -> %d after dedup/alignment", len(news), len(clustered))
    return clustered, [s.name for s in scorers]


MAX_PRICE_GAP_DAYS = 10  # a longer hole between two price rows counts as a data gap


def _drop_rows_after_price_gaps(panel: pd.DataFrame, prices: pd.DataFrame, warmup_rows: int) -> pd.DataFrame:
    """Drop the first ``warmup_rows`` rows after any long hole in a ticker's price history.

    Rolling price features computed across such a hole mix a stale price with a fresh one, so they
    are not comparable with ordinary days. The very first segment is already handled by the
    ordinary warmup drop.
    """
    px = prices.sort_values(["ticker", "date"])[["ticker", "date"]].copy()
    gap = px.groupby("ticker")["date"].diff().dt.days > MAX_PRICE_GAP_DAYS
    px["_segment"] = gap.groupby(px["ticker"]).cumsum()
    px["_pos"] = px.groupby(["ticker", "_segment"]).cumcount()
    bad = px[(px["_segment"] > 0) & (px["_pos"] < warmup_rows)][["ticker", "date"]].assign(_bad=True)
    if bad.empty:
        return panel
    out = panel.merge(bad, on=["ticker", "date"], how="left")
    log.info("Dropped %d rows that follow a gap in the price history", int(out["_bad"].notna().sum()))
    return out[out["_bad"].isna()].drop(columns="_bad")


def build_panel(
    cfg: Config,
    news: pd.DataFrame,
    all_prices: pd.DataFrame,
    tickers: list[str],
    metadata: pd.DataFrame | None = None,
    cache_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(panel, prepared_news)``.

    The panel is flat (ticker, date, px_*, att_*, sent_*, evt_*, tgt_*); the prepared news keeps
    the per-article scores, event ids and feature dates for the dashboard and audits."""
    stock_prices = all_prices[all_prices["ticker"].isin(tickers)].copy()
    calendar = TradingCalendar.from_prices(stock_prices)
    market = market_returns(all_prices, cfg.data.market_symbol, tickers)
    prepared, scorer_names = prepare_news(news[news["ticker"].isin(tickers)], calendar, cfg, cache_dir)

    ctx = FeatureContext(stock_prices, market, prepared, scorer_names, cfg.features)
    features = default_pipeline(include_news=True).build(ctx)
    targets = add_targets(stock_prices, market, cfg.target.horizon)
    panel = features.join(targets, how="left").reset_index()

    longest = max(max(cfg.features.momentum_windows), max(cfg.features.vol_windows))
    panel = panel.dropna(subset=[f"px_mom_{max(cfg.features.momentum_windows)}", f"px_vol_{max(cfg.features.vol_windows)}"])
    panel = _drop_rows_after_price_gaps(panel, stock_prices, longest)
    if cfg.data.start:  # prices before the study start were loaded only to warm up the rolling features
        panel = panel[panel["date"] >= pd.Timestamp(cfg.data.start)]
    if metadata is not None and "sector" in metadata:
        panel = panel.merge(metadata[["ticker", "sector"]], on="ticker", how="left")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    log.info("Panel: %d rows, %d tickers, %s to %s (warmup %d days)", len(panel), panel["ticker"].nunique(),
             panel["date"].min().date(), panel["date"].max().date(), longest)
    return panel, prepared
