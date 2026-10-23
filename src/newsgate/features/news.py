"""News features: attention (``att``), sentiment (``sent``) and event (``evt``) blocks.

Articles are first mapped to the earliest trading date they may be used on (see
``data.alignment``). Daily aggregates are placed on the dense price calendar; days with no news
get zeros and an explicit ``att_has_news`` flag. Rolling baselines use only earlier days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsgate.features.base import FeatureBlock, FeatureContext, per_ticker

_EMPTY_MESSAGE = "news features need aligned news; build the context with news=None only for price features"


def _dense(ctx: FeatureContext, daily: pd.DataFrame) -> pd.DataFrame:
    """Left-join daily news aggregates onto the (ticker, date) price panel."""
    base = ctx.prices[["ticker", "date"]].copy()
    daily = daily.rename(columns={"feature_date": "date"})
    return base.merge(daily, on=["ticker", "date"], how="left")


def _past_mean(series: pd.Series, window: int, min_periods: int = 3) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=min_periods).mean()


class AttentionFeatures(FeatureBlock):
    prefix = "att"

    def _compute(self, ctx: FeatureContext) -> pd.DataFrame:
        if ctx.news is None:
            raise ValueError(_EMPTY_MESSAGE)
        daily = ctx.news.groupby(["ticker", "feature_date"]).size().rename("n").reset_index()
        dense = _dense(ctx, daily)
        window = ctx.cfg.news_baseline_window

        def one(g: pd.DataFrame) -> pd.DataFrame:
            n = g["n"].fillna(0.0)
            out = pd.DataFrame(index=g.index)
            out["att_n"] = n
            out["att_log_n"] = np.log1p(n)
            out["att_has_news"] = (n > 0).astype("float64")
            out["att_n_roll5"] = n.rolling(5, min_periods=1).sum()
            prior_mean = _past_mean(n, window, min_periods=5)
            prior_std = n.shift(1).rolling(window, min_periods=5).std()
            out["att_surprise"] = ((n - prior_mean) / (prior_std.fillna(0.0) + 0.5)).fillna(0.0)
            position = np.arange(len(n), dtype="float64")
            last_news = pd.Series(np.where(n.to_numpy() > 0, position, np.nan)).ffill()
            out["att_days_since"] = (position - last_news).fillna(30.0).clip(upper=30.0).to_numpy()
            return out

        return per_ticker(dense, one)


class SentimentFeatures(FeatureBlock):
    prefix = "sent"

    def _compute(self, ctx: FeatureContext) -> pd.DataFrame:
        if ctx.news is None:
            raise ValueError(_EMPTY_MESSAGE)
        names = ctx.scorer_names
        cols = [f"sent_{s}" for s in names]
        news = ctx.news
        primary = cols[0]
        grouped = news.groupby(["ticker", "feature_date"])
        daily = grouped[cols].mean().add_prefix("m_")
        daily["median"] = grouped[primary].median()
        daily["std"] = grouped[primary].std(ddof=0)
        daily["pos_share"] = grouped[primary].apply(lambda s: float((s > 0.05).mean()))
        daily["neg_share"] = grouped[primary].apply(lambda s: float((s < -0.05).mean()))
        dense = _dense(ctx, daily.reset_index())
        window, roll = ctx.cfg.news_baseline_window, ctx.cfg.sentiment_roll
        means = [f"m_{c}" for c in cols]

        def one(g: pd.DataFrame) -> pd.DataFrame:
            out = pd.DataFrame(index=g.index)
            overall = g[means].mean(axis=1)  # NaN on days without news
            out["sent_mean"] = overall.fillna(0.0)
            for name, col in zip(names, means):
                out[f"sent_{name}_mean"] = g[col].fillna(0.0)
            out["sent_median"] = g["median"].fillna(0.0)
            out["sent_std"] = g["std"].fillna(0.0)
            out["sent_pos_share"] = g["pos_share"].fillna(0.0)
            out["sent_neg_share"] = g["neg_share"].fillna(0.0)
            out["sent_disagree"] = g[means].std(axis=1, ddof=0).fillna(0.0) if len(means) > 1 else 0.0
            out["sent_surprise"] = (overall - _past_mean(overall, window)).fillna(0.0)
            out["sent_mean_lag1"] = overall.shift(1).fillna(0.0)
            out["sent_roll"] = overall.rolling(roll, min_periods=1).mean().fillna(0.0)
            return out

        return per_ticker(dense, one)


class EventFeatures(FeatureBlock):
    prefix = "evt"

    def _compute(self, ctx: FeatureContext) -> pd.DataFrame:
        if ctx.news is None or "event_id" not in ctx.news:
            raise ValueError("event features need clustered news (event_id, novelty, ...)")
        grouped = ctx.news.groupby(["ticker", "feature_date"])
        daily = pd.DataFrame({
            "n_events": grouped["is_new_event"].sum(),
            "novel_share": grouped["is_new_event"].mean(),
            "novelty_mean": grouped["novelty"].mean(),
            "max_cluster": grouped["event_size"].max(),
        }).reset_index()
        dense = _dense(ctx, daily)

        def one(g: pd.DataFrame) -> pd.DataFrame:
            has = g["n_events"].notna()
            out = pd.DataFrame(index=g.index)
            out["evt_n_events"] = g["n_events"].fillna(0.0).astype("float64")
            out["evt_novel_share"] = g["novel_share"].fillna(0.0)
            out["evt_repeat_share"] = np.where(has, 1.0 - g["novel_share"].fillna(0.0), 0.0)
            out["evt_novelty_mean"] = g["novelty_mean"].fillna(0.0)
            out["evt_max_cluster"] = g["max_cluster"].fillna(0.0).astype("float64")
            out["evt_n_events_roll5"] = out["evt_n_events"].rolling(5, min_periods=1).sum()
            return out

        return per_ticker(dense, one)
