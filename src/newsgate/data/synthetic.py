"""Synthetic market and news generator.

The real FNSPID data is large and needs network access, so every stage of the pipeline can
also run on a small synthetic world with a *known* signal. This makes the code testable end
to end and lets us check that the study detects a planted effect and stays quiet when there
is none.

The planted signal: each ticker has an information shock ``s`` on day ``t``. News published
on day ``t`` reflects ``s`` (more articles when ``|s|`` is large, polarity follows ``s``). The
return on the *next* trading day has volatility scaled by ``1 + attention_effect * |s|`` and
a mean shifted by ``sentiment_effect * s``. With both effects at zero the world is a pure
null, and a leakage-free study should find no gain from news.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from newsgate.config import SyntheticConfig
from newsgate.data.schemas import NEWS_COLUMNS, PRICE_COLUMNS
from newsgate.data.sources import NewsSource, PriceSource

MARKET = "SPY"
SECTORS = ["Technology", "Health Care", "Financials", "Energy", "Industrials"]
POSITIVE = ["beats", "surges", "upgraded", "record", "strong", "raises", "wins", "growth", "boosts"]
NEGATIVE = ["misses", "plunges", "downgraded", "probe", "weak", "cuts", "lawsuit", "recalls", "slumps"]
NEUTRAL = ["announces", "schedules", "files", "reports", "updates", "names", "plans"]
TOPICS = [
    "quarterly earnings", "product launch", "board changes", "supply agreement", "guidance",
    "regulatory filing", "share buyback", "partnership", "factory expansion", "analyst meeting",
    "debt offering", "customer contract",
]
CODEWORDS = [f"{a}{b}" for a in "abcdefghijklmnopqrstuvwxyz" for b in ("ra", "to", "mi", "lu", "ko", "ne", "vi", "sa")]
PUBLISHERS = ["wire_a", "wire_b", "daily_post", "market_watch", "finance_blog"]


@dataclass
class _FrameNewsSource(NewsSource):
    frame: pd.DataFrame

    def load(self) -> pd.DataFrame:
        return self.frame.copy()


@dataclass
class _FramePriceSource(PriceSource):
    frame: pd.DataFrame

    def load(self, tickers: Iterable[str]) -> pd.DataFrame:
        wanted = set(tickers) | {MARKET}
        return self.frame[self.frame["ticker"].isin(wanted)].copy()


class SyntheticWorld:
    def __init__(self, tickers: list[str], news: pd.DataFrame, prices: pd.DataFrame, metadata: pd.DataFrame):
        self.tickers, self.news, self.prices, self.metadata = tickers, news, prices, metadata

    # ------------------------------------------------------------------ construction
    @classmethod
    def generate(cls, cfg: SyntheticConfig) -> "SyntheticWorld":
        rng = np.random.default_rng(cfg.seed)
        days = pd.bdate_range(cfg.start, periods=int(cfg.n_days * 1.06))
        days = days[rng.random(len(days)) > 0.035][: cfg.n_days]  # a few market holidays
        n_days, n = len(days), cfg.n_tickers
        tickers = [f"T{i:02d}" for i in range(n)]

        def log_vol(phi: float, scale: float, size: int) -> np.ndarray:
            out = np.zeros(size)
            for t in range(1, size):
                out[t] = phi * out[t - 1] + scale * rng.standard_normal()
            return out

        market_ret = 0.0003 + 0.008 * np.exp(log_vol(0.95, 0.15, n_days)) * rng.standard_normal(n_days)
        beta = rng.uniform(0.6, 1.4, n)
        sigma = rng.uniform(0.008, 0.02, n)
        regimes = np.column_stack([log_vol(0.97, 0.10, n_days) for _ in range(n)])
        shock = rng.standard_normal((n_days, n))
        prev_shock = np.vstack([np.zeros((1, n)), shock[:-1]])

        z = rng.standard_normal((n_days, n))
        vol_scale = sigma * np.exp(regimes) * (1.0 + cfg.attention_effect * np.abs(prev_shock))
        returns = beta * market_ret[:, None] + vol_scale * z + cfg.sentiment_effect * sigma * prev_shock

        prices = cls._prices_frame(days, tickers, returns, market_ret, sigma, rng)

        market_cap = np.exp(rng.normal(24.5, 1.2, n))
        size = (market_cap / np.median(market_cap)) ** 0.3
        metadata = pd.DataFrame({
            "ticker": tickers,
            "sector": rng.choice(SECTORS, n),
            "market_cap": market_cap,
        })
        news = cls._news_frame(days, tickers, shock, size, cfg, rng)
        return cls(tickers, news, prices, metadata)

    @staticmethod
    def _prices_frame(days, tickers, returns, market_ret, sigma, rng) -> pd.DataFrame:
        frames = []
        series = [(MARKET, market_ret, 0.008)] + [(t, returns[:, i], sigma[i]) for i, t in enumerate(tickers)]
        for ticker, ret, vol in series:
            close = rng.uniform(40, 200) * np.exp(np.cumsum(ret))
            prev = np.concatenate([[close[0] / (1 + ret[0])], close[:-1]])
            open_ = prev * np.exp(vol * 0.3 * rng.standard_normal(len(close)))
            high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal(len(close))) * vol * 0.4)
            low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal(len(close))) * vol * 0.4)
            volume = 1e6 * np.exp(0.4 * np.abs(ret) / vol + 0.3 * rng.standard_normal(len(close)))
            frames.append(pd.DataFrame({
                "ticker": ticker, "date": days, "open": open_, "high": high, "low": low,
                "close": close, "adj_close": close, "volume": volume.round(),
            }))
        return pd.concat(frames, ignore_index=True)[PRICE_COLUMNS]

    @staticmethod
    def _news_frame(days, tickers, shock, size, cfg: SyntheticConfig, rng) -> pd.DataFrame:
        lam = cfg.mean_articles * size[None, :] * np.exp(0.5 * np.abs(shock) - 0.3)
        counts = rng.poisson(lam)
        rows = []
        for t_idx, i in zip(*np.nonzero(counts)):
            ticker, day = tickers[i], days[t_idx]
            s = shock[t_idx, i]
            codeword, topic = rng.choice(CODEWORDS), rng.choice(TOPICS)
            for k in range(int(counts[t_idx, i])):
                if k > 0 and rng.random() < 0.55:
                    pass  # follow-up on the same event (similar headline)
                else:
                    codeword, topic = rng.choice(CODEWORDS), rng.choice(TOPICS)
                polarity = np.tanh(0.9 * s + 0.5 * rng.standard_normal())
                if polarity > 0.2:
                    words = " ".join(rng.choice(POSITIVE, size=int(rng.integers(1, 3)), replace=False))
                elif polarity < -0.2:
                    words = " ".join(rng.choice(NEGATIVE, size=int(rng.integers(1, 3)), replace=False))
                else:
                    words = str(rng.choice(NEUTRAL))
                headline = f"{ticker} {words} {topic} project {codeword}"
                if rng.random() < 0.5:  # during market hours, otherwise after hours or overnight
                    hour = rng.uniform(6.0, 15.99)
                else:
                    hour = rng.uniform(16.0, 30.0) % 24.0 if rng.random() < 0.7 else rng.uniform(6.0, 15.99)
                local = day + pd.Timedelta(hours=float(hour))
                rows.append((ticker, local, headline, str(rng.choice(PUBLISHERS))))
                if rng.random() < 0.2:  # syndicated copy from another publisher
                    rows.append((ticker, local + pd.Timedelta(minutes=float(rng.integers(1, 30))), headline, str(rng.choice(PUBLISHERS))))

        frame = pd.DataFrame(rows, columns=["ticker", "local_time", "headline", "publisher"])
        published = (
            pd.DatetimeIndex(frame["local_time"])
            .tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
            .tz_convert("UTC")
            .tz_localize(None)
        )
        frame["published_at"] = published
        frame = frame.dropna(subset=["published_at"]).sort_values("published_at").reset_index(drop=True)
        frame["has_time"] = True
        frame["url"] = [f"https://example.invalid/{i}" for i in range(len(frame))]
        frame["article_id"] = [f"syn{i:07d}" for i in range(len(frame))]
        return frame[NEWS_COLUMNS]

    # ------------------------------------------------------------------ accessors
    def news_source(self) -> NewsSource:
        return _FrameNewsSource(self.news)

    def price_source(self) -> PriceSource:
        return _FramePriceSource(self.prices)

    def write_fnspid_like(self, directory: str | Path) -> tuple[Path, Path]:
        """Write the world in the same file layout the real FNSPID loaders expect."""
        directory = Path(directory)
        price_dir = directory / "prices"
        price_dir.mkdir(parents=True, exist_ok=True)
        stamp = np.where(
            self.news["has_time"],
            self.news["published_at"].dt.strftime("%Y-%m-%d %H:%M:%S") + " UTC",
            self.news["published_at"].dt.strftime("%Y-%m-%d"),
        )
        news = pd.DataFrame({
            "Date": stamp, "Article_title": self.news["headline"], "Stock_symbol": self.news["ticker"],
            "Url": self.news["url"], "Publisher": self.news["publisher"],
        })
        news_path = directory / "news.parquet"
        news.to_parquet(news_path)
        for ticker, group in self.prices.groupby("ticker"):
            group.rename(columns={
                "date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close",
                "adj_close": "Adj Close", "volume": "Volume",
            }).drop(columns="ticker").to_csv(price_dir / f"{ticker}.csv", index=False)
        self.metadata.to_csv(directory / "metadata.csv", index=False)
        return news_path, price_dir


def to_date_only(news: pd.DataFrame) -> pd.DataFrame:
    """Strip the time of day from news (to exercise the conservative date-only rule)."""
    out = news.copy()
    out["published_at"] = out["published_at"].dt.normalize()
    out["has_time"] = False
    return out
