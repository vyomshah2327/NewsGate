"""News and price sources.

Each source is an adapter behind a small abstract interface (strategy pattern), so the
pipeline depends on ``NewsSource`` and ``PriceSource`` and never on where the data came from.
``build_sources`` is the factory that turns a ``DataConfig`` into a matching set of sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from newsgate.config import DataConfig
from newsgate.data.schemas import NEWS_COLUMNS, PRICE_COLUMNS, SchemaError, check_columns
from newsgate.logging_utils import get_logger

log = get_logger(__name__)


class NewsSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return news with the columns in ``NEWS_COLUMNS``; ``published_at`` is naive UTC."""


class PriceSource(ABC):
    @abstractmethod
    def load(self, tickers: Iterable[str]) -> pd.DataFrame:
        """Return daily prices with the columns in ``PRICE_COLUMNS``."""


# --------------------------------------------------------------------------- timestamps
def parse_timestamps(raw: pd.Series, source_tz: str = "UTC", mode: str = "auto") -> tuple[pd.Series, np.ndarray]:
    """Parse raw timestamp strings into naive UTC plus a ``has_time`` flag.

    ``has_time`` is False for date-only values. In ``auto`` mode the whole column is treated as
    date-only when fewer than 5 percent of rows carry a non-midnight time; otherwise each row
    that sits exactly at midnight is treated as date-only (the conservative reading).
    """
    parsed = pd.to_datetime(raw, errors="coerce")
    if parsed.dtype == object:  # mixed offsets
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    aware = getattr(parsed.dt, "tz", None) is not None
    wall = parsed.dt.tz_localize(None) if aware else parsed

    midnight = ((wall.dt.hour == 0) & (wall.dt.minute == 0) & (wall.dt.second == 0)).to_numpy()
    valid = wall.notna().to_numpy()
    if mode == "datetime":
        has_time = np.ones(len(wall), dtype=bool)
    elif mode == "date_only":
        has_time = np.zeros(len(wall), dtype=bool)
    elif mode == "auto":
        share_timed = float((~midnight[valid]).mean()) if valid.any() else 0.0
        has_time = np.zeros(len(wall), dtype=bool) if share_timed < 0.05 else ~midnight
    else:
        raise ValueError(f"Unknown timestamp mode: {mode}")

    if aware:
        utc = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        utc = parsed.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None)
    # Date-only rows keep the date as written, never shifted by a zone conversion.
    published = pd.Series(np.where(has_time, utc, wall.dt.normalize()), index=raw.index)
    return pd.to_datetime(published), has_time


def _make_article_ids(frame: pd.DataFrame) -> pd.Series:
    key = (
        frame["ticker"].astype(str) + "|" + frame["published_at"].astype(str) + "|"
        + frame["headline"].astype(str) + "|" + frame["url"].fillna("").astype(str)
    )
    return pd.util.hash_pandas_object(key, index=False).astype("uint64").astype(str)


# --------------------------------------------------------------------------- FNSPID news
FNSPID_NEWS_COLUMN_MAP = {
    "Date": "published_at",
    "Article_title": "headline",
    "Stock_symbol": "ticker",
    "Url": "url",
    "Publisher": "publisher",
}


class FnspidNewsSource(NewsSource):
    """Reads the FNSPID news table from CSV or Parquet.

    The full table is several gigabytes, so tickers and dates are filtered while reading.
    Only the columns needed for this study are loaded (the article body and the summaries
    are skipped).
    """

    def __init__(
        self,
        path: str | Path,
        tickers: Iterable[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        source_tz: str = "UTC",
        timestamp_mode: str = "auto",
        chunksize: int = 500_000,
    ):
        self.path = Path(path)
        self.tickers = list(tickers) if tickers else None
        self.start, self.end = start, end
        self.source_tz, self.timestamp_mode = source_tz, timestamp_mode
        self.chunksize = chunksize

    def _read(self) -> pd.DataFrame:
        wanted = [c for c in FNSPID_NEWS_COLUMN_MAP if c != "Url" and c != "Publisher"] + ["Url", "Publisher"]
        if self.path.suffix == ".parquet":
            filters = [("Stock_symbol", "in", self.tickers)] if self.tickers else None
            try:
                return pd.read_parquet(self.path, columns=wanted, filters=filters)
            except (KeyError, ValueError):
                return pd.read_parquet(self.path, columns=["Date", "Article_title", "Stock_symbol"], filters=filters)
        chunks = []
        header = pd.read_csv(self.path, nrows=0).columns
        usecols = [c for c in wanted if c in header]
        for chunk in pd.read_csv(self.path, usecols=usecols, chunksize=self.chunksize):
            if self.tickers:
                chunk = chunk[chunk["Stock_symbol"].isin(self.tickers)]
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)

    def load(self) -> pd.DataFrame:
        raw = self._read()
        check_columns(raw, ["Date", "Article_title", "Stock_symbol"], f"FNSPID news file {self.path}")
        frame = raw.rename(columns=FNSPID_NEWS_COLUMN_MAP)
        for col in ("url", "publisher"):
            if col not in frame:
                frame[col] = None
        published, has_time = parse_timestamps(frame["published_at"], self.source_tz, self.timestamp_mode)
        frame["published_at"], frame["has_time"] = published, has_time
        frame = frame.dropna(subset=["published_at", "ticker", "headline"])
        if self.start:
            frame = frame[frame["published_at"] >= pd.Timestamp(self.start)]
        if self.end:
            frame = frame[frame["published_at"] <= pd.Timestamp(self.end) + pd.Timedelta(days=1)]
        frame["article_id"] = _make_article_ids(frame)
        log.info("Loaded %d news rows (%.1f%% with a time of day)", len(frame), 100 * frame["has_time"].mean() if len(frame) else 0.0)
        return frame[NEWS_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- prices
def standardize_price_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Map assorted column spellings (Date, Adj Close, Adj., ...) onto ``PRICE_COLUMNS``."""
    renamed = {}
    for col in frame.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key.startswith("adj"):
            key = "adj_close"
        renamed[col] = key
    frame = frame.rename(columns=renamed)
    if "adj_close" not in frame and "close" in frame:
        frame["adj_close"] = frame["close"]
    check_columns(frame, ["date", "open", "high", "low", "close", "adj_close", "volume"], f"price file for {ticker}")
    frame = frame.assign(ticker=ticker)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    frame = frame.dropna(subset=["date", "adj_close"])
    return frame[PRICE_COLUMNS].sort_values("date").drop_duplicates(["ticker", "date"]).reset_index(drop=True)


class FnspidPriceSource(PriceSource):
    """Reads one CSV (or Parquet) file per ticker from a directory."""

    def __init__(self, price_dir: str | Path):
        self.price_dir = Path(price_dir)

    def load(self, tickers: Iterable[str]) -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            for suffix, reader in ((".csv", pd.read_csv), (".parquet", pd.read_parquet)):
                path = self.price_dir / f"{ticker}{suffix}"
                if path.exists():
                    frames.append(standardize_price_frame(reader(path), ticker))
                    break
            else:
                log.warning("No price file found for %s in %s", ticker, self.price_dir)
        if not frames:
            raise FileNotFoundError(f"No price files found in {self.price_dir}")
        return pd.concat(frames, ignore_index=True)


class YFinancePriceSource(PriceSource):
    """Downloads prices with yfinance and caches them as Parquet (records the download date)."""

    def __init__(self, start: str | None, end: str | None, cache_dir: str | Path):
        self.start, self.end, self.cache_dir = start, end, Path(cache_dir)

    @staticmethod
    def reshape(downloaded: pd.DataFrame) -> pd.DataFrame:
        """Turn yfinance's wide (field, ticker) columns into the long standard schema."""
        if not isinstance(downloaded.columns, pd.MultiIndex):
            raise SchemaError("Expected multi-index columns from yfinance for several tickers")
        long = downloaded.stack(level=1, future_stack=True).reset_index()
        long.columns = [str(c) for c in long.columns]
        long = long.rename(columns={long.columns[0]: "date", long.columns[1]: "ticker"})
        pieces = [standardize_price_frame(g.drop(columns="ticker"), str(t)) for t, g in long.groupby("ticker")]
        return pd.concat(pieces, ignore_index=True)

    def load(self, tickers: Iterable[str]) -> pd.DataFrame:
        import yfinance as yf  # imported lazily: optional dependency

        tickers = sorted(set(tickers))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.today().strftime("%Y%m%d")
        cache = self.cache_dir / f"yfinance_{stamp}_{abs(hash(tuple(tickers))) % 10**8}.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
        data = yf.download(tickers, start=self.start, end=self.end, auto_adjust=False, progress=False)
        frame = self.reshape(data)
        frame.to_parquet(cache)
        log.info("Downloaded %d price rows on %s (cached at %s)", len(frame), stamp, cache)
        return frame


# --------------------------------------------------------------------------- factory
@dataclass
class Sources:
    news: NewsSource
    prices: PriceSource
    metadata: Callable[[], pd.DataFrame | None]
    tickers: list[str]


def build_sources(cfg: DataConfig) -> Sources:
    """Factory: pick the adapters that match ``cfg.source``."""
    if cfg.source == "synthetic":
        from newsgate.data.synthetic import SyntheticWorld

        world = SyntheticWorld.generate(cfg.synthetic)
        return Sources(world.news_source(), world.price_source(), lambda: world.metadata, list(world.tickers))

    tickers = list(cfg.tickers)
    meta_loader: Callable[[], pd.DataFrame | None] = lambda: None
    if cfg.metadata_path:
        meta_loader = lambda: pd.read_csv(cfg.metadata_path)
    if not tickers and cfg.n_tickers and cfg.metadata_path:
        from newsgate.data.sampling import stratified_sample

        tickers = stratified_sample(meta_loader(), cfg.n_tickers)
    if not tickers:
        raise ValueError("Real data needs data.tickers, or data.n_tickers together with data.metadata_path")

    news = FnspidNewsSource(cfg.news_path, tickers, cfg.start, cfg.end, cfg.source_timezone, cfg.timestamp_mode)
    if cfg.source == "fnspid":
        prices: PriceSource = FnspidPriceSource(cfg.price_dir)
    elif cfg.source == "yfinance":
        prices = YFinancePriceSource(cfg.start, cfg.end, cfg.cache_dir)
    else:
        raise ValueError(f"Unknown data.source: {cfg.source}")
    return Sources(news, prices, meta_loader, tickers)
