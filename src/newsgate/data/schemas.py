"""Standard column names shared by every data source.

All sources are adapters that return frames with exactly these columns, so the rest of
the pipeline never sees source-specific names.
"""

from __future__ import annotations

import pandas as pd

NEWS_COLUMNS = ["article_id", "ticker", "published_at", "has_time", "headline", "publisher", "url"]
PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


class SchemaError(ValueError):
    """Raised when a frame does not match the expected schema."""


def check_columns(frame: pd.DataFrame, required: list[str], what: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SchemaError(f"{what} is missing required column(s): {missing}")
