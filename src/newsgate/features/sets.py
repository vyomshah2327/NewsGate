"""Named feature sets: which column prefixes each model in the ablation ladder may use."""

from __future__ import annotations

import pandas as pd

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "historical_rate": (),
    "price_only": ("px",),
    "price_attention": ("px", "att"),
    "price_sentiment": ("px", "sent"),
    "price_news": ("px", "att", "sent", "evt"),
}


def columns_for(columns: pd.Index | list[str], set_name: str) -> list[str]:
    """Panel columns belonging to a named feature set (order preserved)."""
    if set_name not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set '{set_name}'. Available: {sorted(FEATURE_SETS)}")
    prefixes = tuple(p + "_" for p in FEATURE_SETS[set_name])
    return [c for c in columns if c.startswith(prefixes)]
