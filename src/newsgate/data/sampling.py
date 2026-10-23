"""Stratified company sampling by market-cap tercile and sector."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_sample(metadata: pd.DataFrame, n: int, seed: int = 0) -> list[str]:
    """Pick ``n`` tickers, spreading them across market-cap terciles and sectors.

    ``metadata`` needs ``ticker`` and ``market_cap``; ``sector`` is used when present. Strata
    are filled round-robin so no stratum is exhausted before the others get a pick.
    """
    if "ticker" not in metadata or "market_cap" not in metadata:
        raise ValueError("metadata needs 'ticker' and 'market_cap' columns")
    meta = metadata.dropna(subset=["ticker", "market_cap"]).drop_duplicates("ticker").copy()
    if n >= len(meta):
        return sorted(meta["ticker"].tolist())
    meta["_size"] = pd.qcut(meta["market_cap"].rank(method="first"), 3, labels=["small", "mid", "large"])
    meta["_sector"] = meta["sector"] if "sector" in meta else "all"
    rng = np.random.default_rng(seed)

    strata = {
        key: list(rng.permutation(group["ticker"].to_numpy()))
        for key, group in meta.groupby(["_size", "_sector"], observed=True)
    }
    keys = sorted(strata, key=str)
    chosen: list[str] = []
    while len(chosen) < n and any(strata.values()):
        for key in keys:
            if strata[key] and len(chosen) < n:
                chosen.append(strata[key].pop())
    return sorted(chosen)
