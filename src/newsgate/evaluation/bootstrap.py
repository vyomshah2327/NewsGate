"""Paired moving-block bootstrap over dates.

Stocks move together on a given day, so rows from the same date are not independent. The
caller first averages the paired loss difference over tickers within each date; this
function then resamples *blocks of consecutive dates* so that serial dependence across days
is preserved as well.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def date_block_bootstrap(
    diff_by_date: pd.Series, block_length: int = 10, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, Any]:
    """Bootstrap the mean of a date-indexed series of paired differences.

    Returns the observed mean, a percentile confidence interval, and a two-sided p-value for
    "the mean difference is zero" taken from the bootstrap distribution of the mean.
    """
    values = diff_by_date.sort_index().to_numpy(dtype="float64")
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"), "n_dates": n}

    length = max(1, min(block_length, n))
    n_blocks = int(np.ceil(n / length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - length + 1, size=(n_resamples, n_blocks))
    index = (starts[:, :, None] + np.arange(length)[None, None, :]).reshape(n_resamples, -1)[:, :n]
    means = values[index].mean(axis=1)

    observed = float(values.mean())
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p_two_sided = 2.0 * min(np.mean(means <= 0.0), np.mean(means >= 0.0))
    return {
        "mean": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(min(1.0, max(p_two_sided, 1.0 / (n_resamples + 1)))),
        "n_dates": n,
    }
