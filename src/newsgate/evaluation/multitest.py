"""Holm step-down correction for a family of p-values."""

from __future__ import annotations

import numpy as np


def holm_adjust(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Holm-adjusted p-values (controls the family-wise error rate). NaNs are left as NaN."""
    p = np.asarray(p_values, dtype="float64")
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pv[idx])
        adjusted[idx] = min(1.0, running)
    out[valid] = adjusted
    return out
