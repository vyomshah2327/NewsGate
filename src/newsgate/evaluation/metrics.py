"""Probabilistic forecast metrics. Brier score is the primary metric."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype="float64") - np.asarray(y, dtype="float64")) ** 2))


def brier_skill(y: np.ndarray, p: np.ndarray, p_ref: np.ndarray) -> float:
    """1 - Brier(model) / Brier(reference). Positive means better than the reference."""
    ref = brier(y, p_ref)
    return float("nan") if ref == 0 else 1.0 - brier(y, p) / ref


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    y = np.asarray(y, dtype="float64")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted probability versus observed frequency."""
    frame = pd.DataFrame({"y": np.asarray(y, dtype="float64"), "p": np.asarray(p, dtype="float64")})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False, duplicates="drop")
    table = frame.groupby("bin").agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), count=("y", "size"))
    return table.reset_index(drop=True)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    table = reliability_table(y, p, n_bins)
    weights = table["count"] / table["count"].sum()
    return float((weights * (table["mean_pred"] - table["frac_pos"]).abs()).sum())


def risk_coverage(loss: np.ndarray, score: np.ndarray, n_points: int = 100) -> pd.DataFrame:
    """Selective risk: mean loss over the ``coverage`` fraction with the lowest ``score``.

    ``score`` is an uncertainty estimate (higher means less trusted). A useful score gives a
    risk that falls as coverage shrinks.
    """
    loss = np.asarray(loss, dtype="float64")
    order = np.argsort(np.asarray(score, dtype="float64"), kind="stable")
    cum = np.cumsum(loss[order]) / np.arange(1, len(loss) + 1)
    coverage = np.arange(1, len(loss) + 1) / len(loss)
    keep = np.unique(np.linspace(0, len(loss) - 1, n_points).astype(int))
    return pd.DataFrame({"coverage": coverage[keep], "risk": cum[keep]})
