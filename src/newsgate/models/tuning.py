"""Time-aware hyperparameter selection and probability calibration.

Both use an inner chronological split of the *training* window: the last fraction of training
dates is held out for validation, separated from the inner-training dates by a purge gap.
The test window is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from newsgate.models.base import ProbabilisticClassifier, create_model, param_grid

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Sigmoid calibration p' = sigmoid(a * logit(p) + b), fit on held-out predictions."""

    def __init__(self):
        self.a, self.b = 1.0, 0.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        if len(np.unique(y)) < 2:
            return self  # identity when the validation block has a single class
        lr = LogisticRegression(C=1e6, max_iter=200)
        lr.fit(_logit(p).reshape(-1, 1), y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * _logit(p) + self.b)))


@dataclass
class FittedModel:
    model: ProbabilisticClassifier
    calibrator: PlattCalibrator | None = None
    params: dict[str, Any] = field(default_factory=dict)
    inner_brier: float | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X)
        return self.calibrator.transform(p) if self.calibrator is not None else p


def inner_split(row_dates: np.ndarray, val_fraction: float, purge_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (inner-train, inner-validation) over rows, split by date with a purge gap."""
    unique = np.unique(row_dates)
    cut = int(len(unique) * (1.0 - val_fraction))
    val_start = unique[cut]
    train_end_idx = max(cut - purge_days, 1)
    train_last = unique[train_end_idx - 1]
    return row_dates <= train_last, row_dates >= val_start


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def fit_tuned(
    kind: str,
    X: pd.DataFrame,
    y: np.ndarray,
    row_dates: np.ndarray,
    purge_days: int,
    val_fraction: float = 0.2,
    calibrate: bool = True,
    seed: int = 0,
) -> FittedModel:
    """Select hyperparameters by inner-validation Brier score, refit on all training rows.

    The calibrator (if requested) is fit on the inner-validation predictions of the model that
    was fit on the inner-training rows, then applied to the final refit model.
    """
    grid = param_grid(kind)
    if grid == [{}] and not calibrate:
        return FittedModel(create_model(kind).fit(X, y))

    y = np.asarray(y)
    tr, va = inner_split(np.asarray(row_dates), val_fraction, purge_days)
    best_params, best_score, best_val_pred = grid[0], np.inf, None
    if len(grid) > 1 or calibrate:
        for params in grid:
            inner = create_model(kind, seed=seed, **params) if _takes_seed(kind) else create_model(kind, **params)
            inner.fit(X[tr], y[tr])
            pred = inner.predict_proba(X[va])
            score = _brier(pred, y[va])
            if score < best_score:
                best_params, best_score, best_val_pred = params, score, pred

    calibrator = PlattCalibrator().fit(best_val_pred, y[va]) if calibrate and best_val_pred is not None else None
    kwargs = dict(best_params)
    final = create_model(kind, seed=seed, **kwargs) if _takes_seed(kind) else create_model(kind, **kwargs)
    final.fit(X, y)
    return FittedModel(final, calibrator, best_params, best_score if np.isfinite(best_score) else None)


def _takes_seed(kind: str) -> bool:
    return kind in {"logistic", "lightgbm"}
