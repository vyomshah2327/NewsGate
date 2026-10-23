"""Historical-rate baseline: the large-move frequency seen in training, per ticker."""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsgate.models.base import ProbabilisticClassifier, register_model


@register_model("historical_rate")
class HistoricalRate(ProbabilisticClassifier):
    """Per-ticker training rate, shrunk towards the overall rate for stability."""

    def __init__(self, shrinkage: float = 20.0):
        self.shrinkage = shrinkage
        self.global_rate_: float | None = None
        self.rates_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "HistoricalRate":
        y = np.asarray(y, dtype="float64")
        self.global_rate_ = float(y.mean())
        frame = pd.DataFrame({"ticker": X["ticker"].to_numpy(), "y": y})
        stats = frame.groupby("ticker")["y"].agg(["sum", "count"])
        self.rates_ = ((stats["sum"] + self.shrinkage * self.global_rate_) / (stats["count"] + self.shrinkage)).to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_rate_ is None:
            raise RuntimeError("Model has not been fit")
        return X["ticker"].map(self.rates_).fillna(self.global_rate_).to_numpy(dtype="float64")
