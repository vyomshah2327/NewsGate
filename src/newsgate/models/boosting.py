"""LightGBM classifier with conservative settings (small trees, strong regularisation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsgate.models.base import ProbabilisticClassifier, register_model

_GRID = [
    {"num_leaves": 4, "n_estimators": 150},
    {"num_leaves": 4, "n_estimators": 400},
    {"num_leaves": 8, "n_estimators": 150},
    {"num_leaves": 8, "n_estimators": 400},
]


@register_model("lightgbm", grid=_GRID)
class LightGBMModel(ProbabilisticClassifier):
    def __init__(self, num_leaves: int = 4, n_estimators: int = 200, learning_rate: float = 0.03, seed: int = 0):
        self.num_leaves, self.n_estimators, self.learning_rate, self.seed = num_leaves, n_estimators, learning_rate, seed
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LightGBMModel":
        from lightgbm import LGBMClassifier

        self.model_ = LGBMClassifier(
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            random_state=self.seed,
            n_jobs=1,
            verbose=-1,
        )
        self.model_.fit(X.to_numpy(dtype="float64"), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model has not been fit")
        proba = self.model_.predict_proba(X.to_numpy(dtype="float64"))
        classes = list(self.model_.classes_)
        return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
