"""Regularised logistic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from newsgate.models.base import ProbabilisticClassifier, register_model


@register_model("logistic", grid=[{"C": 0.01}, {"C": 0.1}, {"C": 1.0}])
class LogisticModel(ProbabilisticClassifier):
    def __init__(self, C: float = 0.1, seed: int = 0):
        self.C, self.seed = C, seed
        self.pipe_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogisticModel":
        self.pipe_ = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=500, random_state=self.seed),
        )
        self.pipe_.fit(X.to_numpy(dtype="float64"), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipe_ is None:
            raise RuntimeError("Model has not been fit")
        proba = self.pipe_.predict_proba(X.to_numpy(dtype="float64"))
        classes = list(self.pipe_.classes_)
        return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
