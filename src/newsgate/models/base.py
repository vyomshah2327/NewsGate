"""Model interface and registry.

Every model predicts the probability of a large move. The sklearn and LightGBM estimators are
wrapped (adapter pattern) so the runner, tuner and gate only ever see ``fit`` and
``predict_proba``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import pandas as pd


class ProbabilisticClassifier(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ProbabilisticClassifier":
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class, shape (n,)."""


_FACTORIES: dict[str, Callable[..., ProbabilisticClassifier]] = {}
_GRIDS: dict[str, list[dict[str, Any]]] = {}


def register_model(kind: str, grid: list[dict[str, Any]] | None = None):
    def decorator(factory: Callable[..., ProbabilisticClassifier]):
        _FACTORIES[kind] = factory
        _GRIDS[kind] = grid or [{}]
        return factory

    return decorator


def available_kinds() -> list[str]:
    return sorted(_FACTORIES)


def create_model(kind: str, **params: Any) -> ProbabilisticClassifier:
    if kind not in _FACTORIES:
        raise KeyError(f"Unknown model kind '{kind}'. Available: {available_kinds()}")
    return _FACTORIES[kind](**params)


def param_grid(kind: str) -> list[dict[str, Any]]:
    return list(_GRIDS[kind])
