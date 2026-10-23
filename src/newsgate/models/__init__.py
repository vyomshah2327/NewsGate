"""Probabilistic classifiers behind one interface, created by a registry (factory)."""

from newsgate.models.base import ProbabilisticClassifier, available_kinds, create_model, param_grid, register_model
from newsgate.models import baselines, boosting, linear  # noqa: F401  (registers the built-in kinds)
from newsgate.models.tuning import FittedModel, PlattCalibrator, fit_tuned

__all__ = [
    "FittedModel",
    "PlattCalibrator",
    "ProbabilisticClassifier",
    "available_kinds",
    "create_model",
    "fit_tuned",
    "param_grid",
    "register_model",
]
