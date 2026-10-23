"""Evaluation: purged chronological splits, metrics, block bootstrap, multiple-testing correction."""

from newsgate.evaluation.bootstrap import date_block_bootstrap
from newsgate.evaluation.comparisons import compare_models, robustness_table
from newsgate.evaluation.metrics import (
    auroc, brier, brier_skill, expected_calibration_error, log_loss, reliability_table, risk_coverage,
)
from newsgate.evaluation.multitest import holm_adjust
from newsgate.evaluation.runner import Experiment, ModelSpec, build_specs
from newsgate.evaluation.splits import Fold, PurgedExpandingWindowSplit, SplitPlan

__all__ = [
    "Experiment", "Fold", "ModelSpec", "PurgedExpandingWindowSplit", "SplitPlan", "auroc", "brier",
    "brier_skill", "build_specs", "compare_models", "date_block_bootstrap", "expected_calibration_error",
    "holm_adjust", "log_loss", "reliability_table", "risk_coverage", "robustness_table",
]
