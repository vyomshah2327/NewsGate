"""The experiment: fit every model in the ablation ladder on each chronological fold.

For each fold the large-move threshold is learned on the training rows only, every model is
tuned on an inner chronological split of those rows, and out-of-fold probabilities are
recorded for the test rows. ``run_holdout`` evaluates the untouched final period and should be
called once, after the design is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from newsgate.config import Config, EvalConfig
from newsgate.evaluation.splits import Fold, PurgedExpandingWindowSplit, SplitPlan
from newsgate.features.sets import FEATURE_SETS, columns_for
from newsgate.labels import LargeMoveThreshold
from newsgate.logging_utils import get_logger
from newsgate.models import fit_tuned

log = get_logger(__name__)

BASELINE = "historical_rate"


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    feature_set: str

    @property
    def name(self) -> str:
        return BASELINE if self.kind == BASELINE else f"{self.kind}:{self.feature_set}"

    @property
    def column(self) -> str:
        return "p__" + self.name.replace(":", "__")


def column_for(name: str) -> str:
    return "p__" + name.replace(":", "__")


def build_specs(cfg: EvalConfig) -> list[ModelSpec]:
    unknown = [s for s in cfg.feature_sets if s not in FEATURE_SETS]
    if unknown:
        raise KeyError(f"Unknown feature set(s): {unknown}")
    specs = [ModelSpec(BASELINE, BASELINE)]
    specs += [ModelSpec(kind, fs) for kind in cfg.kinds for fs in cfg.feature_sets]
    return specs


class Experiment:
    def __init__(self, panel: pd.DataFrame, cfg: Config):
        self.cfg = cfg
        self.panel = panel[panel["tgt_abs"].notna()].sort_values(["date", "ticker"]).reset_index(drop=True)
        ev = cfg.evaluation
        splitter = PurgedExpandingWindowSplit(ev.min_train_days, ev.test_block_days, ev.purge_days, ev.holdout_fraction)
        self.plan: SplitPlan = splitter.partition(self.panel["date"].unique())
        self.specs = build_specs(ev)
        self.fit_log: list[dict] = []

    # ------------------------------------------------------------------ one fold
    def _rows(self, dates: np.ndarray) -> pd.DataFrame:
        return self.panel[self.panel["date"].isin(dates)]

    def _run(self, fold_index: int, train_dates: np.ndarray, test_dates: np.ndarray) -> pd.DataFrame:
        cfg, ev = self.cfg, self.cfg.evaluation
        train, test = self._rows(train_dates), self._rows(test_dates)
        if train["tgt_end_date"].max() >= test["date"].min():
            raise RuntimeError("Training labels overlap the test period; check purge_days against the horizon")

        threshold = LargeMoveThreshold(cfg.target.quantile, cfg.target.threshold_scope).fit(train)
        y_train, y_test = threshold.label(train), threshold.label(test)

        out = test[["ticker", "date", "tgt_end_date", "tgt_abs"]].copy()
        out["fold"], out["y"], out["threshold"] = fold_index, y_test, threshold.thresholds(test)

        for spec in self.specs:
            if spec.kind == BASELINE:
                cols = ["ticker"]
            else:
                cols = columns_for(self.panel.columns, spec.feature_set)
            fitted = fit_tuned(
                spec.kind, train[cols], y_train, train["date"].to_numpy(), ev.purge_days,
                ev.inner_val_fraction, calibrate=ev.calibrate and spec.kind != BASELINE, seed=ev.seed,
            )
            out[spec.column] = fitted.predict_proba(test[cols])
            self.fit_log.append({"fold": fold_index, "model": spec.name, "n_train": len(train), **fitted.params})
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------ public API
    def run_cv(self) -> pd.DataFrame:
        pieces = []
        for fold in self.plan.folds:
            log.info("Fold %d/%d: train to %s, test %s to %s", fold.index + 1, len(self.plan.folds),
                     pd.Timestamp(fold.train_dates[-1]).date(), pd.Timestamp(fold.test_dates[0]).date(),
                     pd.Timestamp(fold.test_dates[-1]).date())
            pieces.append(self._run(fold.index, fold.train_dates, fold.test_dates))
        return pd.concat(pieces, ignore_index=True)

    def run_holdout(self) -> pd.DataFrame:
        """Train on all development dates and predict the untouched final period."""
        log.info("Final holdout: %s to %s", pd.Timestamp(self.plan.holdout_dates[0]).date(),
                 pd.Timestamp(self.plan.holdout_dates[-1]).date())
        return self._run(-1, self.plan.holdout_train_dates, self.plan.holdout_dates)
