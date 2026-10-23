"""Purged expanding-window splits over *dates*.

The panel has many tickers per date, so splits are made on the unique trading dates and every
row of a date goes to the same side. Training dates stop ``purge_days`` before the test block
starts, which is at least the target horizon, so no training label overlaps the test period.

The final ``holdout_fraction`` of dates is set aside and is only used by ``run_holdout``; the
cross-validation folds never see it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    index: int
    train_dates: np.ndarray
    test_dates: np.ndarray


@dataclass(frozen=True)
class SplitPlan:
    folds: list[Fold]
    dev_dates: np.ndarray
    holdout_train_dates: np.ndarray
    holdout_dates: np.ndarray


class PurgedExpandingWindowSplit:
    def __init__(self, min_train_days: int, test_block_days: int, purge_days: int, holdout_fraction: float):
        if purge_days < 1:
            raise ValueError("purge_days must be at least 1")
        if min_train_days <= purge_days:
            raise ValueError("min_train_days must exceed purge_days")
        self.min_train_days = min_train_days
        self.test_block_days = test_block_days
        self.purge_days = purge_days
        self.holdout_fraction = holdout_fraction

    def partition(self, dates: np.ndarray) -> SplitPlan:
        unique = np.unique(np.asarray(dates, dtype="datetime64[ns]"))
        n = len(unique)
        hold_idx = int(n * (1.0 - self.holdout_fraction))
        dev = unique[: max(hold_idx - self.purge_days, 0)]
        holdout = unique[hold_idx:]
        if len(dev) < self.min_train_days + self.test_block_days // 2:
            raise ValueError(
                f"Not enough dates ({n}) for min_train_days={self.min_train_days}, "
                f"test_block_days={self.test_block_days} and holdout_fraction={self.holdout_fraction}"
            )
        folds: list[Fold] = []
        start = self.min_train_days
        while start < len(dev):
            stop = min(start + self.test_block_days, len(dev))
            if stop - start >= self.test_block_days // 2:  # drop a tiny trailing block
                folds.append(Fold(len(folds), dev[: start - self.purge_days], dev[start:stop]))
            start += self.test_block_days
        return SplitPlan(folds, dev, dev, holdout)
