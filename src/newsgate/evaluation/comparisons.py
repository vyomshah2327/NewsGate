"""Paired model comparisons with date-block bootstrap intervals and Holm correction.

Sign convention: ``mean_diff`` is Brier(challenger) - Brier(baseline), so a **negative** value
means the challenger forecasts better. Intervals come from a moving-block bootstrap over dates.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from newsgate.config import EvalConfig
from newsgate.evaluation.bootstrap import date_block_bootstrap
from newsgate.evaluation.multitest import holm_adjust


def paired_brier_diff_by_date(frame: pd.DataFrame, challenger: str, baseline: str) -> pd.Series:
    """Mean over tickers of (challenger loss - baseline loss), indexed by date."""
    loss_c = (frame[challenger] - frame["y"]) ** 2
    loss_b = (frame[baseline] - frame["y"]) ** 2
    return (loss_c - loss_b).groupby(frame["date"]).mean()


def compare_models(
    frame: pd.DataFrame, pairs: Sequence[tuple[str, str, str]], cfg: EvalConfig
) -> pd.DataFrame:
    """Compare each ``(label, challenger_column, baseline_column)`` pair on rows where both exist."""
    rows = []
    for label, challenger, baseline in pairs:
        both = frame.dropna(subset=[challenger, baseline])
        if both.empty:
            continue
        diff = paired_brier_diff_by_date(both, challenger, baseline)
        boot = date_block_bootstrap(diff, cfg.block_length, cfg.bootstrap_resamples, cfg.alpha, cfg.seed)
        base_brier = float(((both[baseline] - both["y"]) ** 2).mean())
        rows.append({
            "comparison": label,
            "challenger": challenger,
            "baseline": baseline,
            "n_rows": len(both),
            "baseline_brier": base_brier,
            "mean_diff": boot["mean"],
            "relative_change_pct": 100.0 * boot["mean"] / base_brier if base_brier else np.nan,
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "n_dates": boot["n_dates"],
            "p_value": boot["p_value"],
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm_adjust(table["p_value"].to_numpy())
        table["challenger_better"] = (table["ci_high"] < 0)
    return table


def robustness_table(
    frame: pd.DataFrame, challenger: str, baseline: str, group_col: str, cfg: EvalConfig, min_dates: int = 30
) -> pd.DataFrame:
    """The same paired comparison inside each level of ``group_col`` (sector, regime, ...)."""
    rows = []
    for level, part in frame.dropna(subset=[challenger, baseline, group_col]).groupby(group_col, observed=True):
        diff = paired_brier_diff_by_date(part, challenger, baseline)
        if len(diff) < min_dates:
            continue
        boot = date_block_bootstrap(diff, cfg.block_length, cfg.bootstrap_resamples, cfg.alpha, cfg.seed)
        rows.append({"group": group_col, "level": str(level), "n_rows": len(part), **boot})
    return pd.DataFrame(rows)
