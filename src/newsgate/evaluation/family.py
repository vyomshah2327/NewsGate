"""The pre-registered family of tests (see docs/preregistration.md).

For each horizon there are five comparisons against the logistic price-only model:

* T1 price + attention, T2 price + sentiment, T3 price + all news (all rows),
* T4 price + all news on high-attention days only, and
* T5 the gate (rows where the gate has enough history).

With two horizons that is ten tests, and Holm correction is applied across all ten together. The
single primary test is T3 at the longer horizon. Nothing is fit or tuned here: every test only
compares out-of-fold predictions that already exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from newsgate.config import Config, EvalConfig
from newsgate.evaluation.comparisons import compare_models
from newsgate.evaluation.multitest import holm_adjust
from newsgate.evaluation.runner import column_for

KIND = "logistic"
HIGH_ATTENTION_SURPRISE = 1.0  # att_surprise >= 1: article count one smoothed std above the ticker's own average
PRICE_ONLY = column_for(f"{KIND}:price_only")

TESTS = [
    ("T1", "price + attention", column_for(f"{KIND}:price_attention"), "all rows"),
    ("T2", "price + sentiment", column_for(f"{KIND}:price_sentiment"), "all rows"),
    ("T3", "price + all news", column_for(f"{KIND}:price_news"), "all rows"),
    ("T4", "price + all news", column_for(f"{KIND}:price_news"), "high-attention days"),
    ("T5", "gate", "p__gate", "gate rows"),
]


def high_attention_mask(frame: pd.DataFrame) -> pd.Series:
    return (frame["att_has_news"] > 0) & (frame["att_surprise"] >= HIGH_ATTENTION_SURPRISE)


def compute_family(frames: Mapping[int, pd.DataFrame], eval_cfgs: Mapping[int, EvalConfig]) -> pd.DataFrame:
    """Run all tests. ``frames`` maps horizon to out-of-fold rows that also carry the attention columns."""
    primary_horizon = max(frames)
    rows = []
    for horizon in sorted(frames):
        frame = frames[horizon]
        for test_id, label, challenger, subset in TESTS:
            part = frame[high_attention_mask(frame)] if subset == "high-attention days" else frame
            result = compare_models(part, [(label, challenger, PRICE_ONLY)], eval_cfgs[horizon])
            if result.empty:
                raise ValueError(f"{test_id} at horizon {horizon} has no rows; check the prediction columns")
            row = result.iloc[0].to_dict()
            row.update({"test": test_id, "horizon": horizon, "subset": subset,
                        "primary": test_id == "T3" and horizon == primary_horizon})
            rows.append(row)
    table = pd.DataFrame(rows)
    table["p_holm_family"] = holm_adjust(table["p_value"].to_numpy())
    table["family_size"] = len(table)
    table["significant_family"] = (table["p_holm_family"] < eval_cfgs[primary_horizon].alpha) & (table["mean_diff"] < 0)
    cols = ["test", "horizon", "comparison", "subset", "primary", "n_rows", "n_dates", "baseline_brier", "mean_diff",
            "relative_change_pct", "ci_low", "ci_high", "p_value", "p_holm_family", "significant_family", "family_size"]
    return table[cols]


def _load_frame(cfg: Config, split: str) -> pd.DataFrame:
    root = Path(cfg.run.output_dir) / cfg.run.name
    oof = pd.read_parquet(root / ("oof_cv.parquet" if split == "cv" else "oof_holdout.parquet"))
    wanted = [c for c in ("att_has_news", "att_surprise") if c not in oof.columns]  # the gate stage may have added them
    if not wanted:
        return oof
    panel = pd.read_parquet(root / "panel.parquet", columns=["ticker", "date"] + wanted)
    return oof.merge(panel, on=["ticker", "date"], how="left")


def run_family(cfgs: list[Config], split: str = "cv", out_dir: str | Path | None = None) -> pd.DataFrame:
    """Load each configured run, compute the family, and write ``family_<split>.csv``."""
    by_horizon = {c.target.horizon: c for c in cfgs}
    if len(by_horizon) != len(cfgs):
        raise ValueError("Each configuration must use a different horizon")
    table = compute_family({h: _load_frame(c, split) for h, c in by_horizon.items()},
                           {h: c.evaluation for h, c in by_horizon.items()})
    target = Path(out_dir) if out_dir else Path(cfgs[0].run.output_dir) / "family"
    target.mkdir(parents=True, exist_ok=True)
    table.to_csv(target / f"family_{split}.csv", index=False)
    return table
