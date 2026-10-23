"""Run the gate sequentially over folds.

For fold ``k`` the gate is fit only on out-of-fold results from earlier folds whose labels were
already resolved before fold ``k`` begins (``tgt_end_date`` earlier than the first test date).
The first ``min_history_folds`` folds therefore have no gate output.
"""

from __future__ import annotations

import pandas as pd

from newsgate.config import Config
from newsgate.evaluation.runner import column_for
from newsgate.gate.gate import GATE_FEATURES, IncrementalValueGate
from newsgate.logging_utils import get_logger

log = get_logger(__name__)


def _with_features(oof: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    cols = ["ticker", "date"] + [c for c in GATE_FEATURES + ["att_n", "sector"] if c in panel.columns]
    return oof.merge(panel[cols], on=["ticker", "date"], how="left")


def _fit_gate(cfg: Config, history: pd.DataFrame) -> IncrementalValueGate:
    price_col = column_for(f"{cfg.gate.kind}:price_only")
    news_col = column_for(f"{cfg.gate.kind}:price_news")
    for col in (price_col, news_col):
        if col not in history:
            raise KeyError(f"Gate needs prediction column {col}; add it to evaluation.kinds/feature_sets")
    return IncrementalValueGate(cfg.gate, price_col, news_col, cfg.evaluation.seed).fit(history)


def run_gate(cv_oof: pd.DataFrame, panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Cross-validation rows with gate columns added (empty gate columns before enough history)."""
    data = _with_features(cv_oof, panel)
    pieces = []
    for k in sorted(data["fold"].unique()):
        test = data[data["fold"] == k]
        if k < cfg.gate.min_history_folds:
            pieces.append(test)
            continue
        history = data[(data["fold"] < k) & (data["tgt_end_date"] < test["date"].min())]
        gate = _fit_gate(cfg, history)
        pieces.append(test.join(gate.predict(test)))
        log.info("Gate fold %d: fit on %d rows", k, len(history))
    return pd.concat(pieces)


def run_gate_holdout(cv_oof: pd.DataFrame, holdout_oof: pd.DataFrame, panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Holdout rows with gate columns; the gate is fit on all resolved cross-validation results."""
    data = _with_features(cv_oof, panel)
    hold = _with_features(holdout_oof, panel)
    history = data[data["tgt_end_date"] < hold["date"].min()]
    gate = _fit_gate(cfg, history)
    return hold.join(gate.predict(hold))
