"""Risk-coverage comparison of the gate against simple abstention rules.

Every variant is scored on the same rows. A variant supplies a forecast and an uncertainty
score; rows are ranked by that score and the Brier loss of the most trusted ``coverage``
fraction is reported. The controls are matched on coverage by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsgate.config import GateConfig
from newsgate.evaluation.metrics import risk_coverage
from newsgate.evaluation.runner import column_for


def _variants(frame: pd.DataFrame, price_col: str, news_col: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    p_price, p_news = frame[price_col].to_numpy(), frame[news_col].to_numpy()
    confidence = lambda p: p * (1.0 - p)  # expected Brier loss if p is calibrated
    variants = {
        "Gate": (frame["p__gate"].to_numpy(), frame["gate_exp_loss"].to_numpy()),
        "Price-only, confidence": (p_price, confidence(p_price)),
        "News model, confidence": (p_news, confidence(p_news)),
    }
    if "att_n" in frame:
        rule = np.where(frame["att_n"].fillna(0).to_numpy() > 0, p_news, p_price)
        variants["Attention rule, confidence"] = (rule, confidence(rule))
    return variants


def selective_risk_table(frame: pd.DataFrame, cfg: GateConfig) -> pd.DataFrame:
    """Long table (variant, coverage, risk) on the rows where the gate produced output."""
    price_col = column_for(f"{cfg.kind}:price_only")
    news_col = column_for(f"{cfg.kind}:price_news")
    rows = frame.dropna(subset=["p__gate", "gate_exp_loss"])
    if rows.empty:
        return pd.DataFrame(columns=["variant", "coverage", "risk"])
    y = rows["y"].to_numpy(dtype="float64")
    pieces = []
    for name, (p, score) in _variants(rows, price_col, news_col).items():
        curve = risk_coverage((p - y) ** 2, score)
        curve.insert(0, "variant", name)
        pieces.append(curve)
    return pd.concat(pieces, ignore_index=True)
