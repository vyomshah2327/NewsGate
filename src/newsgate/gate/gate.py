"""Second-stage gate.

For each forecast the gate learns, from *earlier out-of-fold* results only, how much lower the
Brier loss of the news-enhanced model is than the price-only model (``delta``), and how
large the price-only loss tends to be (``loss``). It then chooses:

* ``news``  if the predicted gain exceeds ``margin``,
* ``price`` otherwise,
* ``abstain`` when the expected loss of the chosen forecast is among the least trusted,
  using a threshold set on the gate's own training history (never on the test period).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from newsgate.config import GateConfig

GATE_FEATURES = [
    "att_log_n", "att_surprise", "att_days_since", "sent_mean", "sent_std", "sent_disagree",
    "evt_novel_share", "evt_max_cluster", "px_vol_20", "px_absret_0", "px_mkt_vol_20",
]


def _make_regressor(kind: str, seed: int):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))
    if kind == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            num_leaves=4, n_estimators=150, learning_rate=0.03, min_child_samples=300,
            subsample=0.8, subsample_freq=1, reg_lambda=10.0, random_state=seed, n_jobs=1, verbose=-1,
        )
    raise ValueError(f"Unknown gate regressor: {kind}")


class IncrementalValueGate:
    def __init__(self, cfg: GateConfig, price_col: str, news_col: str, seed: int = 0):
        self.cfg, self.price_col, self.news_col, self.seed = cfg, price_col, news_col, seed
        self.features_: list[str] = []
        self.delta_model_ = None
        self.loss_model_ = None
        self.abstain_threshold_: float | None = None

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        p_price, p_news = frame[self.price_col], frame[self.news_col]
        design = frame[self.features_].copy()
        design["p_price"], design["p_news"] = p_price, p_news
        design["p_diff"] = p_news - p_price
        design["p_absdiff"] = (p_news - p_price).abs()
        return design.to_numpy(dtype="float64")

    def fit(self, history: pd.DataFrame) -> "IncrementalValueGate":
        if history.empty:
            raise ValueError("Gate history is empty")
        self.features_ = [c for c in GATE_FEATURES if c in history.columns]
        y = history["y"].to_numpy(dtype="float64")
        loss_price = (history[self.price_col].to_numpy() - y) ** 2
        loss_news = (history[self.news_col].to_numpy() - y) ** 2
        design = self._design(history)
        self.delta_model_ = _make_regressor(self.cfg.regressor, self.seed).fit(design, loss_price - loss_news)
        self.loss_model_ = _make_regressor(self.cfg.regressor, self.seed).fit(design, loss_price)
        _, _, expected = self._predict_parts(history)
        self.abstain_threshold_ = float(np.quantile(expected, self.cfg.default_coverage))
        return self

    def _predict_parts(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        design = self._design(frame)
        delta = np.asarray(self.delta_model_.predict(design))
        loss = np.asarray(self.loss_model_.predict(design))
        expected = loss - np.maximum(delta - self.cfg.margin, 0.0)
        return delta, loss, expected

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``p__gate`` (the chosen forecast), ``gate_action`` and ``gate_exp_loss``."""
        if self.delta_model_ is None:
            raise RuntimeError("Gate has not been fit")
        delta, _, expected = self._predict_parts(frame)
        use_news = delta > self.cfg.margin
        chosen = np.where(use_news, frame[self.news_col].to_numpy(), frame[self.price_col].to_numpy())
        action = np.where(expected > self.abstain_threshold_, "abstain", np.where(use_news, "news", "price"))
        return pd.DataFrame(
            {"p__gate": chosen, "gate_action": action, "gate_exp_loss": expected, "gate_delta": delta},
            index=frame.index,
        )
