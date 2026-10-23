"""Shared fixtures: a small synthetic world that is cheap enough to rebuild per test session."""

from __future__ import annotations

import pytest

from newsgate.config import Config, config_from_dict
from newsgate.data.synthetic import SyntheticWorld


def small_config(**overrides) -> Config:
    raw = {
        "data": {"source": "synthetic", "synthetic": {"n_tickers": 6, "n_days": 700, "seed": 3, "attention_effect": 1.2}},
        "sentiment": {"scorers": ["vader"]},
        "evaluation": {
            "min_train_days": 250, "test_block_days": 80, "purge_days": 5, "holdout_fraction": 0.2,
            "bootstrap_resamples": 200, "kinds": ["logistic"], "feature_sets": ["price_only", "price_attention", "price_news"],
        },
        "gate": {"kind": "logistic", "min_history_folds": 1, "regressor": "ridge"},
    }
    for key, value in overrides.items():
        raw[key] = {**raw.get(key, {}), **value}
    return config_from_dict(raw)


@pytest.fixture(scope="session")
def cfg() -> Config:
    return small_config()


@pytest.fixture(scope="session")
def world(cfg) -> SyntheticWorld:
    return SyntheticWorld.generate(cfg.data.synthetic)
