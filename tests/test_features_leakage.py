"""The strongest leakage control: features for a date must not change when the future is removed."""

import numpy as np
import pandas as pd
import pytest

from newsgate.features.panel import build_panel
from newsgate.features.sets import FEATURE_SETS, columns_for
from newsgate.labels import LargeMoveThreshold


@pytest.fixture(scope="module")
def full_panel(cfg, world):
    panel, _ = build_panel(cfg, world.news, world.prices, world.tickers, world.metadata)
    return panel


def test_features_do_not_depend_on_future_data(cfg, world, full_panel):
    dates = np.sort(full_panel["date"].unique())
    cutoff = pd.Timestamp(dates[len(dates) // 2])
    prices = world.prices[world.prices["date"] <= cutoff]
    news = world.news[world.news["published_at"] < cutoff + pd.Timedelta(days=1)]
    truncated, _ = build_panel(cfg, news, prices, world.tickers, world.metadata)

    feature_cols = [c for c in full_panel.columns if c.split("_")[0] in {"px", "att", "sent", "evt"}]
    key = ["ticker", "date"]
    left = full_panel[full_panel["date"] <= cutoff].set_index(key)[feature_cols].sort_index()
    right = truncated[truncated["date"] <= cutoff].set_index(key)[feature_cols].sort_index()
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-9, atol=1e-12)


def test_targets_look_forward_only_and_end_after_start(full_panel):
    valid = full_panel.dropna(subset=["tgt_end_date"])
    assert (valid["tgt_end_date"] > valid["date"]).all()
    assert np.allclose(valid["tgt_abs"], valid["tgt_fwd_ret"].abs())


def test_feature_sets_partition_columns_by_prefix(full_panel):
    price = set(columns_for(full_panel.columns, "price_only"))
    news = set(columns_for(full_panel.columns, "price_news"))
    assert price < news
    assert all(c.startswith("px_") for c in price)
    assert not any(c.startswith("tgt_") for c in news), "targets must never be features"
    assert columns_for(full_panel.columns, "historical_rate") == []
    assert set(FEATURE_SETS) >= {"price_only", "price_attention", "price_sentiment", "price_news"}


def test_news_after_the_close_is_not_visible_on_its_own_day(cfg, world):
    """Every article published at or after 16:00 New York time is used strictly after its own date.

    (Dropping evening articles and comparing panels is not a valid test: deduplication keeps the
    earliest copy, so removing an evening copy can let a later copy survive on another day.)
    """
    panel, prepared = build_panel(cfg, world.news, world.prices, world.tickers, world.metadata)
    ny = pd.to_datetime(prepared["published_at"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    local_date = ny.dt.tz_localize(None).dt.normalize()
    minutes = ny.dt.hour * 60 + ny.dt.minute
    after_close = (minutes >= 16 * 60).to_numpy()
    assert after_close.any(), "the synthetic world should contain evening articles"
    assert (prepared.loc[after_close, "feature_date"].to_numpy() > local_date[after_close].to_numpy()).all()
    assert (prepared["feature_date"].to_numpy() >= local_date.to_numpy()).all()
    # What the panel sees is exactly what was prepared: article counts per day agree.
    counted = prepared.groupby(["ticker", "feature_date"]).size()
    merged = panel.set_index(["ticker", "date"])["att_n"]
    shared = counted.index.intersection(merged.index)
    assert (merged.loc[shared].to_numpy() == counted.loc[shared].to_numpy()).all()


def test_large_move_threshold_uses_training_rows_only(full_panel):
    dates = np.sort(full_panel["date"].unique())
    train = full_panel[full_panel["date"] < dates[300]].dropna(subset=["tgt_abs"])
    test = full_panel[full_panel["date"] >= dates[300]].dropna(subset=["tgt_abs"]).copy()
    base = LargeMoveThreshold(0.9, "ticker").fit(train)
    before = base.thresholds(test)
    test["tgt_abs"] = test["tgt_abs"] * 100.0  # drastically change the test period
    assert np.array_equal(base.thresholds(test), before)
    labels = base.label(full_panel.dropna(subset=["tgt_abs"]))
    assert set(np.unique(labels)) <= {0, 1}
    assert 0.07 < labels[: len(train)].mean() < 0.13
