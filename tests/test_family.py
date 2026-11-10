import numpy as np
import pandas as pd
import pytest

from newsgate.config import EvalConfig
from newsgate.evaluation.family import PRICE_ONLY, TESTS, compute_family, high_attention_mask
from newsgate.evaluation.runner import column_for


def _frame(seed, news_helps, n_days=300, n_tickers=6):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_days), n_tickers)
    y = (rng.uniform(size=dates.size) < 0.15).astype(int)
    base = np.full(y.size, 0.15)
    news = np.clip(base + (0.5 * (y - 0.15) if news_helps else 0.0) + 0.02 * rng.standard_normal(y.size), 0.01, 0.99)
    frame = pd.DataFrame({"date": dates, "ticker": np.tile(np.arange(n_tickers), n_days), "y": y,
                          PRICE_ONLY: base, "att_has_news": 1.0, "att_surprise": rng.normal(0.5, 1.0, y.size)})
    for _, _, col, _ in TESTS:
        frame[col] = news
    frame["p__gate"] = news
    return frame


def test_family_has_ten_tests_with_family_wide_holm_and_one_primary():
    cfg = EvalConfig(bootstrap_resamples=200)
    table = compute_family({1: _frame(0, True), 5: _frame(1, False)}, {1: cfg, 5: cfg})
    assert len(table) == 10 and set(table["horizon"]) == {1, 5}
    assert table["primary"].sum() == 1
    primary = table[table["primary"]].iloc[0]
    assert primary["test"] == "T3" and primary["horizon"] == 5
    assert (table["p_holm_family"] >= table["p_value"] - 1e-12).all()
    assert (table["family_size"] == 10).all()


def test_helpful_news_is_significant_and_useless_news_is_not():
    cfg = EvalConfig(bootstrap_resamples=300)
    table = compute_family({1: _frame(2, True), 5: _frame(3, False)}, {1: cfg, 5: cfg}).set_index(["horizon", "test"])
    assert table.loc[(1, "T3"), "significant_family"]
    assert not table.loc[(5, "T3"), "significant_family"]


def test_high_attention_subset_only_uses_rows_with_unusual_news():
    frame = _frame(4, True)
    mask = high_attention_mask(frame)
    assert 0.05 < mask.mean() < 0.95
    assert (frame.loc[mask, "att_surprise"] >= 1.0).all()
    cfg = EvalConfig(bootstrap_resamples=100)
    table = compute_family({1: frame}, {1: cfg}).set_index("test")
    assert table.loc["T4", "n_rows"] == int(mask.sum()) and table.loc["T3", "n_rows"] == len(frame)


def test_missing_prediction_column_is_an_error():
    frame = _frame(5, True).drop(columns=["p__gate"])
    with pytest.raises(Exception):
        compute_family({1: frame}, {1: EvalConfig(bootstrap_resamples=50)})


def test_loader_does_not_duplicate_columns_the_gate_already_added(tmp_path):
    from newsgate.config import config_from_dict
    from newsgate.evaluation.family import _load_frame

    root = tmp_path / "r"
    root.mkdir()
    frame = _frame(6, True, n_days=30)
    frame.to_parquet(root / "oof_cv.parquet")  # already has att_has_news and att_surprise
    pd.DataFrame({"ticker": frame["ticker"], "date": frame["date"], "att_has_news": 0.0, "att_surprise": 99.0}).to_parquet(root / "panel.parquet")
    cfg = config_from_dict({"run": {"name": "r", "output_dir": str(tmp_path)}})
    out = _load_frame(cfg, "cv")
    assert "att_surprise_x" not in out and out["att_surprise"].max() < 50, "the oof columns must be kept as they are"
