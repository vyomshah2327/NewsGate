import numpy as np
import pandas as pd
import pytest

from newsgate.config import EvalConfig
from newsgate.evaluation import (
    PurgedExpandingWindowSplit, brier, brier_skill, compare_models, date_block_bootstrap,
    expected_calibration_error, holm_adjust, log_loss, reliability_table, risk_coverage,
)
from newsgate.evaluation.metrics import auroc
from newsgate.models import PlattCalibrator, available_kinds, create_model, fit_tuned
from newsgate.models.tuning import inner_split


# ----------------------------------------------------------------------------- splits
def test_split_has_purge_gap_and_no_overlap():
    dates = pd.bdate_range("2018-01-01", periods=1000).to_numpy()
    plan = PurgedExpandingWindowSplit(300, 60, purge_days=5, holdout_fraction=0.2).partition(dates)
    assert len(plan.folds) >= 5
    for fold in plan.folds:
        gap = np.searchsorted(plan.dev_dates, fold.test_dates[0]) - np.searchsorted(plan.dev_dates, fold.train_dates[-1])
        assert gap > 5, "at least purge_days dates must separate train from test"
        assert fold.train_dates.max() < fold.test_dates.min()
    tests = np.concatenate([f.test_dates for f in plan.folds])
    assert len(tests) == len(np.unique(tests)), "test blocks must not overlap"
    assert tests.max() < plan.holdout_dates.min(), "cross-validation never touches the holdout"
    assert plan.dev_dates.max() < plan.holdout_dates.min()
    assert len(plan.folds[0].train_dates) >= 300 - 5


def test_split_rejects_too_little_data():
    with pytest.raises(ValueError):
        PurgedExpandingWindowSplit(500, 60, 5, 0.2).partition(pd.bdate_range("2020-01-01", periods=300).to_numpy())


def test_split_requires_positive_purge():
    with pytest.raises(ValueError):
        PurgedExpandingWindowSplit(100, 20, 0, 0.2)


# ----------------------------------------------------------------------------- metrics
def test_brier_and_skill():
    y = np.array([1, 0, 0, 1])
    assert brier(y, np.array([1.0, 0.0, 0.0, 1.0])) == 0.0
    assert brier(y, np.full(4, 0.5)) == pytest.approx(0.25)
    assert brier_skill(y, np.full(4, 0.5), np.full(4, 0.5)) == pytest.approx(0.0)
    assert brier_skill(y, y.astype(float), np.full(4, 0.5)) == pytest.approx(1.0)


def test_log_loss_and_auroc_edge_cases():
    assert log_loss(np.array([1, 0]), np.array([1.0, 0.0])) == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(auroc(np.zeros(5), np.linspace(0, 1, 5)))
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_calibration_error_is_small_for_calibrated_forecasts():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.5, 40000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    table = reliability_table(y, p, 10)
    assert len(table) == 10 and (table["mean_pred"].diff().dropna() > 0).all()


def test_risk_coverage_falls_when_score_is_informative():
    rng = np.random.default_rng(1)
    noise = rng.uniform(size=5000)
    loss = noise + 0.05 * rng.standard_normal(5000)
    informative = risk_coverage(loss, noise)
    random = risk_coverage(loss, rng.uniform(size=5000))
    assert informative["risk"].iloc[0] < informative["risk"].iloc[-1]
    assert informative["risk"].iloc[-1] == pytest.approx(random["risk"].iloc[-1])
    assert informative["risk"].iloc[10] < random["risk"].iloc[10]


# ----------------------------------------------------------------------------- statistics
def test_holm_matches_hand_calculation():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = holm_adjust([0.5, np.nan, 0.001])
    assert np.isnan(out[1]) and out[2] == pytest.approx(0.002)
    assert (holm_adjust([0.2, 0.9]) >= np.array([0.2, 0.9])).all()


def test_block_bootstrap_detects_a_real_shift_and_not_noise():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=400)
    noise = pd.Series(rng.standard_normal(400) * 0.01, index=idx)
    null = date_block_bootstrap(noise, block_length=10, n_resamples=1000, seed=1)
    assert null["ci_low"] < 0 < null["ci_high"] and null["p_value"] > 0.05
    shifted = date_block_bootstrap(noise - 0.01, block_length=10, n_resamples=1000, seed=1)
    assert shifted["ci_high"] < 0 and shifted["p_value"] < 0.05
    again = date_block_bootstrap(noise - 0.01, block_length=10, n_resamples=1000, seed=1)
    assert again == shifted, "results must be reproducible for a fixed seed"


def test_block_bootstrap_widens_interval_for_autocorrelated_series():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=600)
    e = rng.standard_normal(600)
    ar = np.zeros(600)
    for t in range(1, 600):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    series = pd.Series(ar, index=idx)
    narrow = date_block_bootstrap(series, block_length=1, n_resamples=1000, seed=0)
    wide = date_block_bootstrap(series, block_length=30, n_resamples=1000, seed=0)
    assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"])


def test_compare_models_sign_convention_and_holm_column():
    rng = np.random.default_rng(4)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=300), 5)
    y = (rng.uniform(size=dates.size) < 0.2).astype(int)
    good = np.clip(0.2 + 0.5 * (y - 0.2) + 0.05 * rng.standard_normal(y.size), 0.01, 0.99)
    base = np.full(y.size, 0.2)
    frame = pd.DataFrame({"date": dates, "y": y, "good": good, "base": base})
    table = compare_models(frame, [("good vs base", "good", "base")], EvalConfig(bootstrap_resamples=300))
    row = table.iloc[0]
    assert row["mean_diff"] < 0 and bool(row["challenger_better"]) and "p_holm" in table


# ----------------------------------------------------------------------------- models
def test_registry_and_historical_rate_shrinkage():
    assert {"historical_rate", "logistic", "lightgbm"} <= set(available_kinds())
    with pytest.raises(KeyError):
        create_model("nope")
    X = pd.DataFrame({"ticker": ["A"] * 100 + ["B"] * 2})
    y = np.array([1] * 20 + [0] * 80 + [1, 1])
    model = create_model("historical_rate", shrinkage=20.0).fit(X, y)
    pa, pb, pc = model.predict_proba(pd.DataFrame({"ticker": ["A", "B", "Z"]}))
    assert pa == pytest.approx((20 + 20 * (22 / 102)) / 120)
    assert pb < 1.0 and pb > pc and pc == pytest.approx(22 / 102)


def test_inner_split_leaves_a_gap():
    dates = pd.bdate_range("2020-01-01", periods=500).to_numpy()
    tr, va = inner_split(np.repeat(dates, 3), val_fraction=0.2, purge_days=5)
    row_dates = np.repeat(dates, 3)
    assert row_dates[tr].max() < row_dates[va].min()
    assert np.searchsorted(dates, row_dates[va].min()) - np.searchsorted(dates, row_dates[tr].max()) > 5


def test_platt_calibrator_fixes_overconfident_forecasts():
    rng = np.random.default_rng(5)
    true_p = rng.uniform(0.05, 0.4, 20000)
    y = (rng.uniform(size=true_p.size) < true_p).astype(int)
    overconfident = np.clip(true_p * 2.0, 0.01, 0.99)
    fixed = PlattCalibrator().fit(overconfident, y).transform(overconfident)
    assert expected_calibration_error(y, fixed) < expected_calibration_error(y, overconfident)


@pytest.mark.parametrize("kind", ["logistic", "lightgbm"])
def test_fit_tuned_learns_a_signal_and_predicts_probabilities(kind):
    rng = np.random.default_rng(6)
    n = 4000
    dates = np.repeat(pd.bdate_range("2019-01-01", periods=n // 4).to_numpy(), 4)
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(2.0 * X["a"] - 2.0)))).astype(int)
    fitted = fit_tuned(kind, X.iloc[:3000], y[:3000], dates[:3000], purge_days=5)
    p = fitted.predict_proba(X.iloc[3000:])
    assert ((p >= 0) & (p <= 1)).all() and auroc(y[3000:], p) > 0.75
