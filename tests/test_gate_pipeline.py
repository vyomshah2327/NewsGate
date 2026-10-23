import json
import os

import numpy as np
import pandas as pd
import pytest

from newsgate import pipeline
from newsgate.config import ConfigError, GateConfig, config_from_dict, load_config
from newsgate.evaluation.runner import build_specs, column_for
from newsgate.gate import IncrementalValueGate, evidence_label, run_gate, selective_risk_table
from newsgate.gate import runner as gate_runner
from tests.conftest import small_config


# ----------------------------------------------------------------------------- config
def test_unknown_config_key_is_rejected():
    with pytest.raises(ConfigError, match="Unknown"):
        config_from_dict({"evaluation": {"min_train_dayz": 5}})


def test_purge_must_cover_the_horizon():
    with pytest.raises(ConfigError, match="purge_days"):
        config_from_dict({"target": {"horizon": 10}, "evaluation": {"purge_days": 5}})


def test_yaml_configs_in_the_repo_load():
    root = os.path.dirname(os.path.dirname(__file__))
    for name in ("default.yaml", "synthetic.yaml"):
        assert load_config(os.path.join(root, "configs", name)).data.source in {"fnspid", "synthetic"}


def test_build_specs_covers_the_ladder(cfg):
    names = [s.name for s in build_specs(cfg.evaluation)]
    assert names == ["historical_rate", "logistic:price_only", "logistic:price_attention", "logistic:price_news"]


# ----------------------------------------------------------------------------- gate
def _history(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = (rng.uniform(size=n) < 0.15).astype(int)
    p_price = np.full(n, 0.15)
    # The news model is sharper only when x > 0 and noisy garbage otherwise.
    p_news = np.where(x > 0, np.where(y == 1, 0.6, 0.05), np.clip(0.15 + 0.3 * rng.standard_normal(n), 0.01, 0.99))
    return pd.DataFrame({"att_log_n": x, "y": y, "p_price": p_price, "p_news": p_news})


def test_gate_learns_when_news_helps_and_abstains_on_the_least_trusted():
    cfg = GateConfig(regressor="lightgbm", default_coverage=0.8)
    hist, test = _history(seed=0), _history(seed=1)
    gate = IncrementalValueGate(cfg, "p_price", "p_news").fit(hist)
    out = gate.predict(test)
    uses_news = out["gate_action"].isin(["news"])
    assert test.loc[test["att_log_n"] > 0.5].index.isin(out.index[uses_news]).mean() > 0.5
    assert test.loc[test["att_log_n"] < -0.5].index.isin(out.index[uses_news]).mean() < 0.2
    assert 0.1 < (out["gate_action"] == "abstain").mean() < 0.35
    assert set(out["gate_action"]) <= {"news", "price", "abstain"}


def test_evidence_labels():
    assert evidence_label(0, 0.0) == "none"
    assert evidence_label(1, 1.0) == "weak"
    assert evidence_label(5, 0.1) == "weak"
    assert evidence_label(4, 0.5) == "moderate"
    assert evidence_label(4, 0.9) == "strong"


def test_gate_only_sees_resolved_history(monkeypatch):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=400)
    rows = []
    for k in range(4):
        block = dates[k * 100:(k + 1) * 100]
        rows.append(pd.DataFrame({
            "ticker": "A", "date": block, "tgt_end_date": block + pd.offsets.BDay(1), "fold": k,
            "y": rng.integers(0, 2, 100), "p__logistic__price_only": 0.5, "p__logistic__price_news": 0.4,
        }))
    oof = pd.concat(rows, ignore_index=True)
    panel = oof[["ticker", "date"]].assign(att_log_n=1.0)
    seen = []
    real_fit = gate_runner._fit_gate

    def spy(cfg, history):
        seen.append((history["tgt_end_date"].max(), history["fold"].max()))
        return real_fit(cfg, history)

    monkeypatch.setattr(gate_runner, "_fit_gate", spy)
    cfg = small_config()
    out = run_gate(oof, panel, cfg)
    assert len(seen) >= 2
    for (last_resolved, last_fold), k in zip(seen, [f for f in range(4) if f >= cfg.gate.min_history_folds]):
        test_start = oof.loc[oof["fold"] == k, "date"].min()
        assert last_resolved < test_start and last_fold < k
    assert out["p__gate"].notna().sum() > 0 and out.loc[out["fold"] < cfg.gate.min_history_folds, "p__gate"].isna().all()


# ----------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    cfg = small_config(run={"name": "t", "output_dir": str(out)})
    pipeline.run_all(cfg, final=True)
    return cfg, pipeline.RunPaths.from_config(cfg).root


def test_pipeline_writes_all_artifacts(run_result):
    _, root = run_result
    for name in ("panel.parquet", "news_prepared.parquet", "oof_cv.parquet", "oof_holdout.parquet", "comparisons.csv",
                 "comparisons_holdout.csv", "robustness.csv", "model_metrics.csv", "selective_risk.csv", "summary.json", "config.yaml"):
        assert (root / name).exists(), name
    for fig in ("fig1_brier_differences.png", "fig2_calibration.png", "fig3_risk_coverage.png",
                "fig4_sentiment_deciles.png", "fig5_event_timeline.png"):
        assert (root / "figures" / fig).stat().st_size > 5000, fig


def test_cv_never_touches_the_holdout_period(run_result):
    _, root = run_result
    cv, hold = pd.read_parquet(root / "oof_cv.parquet"), pd.read_parquet(root / "oof_holdout.parquet")
    assert cv["date"].max() < hold["date"].min()
    assert cv["tgt_end_date"].max() < hold["date"].min() + pd.Timedelta(days=10)
    assert set(hold["fold"]) == {-1}


def test_probabilities_are_valid_and_models_are_not_wildly_worse_than_the_naive_rate(run_result):
    _, root = run_result
    cv = pd.read_parquet(root / "oof_cv.parquet")
    p_cols = [c for c in cv if c.startswith("p__")]
    assert p_cols and all(cv[c].dropna().between(0, 1).all() for c in p_cols)
    metrics = pd.read_csv(root / "model_metrics.csv").set_index("model")
    # A sanity bound, not a skill claim: in a small synthetic world the price-only model may tie the
    # historical rate, but a broken model would be clearly worse.
    assert metrics.loc["logistic:price_only", "brier"] < 1.05 * metrics.loc["historical_rate", "brier"]


def test_planted_attention_effect_is_detected(run_result):
    """The synthetic world plants an attention effect on next-day volatility; the study should find it."""
    _, root = run_result
    table = pd.read_csv(root / "comparisons.csv").set_index("comparison")
    row = table.loc["logistic: + attention vs price-only"]
    assert row["mean_diff"] < 0


def test_null_world_finds_no_significant_gain(tmp_path):
    cfg = small_config(
        data={"synthetic": {"n_tickers": 6, "n_days": 700, "seed": 11, "attention_effect": 0.0, "sentiment_effect": 0.0}},
        run={"name": "null", "output_dir": str(tmp_path)},
    )
    pipeline.run_panel(cfg)
    pipeline.run_cv(cfg)
    tables = pipeline.run_report(cfg, "cv")
    table = tables["comparisons"]
    news_rows = table[~table["comparison"].str.startswith("gate")]
    assert (news_rows["p_holm"] >= 0.05).all(), news_rows[["comparison", "mean_diff", "p_holm"]]


def test_dashboard_renders(run_result, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    _, root = run_result
    monkeypatch.setenv("NEWSGATE_RUN_DIR", str(root))
    app = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "dashboard.py")
    at = AppTest.from_file(app, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5
