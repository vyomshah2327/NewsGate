"""Orchestration of the stages. Each stage reads and writes files in the run directory, so a
stage can be rerun on its own (for example, regenerate figures without refitting models).

    panel    -> panel.parquet, news_prepared.parquet, prices.parquet
    cv       -> oof_cv.parquet (with gate columns), fit_log.csv
    holdout  -> oof_holdout.parquet   (run once, after the design is frozen)
    report   -> comparisons.csv, robustness.csv, selective_risk.csv, metrics.json, figures/*.png
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from newsgate.config import Config, save_config
from newsgate.data import build_sources
from newsgate.evaluation import Experiment, compare_models, robustness_table
from newsgate.evaluation.metrics import auroc, brier, brier_skill, expected_calibration_error
from newsgate.evaluation.runner import BASELINE, column_for
from newsgate.features.panel import build_panel
from newsgate.gate import run_gate, run_gate_holdout, selective_risk_table
from newsgate.logging_utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @classmethod
    def from_config(cls, cfg: Config) -> "RunPaths":
        return cls(Path(cfg.run.output_dir) / cfg.run.name)

    def file(self, name: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / name

    @property
    def figures(self) -> Path:
        path = self.root / "figures"
        path.mkdir(parents=True, exist_ok=True)
        return path


# --------------------------------------------------------------------------- stages
def run_panel(cfg: Config) -> pd.DataFrame:
    paths = RunPaths.from_config(cfg)
    save_config(cfg, paths.file("config.yaml"))
    sources = build_sources(cfg.data)
    news = sources.news.load()
    prices = sources.prices.load(sources.tickers)
    panel, prepared = build_panel(cfg, news, prices, sources.tickers, sources.metadata(), cfg.data.cache_dir)
    panel.to_parquet(paths.file("panel.parquet"))
    prepared.to_parquet(paths.file("news_prepared.parquet"))
    prices.to_parquet(paths.file("prices.parquet"))
    return panel


def run_cv(cfg: Config) -> pd.DataFrame:
    paths = RunPaths.from_config(cfg)
    panel = pd.read_parquet(paths.file("panel.parquet"))
    experiment = Experiment(panel, cfg)
    oof = experiment.run_cv()
    if cfg.gate.enabled:
        oof = run_gate(oof, panel, cfg)
    oof.to_parquet(paths.file("oof_cv.parquet"))
    pd.DataFrame(experiment.fit_log).to_csv(paths.file("fit_log.csv"), index=False)
    return oof


def run_holdout(cfg: Config) -> pd.DataFrame:
    """Evaluate the untouched final period. Call once, after the design is frozen."""
    paths = RunPaths.from_config(cfg)
    panel = pd.read_parquet(paths.file("panel.parquet"))
    experiment = Experiment(panel, cfg)
    hold = experiment.run_holdout()
    if cfg.gate.enabled:
        cv = pd.read_parquet(paths.file("oof_cv.parquet"))
        hold = run_gate_holdout(cv, hold, panel, cfg)
    hold.to_parquet(paths.file("oof_holdout.parquet"))
    return hold


# --------------------------------------------------------------------------- reporting
def comparison_pairs(cfg: Config) -> list[tuple[str, str, str]]:
    """Challenger versus baseline pairs, limited to the feature sets that are actually configured."""
    have = set(cfg.evaluation.feature_sets)
    templates = [
        ("+ attention vs price-only", "price_attention", "price_only"),
        ("+ sentiment vs price-only", "price_sentiment", "price_only"),
        ("+ all news vs price-only", "price_news", "price_only"),
        ("all news vs + attention", "price_news", "price_attention"),
    ]
    pairs = []
    for kind in cfg.evaluation.kinds:
        for label, challenger, baseline in templates:
            if challenger in have and baseline in have:
                pairs.append((f"{kind}: {label}", column_for(f"{kind}:{challenger}"), column_for(f"{kind}:{baseline}")))
    return pairs


def _enrich(oof: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    extra = [c for c in ("sector", "att_n", "att_has_news", "sent_mean", "px_mkt_vol_20") if c in panel and c not in oof]
    out = oof.merge(panel[["ticker", "date"] + extra], on=["ticker", "date"], how="left")
    coverage = out.groupby("ticker")["att_n"].transform("mean") if "att_n" in out else None
    if coverage is not None:
        out["coverage_group"] = pd.qcut(coverage.rank(method="first"), 3, labels=["low coverage", "mid coverage", "high coverage"])
    if "px_mkt_vol_20" in out:
        out["regime"] = pd.qcut(out["px_mkt_vol_20"].rank(method="first"), 3, labels=["calm market", "normal market", "volatile market"])
    out["year"] = out["date"].dt.year.astype(str)
    return out


def model_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    base = oof[column_for(BASELINE)].to_numpy()
    rows = []
    for col in [c for c in oof if c.startswith("p__")]:
        sub = oof.dropna(subset=[col])
        rows.append({
            "model": col[3:].replace("__", ":"),
            "n_rows": len(sub),
            "brier": brier(sub["y"], sub[col]),
            "brier_skill_vs_rate": brier_skill(sub["y"], sub[col], oof.loc[sub.index, column_for(BASELINE)]),
            "auroc": auroc(sub["y"], sub[col]),
            "ece": expected_calibration_error(sub["y"], sub[col]),
        })
    return pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)


def run_report(cfg: Config, split: str = "cv") -> dict[str, pd.DataFrame]:
    """Write tables and figures for ``split`` (``cv`` or ``holdout``)."""
    from newsgate import viz

    paths = RunPaths.from_config(cfg)
    panel = pd.read_parquet(paths.file("panel.parquet"))
    oof = _enrich(pd.read_parquet(paths.file(f"oof_{split}.parquet")), panel)
    suffix = "" if split == "cv" else "_holdout"
    ev = cfg.evaluation

    pairs = comparison_pairs(cfg)
    gate_ready = cfg.gate.enabled and "p__gate" in oof and oof["p__gate"].notna().any()
    if gate_ready:
        px, news = column_for(f"{cfg.gate.kind}:price_only"), column_for(f"{cfg.gate.kind}:price_news")
        pairs += [("gate vs price-only", "p__gate", px), ("gate vs news model", "p__gate", news)]
    comparisons = compare_models(oof, pairs, ev)
    comparisons.to_csv(paths.file(f"comparisons{suffix}.csv"), index=False)

    news_col, px_col = column_for(f"{cfg.gate.kind}:price_news"), column_for(f"{cfg.gate.kind}:price_only")
    robustness = pd.concat(
        [robustness_table(oof, news_col, px_col, g, ev) for g in ("sector", "coverage_group", "regime", "year") if g in oof],
        ignore_index=True,
    )
    robustness.to_csv(paths.file(f"robustness{suffix}.csv"), index=False)

    metrics = model_metrics(oof)
    metrics.to_csv(paths.file(f"model_metrics{suffix}.csv"), index=False)

    selective = selective_risk_table(oof, cfg.gate) if gate_ready else pd.DataFrame()
    selective.to_csv(paths.file(f"selective_risk{suffix}.csv"), index=False)

    fig = paths.figures
    viz.plot_brier_differences(comparisons, fig / f"fig1_brier_differences{suffix}.png")
    viz.plot_calibration(
        oof, {"Price-only": px_col, "With all news": news_col}, fig / f"fig2_calibration{suffix}.png"
    )
    if not selective.empty:
        viz.plot_risk_coverage(selective, fig / f"fig3_risk_coverage{suffix}.png")
    if {"sent_mean", "att_has_news"} <= set(oof.columns):
        viz.plot_decile_rates(oof, fig / f"fig4_sentiment_deciles{suffix}.png")
    news_prepared = pd.read_parquet(paths.file("news_prepared.parquet"))
    prices = pd.read_parquet(paths.file("prices.parquet"))
    busiest = news_prepared["ticker"].value_counts().index[0]
    viz.plot_event_timeline(prices, news_prepared, busiest, fig / "fig5_event_timeline.png")

    summary = {"split": split, "n_rows": int(len(oof)), "n_dates": int(oof["date"].nunique()),
               "positive_rate": float(oof["y"].mean())}
    paths.file(f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    log.info("Report written to %s", paths.root)
    return {"comparisons": comparisons, "robustness": robustness, "metrics": metrics, "selective": selective}


def run_all(cfg: Config, final: bool = False) -> dict[str, pd.DataFrame]:
    run_panel(cfg)
    run_cv(cfg)
    result = run_report(cfg, "cv")
    if final:
        run_holdout(cfg)
        run_report(cfg, "holdout")
    return result
