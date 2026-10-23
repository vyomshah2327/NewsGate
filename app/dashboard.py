"""SignalProof dashboard (Streamlit).

Reads the files written by ``newsgate all`` from a run directory:

    streamlit run app/dashboard.py -- --run-dir outputs/synthetic_demo

It shows the stock and event timeline, price-only versus news-enhanced risk estimates with an
evidence label, model calibration, and a log of when forecasts were issued and how they did.
It supports research and risk awareness, not investment advice.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from newsgate.evaluation.metrics import reliability_table
from newsgate.gate.evidence import evidence_label


def _run_dir() -> Path:
    env = os.environ.get("NEWSGATE_RUN_DIR")
    if env:
        return Path(env)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/synthetic_demo")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return Path(args.run_dir)


@st.cache_data(show_spinner=False)
def _load(run_dir: str) -> dict[str, pd.DataFrame]:
    root = Path(run_dir)
    data = {
        "panel": pd.read_parquet(root / "panel.parquet"),
        "news": pd.read_parquet(root / "news_prepared.parquet"),
        "prices": pd.read_parquet(root / "prices.parquet"),
        "oof": pd.read_parquet(root / "oof_cv.parquet"),
    }
    for name in ("comparisons", "model_metrics"):
        path = root / f"{name}.csv"
        data[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return data


def main() -> None:
    st.set_page_config(page_title="SignalProof", layout="wide")
    run_dir = _run_dir()
    if not (run_dir / "oof_cv.parquet").exists():
        st.error(f"No results found in {run_dir}. Run `newsgate all --config <config>` first.")
        return

    data = _load(str(run_dir))
    oof, panel, news, prices = data["oof"], data["panel"], data["news"], data["prices"]
    price_col, news_col = "p__lightgbm__price_only", "p__lightgbm__price_news"

    st.title("SignalProof")
    st.caption("When does financial news add information about tomorrow's stock-movement risk? "
               "Research and risk awareness only, not investment advice.")

    ticker = st.sidebar.selectbox("Stock", sorted(oof["ticker"].unique()))
    st.sidebar.markdown(f"Run: `{run_dir}`")

    tab_overview, tab_timeline, tab_forecasts, tab_calibration, tab_log = st.tabs(
        ["Overview", "Timeline", "Forecasts", "Calibration", "Forecast log"]
    )

    with tab_overview:
        st.subheader("Does news improve the forecast?")
        st.write("Negative Brier difference means the news model forecasts better than the price-only model.")
        if not data["comparisons"].empty:
            cols = ["comparison", "mean_diff", "ci_low", "ci_high", "p_value", "p_holm"]
            st.dataframe(data["comparisons"][cols].round(5), use_container_width=True, hide_index=True)
        if not data["model_metrics"].empty:
            st.subheader("Out-of-fold accuracy by model")
            st.dataframe(data["model_metrics"].round(5), use_container_width=True, hide_index=True)

    with tab_timeline:
        px = prices[prices["ticker"] == ticker].set_index("date")["adj_close"]
        st.line_chart(px)
        articles = news[news["ticker"] == ticker].sort_values("published_at", ascending=False).head(200)
        shown = ["published_at", "headline", "novelty", "is_new_event", "event_size"]
        shown += [c for c in articles.columns if c.startswith("sent_")]
        st.write("Latest articles (de-duplicated, with novelty and sentiment)")
        st.dataframe(articles[shown].round(3), use_container_width=True, hide_index=True)

    sel = oof[oof["ticker"] == ticker].sort_values("date")
    with tab_forecasts:
        st.write("Probability of a large move on the next session, price-only versus news-enhanced.")
        chart = sel.set_index("date")[[c for c in (price_col, news_col) if c in sel]]
        chart.columns = ["Price-only", "With news"][: chart.shape[1]]
        st.line_chart(chart)
        latest = sel.iloc[-1] if len(sel) else None
        if latest is not None:
            attention = panel[(panel["ticker"] == ticker) & (panel["date"] == latest["date"])]
            n_art = float(attention["att_n"].iloc[0]) if len(attention) else 0.0
            novel = float(attention["evt_novel_share"].iloc[0]) if len(attention) else 0.0
            st.metric("Evidence behind the latest news-enhanced forecast", evidence_label(n_art, novel))
            if "gate_action" in latest and isinstance(latest["gate_action"], str):
                st.write(f"Gate decision for {latest['date'].date()}: **{latest['gate_action']}**")

    with tab_calibration:
        cols = [c for c in (price_col, news_col) if c in oof]
        for col in cols:
            table = reliability_table(oof["y"], oof[col])
            st.write(col[3:].replace("__", " : "))
            st.line_chart(table.set_index("mean_pred")["frac_pos"])

    with tab_log:
        st.write("Every forecast with the date it was issued (close of the prediction date) and what happened next.")
        log = sel[["date", "tgt_end_date", "y"]].copy()
        for col in (price_col, news_col, "p__gate", "gate_action"):
            if col in sel:
                log[col.replace("p__", "")] = sel[col]
        log = log.rename(columns={"date": "issued", "tgt_end_date": "resolved", "y": "large_move"})
        st.dataframe(log.sort_values("issued", ascending=False).round(4), use_container_width=True, hide_index=True)


main()
