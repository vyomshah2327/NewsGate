# NewsGate

NewsGate is a research project on one question: **when does financial news actually improve a forecast of tomorrow's stock risk beyond what past prices already show?**

It is a group project for DS3000/DS9000 (Introduction to Machine Learning, Western University) by Vyom Shah, Harshil Patel, and Harmit Patel. The product and dashboard name used in the proposal is SignalProof.

## What we are doing

1. **Data.** FNSPID (financial news linked to S&P 500 tickers) plus daily adjusted prices, market and sector benchmarks.
2. **Features.** Sentiment scores from three tools (FinBERT, VADER, and the Loughran and McDonald finance dictionary), news attention (article counts and changes), and event groups built from related headlines. Price features such as lagged returns, momentum, volatility, and volume.
3. **Target.** Whether the next trading day is a large-move day (absolute return above a 90th-percentile threshold learned from training data only).
4. **Models.** A historical-rate baseline, price-only, price plus attention, price plus sentiment, and price plus attention plus event and sentiment, using regularized regression and LightGBM.
5. **The gate.** A second-stage model that learns from earlier out-of-fold losses whether to use the price-only forecast, the news-enhanced forecast, or abstain.
6. **Testing.** Strictly chronological: nested, purged, expanding-window tests with past-only tuning, and an untouched final period. Metrics are Brier score (primary), AUROC, calibration, and risk-coverage curves, with date-block bootstrap confidence intervals and corrected tests for multiple comparisons.
7. **Dashboard.** A localhost app showing a stock and event timeline, price-only versus news-enhanced risk estimates, calibration, and a log of when forecasts were issued.

A null result (news adds nothing) is an acceptable and useful finding. Any gain would show association with price moves, not that news causes them, and this is not investment advice.

## Repository layout

| Folder | Purpose |
|---|---|
| `docs/` | The two-page proposal (`proposal.tex`) |
| `data/` | Local data only. Raw and processed data are not committed |
| `src/` | Pipeline, features, models, evaluation, and the dashboard |
| `notebooks/` | Exploration and checks |
| `figures/` | Plots for the report |

## Planned figures

Brier-score differences versus the price-only model with confidence intervals, calibration curves, risk-coverage curves, large-move rate by sentiment decile, and the event timeline.

## First steps

1. Audit the FNSPID timestamps, ticker links, duplicates, and coverage.
2. Build the historical data pipeline with a strict prediction cutoff (US/Eastern market close).
3. Train price-only baselines.
4. Add sentiment, attention, and event features.
5. Build the gate, then the dashboard.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Proposal

`docs/proposal.tex` uses the course NeurIPS 2026 template. To compile it, put the template's `neurips_2026.sty` in the same folder (for example in an Overleaf project).
