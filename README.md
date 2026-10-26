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

| Path | Purpose |
|---|---|
| `docs/proposal.tex` | The two-page proposal |
| `configs/` | `default.yaml` (real FNSPID run) and `synthetic.yaml` (offline demo) |
| `src/newsgate/data/` | Sources (FNSPID, yfinance, synthetic), cleaning, trading calendar, timestamp alignment, sampling |
| `src/newsgate/sentiment/` | VADER, Loughran and McDonald, FinBERT scorers behind one interface |
| `src/newsgate/events/` | Headline clustering into events and novelty |
| `src/newsgate/features/` | Price, attention, sentiment and event feature blocks and the panel builder |
| `src/newsgate/models/` | Historical rate, logistic regression, LightGBM, tuning, Platt calibration |
| `src/newsgate/evaluation/` | Purged expanding-window splits, metrics, block bootstrap, Holm correction, comparisons |
| `src/newsgate/gate/` | The incremental-value gate, evidence labels, selective-risk tables |
| `src/newsgate/viz/` | The five report figures |
| `app/dashboard.py` | Streamlit dashboard (SignalProof) |
| `notebooks/` | Data audit, feature checks, results |
| `tests/` | Unit and end-to-end tests, including leakage checks |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app]"          # add ",nlp" for FinBERT and ",data" for yfinance
```

## Quick start (no downloads needed)

The synthetic world has a planted attention effect, so the whole pipeline can be exercised offline.

```bash
newsgate all --config configs/synthetic.yaml        # panel, cross-validation, gate, tables, figures
newsgate holdout --config configs/synthetic.yaml    # untouched final period, run once
newsgate dashboard --config configs/synthetic.yaml
pytest
```

Results are written to `outputs/<run name>/` (parquet panels, csv tables, `figures/`).
Setting `attention_effect: 0` and `sentiment_effect: 0` in `configs/synthetic.yaml` gives a pure null world, in which the study should find nothing.

## Running on the real data

1. Download `Stock_news/nasdaq_exteral_data.csv` (about 23 GB) and `Stock_price/full_history.zip` from Hugging Face (`Zihan1004/FNSPID`). Put the CSV in `data/raw/fnspid/` and the per-ticker price CSVs directly in `data/raw/fnspid/prices/`.
2. Download the Loughran and McDonald master dictionary into `data/external/` (we do not redistribute it).
3. Shrink the news file once (it drops the article bodies and summaries, and takes a while):
   `python scripts/prepare_fnspid.py data/raw/fnspid/nasdaq_exteral_data.csv data/raw/fnspid/news_slim.parquet`
4. Run `notebooks/01_data_audit.ipynb` against `configs/default.yaml`.
5. `newsgate all --config configs/default.yaml`, then `newsgate holdout --config configs/default.yaml` once the design is frozen.

Finding from the audit of a 30,000 row sample: FNSPID dates end in `UTC` and 99.9% sit at exactly 00:00:00, so they are dates without a time of day. The pipeline therefore uses each article only from the next trading day, which is the conservative rule.

## How leakage is controlled

- News published at or after 16:00 US/Eastern, on weekends or on holidays is used from the next trading day. Date-only timestamps are used from the next trading day strictly after the date.
- Every feature is computed from past data only. A test checks that features do not change when the future is deleted.
- Large-move thresholds, tuning and calibration use training rows only. Validation is expanding-window with a purge gap of at least the horizon, and the gate only learns from earlier folds whose outcomes were already known.
- The final holdout is never used during development.

## Caveats

- Pipelines for the real FNSPID files, yfinance and FinBERT have not yet been run end to end; only the synthetic path is tested.
- A positive result would show association with large moves, not that news causes them. A null result is a valid finding. This is not investment advice.
- On the small synthetic world the gate does not beat the simple abstention controls; we report that as is.
