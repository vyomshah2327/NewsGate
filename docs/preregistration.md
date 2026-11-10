# Pre-registration: round 2 ("the planned round")

Written and committed before any round 2 result exists. The git history is the timestamp.
Anything not listed here is not part of the confirmatory analysis.

## 1. Why this round exists

Round 1 (40 stocks, horizon of 1 trading day, VADER and Loughran and McDonald only, logistic and
LightGBM, cross-validation only) found that price features have real skill (Brier skill 8.7% over the
historical rate, AUROC about 0.70) and that news adds almost nothing (the best case, price plus
sentiment, improved Brier by 0.00013 and was not significant after Holm correction, p = 0.09). The
learned gate was significantly worse than price-only.

Round 1 was exploratory and informed this design in three ways, all disclosed here:

* logistic regression is the only model family in round 2 (it beat LightGBM in round 1 and is much cheaper);
* the horizon of 5 sessions is added because news arrives at least one day late in this dataset
  (FNSPID dates carry no time of day for 98.6% of articles), which should hurt 1-day forecasts most;
* the high-attention subset is added because an average over all days can hide days on which news matters.

The final holdout (the last 20% of dates) has not been used for any purpose, including round 1.

## 2. Data and universe

* FNSPID news, 2012-01-01 to 2023-12-31, and the FNSPID daily price files. SPY is the market benchmark.
* Universe: all 523 tickers that pass `scripts/make_metadata.py` (listed in `configs/round2_h1.yaml`):
  a price file from the study start to the study end, at least 1,500 articles and 500 distinct news days
  in the window, and a share of fund-style headlines of at most 20%.
* Known limitation: that rule is imperfect and some ETFs may remain. We do not remove tickers by hand after
  seeing results.
* Articles are used from the next trading day after their date (timestamps have no usable time of day).
* Sentiment scorers: VADER, Loughran and McDonald, and FinBERT (ProsusAI/finbert, P(positive) minus P(negative)
  on the headline). FinBERT scores are computed once, outside the pipeline, by `scripts/score_finbert.py`.

## 3. Targets

Large move: the absolute forward log return over the next `h` sessions exceeds that ticker's 90th percentile,
estimated on training rows only (global fallback for short histories). Horizons: `h = 1` and `h = 5`.
Prices before 2012 are used only as warmup. Labels that span more than 4 + 7h calendar days are treated as
missing (data gaps).

## 4. Models and validation

* Logistic regression with inner chronological tuning and Platt calibration, and the historical-rate baseline.
* Feature sets: price_only, price_attention, price_sentiment, price_news (all news features together).
* Expanding-window validation on unique dates: at least 756 training days, test blocks of 126 days, a purge
  gap of 5 days (h = 1) or 10 days (h = 5), and the last 20% of dates held out.
* Intervals and p-values: paired moving-block bootstrap over dates, 2000 resamples, block length 10 (h = 1)
  or 20 (h = 5), seed 0. Metric: Brier score. Difference is challenger minus baseline, so negative means better.

## 5. The family of tests (10 in total)

High-attention day: the ticker has news that day and `att_surprise >= 1.0`, meaning the article count is at least one
(smoothed) standard deviation above the ticker's own earlier average. The subset is evaluated on the same
out-of-fold predictions, with no refit.

For each horizon `h` in {1, 5}, versus the logistic price_only model:

| Test | Comparison | Rows |
|---|---|---|
| T1 | price_attention | all rows |
| T2 | price_sentiment | all rows |
| T3 | price_news | all rows |
| T4 | price_news | high-attention rows only |
| T5 | gate (chooses news, price or abstain) | rows where the gate has history |

That is 5 tests per horizon and 10 in total. Holm correction is applied across all 10 together, at alpha = 0.05.

The single primary test is T3 at `h = 5`. It is also reported on its own, uncorrected, next to the corrected family.

## 6. Decision rule

* The cross-validation results are development evidence.
* The confirmatory evidence is the single holdout run. A claim that "news improves forecasts" requires a Holm-adjusted
  p below 0.05, a negative mean Brier difference in the holdout, and the same sign in cross-validation.
* A claim about the gate requires the same.
* Anything else is reported as "no evidence of improvement".

## 7. Rules for the run

1. The holdout is run exactly once, after cross-validation is complete and the configuration is frozen.
2. Stage A (cross-validation without FinBERT) is a development step that checks the pipeline at full scale. Stage B
   (cross-validation and holdout with FinBERT) is the registered analysis. If FinBERT scores are never supplied, the
   holdout is run once without them and the report says so plainly.
3. No variant that is not listed here is added after seeing results. If one is explored, it is labelled exploratory,
   kept out of the family, and not used to support a claim.
4. All results are reported, including every null result, with the confidence intervals.
5. Bug fixes that do not change the specification are allowed and are logged below. Changes to the specification
   are not.

## 8. Deviations log

No change to the specification. Two logged events:

* FinBERT scores were supplied after Stage A, as rule 7.2 allows. Stage B used the frozen configs with only the scorer list and run name changed (scorers vader, lm, finbert).
* The first Stage B launch of both horizons at once ran out of memory (exit 137) and was rerun one horizon at a time. This changed no setting.

Outcome (Stage B holdout, run once per horizon): no test passes the decision rule. The primary test (T3, h = 5) has mean Brier difference -0.000033, uncorrected p = 0.013, Holm p = 0.104. The smallest Holm p in the family is 0.100 (T1 h5 and T5 h5). Reported as no evidence of improvement. See results/round2_stageB_family_holdout.csv.
