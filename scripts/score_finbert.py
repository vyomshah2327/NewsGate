"""Score the study's headlines with FinBERT (run once, on a computer with internet).

    pip install -e ".[nlp]"
    python scripts/score_finbert.py --config configs/round2_h5.yaml

Reads the slimmed FNSPID news file named in the config, keeps the configured tickers and dates,
scores each distinct headline once, and writes ``<cache_dir>/scores_finbert.parquet``. It saves
progress every few minutes and resumes where it stopped if you run it again. Use ``--limit 2000``
for a quick trial. Then copy ``scores_finbert.parquet`` anywhere it is needed: the pipeline reads
it from ``data.cache_dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from newsgate.config import load_config
from newsgate.logging_utils import configure_logging
from newsgate.sentiment.finbert_batch import distinct_headlines, load_hf_predictor, score_distinct, write_cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: the best available)")
    ap.add_argument("--limit", type=int, default=None, help="score only this many headlines (trial run)")
    ap.add_argument("--out", default=None, help="output file (default: <cache_dir>/scores_finbert.parquet)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    configure_logging("INFO")
    out = Path(args.out) if args.out else Path(cfg.data.cache_dir or "data/cache") / "scores_finbert.parquet"

    news = pd.read_parquet(cfg.data.news_path, columns=["Date", "Article_title", "Stock_symbol"])
    day = news["Date"].str[:10]
    keep = news["Stock_symbol"].isin(cfg.data.tickers) & (day >= cfg.data.start) & (day <= cfg.data.end)
    headlines = distinct_headlines(news.loc[keep, "Article_title"])
    if args.limit:
        headlines = headlines.iloc[: args.limit]
    print(f"{int(keep.sum()):,} articles, {len(headlines):,} distinct headlines to score")

    scores = score_distinct(
        headlines, load_hf_predictor(cfg.sentiment.finbert_model, args.device),
        part_dir=out.with_suffix(".parts"), batch_size=args.batch_size,
    )
    write_cache(scores, out)
    print(f"wrote {len(scores):,} scores to {out}")


if __name__ == "__main__":
    main()
