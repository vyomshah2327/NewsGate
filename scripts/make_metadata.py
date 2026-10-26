"""Build data/raw/metadata.csv (the ticker universe) from the downloaded FNSPID files.

FNSPID has no sector or market-cap table, so this derives what the sampler needs from the data:

* eligibility: a price file that starts by the study start and runs to the study end, and at
  least ``--min-articles`` articles in the study window;
* ``market_cap``: a PROXY, the average daily dollar volume (close x volume) over the last 250
  trading days. It orders companies by size and liquidity; it is not a market capitalisation;
* ``sector`` is left empty, so sector stratification and the sector robustness check need a
  real sector table (add a ``sector`` column to the output to enable them).

    python scripts/make_metadata.py --news data/raw/fnspid/news_slim.parquet \
        --prices data/raw/fnspid/prices --out data/raw/metadata.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build(news_path: Path, price_dir: Path, start: str, end: str, min_articles: int, exclude: set[str]) -> pd.DataFrame:
    news = pd.read_parquet(news_path, columns=["Date", "Stock_symbol"])
    day = pd.to_datetime(news["Date"].str[:10], errors="coerce")
    news = news.assign(day=day)[(day >= start) & (day <= end)]
    counts = news.groupby("Stock_symbol").agg(n_articles=("day", "size"), n_news_days=("day", "nunique"))
    counts = counts[counts["n_articles"] >= min_articles]

    rows = []
    for ticker in counts.index:
        path = price_dir / f"{ticker}.csv"
        if ticker in exclude or not path.exists():
            continue
        px = pd.read_csv(path, usecols=lambda c: c.lower() in {"date", "close", "volume"})
        px.columns = [c.lower() for c in px.columns]
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        px = px.dropna(subset=["date"]).sort_values("date")
        if px.empty or px["date"].iloc[0] > pd.Timestamp(start) or px["date"].iloc[-1] < pd.Timestamp(end) - pd.Timedelta(days=45):
            continue
        recent = px.tail(250)
        rows.append({"ticker": ticker, "market_cap": float((recent["close"] * recent["volume"]).mean()),
                     "first_price_date": px["date"].iloc[0].date(), "last_price_date": px["date"].iloc[-1].date()})
    meta = pd.DataFrame(rows).merge(counts.reset_index().rename(columns={"Stock_symbol": "ticker"}), on="ticker")
    meta["sector"] = ""
    return meta.sort_values("ticker").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--news", type=Path, required=True)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--min-articles", type=int, default=1500)
    ap.add_argument("--exclude", default="SPY,QQQ,DIA,IWM", help="comma separated tickers to leave out (benchmarks)")
    a = ap.parse_args()
    meta = build(a.news, a.prices, a.start, a.end, a.min_articles, set(a.exclude.split(",")))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(a.out, index=False)
    print(f"{len(meta)} eligible tickers -> {a.out}")
