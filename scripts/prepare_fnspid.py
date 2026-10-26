"""One-time conversion of the 23 GB FNSPID news CSV into a small Parquet file.

The study only needs the headline, ticker, timestamp, URL and publisher. The article body and the
four summaries make up almost all of the file, so they are dropped. Reading is chunked, so memory
use stays small.

    python scripts/prepare_fnspid.py data/raw/fnspid/nasdaq_exteral_data.csv data/raw/fnspid/news_slim.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KEEP = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]


def convert(src: Path, dst: Path, chunksize: int = 100_000, max_rows: int | None = None) -> int:
    header = pd.read_csv(src, nrows=0).columns
    usecols = [c for c in KEEP if c in header]
    missing = {"Date", "Article_title", "Stock_symbol"} - set(usecols)
    if missing:
        raise ValueError(f"{src} is missing required columns: {sorted(missing)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer, total = None, 0
    for i, chunk in enumerate(pd.read_csv(src, usecols=usecols, chunksize=chunksize, dtype=str)):
        chunk = chunk.dropna(subset=["Date", "Article_title", "Stock_symbol"])
        table = pa.Table.from_pandas(chunk[usecols], preserve_index=False)
        writer = writer or pq.ParquetWriter(dst, table.schema, compression="zstd")
        writer.write_table(table)
        total += len(chunk)
        print(f"chunk {i + 1}: {total:,} rows written", file=sys.stderr, flush=True)
        if max_rows and total >= max_rows:
            break
    if writer:
        writer.close()
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--max-rows", type=int, default=None, help="stop early (for testing)")
    args = parser.parse_args()
    print(f"done: {convert(args.src, args.dst, max_rows=args.max_rows):,} rows -> {args.dst}")
