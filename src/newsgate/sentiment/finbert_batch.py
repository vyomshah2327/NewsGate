"""Score every distinct headline with FinBERT once, offline, and write the pipeline's score cache.

FinBERT's weights come from Hugging Face, which is only reachable from an ordinary internet
connection, so scoring is a separate one-off step (``scripts/score_finbert.py``). The result is a
Parquet file with columns ``norm`` (the normalised headline) and ``score`` (P(positive) minus
P(negative)); the feature pipeline picks it up through its normal sentiment cache and never loads
the model itself.

The scoring loop takes the model as a plain function, so it is tested without any download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from newsgate.data.cleaning import normalize_headline
from newsgate.logging_utils import get_logger

log = get_logger(__name__)

Predictor = Callable[[Sequence[str]], np.ndarray]


def distinct_headlines(headlines: pd.Series) -> pd.Series:
    """One representative headline per normalised form, indexed by that form."""
    norm = headlines.map(normalize_headline)
    keep = (norm != "") & ~norm.duplicated()
    return pd.Series(headlines[keep].to_numpy(), index=norm[keep].to_numpy(), name="headline")


def score_distinct(
    headlines: pd.Series,
    predict: Predictor,
    part_dir: str | Path,
    batch_size: int = 128,
    flush_every: int = 200,
) -> pd.Series:
    """Score ``headlines`` (indexed by normalised form), resuming from ``part_dir``.

    Headlines are scored in order of length, so batches need little padding. Every
    ``flush_every`` batches the scores so far are written to ``part_dir``, and a rerun skips
    anything already there.
    """
    part_dir = Path(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    done = [pd.read_parquet(p).set_index("norm")["score"] for p in sorted(part_dir.glob("part_*.parquet"))]
    known = pd.concat(done) if done else pd.Series(dtype="float64")
    todo = headlines[~headlines.index.isin(known.index)]
    todo = todo.iloc[np.argsort(todo.str.len().to_numpy(), kind="stable")]
    log.info("%d distinct headlines, %d already scored, %d to do", len(headlines), len(known), len(todo))

    buffer_idx: list[str] = []
    buffer_val: list[np.ndarray] = []
    next_part = len(list(part_dir.glob("part_*.parquet")))

    def flush() -> None:
        nonlocal buffer_idx, buffer_val, next_part
        if buffer_idx:
            frame = pd.DataFrame({"norm": buffer_idx, "score": np.concatenate(buffer_val)})
            frame.to_parquet(part_dir / f"part_{next_part:05d}.parquet")
            next_part += 1
            buffer_idx, buffer_val = [], []

    texts = todo.tolist()
    for number, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[start : start + batch_size]
        values = np.asarray(predict(batch), dtype="float64")
        if values.shape != (len(batch),):
            raise ValueError(f"predictor returned shape {values.shape} for a batch of {len(batch)}")
        buffer_idx.extend(todo.index[start : start + len(batch)])
        buffer_val.append(values)
        if number % flush_every == 0:
            flush()
            log.info("scored %d / %d", start + len(batch), len(texts))
    flush()

    parts = [pd.read_parquet(p).set_index("norm")["score"] for p in sorted(part_dir.glob("part_*.parquet"))]
    scores = pd.concat(parts) if parts else pd.Series(dtype="float64")
    scores = scores[~scores.index.duplicated()]
    missing = headlines.index.difference(scores.index)
    if len(missing):
        raise RuntimeError(f"{len(missing)} headlines were not scored")
    return scores.loc[headlines.index]


def load_hf_predictor(model_name: str = "ProsusAI/finbert", device: str | None = None, max_length: int = 128) -> Predictor:
    """Build a predictor from Hugging Face ``transformers`` (needs the ``nlp`` extra and internet)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    labels = {int(i): str(name).lower() for i, name in model.config.id2label.items()}
    pos = next(i for i, name in labels.items() if name == "positive")
    neg = next(i for i, name in labels.items() if name == "negative")
    log.info("FinBERT on %s, labels %s", device, labels)

    @torch.inference_mode()
    def predict(texts: Sequence[str]) -> np.ndarray:
        enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
        return probs[:, pos] - probs[:, neg]

    return predict


def write_cache(scores: pd.Series, path: str | Path) -> None:
    """Write scores in the layout ``score_news`` reads: columns ``norm`` and ``score``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.rename("score").rename_axis("norm").reset_index().to_parquet(path)
