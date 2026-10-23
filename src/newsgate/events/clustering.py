"""Causal online clustering of headlines into events.

Each article is compared only with earlier articles about the same ticker inside a trailing
window. If its best match is similar enough it joins that event, otherwise it starts a new
event. Novelty is ``1 - best similarity`` (1.0 for the first article about something new).

Headlines are vectorised with a stateless hashing vectoriser (no corpus-wide IDF), so the
representation of one article never depends on articles published later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from newsgate.config import EventConfig
from newsgate.data.cleaning import normalize_headline


def cluster_headlines(news: pd.DataFrame, cfg: EventConfig) -> pd.DataFrame:
    """Add ``event_id``, ``novelty``, ``is_new_event`` and ``event_size`` (size when published)."""
    out = news.sort_values(["ticker", "published_at", "article_id"], kind="stable").reset_index(drop=True)
    if out.empty:
        for col, dtype in (("event_id", "object"), ("novelty", "float64"), ("is_new_event", "bool"), ("event_size", "int64")):
            out[col] = pd.Series(dtype=dtype)
        return out

    vectorizer = HashingVectorizer(
        n_features=cfg.n_features, alternate_sign=False, norm="l2", ngram_range=(1, 2), stop_words="english"
    )
    matrix = vectorizer.transform(out["headline"].map(normalize_headline))
    times = out["published_at"].to_numpy("datetime64[ns]")
    window = np.timedelta64(int(cfg.window_days * 24 * 3600), "s")

    event_id = np.empty(len(out), dtype=object)
    novelty = np.ones(len(out))
    event_size = np.ones(len(out), dtype=np.int64)
    is_new = np.ones(len(out), dtype=bool)

    for ticker, positions in out.groupby("ticker", sort=False).indices.items():
        positions = np.asarray(positions)
        sizes: dict[str, int] = {}
        left = 0  # first position inside the trailing window
        for k, i in enumerate(positions):
            while times[positions[left]] < times[i] - window:
                left += 1
            if k > left:
                prior = positions[left:k]
                sims = (matrix[prior] @ matrix[i].T).toarray().ravel()
                best = int(np.argmax(sims))
                if sims[best] >= cfg.similarity_threshold:
                    event_id[i] = event_id[prior[best]]
                    novelty[i] = 1.0 - float(sims[best])
                    is_new[i] = False
                else:
                    event_id[i] = f"{ticker}-{i}"
                    novelty[i] = 1.0 - float(sims[best])
            else:
                event_id[i] = f"{ticker}-{i}"
            sizes[event_id[i]] = sizes.get(event_id[i], 0) + 1
            event_size[i] = sizes[event_id[i]]

    out["event_id"], out["novelty"], out["is_new_event"], out["event_size"] = event_id, novelty, is_new, event_size
    return out
