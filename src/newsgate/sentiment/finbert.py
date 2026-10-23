"""FinBERT scorer (pretrained financial language model).

The Hugging Face ``transformers`` pipeline is created lazily and can be injected, so the
class can be tested without downloading the model. Score = P(positive) - P(negative).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from newsgate.config import SentimentConfig
from newsgate.sentiment.base import SentimentScorer, register_scorer


class FinBertScorer(SentimentScorer):
    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 32, pipeline_factory: Callable | None = None):
        self.model_name, self.batch_size = model_name, batch_size
        self._factory = pipeline_factory
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            if self._factory is not None:
                self._pipe = self._factory()
            else:
                from transformers import pipeline  # optional dependency

                self._pipe = pipeline(
                    "text-classification", model=self.model_name, top_k=None, truncation=True, max_length=128
                )
        return self._pipe

    def score(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        out = np.zeros(len(texts))
        pipe = self._pipeline()
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            for offset, labels in enumerate(pipe(batch)):
                probs = {item["label"].lower(): item["score"] for item in labels}
                out[start + offset] = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        return out


@register_scorer("finbert")
def _make_finbert(cfg: SentimentConfig) -> SentimentScorer:
    return FinBertScorer(cfg.finbert_model, cfg.finbert_batch_size)
