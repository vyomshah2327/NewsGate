"""Feature pipeline (builder): run blocks and join their outputs on (ticker, date)."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from newsgate.features.base import FeatureBlock, FeatureContext
from newsgate.features.news import AttentionFeatures, EventFeatures, SentimentFeatures
from newsgate.features.price import PriceFeatures


class FeaturePipeline:
    def __init__(self, blocks: Sequence[FeatureBlock]):
        prefixes = [b.prefix for b in blocks]
        if len(set(prefixes)) != len(prefixes):
            raise ValueError(f"Feature blocks must have unique prefixes, got {prefixes}")
        self.blocks = list(blocks)

    def build(self, ctx: FeatureContext) -> pd.DataFrame:
        frames = [block.build(ctx) for block in self.blocks]
        out = frames[0]
        for frame in frames[1:]:
            overlap = out.columns.intersection(frame.columns)
            if len(overlap):
                raise ValueError(f"Duplicate feature columns across blocks: {list(overlap)}")
            out = out.join(frame, how="outer")
        return out.sort_index()


def default_pipeline(include_news: bool = True) -> FeaturePipeline:
    blocks: list[FeatureBlock] = [PriceFeatures()]
    if include_news:
        blocks += [AttentionFeatures(), SentimentFeatures(), EventFeatures()]
    return FeaturePipeline(blocks)
