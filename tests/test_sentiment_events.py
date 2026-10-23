import numpy as np
import pandas as pd
import pytest

from newsgate.config import EventConfig, SentimentConfig
from newsgate.events import cluster_headlines
from newsgate.sentiment import available_scorers, create_scorer, score_news
from newsgate.sentiment.finbert import FinBertScorer
from newsgate.sentiment.lexicon import LoughranMcDonaldScorer


class FakeScorer:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def score(self, texts):
        self.calls += len(texts)
        return np.array([1.0 if "good" in t.lower() else -1.0 for t in texts])


def test_registry_lists_builtin_scorers():
    assert {"vader", "lm", "finbert"} <= set(available_scorers())
    with pytest.raises(KeyError):
        create_scorer("nope", SentimentConfig())


def test_lexicon_scorer():
    scorer = LoughranMcDonaldScorer(["gain", "strong"], ["loss", "weak"])
    out = scorer.score(["Strong gain", "weak loss, weak", "nothing here", "gain and loss"])
    assert list(out) == [1.0, -1.0, 0.0, 0.0]


def test_lexicon_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Loughran"):
        LoughranMcDonaldScorer.from_dictionary_csv(tmp_path / "missing.csv")


def test_finbert_scorer_with_injected_pipeline():
    def factory():
        return lambda batch: [[{"label": "positive", "score": 0.7}, {"label": "negative", "score": 0.1},
                               {"label": "neutral", "score": 0.2}] for _ in batch]

    scorer = FinBertScorer(batch_size=2, pipeline_factory=factory)
    assert np.allclose(scorer.score(["a", "b", "c"]), 0.6)


def test_score_news_scores_each_distinct_headline_once_and_caches(tmp_path):
    news = pd.DataFrame({"headline": ["Good news", "good news!", "Bad day"]})
    first = FakeScorer()
    out = score_news(news, [first], cache_dir=tmp_path)
    assert first.calls == 2 and list(out["sent_fake"]) == [1.0, 1.0, -1.0]
    second = FakeScorer()
    score_news(news, [second], cache_dir=tmp_path)
    assert second.calls == 0


def _articles(rows):
    return pd.DataFrame({
        "article_id": [f"a{i}" for i in range(len(rows))],
        "ticker": [r[0] for r in rows],
        "published_at": pd.to_datetime([r[1] for r in rows]),
        "headline": [r[2] for r in rows],
    })


def test_event_clustering_groups_followups_and_flags_novelty():
    news = _articles([
        ("X", "2024-01-02 10:00", "Acme announces factory expansion project kora"),
        ("X", "2024-01-02 15:00", "Acme announces factory expansion project kora update"),
        ("X", "2024-01-03 09:00", "Acme names new chief financial officer"),
        ("Y", "2024-01-02 11:00", "Acme announces factory expansion project kora"),
    ])
    out = cluster_headlines(news, EventConfig(window_days=3, similarity_threshold=0.5)).set_index("article_id")
    assert out.loc["a0", "is_new_event"] and not out.loc["a1", "is_new_event"]
    assert out.loc["a0", "event_id"] == out.loc["a1", "event_id"]
    assert out.loc["a1", "event_size"] == 2 and out.loc["a2", "is_new_event"]
    assert out.loc["a3", "is_new_event"], "events never merge across tickers"


def test_event_clustering_is_causal():
    """An article's cluster output must not change when later articles are removed."""
    news = _articles([
        ("X", "2024-01-02 10:00", "Acme beats quarterly earnings project kora"),
        ("X", "2024-01-02 12:00", "Acme beats quarterly earnings project kora again"),
        ("X", "2024-01-05 12:00", "Totally unrelated product launch"),
        ("X", "2024-01-06 12:00", "Acme beats quarterly earnings project kora"),
    ])
    cfg = EventConfig()
    full = cluster_headlines(news, cfg).set_index("article_id")
    for keep in (2, 3):
        part = cluster_headlines(news.iloc[:keep], cfg).set_index("article_id")
        cols = ["novelty", "is_new_event", "event_size"]
        pd.testing.assert_frame_equal(full.loc[part.index, cols], part[cols])
