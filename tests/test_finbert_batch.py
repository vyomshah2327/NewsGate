import numpy as np
import pandas as pd
import pytest

from newsgate.data.cleaning import normalize_headline
from newsgate.sentiment.base import score_news
from newsgate.sentiment.finbert_batch import distinct_headlines, score_distinct, write_cache
from newsgate.sentiment.base import SentimentScorer


def fake_predict(texts):
    return np.array([1.0 if "up" in t.lower() else -1.0 for t in texts])


def test_distinct_headlines_collapse_case_and_punctuation():
    s = pd.Series(["Stocks UP!", "stocks up", "Shares down", "", None])
    d = distinct_headlines(s.fillna(""))
    assert list(d.index) == ["stocks up", "shares down"]


def test_scoring_is_resumable_and_keeps_order(tmp_path):
    heads = distinct_headlines(pd.Series([f"headline {i} up" if i % 2 else f"headline {i} down long text" for i in range(25)]))
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return fake_predict(texts)

    first = score_distinct(heads, counting, tmp_path, batch_size=4, flush_every=2)
    assert list(first.index) == list(heads.index) and first.notna().all()
    n_calls = len(calls)
    again = score_distinct(heads, counting, tmp_path, batch_size=4, flush_every=2)
    assert len(calls) == n_calls, "a rerun must not rescore anything"
    pd.testing.assert_series_equal(first, again)


def test_interrupted_run_resumes(tmp_path):
    heads = distinct_headlines(pd.Series([f"item {i} up" for i in range(30)]))
    state = {"n": 0}

    def flaky(texts):
        state["n"] += 1
        if state["n"] == 6:
            raise KeyboardInterrupt
        return fake_predict(texts)

    with pytest.raises(KeyboardInterrupt):
        score_distinct(heads, flaky, tmp_path, batch_size=5, flush_every=2)
    seen = []
    done = score_distinct(heads, lambda t: (seen.extend(t), fake_predict(t))[1], tmp_path, batch_size=5, flush_every=2)
    assert len(seen) < len(heads), "only the unfinished headlines are scored on resume"
    assert len(done) == len(heads)


def test_wrong_shape_is_rejected(tmp_path):
    heads = distinct_headlines(pd.Series(["a up", "b down"]))
    with pytest.raises(ValueError):
        score_distinct(heads, lambda t: np.zeros(len(t) + 1), tmp_path)


def test_cache_file_is_read_by_the_pipeline_without_loading_a_model(tmp_path):
    news = pd.DataFrame({"headline": ["Stocks UP today", "Shares fall hard", "Stocks up today"]})
    heads = distinct_headlines(news["headline"])
    write_cache(score_distinct(heads, fake_predict, tmp_path / "parts"), tmp_path / "scores_finbert.parquet")

    class NoModel(SentimentScorer):
        name = "finbert"

        def score(self, texts):
            raise AssertionError("the model must not be used when every score is cached")

    out = score_news(news, [NoModel()], cache_dir=tmp_path)
    assert out["sent_finbert"].tolist() == [1.0, -1.0, 1.0]
    assert normalize_headline("Stocks UP today") == "stocks up today"
