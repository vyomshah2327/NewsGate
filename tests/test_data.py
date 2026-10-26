import numpy as np
import pandas as pd
import pytest

from newsgate.data.calendar import TradingCalendar
from newsgate.data.cleaning import deduplicate_news, normalize_headline
from newsgate.data.sampling import stratified_sample
from newsgate.data.sources import FnspidNewsSource, FnspidPriceSource, parse_timestamps
from newsgate.data.synthetic import to_date_only


def test_normalize_headline():
    assert normalize_headline("  ACME Beats, Earnings!! ") == "acme beats earnings"


def test_deduplicate_keeps_earliest_copy_and_drops_syndication():
    news = pd.DataFrame({
        "article_id": ["a", "b", "c"],
        "ticker": ["X", "X", "X"],
        "published_at": pd.to_datetime(["2024-01-02 10:05", "2024-01-02 10:00", "2024-01-03 10:00"]),
        "headline": ["ACME beats!", "acme beats", "Different story"],
        "url": [None, None, None],
    })
    out = deduplicate_news(news)
    assert list(out["article_id"]) == ["b", "c"]


def test_calendar_next_trading_day():
    cal = TradingCalendar(pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-12"]))
    nxt = cal.next_trading_day(pd.to_datetime(["2024-01-09", "2024-01-10", "2024-01-13"]), strict=True)
    assert nxt[0] == pd.Timestamp("2024-01-12") and nxt[1] == pd.Timestamp("2024-01-12") and pd.isna(nxt[2])
    same = cal.next_trading_day(pd.to_datetime(["2024-01-09", "2024-01-10"]), strict=False)
    assert same[0] == pd.Timestamp("2024-01-09") and same[1] == pd.Timestamp("2024-01-12")
    assert list(cal.contains(pd.to_datetime(["2024-01-09", "2024-01-10"]))) == [True, False]


def test_parse_timestamps_detects_date_only_and_timed():
    dates = pd.Series(["2024-01-02", "2024-01-03"])
    published, has_time = parse_timestamps(dates)
    assert not has_time.any()
    timed = pd.Series(["2024-01-02 14:30:00 UTC", "2024-01-03 09:00:00 UTC", "2024-01-04 00:00:00 UTC"])
    published, has_time = parse_timestamps(timed)
    assert list(has_time) == [True, True, False]
    assert published.iloc[0] == pd.Timestamp("2024-01-02 14:30:00")


def test_parse_timestamps_converts_source_timezone_to_utc():
    published, has_time = parse_timestamps(pd.Series(["2024-01-02 10:00:00"]), source_tz="America/New_York", mode="datetime")
    assert published.iloc[0] == pd.Timestamp("2024-01-02 15:00:00") and has_time.all()


def test_fnspid_files_round_trip(world, tmp_path):
    news_path, price_dir = world.write_fnspid_like(tmp_path)
    news = FnspidNewsSource(news_path, world.tickers[:2]).load()
    assert set(news["ticker"]) <= set(world.tickers[:2]) and news["has_time"].mean() > 0.9
    original = world.news[world.news["ticker"].isin(world.tickers[:2])]
    assert len(news) == len(original)
    prices = FnspidPriceSource(price_dir).load(world.tickers[:2])
    assert set(prices.columns) >= {"ticker", "date", "adj_close", "volume"}
    assert prices.groupby("ticker").size().min() > 100


def test_date_only_world_has_no_times(world):
    assert not to_date_only(world.news)["has_time"].any()


def test_stratified_sample_spreads_across_size_and_sector():
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(60)],
        "sector": rng.choice(["A", "B", "C"], 60),
        "market_cap": np.exp(rng.normal(20, 2, 60)),
    })
    chosen = stratified_sample(meta, 12, seed=1)
    assert len(chosen) == len(set(chosen)) == 12
    sizes = pd.qcut(meta["market_cap"].rank(method="first"), 3, labels=False)
    picked = sizes[meta["ticker"].isin(chosen)].value_counts()
    assert picked.min() >= 3


def test_fnspid_price_source_clips_to_the_study_window(tmp_path):
    dates = pd.bdate_range("2000-01-03", "2024-03-01")
    frame = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                          "adj close": 1.0, "volume": 10})
    frame.to_csv(tmp_path / "AAA.csv", index=False)
    from newsgate.data.sources import FnspidPriceSource
    out = FnspidPriceSource(tmp_path, "2012-01-01", "2023-12-31").load(["AAA"])
    assert out["date"].min() >= pd.Timestamp("2012-01-01") - pd.Timedelta(days=400)
    assert out["date"].min() < pd.Timestamp("2011-01-01"), "a warmup margin before the start is kept"
    assert out["date"].max() <= pd.Timestamp("2023-12-31") + pd.Timedelta(days=15)
    unclipped = FnspidPriceSource(tmp_path).load(["AAA"])
    assert unclipped["date"].min() == pd.Timestamp("2000-01-03")
