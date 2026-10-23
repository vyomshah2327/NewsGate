"""Timestamp discipline: the single most important leakage control."""

import pandas as pd
import pytest

from newsgate.data.alignment import assign_feature_dates
from newsgate.data.calendar import TradingCalendar

# Mon 2024-01-08 .. Fri 2024-01-12, then Mon 15th is a holiday, trading resumes Tue 16th.
DAYS = ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-16"]
CAL = TradingCalendar(pd.to_datetime(DAYS))


def feature_date(utc_text: str, has_time: bool = True) -> str:
    news = pd.DataFrame({"published_at": [pd.Timestamp(utc_text)], "has_time": [has_time]})
    out = assign_feature_dates(news, CAL)
    return str(out["feature_date"].iloc[0].date())


def test_morning_news_counts_for_same_day():
    # 14:00 UTC in January is 09:00 New York.
    assert feature_date("2024-01-09 14:00") == "2024-01-09"


def test_news_just_before_close_counts_for_same_day():
    # 20:59 UTC is 15:59 New York.
    assert feature_date("2024-01-09 20:59") == "2024-01-09"


def test_news_at_close_moves_to_next_day():
    # 21:00 UTC is exactly 16:00 New York.
    assert feature_date("2024-01-09 21:00") == "2024-01-10"


def test_friday_evening_moves_to_next_trading_day_after_holiday_weekend():
    assert feature_date("2024-01-12 23:30") == "2024-01-16"


def test_weekend_news_moves_to_next_trading_day():
    assert feature_date("2024-01-13 15:00") == "2024-01-16"


def test_holiday_news_moves_to_next_trading_day():
    assert feature_date("2024-01-15 15:00") == "2024-01-16"


def test_late_utc_evening_belongs_to_new_york_same_date():
    # 02:00 UTC on the 10th is 21:00 New York on the 9th, after the close.
    assert feature_date("2024-01-10 02:00") == "2024-01-10"


def test_date_only_news_is_never_used_on_its_own_date():
    assert feature_date("2024-01-09 00:00", has_time=False) == "2024-01-10"


def test_news_after_close_on_last_calendar_day_is_dropped():
    news = pd.DataFrame({"published_at": [pd.Timestamp("2024-01-16 22:00")], "has_time": [True]})
    out = assign_feature_dates(news, CAL)
    assert out.empty


@pytest.mark.parametrize("utc_text", ["2024-01-09 14:00", "2024-01-12 23:30", "2024-01-13 15:00"])
def test_feature_date_is_never_before_publication_date_in_new_york(utc_text):
    published = pd.Timestamp(utc_text).tz_localize("UTC").tz_convert("America/New_York")
    assert pd.Timestamp(feature_date(utc_text)) >= published.tz_localize(None).normalize()
