"""Data access: sources, cleaning, trading calendar and timestamp alignment."""

from newsgate.data.alignment import assign_feature_dates
from newsgate.data.calendar import TradingCalendar
from newsgate.data.cleaning import deduplicate_news, normalize_headline
from newsgate.data.sources import NewsSource, PriceSource, build_sources

__all__ = [
    "NewsSource",
    "PriceSource",
    "TradingCalendar",
    "assign_feature_dates",
    "build_sources",
    "deduplicate_news",
    "normalize_headline",
]
