"""Typed configuration.

Configuration is a tree of frozen dataclasses. ``load_config`` reads a YAML file and
fills every field it names, rejecting unknown keys so that typos fail loudly instead of
silently falling back to a default.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is malformed."""


@dataclass(frozen=True)
class SyntheticConfig:
    n_tickers: int = 24
    n_days: int = 1400
    seed: int = 7
    start: str = "2016-01-04"
    mean_articles: float = 1.3
    # How strongly news activity raises the next-day volatility (0 means no effect).
    attention_effect: float = 0.45
    # How strongly (signed) sentiment shifts next-day returns (0 means no effect).
    sentiment_effect: float = 0.0


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"  # synthetic | fnspid | yfinance
    news_path: str = "data/raw/fnspid/news.parquet"
    price_dir: str = "data/raw/fnspid/prices"
    metadata_path: str | None = None  # optional CSV with ticker, sector, market_cap
    tickers: list[str] = field(default_factory=list)
    n_tickers: int | None = None  # stratified sample size when tickers is empty
    start: str | None = None
    end: str | None = None
    source_timezone: str = "UTC"  # zone of timestamps that carry no explicit offset
    timestamp_mode: str = "auto"  # auto | datetime | date_only
    market_symbol: str = "SPY"
    warmup_days: int = 60
    cache_dir: str = "data/processed"
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)


@dataclass(frozen=True)
class SentimentConfig:
    scorers: list[str] = field(default_factory=lambda: ["vader"])
    lm_dictionary_path: str = "data/external/Loughran-McDonald_MasterDictionary.csv"
    finbert_model: str = "ProsusAI/finbert"
    finbert_batch_size: int = 32


@dataclass(frozen=True)
class EventConfig:
    window_days: int = 3
    similarity_threshold: float = 0.55
    n_features: int = 2**16


@dataclass(frozen=True)
class FeatureConfig:
    return_lags: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    momentum_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 60])
    vol_windows: list[int] = field(default_factory=lambda: [5, 20, 60])
    volume_window: int = 20
    news_baseline_window: int = 20
    sentiment_roll: int = 5


@dataclass(frozen=True)
class TargetConfig:
    horizon: int = 1  # trading days ahead
    quantile: float = 0.9  # large-move threshold, learned on training data only
    threshold_scope: str = "ticker"  # ticker | global


@dataclass(frozen=True)
class EvalConfig:
    min_train_days: int = 504
    test_block_days: int = 63
    purge_days: int = 5  # must be >= target horizon
    holdout_fraction: float = 0.2
    inner_val_fraction: float = 0.2
    calibrate: bool = True
    kinds: list[str] = field(default_factory=lambda: ["logistic", "lightgbm"])
    feature_sets: list[str] = field(
        default_factory=lambda: ["price_only", "price_attention", "price_sentiment", "price_news"]
    )
    bootstrap_resamples: int = 2000
    block_length: int = 10
    alpha: float = 0.05
    seed: int = 0


@dataclass(frozen=True)
class GateConfig:
    enabled: bool = True
    kind: str = "lightgbm"  # which model family the gate chooses between
    regressor: str = "lightgbm"  # lightgbm | ridge
    min_history_folds: int = 2
    default_coverage: float = 0.8
    coverage_levels: list[float] = field(default_factory=lambda: [0.5, 0.7, 0.8, 0.9, 1.0])
    margin: float = 0.0


@dataclass(frozen=True)
class RunConfig:
    name: str = "default"
    output_dir: str = "outputs"
    log_level: str = "INFO"


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    events: EventConfig = field(default_factory=EventConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def validate(self) -> "Config":
        if self.evaluation.purge_days < self.target.horizon:
            raise ConfigError(
                f"purge_days ({self.evaluation.purge_days}) must be at least the target "
                f"horizon ({self.target.horizon}) or overlapping labels will leak."
            )
        if not 0.5 < self.target.quantile < 1.0:
            raise ConfigError("target.quantile must be between 0.5 and 1.0")
        if self.target.threshold_scope not in {"ticker", "global"}:
            raise ConfigError("target.threshold_scope must be 'ticker' or 'global'")
        if not 0.0 < self.evaluation.holdout_fraction < 0.5:
            raise ConfigError("evaluation.holdout_fraction must be in (0, 0.5)")
        if self.data.timestamp_mode not in {"auto", "datetime", "date_only"}:
            raise ConfigError("data.timestamp_mode must be auto, datetime or date_only")
        if not self.sentiment.scorers:
            raise ConfigError("at least one sentiment scorer is required")
        return self


def _build(cls: type, data: Mapping[str, Any], path: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        where = path or cls.__name__
        raise ConfigError(f"Unknown configuration key(s) in {where}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint) and isinstance(value, Mapping):
            value = _build(hint, value, f"{path}.{name}" if path else name)
        kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: Mapping[str, Any]) -> Config:
    return _build(Config, data).validate()


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load a YAML config (or defaults when ``path`` is None) and apply nested overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    if overrides:
        raw = _deep_merge(raw, dict(overrides))
    return config_from_dict(raw)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(cfg), handle, sort_keys=False)
