"""The five report figures.

Styling follows one small system: thin 1.5 pt lines, recessive grid, text in ink colours (never
the series colour), direct or legend labels for every series, and a categorical order that
is fixed (blue, orange, aqua, yellow). Every function returns the matplotlib ``Figure`` and
saves it when ``path`` is given.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from newsgate.evaluation.metrics import reliability_table  # noqa: E402

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _finish(fig, path: str | Path | None):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig


def plot_brier_differences(table: pd.DataFrame, path: str | Path | None = None):
    """Figure 1: Brier-score difference versus the baseline with confidence intervals."""
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * max(len(table), 3) + 1.4))
    ypos = np.arange(len(table))[::-1]
    for y, (_, row) in zip(ypos, table.iterrows()):
        significant = row.get("p_holm", 1.0) < 0.05
        ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=BLUE, linewidth=1.5, solid_capstyle="round")
        ax.plot(row["mean_diff"], y, "o", markersize=7, markerfacecolor=BLUE if significant else SURFACE,
                markeredgecolor=BLUE, markeredgewidth=1.5)
    ax.axvline(0.0, color=INK_2, linewidth=1.0)
    ax.set_yticks(ypos, table["comparison"], color=INK)
    _style(ax, "Does news improve the forecast?", "Brier score difference (negative = news helps)")
    ax.grid(False, axis="y")
    fig.text(0.99, 0.005, "filled marker = significant after Holm correction", ha="right", va="bottom",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.patch.set_facecolor(SURFACE)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig


def plot_calibration(oof: pd.DataFrame, columns: dict[str, str], path: str | Path | None = None):
    """Figure 2: reliability curves. ``columns`` maps a legend label to a prediction column."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    top = 0.0
    for color, (label, col) in zip(SERIES, columns.items()):
        table = reliability_table(oof["y"], oof[col], n_bins=10)
        ax.plot(table["mean_pred"], table["frac_pos"], "-o", color=color, linewidth=1.5, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=label)
        top = max(top, table["mean_pred"].max(), table["frac_pos"].max())
    top = max(top * 1.1, 0.2)
    ax.plot([0, top], [0, top], color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    _style(ax, "Calibration of predicted large-move probability", "Mean predicted probability", "Observed frequency")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    return _finish(fig, path)


def plot_risk_coverage(table: pd.DataFrame, path: str | Path | None = None):
    """Figure 3: selective Brier loss as coverage shrinks (lower is better)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for color, (variant, part) in zip(SERIES, table.groupby("variant", sort=False)):
        ax.plot(part["coverage"], part["risk"], color=color, linewidth=1.5, label=variant)
    _style(ax, "Risk versus coverage", "Coverage (share of forecasts kept)", "Brier loss of kept forecasts")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    return _finish(fig, path)


def plot_decile_rates(frame: pd.DataFrame, path: str | Path | None = None):
    """Figure 4: large-move rate by sentiment decile (days with news only).

    ``frame`` needs ``sent_mean``, ``att_has_news`` and ``y``.
    """
    rows = frame[frame["att_has_news"] > 0].dropna(subset=["sent_mean", "y"])
    deciles = pd.qcut(rows["sent_mean"].rank(method="first"), 10, labels=False)
    rate = rows.groupby(deciles)["y"].mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(rate.index + 1, rate.to_numpy(), color=BLUE, width=0.72)
    ax.axhline(rows["y"].mean(), color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(10.4, rows["y"].mean(), "overall rate", va="bottom", ha="right", fontsize=8, color=INK_2)
    ax.set_xticks(range(1, 11))
    _style(ax, "Large-move rate by sentiment decile", "Sentiment decile (1 = most negative)", "Share of next-day large moves")
    ax.grid(False, axis="x")
    return _finish(fig, path)


def plot_event_timeline(prices: pd.DataFrame, news: pd.DataFrame, ticker: str, path: str | Path | None = None, last_days: int = 250):
    """Figure 5: price with article activity underneath, coloured by sentiment sign.

    ``prices`` has ticker, date, adj_close. ``news`` has ticker, feature_date, sent_* and is_new_event.
    """
    px = prices[prices["ticker"] == ticker].sort_values("date").tail(last_days)
    articles = news[(news["ticker"] == ticker) & (news["feature_date"] >= px["date"].min())]
    sent_col = next((c for c in articles.columns if c.startswith("sent_")), None)
    daily = articles.groupby("feature_date").agg(n=("article_id", "size"), new=("is_new_event", "sum"))
    if sent_col:
        daily["sent"] = articles.groupby("feature_date")[sent_col].mean()

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.0, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.6]})
    top.plot(px["date"], px["adj_close"], color=INK, linewidth=1.5)
    _style(top, f"{ticker}: price and news activity", "", "Adjusted close")
    colors = np.where(daily.get("sent", pd.Series(0.0, index=daily.index)) >= 0, BLUE, ORANGE)
    bottom.bar(daily.index, daily["n"], color=colors, width=1.0)
    bottom.bar(daily.index, daily["new"], color=INK_2, width=1.0, alpha=0.55)
    _style(bottom, "", "", "Articles per day")
    bottom.grid(False, axis="x")
    handles = [Patch(color=BLUE, label="net positive tone"), Patch(color=ORANGE, label="net negative tone"),
               Patch(color=INK_2, alpha=0.55, label="of which new events")]
    bottom.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", ncol=3)
    bottom.set_ylim(0, max(daily["n"].max() * 1.35, 2))
    return _finish(fig, path)
