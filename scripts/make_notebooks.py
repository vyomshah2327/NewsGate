"""Regenerate the notebooks in notebooks/ from the cell lists below.

Keeping the cells in a script (instead of hand-editing JSON) means the notebooks stay short,
reviewable in diffs, and thin: they call the package, they do not reimplement it.
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
    return nb


AUDIT = [
    ("md", "# 01. Data audit\n\nCheck the news timestamps, ticker coverage and duplicates **before** trusting any result. "
           "Set `CONFIG` to `configs/default.yaml` to audit the real FNSPID files; the default runs on the synthetic world."),
    ("code", "import pandas as pd\nimport matplotlib.pyplot as plt\nfrom newsgate.config import load_config\nfrom newsgate.data import build_sources, TradingCalendar, assign_feature_dates, deduplicate_news\n\n"
             "CONFIG = 'configs/synthetic.yaml'\ncfg = load_config(CONFIG)\nsources = build_sources(cfg.data)\nnews = sources.news.load()\nprices = sources.prices.load(sources.tickers)\nlen(news), len(prices)"),
    ("md", "## 1. Do the timestamps carry a time of day?\n\nThis decides whether articles can be placed before or after the 4:00 pm close. "
           "If almost nothing has a time, the conservative date-only rule applies (use news from day *t* only for day *t+1* onward)."),
    ("code", "print('share with a time of day:', round(news['has_time'].mean(), 3))\n"
             "ny = pd.to_datetime(news.loc[news['has_time'], 'published_at']).dt.tz_localize('UTC').dt.tz_convert('America/New_York')\n"
             "ny.dt.hour.value_counts().sort_index().plot(kind='bar', title='Publication hour (New York time)', figsize=(8, 3))\nplt.show()"),
    ("md", "## 2. How many articles survive cleaning and alignment?"),
    ("code", "cal = TradingCalendar.from_prices(prices)\ncleaned = deduplicate_news(news)\naligned = assign_feature_dates(cleaned, cal)\n"
             "pd.Series({'raw': len(news), 'after dedup': len(cleaned), 'aligned to a trading day': len(aligned)})"),
    ("code", "after_close = aligned['published_at'].dt.normalize() != aligned['feature_date']\nprint('articles moved to a later trading day:', round(after_close.mean(), 3))"),
    ("md", "## 3. Coverage per ticker (articles per trading day)"),
    ("code", "per_ticker = aligned.groupby('ticker').size() / len(cal)\nper_ticker.sort_values().plot(kind='barh', figsize=(6, 6), title='Articles per trading day')\nplt.show()\nper_ticker.describe()"),
    ("md", "## 4. Coverage drift over time\n\nA rising or falling number of articles per year can create spurious signals."),
    ("code", "aligned.groupby(aligned['feature_date'].dt.year).size().plot(kind='bar', title='Articles per year', figsize=(8, 3))\nplt.show()"),
]

FEATURES = [
    ("md", "# 02. Feature checks\n\nSanity checks on the modelling panel and a demonstration of the leakage control."),
    ("code", "import numpy as np\nimport pandas as pd\nfrom newsgate.config import load_config\nfrom newsgate.pipeline import RunPaths\n\n"
             "cfg = load_config('configs/synthetic.yaml')\npanel = pd.read_parquet(RunPaths.from_config(cfg).root / 'panel.parquet')\npanel.shape"),
    ("code", "feature_cols = [c for c in panel.columns if c.split('_')[0] in {'px', 'att', 'sent', 'evt'}]\npanel[feature_cols].describe().T.round(3)"),
    ("md", "## Missing values and extreme values"),
    ("code", "panel[feature_cols].isna().mean().sort_values(ascending=False).head(10)"),
    ("md", "## Which features line up with tomorrow's absolute move? (association only)"),
    ("code", "panel[feature_cols + ['tgt_abs']].corr(method='spearman')['tgt_abs'].drop('tgt_abs').sort_values(key=abs, ascending=False).head(15).round(3)"),
    ("md", "## Leakage control\n\nFeatures for a date must not change when later data is removed. "
           "`tests/test_features_leakage.py` checks this on every run; this cell repeats the idea by hand."),
    ("code", "from newsgate.data import build_sources\nfrom newsgate.features.panel import build_panel\n\nsources = build_sources(cfg.data)\nnews, prices = sources.news.load(), sources.prices.load(sources.tickers)\n"
             "cut = pd.Timestamp(np.sort(panel['date'].unique())[len(panel['date'].unique()) // 2])\n"
             "short, _ = build_panel(cfg, news[news['published_at'] < cut + pd.Timedelta(days=1)], prices[prices['date'] <= cut], sources.tickers)\n"
             "key = ['ticker', 'date']\na = panel[panel['date'] <= cut].set_index(key)[feature_cols].sort_index()\nb = short[short['date'] <= cut].set_index(key)[feature_cols].sort_index()\n"
             "print('max absolute difference:', float((a - b).abs().max().max()))"),
]

RESULTS = [
    ("md", "# 03. Results\n\nRun `newsgate all --config configs/synthetic.yaml` first (add `--final` for the holdout)."),
    ("code", "import pandas as pd\nfrom IPython.display import Image, display\nfrom newsgate.config import load_config\nfrom newsgate.pipeline import RunPaths\n\n"
             "cfg = load_config('configs/synthetic.yaml')\nroot = RunPaths.from_config(cfg).root\nroot"),
    ("md", "## Does news improve the forecast? (negative Brier difference = news helps)"),
    ("code", "pd.read_csv(root / 'comparisons.csv')[['comparison', 'mean_diff', 'ci_low', 'ci_high', 'p_value', 'p_holm']].round(5)"),
    ("code", "display(Image(root / 'figures' / 'fig1_brier_differences.png'))"),
    ("md", "## Accuracy of each model"),
    ("code", "pd.read_csv(root / 'model_metrics.csv').round(4)"),
    ("md", "## Calibration"),
    ("code", "display(Image(root / 'figures' / 'fig2_calibration.png'))"),
    ("md", "## The gate against simple abstention rules"),
    ("code", "display(Image(root / 'figures' / 'fig3_risk_coverage.png'))"),
    ("md", "## Does the benefit survive across sectors, coverage levels, market regimes and years? (H3)"),
    ("code", "pd.read_csv(root / 'robustness.csv').round(5)"),
    ("md", "## Sentiment deciles and the event timeline"),
    ("code", "display(Image(root / 'figures' / 'fig4_sentiment_deciles.png'))\ndisplay(Image(root / 'figures' / 'fig5_event_timeline.png'))"),
    ("md", "## Final holdout (only after the design is frozen)"),
    ("code", "path = root / 'comparisons_holdout.csv'\npd.read_csv(path).round(5) if path.exists() else 'Run `newsgate holdout` once the design is frozen.'"),
]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, cells in (("01_data_audit", AUDIT), ("02_feature_checks", FEATURES), ("03_results", RESULTS)):
        nbf.write(notebook(cells), OUT / f"{name}.ipynb")
        print("wrote", OUT / f"{name}.ipynb")
