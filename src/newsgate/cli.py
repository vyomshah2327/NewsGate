"""Command line interface: ``newsgate <command> --config configs/synthetic.yaml``."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from newsgate import pipeline
from newsgate.config import load_config
from newsgate.logging_utils import configure_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="newsgate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", "-c", default=None, help="YAML config (defaults are used when omitted)")
        return p

    add("panel", "build the feature panel from the configured data source")
    add("cv", "run chronological cross-validation (and the gate)")
    add("holdout", "evaluate the untouched final period (run once, after freezing the design)")
    report = add("report", "write comparison tables and figures")
    report.add_argument("--split", choices=["cv", "holdout"], default="cv")
    run_all = add("all", "panel, cv and report in one go")
    run_all.add_argument("--final", action="store_true", help="also evaluate the final holdout")
    synth = add("write-synthetic", "write a synthetic world in FNSPID-like file layout")
    synth.add_argument("--out", default="data/synthetic", help="output directory")
    dash = add("dashboard", "launch the Streamlit dashboard")
    dash.add_argument("--port", default="8501")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    configure_logging(cfg.run.log_level)

    if args.command == "panel":
        pipeline.run_panel(cfg)
    elif args.command == "cv":
        pipeline.run_cv(cfg)
    elif args.command == "holdout":
        pipeline.run_holdout(cfg)
        pipeline.run_report(cfg, "holdout")
    elif args.command == "report":
        pipeline.run_report(cfg, args.split)
    elif args.command == "all":
        pipeline.run_all(cfg, final=args.final)
    elif args.command == "write-synthetic":
        from newsgate.data.synthetic import SyntheticWorld

        world = SyntheticWorld.generate(cfg.data.synthetic)
        news_path, price_dir = world.write_fnspid_like(args.out)
        print(f"Wrote {news_path} and prices in {price_dir}")
    elif args.command == "dashboard":
        app = Path(__file__).resolve().parents[2] / "app" / "dashboard.py"
        run_dir = pipeline.RunPaths.from_config(cfg).root
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", args.port, "--", "--run-dir", str(run_dir)]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
