#!/usr/bin/env python3
"""
Prepare report assets for presentation_docs.

Runs the hourly analysis pipeline and persists required artifacts for the
LaTeX reports and figure-generation scripts:

  - outputs/panels/hourly_panel.parquet
  - outputs/roll_signals/hourly_widening.csv
  - outputs/roll_signals/hourly_spread.csv
  - outputs/roll_signals/multi_spreads.csv
  - outputs/roll_signals/multi_spread_events.csv

Usage (from repo root or via Makefile):
  python scripts/presentation_docs/prepare_report_assets.py \
    --settings config/settings.yaml [--max-files 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def _import_package(root: Path) -> None:
    """Import project package, falling back to src on sys.path when needed."""
    try:
        import futures_roll_analysis  # noqa: F401
        return
    except Exception:
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import futures_roll_analysis  # type: ignore  # noqa: F401


def run(settings_path: Path, max_files: Optional[int] = None) -> None:
    from futures_roll_analysis.config import load_settings
    from futures_roll_analysis import analysis

    root = settings_path.resolve().parents[1]
    outputs = root / "outputs"
    panels_dir = outputs / "panels"
    rolls_dir = outputs / "roll_signals"
    panels_dir.mkdir(parents=True, exist_ok=True)
    rolls_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(settings_path)

    # Hourly (bucket) analysis only — sufficient for report figures
    bucket = analysis.run_bucket_analysis(settings=settings, max_files=max_files)

    # Persist panel
    panel_path = panels_dir / "hourly_panel.parquet"
    try:
        bucket.panel.to_parquet(panel_path)
    except Exception:
        # Fallback: CSV if parquet engine unavailable
        (panels_dir / "hourly_panel.csv").write_text(
            bucket.panel.to_csv(), encoding="utf-8"
        )

    # Persist widening (boolean) with timestamp index
    widen_df = bucket.widening.astype(bool).to_frame(name="spread_widening")
    widen_df.index.name = "bucket_start_ts"
    widening_path = rolls_dir / "hourly_widening.csv"
    widen_df.to_csv(widening_path)

    # Persist spread series
    spread_df = bucket.spread.to_frame(name="spread")
    spread_df.index.name = "bucket_start_ts"
    (rolls_dir / "hourly_spread.csv").write_text(
        spread_df.to_csv(), encoding="utf-8"
    )

    # Persist multi-spread diagnostics (optional consumers)
    if bucket.multi_spreads is not None:
        bucket.multi_spreads.to_csv(rolls_dir / "multi_spreads.csv", index=True)
    if bucket.multi_events is not None:
        bucket.multi_events.to_csv(rolls_dir / "multi_spread_events.csv", index=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare assets for LaTeX reports")
    parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        help="Path to YAML settings (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Limit number of files for a quick run",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    _import_package(root)
    run(Path(args.settings), max_files=args.max_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
