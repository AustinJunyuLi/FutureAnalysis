from __future__ import annotations

"""
Overlay S1 (front-next spread) by trading day-of-month for a single year.

Outputs:
- PNG figure at outputs/exploratory/s1_overlay_{year}.png
- CSV with aligned values at outputs/exploratory/s1_overlay_{year}.csv

Defaults:
- Product from settings (first entry), usually HG
- Metric: widening (daily diff of S1)
- BDOM cap: 20 (1..20) for consistent coverage across months

Usage:
    python scripts/s1_overlay.py --settings config/settings.yaml --year 2023 \
        --metric diff --bdom-max 20
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from futures_roll_analysis.config import load_settings
from futures_roll_analysis.ingest import build_contract_frames, find_contract_files
from futures_roll_analysis.panel import assemble_panel
from futures_roll_analysis.rolls import (
    build_expiry_map,
    compute_spread,
    identify_front_to_f12,
)


LOGGER = logging.getLogger(__name__)


def _ensure_output_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base


def _filter_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31 23:59:59")
    return df.loc[(df.index >= start) & (df.index <= end)]


def _month_abbr(m: int) -> str:
    return ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m]


def compute_s1_series(
    settings_path: Path,
    year: int,
    *,
    product: str | None = None,
    metric: str = "diff",  # "diff" (ΔS1) or "level" (S1)
) -> pd.Series:
    """Compute S1 (or ΔS1) daily series for the given year using settings.

    Returns a pandas Series indexed by timestamp.
    """
    settings = load_settings(settings_path)

    root_symbol = (product or (list(settings.products)[0] if settings.products else "HG")).upper()
    data_root = Path(settings.data["minute_root"]).resolve()
    tz = settings.data.get("timezone", "US/Central")
    ts_format = settings.data.get("timestamp_format")

    files = find_contract_files(data_root, root_symbol=root_symbol)
    if not files:
        raise FileNotFoundError(f"No {root_symbol} files found under {data_root}")

    # Build daily frames per contract, then filter to target year
    daily_by_contract = build_contract_frames(
        files,
        root_symbol=root_symbol,
        tz=tz,
        aggregate="daily",
        timestamp_format=ts_format,
    )
    # Restrict to the requested year and drop empties
    for k in list(daily_by_contract.keys()):
        df = _filter_year(daily_by_contract[k], year)
        if df.empty:
            daily_by_contract.pop(k)
        else:
            daily_by_contract[k] = df

    if not daily_by_contract:
        raise RuntimeError(f"No daily data in {year} after filtering for product {root_symbol}")

    # Load metadata and assemble panel
    metadata_path = Path(settings.metadata_path).resolve()
    metadata = pd.read_csv(metadata_path)
    if not {"contract", "expiry_date"}.issubset(metadata.columns):
        raise ValueError("Metadata CSV must contain 'contract' and 'expiry_date' columns")
    metadata["expiry_date"] = pd.to_datetime(metadata["expiry_date"]).dt.normalize()

    # Keep only contracts we actually have
    contracts = list(daily_by_contract.keys())
    meta_subset = metadata[metadata["contract"].isin(contracts)]
    if len(meta_subset) < len(contracts):
        missing = sorted(set(contracts) - set(meta_subset["contract"]))
        raise ValueError(f"Missing explicit expiries for contracts: {missing[:10]}")

    panel = assemble_panel(daily_by_contract, meta_subset, include_bucket_meta=False)
    expiry_map = build_expiry_map(meta_subset)
    tz_ex = (settings.time or {}).get("tz_exchange", "America/Chicago")
    chain = identify_front_to_f12(panel, expiry_map, tz_exchange=tz_ex, max_contracts=2)
    front_next = chain.rename(columns={"F1": "front_contract", "F2": "next_contract"})

    price_field = settings.data.get("price_field", "close")
    s1 = compute_spread(panel, front_next, price_field=price_field)
    # Restrict index to target year again (panel alignment can extend slightly)
    s1 = s1.loc[(s1.index >= pd.Timestamp(f"{year}-01-01")) & (s1.index <= pd.Timestamp(f"{year}-12-31 23:59:59"))]

    if metric == "diff":
        return s1.diff().rename("ΔS1")
    elif metric == "level":
        return s1.rename("S1")
    else:
        raise ValueError("metric must be 'diff' or 'level'")


def align_by_bdom(series: pd.Series, year: int, *, bdom_max: int = 20) -> Dict[str, pd.Series]:
    """Return a dict of month label -> Series indexed by BDOM (1..bdom_max)."""
    out: Dict[str, pd.Series] = {}
    # Use observed trading days in the series as the calendar for ordinal positions per month
    df = series.dropna().copy()
    dates = pd.DatetimeIndex(df.index).tz_localize(None) if df.index.tz is not None else pd.DatetimeIndex(df.index)
    df.index = dates

    for month in range(1, 13):
        month_mask = (df.index.year == year) & (df.index.month == month)
        m = df.loc[month_mask]
        if m.empty:
            continue
        # Ordinal within observed trading days of this month
        bdom = pd.Series(np.arange(1, len(m) + 1), index=m.index)
        m_bounded = m.iloc[: min(bdom_max, len(m))]
        bdom_bounded = bdom.iloc[: min(bdom_max, len(m))]
        s = pd.Series(m_bounded.values, index=bdom_bounded.values)
        s.index.name = "BDOM"
        out[_month_abbr(month)] = s
    return out


def plot_overlay(month_to_series: Dict[str, pd.Series], *, title: str, ylabel: str, output_png: Path, output_csv: Path) -> None:
    # Create a common index (1..max length present), but we will draw each month as given
    plt.figure(figsize=(10, 6))
    for label, series in month_to_series.items():
        if series.empty:
            continue
        x = series.index.to_numpy()
        y = series.to_numpy(dtype=float)
        plt.plot(x, y, label=label, linewidth=1.4)

    plt.title(title)
    plt.xlabel("Business Day of Month (BDOM)")
    plt.ylabel(ylabel)
    plt.xlim(1, max((int(s.index.max()) for s in month_to_series.values() if not s.empty), default=20))
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=9)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()

    # Save a wide CSV with each month as a column, index=BDOM
    if month_to_series:
        # Build a DataFrame aligned by BDOM (outer join)
        frames = []
        for label, s in month_to_series.items():
            ss = s.copy()
            ss.name = label
            frames.append(ss)
        df = pd.concat(frames, axis=1)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index_label="BDOM")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot overlay of monthly S1 or ΔS1 by business day-of-month for a given year.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--year", type=int, default=2023, help="Target year (e.g., 2023).")
    parser.add_argument("--product", help="Override product symbol (defaults to first in settings).")
    parser.add_argument("--metric", choices=["diff", "level"], default="diff", help="'diff' for ΔS1, 'level' for S1 level.")
    parser.add_argument("--bdom-max", type=int, default=20, help="Maximum business day-of-month to include (default 20).")
    parser.add_argument("--outdir", default="outputs/exploratory", help="Output directory for PNG/CSV.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    settings_path = Path(args.settings)
    outdir = _ensure_output_dir(Path(args.outdir))

    LOGGER.info("Computing %s for %s in %d...", "ΔS1" if args.metric == "diff" else "S1", args.product or "(settings product)", args.year)
    s1 = compute_s1_series(settings_path, args.year, product=args.product, metric=args.metric)
    months = align_by_bdom(s1, args.year, bdom_max=args.bdom_max)

    ylabel = "ΔS1 (next − front), daily change" if args.metric == "diff" else "S1 level (next − front)"
    title = f"{(args.product or '').upper() or ''} S1 overlay by BDOM — {args.year} ({'Δ' if args.metric=='diff' else 'level'})".strip()
    output_png = outdir / f"s1_overlay_{args.year}.png"
    output_csv = outdir / f"s1_overlay_{args.year}.csv"
    plot_overlay(months, title=title, ylabel=ylabel, output_png=output_png, output_csv=output_csv)

    LOGGER.info("Wrote %s", output_png)
    LOGGER.info("Wrote %s", output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

