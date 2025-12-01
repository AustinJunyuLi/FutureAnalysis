from __future__ import annotations

"""
Overlay any calendar spread (S1, S2, S3, ...) by business day-of-month for a single year.

This script consolidates the functionality of s1_overlay.py and s2_overlay.py into a
single parameterized script that works for any spread in the strip.

Outputs:
- PNG: outputs/exploratory/{spread}_overlay_{year}.png
- CSV: outputs/exploratory/{spread}_overlay_{year}.csv (columns per month, index=BDOM)

Usage:
    python scripts/spread_overlay.py --spread S1 --year 2023 --metric diff
    python scripts/spread_overlay.py --spread S2 --year 2023 --metric level --bdom-max 20
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from futures_roll_analysis.config import load_settings
from futures_roll_analysis.ingest import build_contract_frames, find_contract_files
from futures_roll_analysis.panel import assemble_panel
from futures_roll_analysis.rolls import (
    build_expiry_map,
    identify_front_to_f12,
    compute_multi_spreads,
)


LOGGER = logging.getLogger(__name__)

# Color palette for different spreads
SPREAD_COLORS = {
    "S1": "#1f77b4",  # blue
    "S2": "#d62728",  # red
    "S3": "#2ca02c",  # green
    "S4": "#ff7f0e",  # orange
    "S5": "#9467bd",  # purple
    "S6": "#8c564b",  # brown
}


def _ensure_output_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _filter_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31 23:59:59")
    return df.loc[(df.index >= start) & (df.index <= end)]


def _month_abbr(m: int) -> str:
    return ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m]


def compute_spread_series(
    settings_path: Path,
    year: int,
    *,
    spread: str = "S1",
    product: str | None = None,
    metric: str = "level",
) -> pd.Series:
    """Compute any spread (S1, S2, ...) daily series for the given year.

    Args:
        settings_path: Path to settings YAML
        year: Target year
        spread: Spread name (S1, S2, S3, etc.)
        product: Override product symbol
        metric: 'level' for spread value, 'diff' for daily change

    Returns:
        pandas Series indexed by timestamp
    """
    settings = load_settings(settings_path)
    root = (product or (list(settings.products)[0] if settings.products else "HG")).upper()
    data_root = Path(settings.data["minute_root"]).resolve()
    tz = settings.data.get("timezone", "US/Central")
    ts_fmt = settings.data.get("timestamp_format")

    files = find_contract_files(data_root, root_symbol=root)
    if not files:
        raise FileNotFoundError(f"No {root} files found under {data_root}")

    daily_by_contract = build_contract_frames(
        files,
        root_symbol=root,
        tz=tz,
        aggregate="daily",
        timestamp_format=ts_fmt,
    )

    # Filter contracts to year
    for k in list(daily_by_contract.keys()):
        df = _filter_year(daily_by_contract[k], year)
        if df.empty:
            daily_by_contract.pop(k)
        else:
            daily_by_contract[k] = df

    if not daily_by_contract:
        raise RuntimeError(f"No daily data for {root} in {year}")

    # Metadata subset
    meta = pd.read_csv(Path(settings.metadata_path).resolve())
    if not {"contract", "expiry_date"}.issubset(meta.columns):
        raise ValueError("Metadata CSV must contain 'contract' and 'expiry_date'")
    meta["expiry_date"] = pd.to_datetime(meta["expiry_date"]).dt.normalize()
    contracts = list(daily_by_contract.keys())
    meta = meta[meta["contract"].isin(contracts)]
    if len(meta) < len(contracts):
        missing = sorted(set(contracts) - set(meta["contract"]))
        raise ValueError(f"Missing expiry metadata for contracts: {missing[:10]}")

    panel = assemble_panel(daily_by_contract, meta, include_bucket_meta=False)
    expiry_map = build_expiry_map(meta)
    tz_ex = (settings.time or {}).get("tz_exchange", "America/Chicago")

    # Determine required contract depth from spread name (S1 needs 2, S2 needs 3, etc.)
    spread_num = int(spread[1:]) if spread.startswith("S") and spread[1:].isdigit() else 1
    max_contracts = max(6, spread_num + 1)

    chain = identify_front_to_f12(panel, expiry_map, tz_exchange=tz_ex, max_contracts=max_contracts)
    price_field = settings.data.get("price_field", "close")
    spreads = compute_multi_spreads(panel, chain, price_field=price_field)

    if spread not in spreads.columns:
        raise RuntimeError(f"{spread} not available — insufficient contract depth")

    s = spreads[spread]
    # Restrict index to year
    s = s.loc[(s.index >= pd.Timestamp(f"{year}-01-01")) & (s.index <= pd.Timestamp(f"{year}-12-31 23:59:59"))]

    if metric == "level":
        return s.rename(spread)
    elif metric == "diff":
        return s.diff().rename(f"Δ{spread}")
    else:
        raise ValueError("metric must be 'level' or 'diff'")


def align_by_bdom(series: pd.Series, year: int, *, bdom_max: int = 20) -> Dict[str, pd.Series]:
    """Return a dict of month label -> Series indexed by BDOM (1..bdom_max)."""
    out: Dict[str, pd.Series] = {}
    s = series.dropna().copy()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    s.index = idx

    for month in range(1, 13):
        m = s[(s.index.year == year) & (s.index.month == month)]
        if m.empty:
            continue
        bdom = pd.Series(np.arange(1, len(m) + 1), index=m.index)
        m_bounded = m.iloc[: min(bdom_max, len(m))]
        bdom_bounded = bdom.iloc[: min(bdom_max, len(m))]
        out[_month_abbr(month)] = pd.Series(m_bounded.values, index=bdom_bounded.values)
    return out


def plot_overlay(
    month_map: Dict[str, pd.Series],
    *,
    title: str,
    ylabel: str,
    output_png: Path,
    output_csv: Path,
) -> None:
    """Plot overlay of monthly series and save CSV."""
    plt.figure(figsize=(10, 6))
    for label, s in month_map.items():
        if s.empty:
            continue
        plt.plot(s.index.values, s.values, label=label, linewidth=1.4)

    plt.title(title)
    plt.xlabel("Business Day of Month (BDOM)")
    plt.ylabel(ylabel)
    plt.xlim(1, max((int(s.index.max()) for s in month_map.values() if not s.empty), default=20))
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=9)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()

    # Save wide CSV with each month as a column
    if month_map:
        frames = []
        for label, s in month_map.items():
            ss = s.copy()
            ss.name = label
            frames.append(ss)
        df = pd.concat(frames, axis=1)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index_label="BDOM")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Overlay any spread (S1, S2, ...) by business day-of-month for a given year."
    )
    ap.add_argument("--spread", default="S1", help="Spread name: S1, S2, S3, etc. (default: S1)")
    ap.add_argument("--settings", default="config/settings.yaml", help="Path to settings YAML")
    ap.add_argument("--year", type=int, required=True, help="Target year (e.g., 2023)")
    ap.add_argument("--product", help="Override product symbol (default from settings)")
    ap.add_argument("--metric", choices=["level", "diff"], default="level", help="'level' or 'diff' (daily change)")
    ap.add_argument("--bdom-max", type=int, default=20, help="Max business day-of-month to include (default: 20)")
    ap.add_argument("--outdir", default="outputs/exploratory", help="Output directory for PNG/CSV")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    spread = args.spread.upper()
    series = compute_spread_series(
        Path(args.settings), args.year, spread=spread, product=args.product, metric=args.metric
    )
    months = align_by_bdom(series, args.year, bdom_max=args.bdom_max)

    if args.metric == "level":
        ylabel = f"{spread} level"
        title = f"{spread} overlay by BDOM — {args.year} (level)"
    else:
        ylabel = f"Δ{spread} (daily change)"
        title = f"{spread} overlay by BDOM — {args.year} (Δ)"

    outdir = _ensure_output_dir(Path(args.outdir))
    out_png = outdir / f"{spread.lower()}_overlay_{args.year}.png"
    out_csv = outdir / f"{spread.lower()}_overlay_{args.year}.csv"
    plot_overlay(months, title=title, ylabel=ylabel, output_png=out_png, output_csv=out_csv)

    LOGGER.info("Wrote %s", out_png)
    LOGGER.info("Wrote %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
