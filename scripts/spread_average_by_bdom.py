from __future__ import annotations

"""
Average any calendar spread (S1, S2, ...) by business day-of-month for a given year.

This script consolidates the functionality of s1_average_by_bdom.py and s2_average_by_bdom.py
into a single parameterized script that works for any spread in the strip.

Computes the spread level daily from minute data aggregated to daily, optionally
differences to obtain the daily change, aligns each month to BDOM=1..N, then
averages across months for each BDOM.

Outputs:
- PNG: {outdir}/{spread}_average_{year}.png
- CSV: {outdir}/{spread}_average_{year}.csv (columns: mean, p25, p75)

Usage:
    python scripts/spread_average_by_bdom.py --spread S1 --year 2023 --metric level
    python scripts/spread_average_by_bdom.py --spread S2 --year 2023 --metric diff
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


def compute_spread_level(
    settings_path: Path,
    year: int,
    *,
    spread: str = "S1",
    product: str | None = None,
) -> pd.Series:
    """Compute spread level daily series for the given year.

    Args:
        settings_path: Path to settings YAML
        year: Target year
        spread: Spread name (S1, S2, S3, etc.)
        product: Override product symbol

    Returns:
        pandas Series of spread levels indexed by timestamp
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

    # Determine required contract depth from spread name
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
    return s.rename(spread)


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
        out[m.index[0].strftime("%b")] = pd.Series(
            m_bounded.values,
            index=bdom_bounded.values,
            name=series.name,
        )
    return out


def plot_average(
    mean: pd.Series,
    p25: pd.Series | None,
    p75: pd.Series | None,
    *,
    title: str,
    ylabel: str,
    output_png: Path,
    color: str = "#1f77b4",
) -> None:
    """Plot average with IQR band."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = mean.index.values
    ax.plot(x, mean.values, color=color, linewidth=2.0, label="Mean")
    if p25 is not None and p75 is not None:
        ax.fill_between(x, p25.values, p75.values, color=color, alpha=0.15, label="IQR (25-75%)")
    ax.set_title(title)
    ax.set_xlabel("Business Day of Month (BDOM)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Average any spread (S1, S2, ...) by business day-of-month for a given year."
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
    LOGGER.info("Computing %s for %d...", f"Δ{spread}" if args.metric == "diff" else f"{spread} levels", args.year)

    s = compute_spread_level(Path(args.settings), args.year, spread=spread, product=args.product)
    if args.metric == "diff":
        s = s.diff().rename(f"Δ{spread}").dropna()

    months = align_by_bdom(s, args.year, bdom_max=args.bdom_max)
    if not months:
        raise RuntimeError("No monthly data after alignment")

    # Build wide BDOM x month frame and compute stats
    frames = []
    for label, series in months.items():
        ss = series.copy()
        ss.name = label
        frames.append(ss)
    wide = pd.concat(frames, axis=1)
    mean = wide.mean(axis=1)
    p25 = wide.quantile(0.25, axis=1)
    p75 = wide.quantile(0.75, axis=1)

    if args.metric == "level":
        title = f"Average {spread} level by BDOM — {args.year}"
        ylabel = f"{spread} level"
    else:
        title = f"Average Δ{spread} by BDOM — {args.year}"
        ylabel = f"Δ{spread} (daily change)"

    color = SPREAD_COLORS.get(spread, "#1f77b4")
    outdir = _ensure_output_dir(Path(args.outdir))
    out_png = outdir / f"{spread.lower()}_average_{args.year}.png"
    plot_average(mean, p25, p75, title=title, ylabel=ylabel, output_png=out_png, color=color)

    # Save CSV
    out_csv = outdir / f"{spread.lower()}_average_{args.year}.csv"
    out = pd.DataFrame({"mean": mean, "p25": p25, "p75": p75})
    out.index.name = "BDOM"
    out.to_csv(out_csv)

    LOGGER.info("Wrote %s", out_png)
    LOGGER.info("Wrote %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
