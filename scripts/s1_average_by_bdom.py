from __future__ import annotations

"""
Average S1 spread (level or ΔS1) by business day-of-month (BDOM) for a given year.

Computes S1 level (F2 − F1) daily from minute data aggregated to daily,
optionally differences to obtain ΔS1 (daily change), aligns each month to
BDOM=1..N, then averages across months for each BDOM.

Outputs (per metric):
- PNG: {outdir}/s1_average_{year}.png
- CSV: {outdir}/s1_average_{year}.csv (columns: mean, p25, p75)

Usage:
    python scripts/s1_average_by_bdom.py --settings config/settings.yaml --year 2023 --metric level
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


def compute_s1_level(settings_path: Path, year: int, *, product: str | None = None) -> pd.Series:
    settings = load_settings(settings_path)
    root_symbol = (product or (list(settings.products)[0] if settings.products else "HG")).upper()
    data_root = Path(settings.data["minute_root"]).resolve()
    tz = settings.data.get("timezone", "US/Central")
    ts_format = settings.data.get("timestamp_format")

    files = find_contract_files(data_root, root_symbol=root_symbol)
    if not files:
        raise FileNotFoundError(f"No {root_symbol} files found under {data_root}")

    daily_by_contract = build_contract_frames(
        files,
        root_symbol=root_symbol,
        tz=tz,
        aggregate="daily",
        timestamp_format=ts_format,
    )

    for k in list(daily_by_contract.keys()):
        df = _filter_year(daily_by_contract[k], year)
        if df.empty:
            daily_by_contract.pop(k)
        else:
            daily_by_contract[k] = df

    if not daily_by_contract:
        raise RuntimeError(f"No daily data in {year} after filtering for {root_symbol}")

    # Metadata subset
    settings_meta = Path(settings.metadata_path).resolve()
    meta = pd.read_csv(settings_meta)
    meta["expiry_date"] = pd.to_datetime(meta["expiry_date"]).dt.normalize()
    contracts = list(daily_by_contract.keys())
    meta = meta[meta["contract"].isin(contracts)]
    if len(meta) < len(contracts):
        missing = sorted(set(contracts) - set(meta["contract"]))
        raise ValueError(f"Missing expiry metadata: {missing[:10]}")

    panel = assemble_panel(daily_by_contract, meta, include_bucket_meta=False)
    expiry_map = build_expiry_map(meta)
    tz_ex = (settings.time or {}).get("tz_exchange", "America/Chicago")
    chain = identify_front_to_f12(panel, expiry_map, tz_exchange=tz_ex, max_contracts=2)
    front_next = chain.rename(columns={"F1": "front_contract", "F2": "next_contract"})
    price_field = settings.data.get("price_field", "close")
    s1 = compute_spread(panel, front_next, price_field=price_field)
    s1 = s1.loc[(s1.index >= pd.Timestamp(f"{year}-01-01")) & (s1.index <= pd.Timestamp(f"{year}-12-31 23:59:59"))]
    return s1.rename("S1")


def align_by_bdom(series: pd.Series, year: int, *, bdom_max: int = 20) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    work = series.dropna().copy()
    idx = pd.DatetimeIndex(work.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    work.index = idx

    for month in range(1, 13):
        m = work[(work.index.year == year) & (work.index.month == month)]
        if m.empty:
            continue
        bdom = pd.Series(np.arange(1, len(m) + 1), index=m.index)
        m_bounded = m.iloc[: min(bdom_max, len(m))]
        bdom_bounded = bdom.iloc[: min(bdom_max, len(m))]
        out[m.index[0].strftime("%b")] = pd.Series(
            m_bounded.values,
            index=bdom_bounded.values,
            name=series.name or "S1",
        )
    return out


def plot_average(mean: pd.Series, p25: pd.Series | None, p75: pd.Series | None, *, title: str, ylabel: str, output_png: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = mean.index.values
    ax.plot(x, mean.values, color="#1f77b4", linewidth=2.0, label="Mean")
    if p25 is not None and p75 is not None:
        ax.fill_between(x, p25.values, p75.values, color="#1f77b4", alpha=0.15, label="IQR (25–75%)")
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
    ap = argparse.ArgumentParser(description="Average S1 (level or Δ) by business day-of-month for a given year.")
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--product", help="Override product (default from settings)")
    ap.add_argument("--bdom-max", type=int, default=20)
    ap.add_argument("--outdir", default="outputs/exploratory")
    ap.add_argument(
        "--metric",
        choices=["level", "diff"],
        default="level",
        help="Compute averages on S1 level or ΔS1 (daily change)",
    )
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    settings_path = Path(args.settings)
    outdir = _ensure_output_dir(Path(args.outdir))

    LOGGER.info("Computing %s for %d...", "ΔS1" if args.metric == "diff" else "S1 levels", args.year)
    s1 = compute_s1_level(settings_path, args.year, product=args.product)
    if args.metric == "diff":
        s1 = s1.diff().rename("ΔS1").dropna()
    months = align_by_bdom(s1, args.year, bdom_max=args.bdom_max)
    if not months:
        raise RuntimeError("No monthly data after alignment")

    # Build wide BDOM×month frame and compute stats
    frames = []
    for label, series in months.items():
        ss = series.copy()
        ss.name = label
        frames.append(ss)
    wide = pd.concat(frames, axis=1)
    mean = wide.mean(axis=1)
    p25 = wide.quantile(0.25, axis=1)
    p75 = wide.quantile(0.75, axis=1)

    title = f"Average {'ΔS1' if args.metric == 'diff' else 'S1 level'} by BDOM — {args.year}"
    ylabel = "ΔS1 (daily change)" if args.metric == "diff" else "S1 level (next − front)"
    output_png = outdir / f"s1_average_{args.year}.png"
    plot_average(mean, p25, p75, title=title, ylabel=ylabel, output_png=output_png)

    # Save CSV
    out_csv = outdir / f"s1_average_{args.year}.csv"
    out = pd.DataFrame({"mean": mean, "p25": p25, "p75": p75})
    out.index.name = "BDOM"
    out.to_csv(out_csv)
    LOGGER.info("Wrote %s", output_png)
    LOGGER.info("Wrote %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
