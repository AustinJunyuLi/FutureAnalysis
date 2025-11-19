from __future__ import annotations

"""
Overlay S2 (second spread: F3 − F2) by business day-of-month for a single year.

Outputs:
- PNG: outputs/exploratory/s2_overlay_{year}.png
- CSV: outputs/exploratory/s2_overlay_{year}.csv (columns per month, index=BDOM)

Usage:
  PYTHONPATH=src python scripts/s2_overlay.py --settings config/settings.yaml --year 2023 --metric level --bdom-max 20
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


def _ensure_output_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _filter_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31 23:59:59")
    return df.loc[(df.index >= start) & (df.index <= end)]


def compute_s2_series(settings_path: Path, year: int, *, product: str | None = None, metric: str = "level") -> pd.Series:
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
    # filter contracts to year
    for k in list(daily_by_contract.keys()):
        df = _filter_year(daily_by_contract[k], year)
        if df.empty:
            daily_by_contract.pop(k)
        else:
            daily_by_contract[k] = df
    if not daily_by_contract:
        raise RuntimeError(f"No daily data for {root} in {year}")

    # metadata subset
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
    # need at least 3 contracts depth
    chain = identify_front_to_f12(panel, expiry_map, tz_exchange=tz_ex, max_contracts=6)

    price_field = settings.data.get("price_field", "close")
    spreads = compute_multi_spreads(panel, chain, price_field=price_field)
    if "S2" not in spreads.columns:
        raise RuntimeError("S2 not available — insufficient contract depth")
    s2 = spreads["S2"]
    # restrict index to year
    s2 = s2.loc[(s2.index >= pd.Timestamp(f"{year}-01-01")) & (s2.index <= pd.Timestamp(f"{year}-12-31 23:59:59"))]

    if metric == "level":
        return s2.rename("S2")
    elif metric == "diff":
        return s2.diff().rename("ΔS2")
    else:
        raise ValueError("metric must be 'level' or 'diff'")


def align_by_bdom(series: pd.Series, year: int, *, bdom_max: int = 20) -> Dict[str, pd.Series]:
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
        out[m.index[0].strftime("%b")] = pd.Series(m_bounded.values, index=bdom_bounded.values)
    return out


def plot_overlay(month_map: Dict[str, pd.Series], *, title: str, ylabel: str, output_png: Path, output_csv: Path) -> None:
    plt.figure(figsize=(10, 6))
    for label, s in month_map.items():
        if s.empty:
            continue
        plt.plot(s.index.values, s.values, label=label, linewidth=1.4)
    plt.title(title)
    plt.xlabel("Business Day of Month (BDOM)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=9)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()

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
    ap = argparse.ArgumentParser(description="Overlay S2 (F3−F2) by BDOM for a given year.")
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--product", help="Override product (default from settings)")
    ap.add_argument("--metric", choices=["level", "diff"], default="level")
    ap.add_argument("--bdom-max", type=int, default=20)
    ap.add_argument("--outdir", default="outputs/exploratory")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    s2 = compute_s2_series(Path(args.settings), args.year, product=args.product, metric=args.metric)
    months = align_by_bdom(s2, args.year, bdom_max=args.bdom_max)

    ylabel = "S2 level (F3 − F2)" if args.metric == "level" else "ΔS2 (daily change)"
    title = f"S2 overlay by BDOM — {args.year} ({'level' if args.metric=='level' else 'Δ'})"
    out_png = Path(args.outdir) / f"s2_overlay_{args.year}.png"
    out_csv = Path(args.outdir) / f"s2_overlay_{args.year}.csv"
    plot_overlay(months, title=title, ylabel=ylabel, output_png=out_png, output_csv=out_csv)
    LOGGER.info("Wrote %s", out_png)
    LOGGER.info("Wrote %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

