from __future__ import annotations

"""
Average S2 (F3 − F2) or ΔS2 by business day-of-month for a given year.

Outputs (per metric):
- PNG: {outdir}/s2_average_{year}.png
- CSV: {outdir}/s2_average_{year}.csv (mean, p25, p75 by BDOM)
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from futures_roll_analysis.config import load_settings
from futures_roll_analysis.ingest import build_contract_frames, find_contract_files
from futures_roll_analysis.panel import assemble_panel
from futures_roll_analysis.rolls import build_expiry_map, identify_front_to_f12, compute_multi_spreads


LOGGER = logging.getLogger(__name__)


def _ensure_output_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _filter_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31 23:59:59")
    return df.loc[(df.index >= start) & (df.index <= end)]


def compute_s2_level(settings_path: Path, year: int, *, product: str | None = None) -> pd.Series:
    settings = load_settings(settings_path)
    root = (product or (list(settings.products)[0] if settings.products else "HG")).upper()
    data_root = Path(settings.data["minute_root"]).resolve()
    tz = settings.data.get("timezone", "US/Central")
    ts_fmt = settings.data.get("timestamp_format")

    files = find_contract_files(data_root, root_symbol=root)
    if not files:
        raise FileNotFoundError(f"No {root} files found under {data_root}")

    daily_by_contract = build_contract_frames(files, root_symbol=root, tz=tz, aggregate="daily", timestamp_format=ts_fmt)
    for k in list(daily_by_contract.keys()):
        df = _filter_year(daily_by_contract[k], year)
        if df.empty:
            daily_by_contract.pop(k)
        else:
            daily_by_contract[k] = df
    if not daily_by_contract:
        raise RuntimeError(f"No daily data in {year}")

    meta = pd.read_csv(Path(settings.metadata_path).resolve())
    meta["expiry_date"] = pd.to_datetime(meta["expiry_date"]).dt.normalize()
    contracts = list(daily_by_contract.keys())
    meta = meta[meta["contract"].isin(contracts)]
    if len(meta) < len(contracts):
        missing = sorted(set(contracts) - set(meta["contract"]))
        raise ValueError(f"Missing expiry metadata: {missing[:10]}")

    panel = assemble_panel(daily_by_contract, meta, include_bucket_meta=False)
    tz_ex = (settings.time or {}).get("tz_exchange", "America/Chicago")
    chain = identify_front_to_f12(panel, build_expiry_map(meta), tz_exchange=tz_ex, max_contracts=6)
    spreads = compute_multi_spreads(panel, chain, price_field=settings.data.get("price_field", "close"))
    if "S2" not in spreads.columns:
        raise RuntimeError("S2 not available")
    s2 = spreads["S2"]
    s2 = s2.loc[(s2.index >= pd.Timestamp(f"{year}-01-01")) & (s2.index <= pd.Timestamp(f"{year}-12-31 23:59:59"))]
    return s2.rename("S2")


def align_by_bdom(series: pd.Series, year: int, *, bdom_max: int = 20) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
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
            name=series.name or "S2",
        )
    return out


def plot_average(mean: pd.Series, p25: pd.Series, p75: pd.Series, *, title: str, ylabel: str, output_png: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = mean.index.values
    ax.plot(x, mean.values, color="#d62728", linewidth=2.0, label="Mean")
    ax.fill_between(x, p25.values, p75.values, color="#d62728", alpha=0.15, label="IQR (25–75%)")
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
    ap = argparse.ArgumentParser(description="Average S2 (level or Δ) by business day-of-month for a year.")
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--product", help="Override product symbol")
    ap.add_argument("--bdom-max", type=int, default=20)
    ap.add_argument("--outdir", default="outputs/exploratory")
    ap.add_argument(
        "--metric",
        choices=["level", "diff"],
        default="level",
        help="Use S2 levels or ΔS2 (daily change) for the averages",
    )
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    s2 = compute_s2_level(Path(args.settings), args.year, product=args.product)
    if args.metric == "diff":
        s2 = s2.diff().rename("ΔS2").dropna()
    months = align_by_bdom(s2, args.year, bdom_max=args.bdom_max)
    if not months:
        raise RuntimeError("No monthly data after alignment")
    wide = pd.concat([s.rename(m) for m, s in months.items()], axis=1)
    mean = wide.mean(axis=1)
    p25 = wide.quantile(0.25, axis=1)
    p75 = wide.quantile(0.75, axis=1)
    title = f"Average {'ΔS2' if args.metric == 'diff' else 'S2 level'} by BDOM — {args.year}"
    ylabel = "ΔS2 (daily change)" if args.metric == "diff" else "S2 level (F3 − F2)"
    out_png = Path(args.outdir) / f"s2_average_{args.year}.png"
    plot_average(mean, p25, p75, title=title, ylabel=ylabel, output_png=out_png)
    out_csv = Path(args.outdir) / f"s2_average_{args.year}.csv"
    out = pd.DataFrame({"mean": mean, "p25": p25, "p75": p75})
    out.index.name = "BDOM"
    out.to_csv(out_csv)
    LOGGER.info("Wrote %s", out_png)
    LOGGER.info("Wrote %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
