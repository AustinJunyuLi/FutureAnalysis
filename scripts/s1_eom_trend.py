from __future__ import annotations

"""
End-of-month S1 trend analysis (2015–2024).

Computes intraday-average S1 (front spread F2−F1) per trading day using
bucket-level aggregation, then selects the last five trading days of each
month (EOM-4..EOM). For each offset, runs a time-trend regression across
months (y = α + β·t + ε), where t is a month index starting at 0 for 2015-01.

Outputs (default folder: outputs/s1_eom_trend_2015_2024):
  - eom_data_long.csv: date, year, month, offset, s1_us_mean, t_index
  - regression_summary.csv: one row per offset (beta_month, beta_year, se, t, p_norm, r2, n)
  - scatter_trends.png: scatter colored by offset with fitted trend lines
  - beta_bars.png: bar chart of β (per month) with error bars
  - README.md: method notes and regeneration command

Usage:
  PYTHONPATH=src python scripts/s1_eom_trend.py --settings config/settings.yaml \
      --start 2015-01 --end 2024-12 --outdir outputs/s1_eom_trend_2015_2024
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from futures_roll_analysis.config import load_settings
from futures_roll_analysis.ingest import build_contract_frames, find_contract_files
from futures_roll_analysis.panel import assemble_panel
from futures_roll_analysis.rolls import build_expiry_map, identify_front_to_f12, compute_spread
from futures_roll_analysis.trading_days import load_calendar


LOGGER = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _month_index(dt: pd.Timestamp, base: pd.Timestamp) -> int:
    return (dt.year - base.year) * 12 + (dt.month - base.month)


def compute_us_session_s1_daily(settings_path: Path, start: str, end: str, *, product: str | None = None) -> pd.Series:
    """Compute daily S1 as the equal-weight mean over US regular-hour buckets (1..7)."""
    settings = load_settings(settings_path)
    root = (product or (list(settings.products)[0] if settings.products else "HG")).upper()

    data_root = Path(settings.data["minute_root"]).resolve()
    tz = settings.data.get("timezone", "US/Central")
    ts_fmt = settings.data.get("timestamp_format")

    files = find_contract_files(data_root, root_symbol=root)
    if not files:
        raise FileNotFoundError(f"No {root} files found under {data_root}")

    # Bucket aggregation per contract
    buckets = build_contract_frames(
        files,
        root_symbol=root,
        tz=tz,
        aggregate="bucket",
        timestamp_format=ts_fmt,
    )
    # Clip to date window to reduce memory
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    for k in list(buckets.keys()):
        df = buckets[k]
        win = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if win.empty:
            buckets.pop(k)
        else:
            buckets[k] = win

    if not buckets:
        raise RuntimeError("No bucket data in requested window")

    # Metadata subset and panel
    meta = pd.read_csv(Path(settings.metadata_path).resolve())
    if not {"contract", "expiry_date"}.issubset(meta.columns):
        raise ValueError("Metadata CSV must contain 'contract' and 'expiry_date'")
    meta["expiry_date"] = pd.to_datetime(meta["expiry_date"]).dt.normalize()
    contracts = list(buckets.keys())
    meta = meta[meta["contract"].isin(contracts)]
    if len(meta) < len(contracts):
        missing = sorted(set(contracts) - set(meta["contract"]))
        LOGGER.warning("Missing expiries for some contracts; dropping: %s", ", ".join(missing[:10]))

    panel = assemble_panel(buckets, meta, include_bucket_meta=True)
    expiry_map = build_expiry_map(meta)
    tz_ex = (settings.time or {}).get("tz_exchange", "America/Chicago")
    chain = identify_front_to_f12(panel, expiry_map, tz_exchange=tz_ex, max_contracts=4)
    front_next = chain.rename(columns={"F1": "front_contract", "F2": "next_contract"})

    # S1 at bucket timestamps
    s1 = compute_spread(panel, front_next, price_field=settings.data.get("price_field", "close"))
    # US regular-hour buckets: 1..7
    bucket_ids = pd.to_numeric(panel[("meta", "bucket")], errors="coerce").astype("Int64")
    # Align spread to bucket rows
    df = pd.DataFrame({"spread": s1}).copy()
    df["bucket"] = bucket_ids.reindex(df.index)
    df = df.dropna(subset=["spread", "bucket"])
    df["bucket"] = df["bucket"].astype(int)
    us = df[(df["bucket"] >= 1) & (df["bucket"] <= 7)]
    # Daily mean over available US buckets
    daily = us.groupby(us.index.normalize())["spread"].mean().rename("s1_us_mean")
    # Clip to window (normalize index already naive)
    daily = daily[(daily.index >= start_ts.normalize()) & (daily.index <= pd.Timestamp(end).normalize())]
    return daily


def last_five_trading_days(calendar_df: pd.DataFrame, year: int, month: int) -> List[pd.Timestamp]:
    """Compute last five trading days using weekdays minus explicit full-closed holidays."""
    cal = calendar_df.copy()
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    # Dates explicitly marked closed/full close
    closed = set(cal.loc[cal["is_trading_day"].astype(bool) == False, "date"].tolist())
    # Month range
    first = pd.Timestamp(year=year, month=month, day=1)
    last = first + pd.offsets.MonthEnd(0)
    days = pd.date_range(first, last, freq="D")
    # Weekdays only (Mon-Fri)
    weekdays = [d for d in days if d.weekday() < 5]
    # Remove explicit closed
    trading = [d for d in weekdays if d.normalize() not in closed]
    if len(trading) == 0:
        return []
    return trading[-5:]


@dataclass
class OLSResult:
    beta_month: float
    alpha: float
    se_beta: float
    t_beta: float
    p_norm: float
    r2: float
    n: int


def ols_trend(y: np.ndarray, t: np.ndarray) -> OLSResult:
    """Simple OLS with normal-approx p-value for beta (no SciPy dependency)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(y)
    X = np.column_stack([np.ones(n), t])
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    yhat = X @ beta
    resid = y - yhat
    dof = max(n - 2, 1)
    s2 = float((resid @ resid) / dof)
    var = s2 * XtX_inv
    se = np.sqrt(np.diag(var))
    se_beta = float(se[1])
    b1 = float(beta[1])
    t_beta = float(b1 / se_beta) if se_beta > 0 else np.nan
    # Normal approx p-value
    import math
    p_norm = float(2 * (1 - 0.5 * (1 + math.erf(abs(t_beta) / math.sqrt(2)))))
    # R^2
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return OLSResult(beta_month=b1, alpha=float(beta[0]), se_beta=se_beta, t_beta=t_beta, p_norm=p_norm, r2=r2, n=n)


def plot_scatter_with_trends(df_long: pd.DataFrame, summary: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(11, 6))
    colors = {
        "EOM-4": "#1f77b4",
        "EOM-3": "#ff7f0e",
        "EOM-2": "#2ca02c",
        "EOM-1": "#d62728",
        "EOM": "#9467bd",
    }
    base = pd.Timestamp("2015-01-01")
    for off, group in df_long.groupby("offset"):
        c = colors.get(off, None)
        plt.scatter(group["date"], group["s1_us_mean"], s=12, color=c, alpha=0.65, label=off)
        # Fitted line using month index
        res = summary.loc[summary["offset"] == off].iloc[0]
        # Build fitted values across observed dates for that offset
        tvals = group["t_index"].to_numpy()
        yhat = res["alpha"] + res["beta_month"] * tvals
        plt.plot(group["date"], yhat, color=c, linewidth=1.6)
    plt.title("S1 (US session mean) on last 5 trading days — 2015–2024")
    plt.xlabel("Date")
    plt.ylabel("S1 level (F2 − F1)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=5, fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_beta_bars(summary: pd.DataFrame, out_png: Path) -> None:
    order = ["EOM-4", "EOM-3", "EOM-2", "EOM-1", "EOM"]
    sub = summary.set_index("offset").loc[order]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(order))
    ax.bar(x, sub["beta_month"], yerr=sub["se_beta"], color="#1f77b4", alpha=0.8, capsize=4)
    ax.set_xticks(x, order)
    ax.set_ylabel("β (per month)")
    ax.set_title("Time trend β for S1 at EOM offsets (2015–2024)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="EOM S1 trend regressions (2015–2024).")
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--start", default="2015-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--product", help="Override product symbol (default from settings)")
    ap.add_argument("--outdir", default="outputs/s1_eom_trend_2015_2024")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    outdir = _ensure_dir(Path(args.outdir))

    # Date window
    start_month = pd.Timestamp(str(args.start) + "-01")
    end_month = pd.Timestamp(str(args.end) + "-01")
    end_month = end_month + pd.offsets.MonthEnd(0)

    # Compute daily S1 US-session mean
    daily = compute_us_session_s1_daily(Path(args.settings), start=str(start_month.date()), end=str(end_month.date()), product=args.product)

    # Load trading calendar from settings
    settings = load_settings(Path(args.settings))
    cal_paths = settings.business_days.get("calendar_paths", [])
    if not cal_paths:
        raise RuntimeError("Trading calendar paths are required for EOM analysis")
    calendar_df = load_calendar(cal_paths, hierarchy=settings.business_days.get("calendar_hierarchy", "override"))

    # Build last five trading days per month and collect observations
    rows: List[Dict[str, object]] = []
    cur = pd.Timestamp(start_month)
    base = pd.Timestamp("2015-01-01")
    while cur <= end_month:
        y, m = cur.year, cur.month
        last5 = last_five_trading_days(calendar_df, y, m)
        if last5:
            off_labels = ["EOM-4", "EOM-3", "EOM-2", "EOM-1", "EOM"]
            for idx, d in enumerate(last5):
                val = daily.get(d, np.nan)
                rows.append({
                    "date": d,
                    "year": y,
                    "month": m,
                    "offset": off_labels[idx],
                    "s1_us_mean": val,
                    "t_index": _month_index(pd.Timestamp(year=y, month=m, day=1), base),
                })
        cur = (cur + pd.offsets.MonthBegin(1))

    df_long = pd.DataFrame(rows)
    # Keep only exact 5 offsets per month when values exist; drop NaNs
    df_long = df_long.dropna(subset=["s1_us_mean"]).sort_values("date").reset_index(drop=True)
    # Sanity: constrain offsets to expected set
    df_long = df_long[df_long["offset"].isin(["EOM-4", "EOM-3", "EOM-2", "EOM-1", "EOM"])]

    # Regressions per offset
    summary_rows = []
    for off in ["EOM-4", "EOM-3", "EOM-2", "EOM-1", "EOM"]:
        sub = df_long[df_long["offset"] == off]
        if len(sub) < 10:
            LOGGER.warning("Offset %s has few observations (n=%d)", off, len(sub))
        res = ols_trend(sub["s1_us_mean"].to_numpy(), sub["t_index"].to_numpy())
        summary_rows.append({
            "offset": off,
            "alpha": res.alpha,
            "beta_month": res.beta_month,
            "beta_year": res.beta_month * 12.0,
            "se_beta": res.se_beta,
            "t_beta": res.t_beta,
            "p_norm": res.p_norm,
            "r2": res.r2,
            "n": res.n,
        })
    summary = pd.DataFrame(summary_rows)

    # Save outputs
    df_long.to_csv(outdir / "eom_data_long.csv", index=False)
    summary.to_csv(outdir / "regression_summary.csv", index=False)

    # Plots
    plot_scatter_with_trends(df_long, summary, outdir / "scatter_trends.png")
    plot_beta_bars(summary, outdir / "beta_bars.png")

    # README
    readme = f"""
S1 EOM Trend Regressions (2015–2024)
====================================

Definition
- S1 = F2 − F1 (front calendar spread).
- Daily S1 is computed as the equal-weight mean across US regular-hour buckets (09:00–15:00 CT, buckets 1..7).
- For each month, we select the last five trading days (EOM−4 … EOM) using the configured trading calendar.

Regression
- For each offset k ∈ {{EOM−4, EOM−3, EOM−2, EOM−1, EOM}} we regress S1_k(m) = α_k + β_k · t_m + ε,
  with t_m = month index from Jan 2015 = 0 to Dec 2024 = 119.
- Reported β is per month; β_year = 12 × β.
- Standard errors and p-values use a normal approximation (no SciPy dependency).

Files
- eom_data_long.csv          — long-format data (date, offset, S1, t_index)
- regression_summary.csv     — per-offset coefficients and diagnostics
- scatter_trends.png         — scatter by offset with fitted lines
- beta_bars.png              — bar chart of β (per month) with error bars

Reproduce
```
PYTHONPATH=src python scripts/s1_eom_trend.py \
  --settings config/settings.yaml --start 2015-01 --end 2024-12 \
  --outdir outputs/s1_eom_trend_2015_2024
```
"""
    (outdir / "README.md").write_text(readme.strip(), encoding="utf-8")

    LOGGER.info("Wrote outputs to %s", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
