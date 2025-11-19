#!/usr/bin/env python
from __future__ import annotations

"""
Analyze S1 behavior during the last five trading days of each month.

This script consumes the long-format dataset produced by s1_eom_trend.py
(`eom_data_long.csv`), computes the intra-month slope of S1 (F2−F1) for each month,
and runs a regression of those slopes against calendar time to test whether the
within-month run-up is persistent or changing.

Outputs (defaults to outputs/s1_eom_trend_2015_2024/):
  - intra_month_slopes.csv          : per-month slope/intercept diagnostics
  - intra_month_regression.csv      : α/β summary for slope_m = α + β·t_m
  - intra_month_slopes.png          : visualization of slopes over time
"""

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # Ensure headless-friendly backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


OFFSET_ORDER: Dict[str, int] = {
    "EOM-4": 1,
    "EOM-3": 2,
    "EOM-2": 3,
    "EOM-1": 4,
    "EOM": 5,
}


@dataclass(frozen=True)
class OLSResult:
    alpha: float
    beta: float
    se_beta: float
    t_beta: float
    p_norm: float
    r2: float
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute intra-month S1 slopes and regress them on time."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/s1_eom_trend_2015_2024/eom_data_long.csv"),
        help="Path to eom_data_long.csv from s1_eom_trend.py.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/s1_eom_trend_2015_2024"),
        help="Output directory for regression tables and plots.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum number of offsets required per month to compute a slope.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def load_eom_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"eom_data_long.csv not found at {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    required_cols = {"date", "offset", "s1_us_mean", "t_index"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df = df[df["offset"].isin(OFFSET_ORDER.keys())].copy()
    df["offset_num"] = df["offset"].map(OFFSET_ORDER).astype(float)
    df["month"] = df["date"].dt.to_period("M")
    df["month_start"] = df["month"].dt.to_timestamp()
    df = df.dropna(subset=["s1_us_mean", "offset_num", "t_index"])
    LOGGER.info("Loaded %s rows from %s", len(df), path)
    return df


def compute_monthly_slopes(df: pd.DataFrame, min_points: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for month, group in df.groupby("month"):
        group = group.sort_values("offset_num")
        if len(group) < min_points:
            continue
        x = group["offset_num"].to_numpy(dtype=float)
        y = group["s1_us_mean"].to_numpy(dtype=float)
        if np.isnan(y).all():
            continue
        # Linear regression y = a + b * x
        Xmat = np.column_stack([np.ones_like(x), x])
        beta_hat, _, _, _ = np.linalg.lstsq(Xmat, y, rcond=None)
        yhat = Xmat @ beta_hat
        resid = y - yhat
        ss_tot = float(((y - y.mean()) ** 2).sum())
        ss_res = float((resid ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rows.append(
            {
                "month": month.to_timestamp(),
                "t_index": float(group["t_index"].iloc[0]),
                "slope_per_day": float(beta_hat[1]),
                "intercept": float(beta_hat[0]),
                "r2": r2,
                "n_points": len(group),
                "available_offsets": ",".join(group["offset"].tolist()),
            }
        )
    result = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    if result.empty:
        raise RuntimeError("No monthly slopes computed; check min-points or input data.")
    LOGGER.info("Computed slopes for %s months", len(result))
    return result


def ols_with_time(y: np.ndarray, t: np.ndarray) -> OLSResult:
    n = len(y)
    if n < 2:
        raise ValueError("Need at least two observations for regression.")
    X = np.column_stack([np.ones_like(t), t])
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    yhat = X @ beta
    resid = y - yhat
    dof = max(n - 2, 1)
    s2 = float((resid @ resid) / dof)
    cov = s2 * XtX_inv
    se_beta = float(np.sqrt(cov[1, 1]))
    t_beta = float(beta[1] / se_beta) if se_beta > 0 else float("nan")
    p_norm = float(2 * (1 - 0.5 * (1 + math.erf(abs(t_beta) / math.sqrt(2))))) if not math.isnan(t_beta) else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return OLSResult(
        alpha=float(beta[0]),
        beta=float(beta[1]),
        se_beta=se_beta,
        t_beta=t_beta,
        p_norm=p_norm,
        r2=r2,
        n=n,
    )


def save_regression_summary(result: OLSResult, out_csv: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "alpha": result.alpha,
                "beta_per_month": result.beta,
                "beta_per_year": result.beta * 12,
                "se_beta": result.se_beta,
                "t_beta": result.t_beta,
                "p_norm": result.p_norm,
                "r2": result.r2,
                "n": result.n,
            }
        ]
    )
    df.to_csv(out_csv, index=False)
    LOGGER.info("Wrote regression summary to %s", out_csv)


def plot_slopes(slopes: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(slopes["month"], slopes["slope_per_day"], color="#1f77b4", linewidth=1.2, label="Monthly slope")
    if len(slopes) >= 6:
        rolling = slopes["slope_per_day"].rolling(6, center=True).mean()
        ax.plot(slopes["month"], rolling, color="#ff7f0e", linewidth=2.0, linestyle="--", label="6-mo avg")
    ax.axhline(0.0, color="#444444", linewidth=1, linestyle=":")
    ax.set_title("S1 slope across last five trading days (per day change)")
    ax.set_ylabel("Slope (ΔS1 per trading day)")
    ax.set_xlabel("Month")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved slope plot to %s", out_png)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    df = load_eom_long(args.input)
    slopes = compute_monthly_slopes(df, min_points=args.min_points)

    args.outdir.mkdir(parents=True, exist_ok=True)
    slopes_csv = args.outdir / "intra_month_slopes.csv"
    slopes.to_csv(slopes_csv, index=False)
    LOGGER.info("Wrote monthly slopes to %s", slopes_csv)

    reg = ols_with_time(slopes["slope_per_day"].to_numpy(), slopes["t_index"].to_numpy())
    save_regression_summary(reg, args.outdir / "intra_month_regression.csv")
    plot_slopes(slopes, args.outdir / "intra_month_slopes.png")

    print("Intra-month slope regression:")
    print(f"  alpha          = {reg.alpha:.6f}")
    print(f"  beta_per_month = {reg.beta:.6f} (per calendar month)")
    print(f"  beta_per_year  = {reg.beta * 12:.6f} (per year)")
    print(f"  t_beta         = {reg.t_beta:.3f}  (p_norm={reg.p_norm:.4f})")
    print(f"  R^2            = {reg.r2:.4f}  (n={reg.n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
