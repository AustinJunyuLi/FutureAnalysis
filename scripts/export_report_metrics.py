#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, **kwargs)


def _session_metrics(hourly_analysis_dir: Path) -> List[Dict[str, float]]:
    session_path = hourly_analysis_dir / "session_event_summary.csv"
    if not session_path.exists():
        return []
    df = _load_csv(session_path)
    return df.to_dict(orient="records")


def _spread_metrics(hourly_analysis_dir: Path) -> List[Dict[str, float]]:
    spread_path = hourly_analysis_dir / "spread_signal_comparison.csv"
    df = _load_csv(spread_path)
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def _timing_metrics(hourly_analysis_dir: Path) -> Dict[str, float]:
    timing_path = hourly_analysis_dir / "spread_event_timing.csv"
    df = _load_csv(timing_path)
    s1 = df[df["spread"] == "S1"]
    if s1.empty:
        return {"median_days": None, "iqr_days": [None, None]}
    median = float(s1["days_to_expiry"].median())
    q1 = float(s1["days_to_expiry"].quantile(0.25))
    q3 = float(s1["days_to_expiry"].quantile(0.75))
    return {"median_days": median, "iqr_days": [q1, q3]}


def _strip_class_metrics(hourly_analysis_dir: Path) -> List[Dict[str, float]]:
    diag_path = hourly_analysis_dir / "strip_spread_diagnostics.csv"
    if not diag_path.exists():
        return []
    df = _load_csv(diag_path)
    counts = df["classification"].value_counts().sort_index()
    total = counts.sum()
    metrics = []
    for cls, count in counts.items():
        metrics.append({
            "classification": cls,
            "days": int(count),
            "share_pct": float(count / total * 100.0) if total else 0.0,
        })
    return metrics


def _hourly_event_count(hourly_analysis_dir: Path) -> int:
    summary = _load_csv(hourly_analysis_dir / "hourly_widening_summary.csv")
    return len(summary)


def _daily_event_metrics(daily_analysis_dir: Path, daily_signals_dir: Path) -> Dict[str, float]:
    summary = _load_csv(daily_analysis_dir / "daily_widening_summary.csv")
    widening = _load_csv(daily_signals_dir / "hg_widening_filtered.csv")
    liquidity = _load_csv(daily_signals_dir / "hg_liquidity_roll_filtered.csv")
    merged = widening.merge(liquidity, on="timestamp", suffixes=("_spread", "_liq"))
    spread_col = merged.columns[1]
    liq_col = merged.columns[2]
    total_spread = int(merged[spread_col].sum())
    overlap = int((merged[spread_col] & merged[liq_col]).sum())
    overlap_pct = float(overlap / total_spread * 100.0) if total_spread else 0.0
    return {
        "daily_event_count": len(summary),
        "spread_true_count": total_spread,
        "liquidity_true_count": int(merged[liq_col].sum()),
        "overlap_count": overlap,
        "overlap_pct": overlap_pct,
    }


def build_metrics(hourly_dir: Path, daily_dir: Path) -> Dict[str, object]:
    hourly_analysis = hourly_dir / "analysis"
    daily_analysis = daily_dir / "analysis"
    hourly_signals = hourly_dir / "roll_signals"
    daily_signals = daily_dir / "roll_signals"

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "hourly": {
            "event_count": _hourly_event_count(hourly_analysis),
            "sessions": _session_metrics(hourly_analysis),
            "spreads": _spread_metrics(hourly_analysis),
            "timing": _timing_metrics(hourly_analysis),
            "strip_classification": _strip_class_metrics(hourly_analysis),
        },
        "daily": _daily_event_metrics(daily_analysis, daily_signals),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Export report metrics from analysis outputs")
    parser.add_argument("--hourly", required=True, type=Path, help="Path to hourly output directory")
    parser.add_argument("--daily", required=True, type=Path, help="Path to daily output directory")
    parser.add_argument("--out", required=True, type=Path, help="Destination JSON path")
    args = parser.parse_args()

    metrics = build_metrics(args.hourly.resolve(), args.daily.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()
