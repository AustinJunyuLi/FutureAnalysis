#!/usr/bin/env python3
"""
Hourly (bucket) Futures Roll Analysis Pipeline

Runs the bucket-only pipeline: ingest minute files → bucket aggregation → panel assembly
→ front/next detection → bucket spreads → bucket widening events → outputs.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Local imports
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from roll_analysis.ingest import find_hg_files, build_buckets_by_contract
from roll_analysis.bucket_panel import (
    assemble_bucket_panel,
    mark_front_next_buckets,
    compute_bucket_spread,
    calculate_bucket_liquidity_transitions,
)
from roll_analysis.bucket_events import detect_bucket_widening, generate_bucket_report


def load_settings(settings_path: Path) -> dict:
    import yaml
    with open(settings_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="HG Futures Roll Analysis (Hourly Buckets)")
    project_root = Path(__file__).resolve().parents[1]
    default_data_root = project_root.parent / "organized_data" / "copper"

    parser.add_argument("--root", default=str(default_data_root), help="Root directory to search for HG minute files")
    parser.add_argument("--settings", default=str(project_root / 'config' / 'settings.yaml'), help="Path to settings.yaml")
    parser.add_argument("--metadata", default=str(project_root / 'metadata' / 'contracts_metadata.csv'), help="CSV with explicit expiry dates")
    parser.add_argument("--output_dir", default=str(project_root / 'outputs'), help="Output directory")
    parser.add_argument("--max_files", type=int, default=None, help="Optional: process only the first N files for quick runs")
    args = parser.parse_args()

    settings = load_settings(Path(args.settings))
    tz = settings.get('data', {}).get('timezone', 'US/Central')
    price_field = settings.get('data', {}).get('price_field', 'close')

    spread_cfg = settings.get('spread', {})
    spread_method = spread_cfg.get('method', 'zscore')
    window_buckets = spread_cfg.get('window_buckets', 20)
    spread_z = spread_cfg.get('z_threshold', 1.5)
    spread_abs_min = spread_cfg.get('abs_min', None)
    cool_down_hours = spread_cfg.get('cool_down_buckets', None)
    # Prefer explicit hours if provided
    cool_down_hours = spread_cfg.get('cool_down_hours', 3.0)

    out_root = Path(args.output_dir)
    (out_root / 'panels').mkdir(parents=True, exist_ok=True)
    (out_root / 'roll_signals').mkdir(parents=True, exist_ok=True)
    (out_root / 'analysis').mkdir(parents=True, exist_ok=True)

    # 1) Discover HG files
    files = find_hg_files(args.root)
    if args.max_files is not None and args.max_files > 0:
        files = files[:args.max_files]
        print(f"Limiting to first {len(files)} files for this run")
    if not files:
        print("No HG files found under", args.root)
        return 0

    # 2) Build bucket aggregates per contract
    buckets_by_contract = build_buckets_by_contract(files, tz=tz)
    if not buckets_by_contract:
        print("No bucket data built.")
        return 0

    # 3) Load explicit expiry metadata (required)
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        print("ERROR: Metadata CSV not found:", meta_path)
        return 1
    meta_df = pd.read_csv(meta_path)
    if 'contract' not in meta_df.columns or 'expiry_date' not in meta_df.columns:
        print("ERROR: Metadata CSV must include columns: contract, expiry_date")
        return 1

    # Filter to discovered contracts
    contracts = sorted(buckets_by_contract.keys())
    meta_df['contract'] = meta_df['contract'].astype(str)
    expiry_df = meta_df.loc[meta_df['contract'].isin(contracts)].copy()

    # Validate all required contracts have explicit expiry
    missing = sorted(set(contracts) - set(expiry_df['contract'].unique()))
    if missing:
        print("ERROR: Missing explicit expiry for contracts:", ", ".join(missing))
        return 1

    # Coerce expiry_date to datetime
    expiry_df['expiry_date'] = pd.to_datetime(expiry_df['expiry_date'], errors='coerce').dt.normalize()
    if expiry_df['expiry_date'].isna().any():
        bad = expiry_df.loc[expiry_df['expiry_date'].isna(), 'contract'].tolist()
        print("ERROR: Some expiry_date values could not be parsed for:", ", ".join(bad))
        return 1

    # 4) Assemble bucket panel and mark front/next
    panel = assemble_bucket_panel(buckets_by_contract, expiry_df)
    panel = mark_front_next_buckets(panel)

    # 5) Compute spread and widening
    spread = compute_bucket_spread(panel, price_field=price_field)
    widening = detect_bucket_widening(
        spread,
        method=spread_method,
        window_buckets=window_buckets,
        z_threshold=spread_z,
        abs_min=spread_abs_min,
        cool_down_hours=cool_down_hours,
    )

    # 6) Save outputs
    panel_out = out_root / 'panels' / 'hourly_panel.parquet'
    try:
        panel.to_parquet(panel_out)
    except Exception as e:
        print("Warning: failed to write parquet (install pyarrow or fastparquet). Error:", e)
        panel.to_csv(out_root / 'panels' / 'hourly_panel.csv')

    spread_out = out_root / 'roll_signals' / 'hourly_spread.csv'
    spread.to_csv(spread_out, header=True)

    widen_out = out_root / 'roll_signals' / 'hourly_widening.csv'
    widening.to_csv(widen_out, header=True)

    # 7) Analysis reports
    # Build a volume series for preference scores if possible (use front contract volume as proxy)
    volume_series = None
    try:
        f = panel[('meta','front_contract')].astype('string')
        vol_vals = []
        for idx in panel.index:
            fc = f.loc[idx]
            try:
                v = panel.at[idx, (fc, 'volume')]
            except Exception:
                v = float('nan')
            vol_vals.append(v)
        volume_series = pd.Series(vol_vals, index=panel.index, name='volume')
    except Exception:
        pass

    report = generate_bucket_report(widening, spread, volume=volume_series)
    # Save bucket summary
    bucket_summary = report.get('bucket_summary')
    if isinstance(bucket_summary, pd.DataFrame):
        bucket_summary.to_csv(out_root / 'analysis' / 'bucket_summary.csv', index=False)
    # Save preference scores
    prefs = report.get('preference_scores')
    if prefs is not None:
        prefs.to_csv(out_root / 'analysis' / 'preference_scores.csv', header=True)
    # Save transition matrix
    trans = report.get('transition_matrix')
    if isinstance(trans, pd.DataFrame):
        trans.to_csv(out_root / 'analysis' / 'transition_matrix.csv')

    # Summary to stdout
    print("Bucket panel rows:", len(panel))
    print("Valid spreads:", spread.notna().sum(), "/", len(spread))
    print("Widening events detected:", int(widening.sum()))
    print(f"Parameters: method={spread_method}, window_buckets={window_buckets}, z_threshold={spread_z}, cool_down_hours={cool_down_hours}")
    print("Outputs:", panel_out, spread_out, widen_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
