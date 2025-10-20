#!/usr/bin/env python3
"""
Unified CLI for Futures Roll Analysis

Usage examples:
  Hourly (bucket) analysis:
    roll-analyze --granularity hourly \
      --root ../organized_data/copper \
      --settings etf_roll_analysis/config/settings.yaml \
      --metadata etf_roll_analysis/metadata/contracts_metadata.csv \
      --output_dir etf_roll_analysis/outputs

  Daily analysis:
    roll-analyze --granularity daily \
      --root ../organized_data/copper \
      --settings etf_roll_analysis/config/settings.yaml \
      --metadata etf_roll_analysis/metadata/contracts_metadata.csv \
      --output_dir etf_roll_analysis/outputs
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def _load_settings(settings_path: Path) -> dict:
    import yaml
    with open(settings_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Unified Futures Roll Analysis CLI")
    project_root = Path(__file__).resolve().parents[1]

    parser.add_argument("--granularity", choices=["daily", "hourly"], required=True,
                        help="Analysis granularity: daily or hourly (bucket)")
    parser.add_argument("--root", default=str(project_root.parent / "organized_data" / "copper"),
                        help="Root directory of minute files for the product (e.g., organized_data/copper)")
    parser.add_argument("--settings", default=str(project_root / 'config' / 'settings.yaml'),
                        help="Path to settings.yaml")
    parser.add_argument("--metadata", default=str(project_root / 'metadata' / 'contracts_metadata.csv'),
                        help="CSV with explicit expiry dates")
    parser.add_argument("--output_dir", default=str(project_root / 'outputs'),
                        help="Output directory root")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Optional: process only the first N files")
    parser.add_argument("--quality_report", action="store_true",
                        help="If set, write data quality CSV/JSON reports (daily only)")
    args = parser.parse_args()

    # Load settings
    settings = _load_settings(Path(args.settings))
    tz = settings.get('data', {}).get('timezone', 'US/Central')
    price_field = settings.get('data', {}).get('price_field', 'close')
    out_root = Path(args.output_dir)
    (out_root / 'panels').mkdir(parents=True, exist_ok=True)
    (out_root / 'roll_signals').mkdir(parents=True, exist_ok=True)
    (out_root / 'analysis').mkdir(parents=True, exist_ok=True)

    # Import library modules from src
    sys.path.append(str(project_root / 'src'))

    if args.granularity == "hourly":
        # Hourly (bucket) pipeline
        from roll_analysis.ingest import find_hg_files, build_buckets_by_contract
        from roll_analysis.bucket_panel import (
            assemble_bucket_panel,
            mark_front_next_buckets,
            compute_bucket_spread,
        )
        from roll_analysis.bucket_events import (
            detect_bucket_widening,
            generate_bucket_report,
        )

        files = find_hg_files(args.root)
        if args.max_files is not None and args.max_files > 0:
            files = files[:args.max_files]
            print(f"Limiting to first {len(files)} files for this run")
        if not files:
            print("No HG files found under", args.root)
            return 0

        buckets_by_contract = build_buckets_by_contract(files, tz=tz)
        if not buckets_by_contract:
            print("No bucket data built.")
            return 0

        meta_path = Path(args.metadata)
        if not meta_path.exists():
            print("ERROR: Metadata CSV not found:", meta_path)
            return 1
        meta_df = pd.read_csv(meta_path)
        if 'contract' not in meta_df.columns or 'expiry_date' not in meta_df.columns:
            print("ERROR: Metadata CSV must include columns: contract, expiry_date")
            return 1
        contracts = sorted(buckets_by_contract.keys())
        meta_df['contract'] = meta_df['contract'].astype(str)
        expiry_df = meta_df.loc[meta_df['contract'].isin(contracts)].copy()

        missing = sorted(set(contracts) - set(expiry_df['contract'].unique()))
        if missing:
            print("ERROR: Missing explicit expiry for contracts:", ", ".join(missing))
            return 1
        expiry_df['expiry_date'] = pd.to_datetime(expiry_df['expiry_date'], errors='coerce').dt.normalize()
        if expiry_df['expiry_date'].isna().any():
            bad = expiry_df.loc[expiry_df['expiry_date'].isna(), 'contract'].tolist()
            print("ERROR: Some expiry_date values could not be parsed for:", ", ".join(bad))
            return 1

        panel = assemble_bucket_panel(buckets_by_contract, expiry_df)
        panel = mark_front_next_buckets(panel)

        spread = compute_bucket_spread(panel, price_field=price_field)

        spread_cfg = settings.get('spread', {})
        widening = detect_bucket_widening(
            spread,
            method=spread_cfg.get('method', 'zscore'),
            window_buckets=spread_cfg.get('window_buckets', 20),
            z_threshold=spread_cfg.get('z_threshold', 1.5),
            abs_min=spread_cfg.get('abs_min', None),
            cool_down_hours=spread_cfg.get('cool_down_hours', 3.0),
        )

        # Save outputs
        panel_out = out_root / 'panels' / 'hourly_panel.parquet'
        try:
            panel.to_parquet(panel_out)
        except Exception as e:
            print("Warning: failed to write parquet. Error:", e)
            panel.to_csv(out_root / 'panels' / 'hourly_panel.csv')

        spread.to_csv(out_root / 'roll_signals' / 'hourly_spread.csv', header=True)
        widening.to_csv(out_root / 'roll_signals' / 'hourly_widening.csv', header=True)

        # Optional analytics
        volume_series = None
        try:
            f = panel[('meta','front_contract')].astype('string')
            vals = []
            for idx in panel.index:
                fc = f.loc[idx]
                try:
                    v = panel.at[idx, (fc, 'volume')]
                except Exception:
                    v = float('nan')
                vals.append(v)
            volume_series = pd.Series(vals, index=panel.index, name='volume')
        except Exception:
            pass

        report = generate_bucket_report(widening, spread, volume=volume_series)
        if isinstance(report.get('bucket_summary'), pd.DataFrame):
            report['bucket_summary'].to_csv(out_root / 'analysis' / 'bucket_summary.csv', index=False)
        if isinstance(report.get('preference_scores'), pd.Series):
            report['preference_scores'].to_csv(out_root / 'analysis' / 'preference_scores.csv', header=True)
        if isinstance(report.get('transition_matrix'), pd.DataFrame):
            report['transition_matrix'].to_csv(out_root / 'analysis' / 'transition_matrix.csv')

        print("Bucket panel rows:", len(panel))
        print("Valid spreads:", spread.notna().sum(), "/", len(spread))
        print("Widening events detected:", int(widening.sum()))
        return 0

    # Daily pipeline
    from roll_analysis.ingest import find_hg_files, build_daily_by_contract
    from roll_analysis.panel import assemble_panel
    from roll_analysis.rolls import mark_front_next, liquidity_roll_signal
    from roll_analysis.spread import compute_spread
    from roll_analysis.events import detect_widening

    files = find_hg_files(args.root)
    if args.max_files is not None and args.max_files > 0:
        files = files[:args.max_files]
        print(f"Limiting to first {len(files)} files for this run")
    if not files:
        print("No HG files found under", args.root)
        return 0

    daily_by_contract = build_daily_by_contract(files, tz=tz)
    if not daily_by_contract:
        print("No daily data built.")
        return 0

    # Data quality (optional via settings)
    data_quality_config = settings.get('data_quality', {})
    if data_quality_config.get('filter_enabled', False):
        print("\nApplying data quality filtering...")
        from roll_analysis.data_quality import DataQualityFilter
        filt = DataQualityFilter(data_quality_config)
        daily_by_contract, quality_metrics = filt.apply(daily_by_contract)
        if args.quality_report:
            filt.save_exclusion_report(quality_metrics, out_root)
        if not daily_by_contract:
            print("ERROR: No contracts passed quality filtering!")
            return 1
        print(f"Continuing with {len(daily_by_contract)} quality-filtered contracts")

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        print("ERROR: Explicit expiry metadata CSV not found at", meta_path)
        return 1
    meta_df = pd.read_csv(meta_path)
    if 'contract' not in meta_df.columns or 'expiry_date' not in meta_df.columns:
        print("ERROR: Metadata CSV must include columns: contract, expiry_date")
        return 1

    contracts = sorted(daily_by_contract.keys())
    meta_df['contract'] = meta_df['contract'].astype(str)
    expiry_df = meta_df.loc[meta_df['contract'].isin(contracts)].copy()

    missing = sorted(set(contracts) - set(expiry_df['contract'].unique()))
    if missing:
        print("ERROR: Missing explicit expiry for contracts:", ", ".join(missing))
        return 1

    expiry_df['expiry_date'] = pd.to_datetime(expiry_df['expiry_date'], errors='coerce').dt.normalize()
    if expiry_df['expiry_date'].isna().any():
        bad = expiry_df.loc[expiry_df['expiry_date'].isna(), 'contract'].tolist()
        print("ERROR: Some expiry_date values could not be parsed for:", ", ".join(bad))
        return 1

    panel = assemble_panel(daily_by_contract, expiry_df)
    panel = mark_front_next(panel)

    liq_cfg = settings.get('roll_rules', {})
    liq_signal = liquidity_roll_signal(
        panel,
        alpha=liq_cfg.get('liquidity_threshold', 0.8),
        confirm_days=liq_cfg.get('confirm_days', 1),
        price_field=price_field,
    )

    spread_cfg = settings.get('spread', {})
    spread = compute_spread(panel, price_field=price_field)
    widening = detect_widening(
        spread,
        method=spread_cfg.get('method', 'zscore'),
        window=spread_cfg.get('window', 30),
        z_threshold=spread_cfg.get('z_threshold', 1.5),
        abs_min=spread_cfg.get('abs_min', None),
        cool_down=spread_cfg.get('cool_down', 0),
    )

    print(f"\nSpread analysis:")
    print(f"  Valid spreads: {spread.notna().sum():,} / {len(spread):,} ({spread.notna().sum()/len(spread)*100:.1f}%)")
    print(f"  Widening events detected: {widening.sum():,} ({widening.sum()/len(widening)*100:.2f}%)")

    # Validate lengths
    assert len(liq_signal) == len(panel.index)
    assert len(spread) == len(panel.index)
    assert len(widening) == len(panel.index)

    # Save outputs (keep naming consistent with prior scripts)
    suffix = '_filtered' if data_quality_config.get('filter_enabled', False) else ''
    try:
        panel.to_parquet(out_root / 'panels' / f'hg_panel{suffix}.parquet')
    except Exception as e:
        print("Warning: failed to write parquet. Error:", e)

    # Simple CSV with one column per contract and meta row of expiries
    contracts = [c for c in panel.columns.get_level_values(0).unique().tolist() if c != 'meta']
    simple_panel = pd.DataFrame(index=panel.index)
    expiry_row = {}
    for contract in contracts:
        try:
            simple_panel[contract] = panel[(contract, price_field)]
            exp = panel[(contract, 'expiry')].dropna()
            expiry_row[contract] = pd.to_datetime(exp.iloc[0]).strftime('%Y-%m-%d') if not exp.empty else ''
        except Exception:
            continue
    simple_panel['front_contract'] = panel[('meta', 'front_contract')]
    simple_panel['next_contract'] = panel[('meta', 'next_contract')]
    expiry_df_row = pd.DataFrame([expiry_row], columns=simple_panel.columns)
    expiry_df_row.index = ['expiry_date']
    expiry_df_row['front_contract'] = ''
    expiry_df_row['next_contract'] = ''
    simple_csv = pd.concat([expiry_df_row, simple_panel])
    simple_csv.to_csv(out_root / 'panels' / f'hg_panel_simple{suffix}.csv')

    panel.to_csv(out_root / 'panels' / f'hg_panel_full{suffix}.csv')
    expiry_df.to_csv(out_root / 'panels' / f'hg_expiry_table{suffix}.csv', index=False)

    liq_signal.to_csv(out_root / 'roll_signals' / f'hg_liquidity_roll{suffix}.csv', header=True)
    spread.to_csv(out_root / 'roll_signals' / f'hg_spread{suffix}.csv', header=True)
    widening.to_csv(out_root / 'roll_signals' / f'hg_widening{suffix}.csv', header=True)

    print("Daily analysis complete. Outputs written to:")
    print("  panels/ (hg_panel*.parquet, hg_panel_simple*.csv, hg_panel_full*.csv, hg_expiry_table*.csv)")
    print("  roll_signals/ (hg_liquidity_roll*.csv, hg_spread*.csv, hg_widening*.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
