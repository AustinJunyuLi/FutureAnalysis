#!/usr/bin/env python3
"""
Bucket-level analysis pipeline for futures roll detection.
Implements variable-granularity bucketing with hourly precision during US hours.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from roll_analysis.ingest import find_hg_files, build_buckets_by_contract
from roll_analysis.bucket_panel import (
    assemble_bucket_panel, 
    mark_front_next_buckets,
    compute_bucket_spread,
    calculate_bucket_liquidity_transitions,
    aggregate_buckets_to_daily
)
from roll_analysis.bucket_events import (
    detect_bucket_widening,
    summarize_bucket_events,
    calculate_preference_scores,
    generate_bucket_report
)
from roll_analysis.bucket import calculate_bucket_statistics


def load_settings(settings_path: Path) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(settings_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main bucket analysis pipeline."""
    parser = argparse.ArgumentParser(description="Bucket-Level Futures Roll Analysis")
    project_root = Path(__file__).resolve().parents[1]
    default_data_root = project_root.parent / "organized_data" / "copper"
    
    parser.add_argument("--root", default=str(default_data_root), 
                       help="Root directory for minute files")
    parser.add_argument("--settings", default=str(project_root / 'config' / 'settings.yaml'),
                       help="Path to settings.yaml")
    parser.add_argument("--metadata", default=str(project_root / 'metadata' / 'contracts_metadata.csv'),
                       help="Metadata CSV with expiry dates")
    parser.add_argument("--output_dir", default=str(project_root / 'outputs'),
                       help="Output directory")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation against daily aggregation")
    args = parser.parse_args()
    
    # Load settings
    settings = load_settings(Path(args.settings))
    tz = settings.get('data', {}).get('timezone', 'US/Central')
    price_field = settings.get('data', {}).get('price_field', 'close')
    
    # Bucket-specific settings
    bucket_cfg = settings.get('bucket_config', {})
    if not bucket_cfg.get('enabled', True):
        print("Bucket analysis is disabled in settings.")
        return 0
    
    # Spread detection settings
    spread_cfg = settings.get('spread', {})
    spread_method = spread_cfg.get('method', 'zscore')
    spread_window_buckets = spread_cfg.get('window_buckets', 480)  # 48 hours
    spread_z = spread_cfg.get('z_threshold', 1.5)
    spread_abs_min = spread_cfg.get('abs_min', None)
    spread_cool_down_buckets = spread_cfg.get('cool_down_buckets', 30)  # 3 hours
    
    # Liquidity settings
    roll_cfg = settings.get('roll_rules', {})
    alpha = roll_cfg.get('liquidity_threshold', 0.8)
    
    # Create output directories
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    bucket_panels_dir = out_root / 'bucket_panels'
    bucket_signals_dir = out_root / 'bucket_signals'
    bucket_analysis_dir = out_root / 'bucket_analysis'
    
    for dir_path in [bucket_panels_dir, bucket_signals_dir, bucket_analysis_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BUCKET-LEVEL FUTURES ROLL ANALYSIS")
    print("=" * 60)
    print(f"Data root: {args.root}")
    print(f"Timezone: {tz}")
    print(f"Price field: {price_field}")
    print(f"Window: {spread_window_buckets} buckets (~{spread_window_buckets/10:.1f} trading days)")
    print(f"Z-threshold: {spread_z}")
    print(f"Cool-down: {spread_cool_down_buckets} buckets (~{spread_cool_down_buckets/10:.1f} hours)")
    print()
    
    # 1) Discover files
    print("Step 1: Discovering HG files...")
    files = find_hg_files(args.root)
    if not files:
        print("ERROR: No HG files found under", args.root)
        return 1
    print(f"Found {len(files)} HG contract files")
    
    # 2) Build bucket-level data
    print("\nStep 2: Building bucket-level data...")
    buckets_by_contract = build_buckets_by_contract(files, tz=tz)
    if not buckets_by_contract:
        print("ERROR: No bucket data built.")
        return 1
    print(f"Processed {len(buckets_by_contract)} contracts into buckets")
    
    # 3) Load expiry metadata
    print("\nStep 3: Loading expiry metadata...")
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        print(f"ERROR: Metadata file not found at {meta_path}")
        return 1
    
    meta_df = pd.read_csv(meta_path)
    meta_df['contract'] = meta_df['contract'].astype(str)
    meta_df['expiry_date'] = pd.to_datetime(meta_df['expiry_date'], errors='coerce').dt.normalize()
    
    # Filter to discovered contracts
    contracts = sorted(buckets_by_contract.keys())
    expiry_df = meta_df[meta_df['contract'].isin(contracts)].copy()
    
    # Check for missing expiries
    missing = sorted(set(contracts) - set(expiry_df['contract'].unique()))
    if missing:
        print(f"WARNING: Missing expiry for contracts: {', '.join(missing[:5])}")
        if len(missing) > 5:
            print(f"         ... and {len(missing)-5} more")
    
    # 4) Assemble bucket panel
    print("\nStep 4: Assembling bucket panel...")
    panel = assemble_bucket_panel(buckets_by_contract, expiry_df)
    print(f"Panel shape: {panel.shape}")
    
    # 5) Mark front/next contracts at bucket level
    print("\nStep 5: Identifying front/next contracts per bucket...")
    panel = mark_front_next_buckets(panel)
    
    # 6) Calculate spreads
    print("\nStep 6: Computing calendar spreads...")
    spread = compute_bucket_spread(panel, price_field=price_field)
    valid_spreads = spread.notna().sum()
    print(f"Valid spreads: {valid_spreads:,} / {len(spread):,} ({valid_spreads/len(spread)*100:.1f}%)")
    
    # 7) Detect widening events
    print("\nStep 7: Detecting spread widening events...")
    widening = detect_bucket_widening(
        spread,
        method=spread_method,
        window_buckets=spread_window_buckets,
        z_threshold=spread_z,
        abs_min=spread_abs_min,
        cool_down_buckets=spread_cool_down_buckets
    )
    
    event_count = widening.sum()
    print(f"Widening events detected: {event_count:,} ({event_count/len(widening)*100:.2f}%)")
    
    # 8) Calculate liquidity signals
    print("\nStep 8: Computing liquidity transition signals...")
    liquidity_signal = calculate_bucket_liquidity_transitions(panel, alpha=alpha, price_field=price_field)
    liquidity_count = liquidity_signal.sum()
    print(f"Liquidity signals: {liquidity_count:,} ({liquidity_count/len(liquidity_signal)*100:.2f}%)")
    
    # 9) Generate bucket statistics
    print("\nStep 9: Calculating bucket statistics...")
    
    # Extract volume data for preference score calculation
    volume_series = pd.Series(dtype=float)
    for contract in contracts[:10]:  # Sample from first 10 contracts
        if (contract, 'volume') in panel.columns:
            contract_volume = panel[(contract, 'volume')]
            volume_series = volume_series.combine_first(contract_volume)
    
    # Generate comprehensive report
    bucket_report = generate_bucket_report(widening, spread, volume_series)
    
    # 10) Save outputs
    print("\nStep 10: Saving outputs...")
    
    # Save panel data
    panel_parquet = bucket_panels_dir / 'hg_bucket_panel.parquet'
    try:
        panel.to_parquet(panel_parquet)
        print(f"  [OK] Panel saved to {panel_parquet.name}")
    except Exception as e:
        print(f"  [ERROR] Failed to save parquet: {e}")
    
    # Save simplified CSV (similar to daily version)
    simple_panel = pd.DataFrame(index=panel.index)
    for contract in contracts[:50]:  # Limit for manageable file size
        if (contract, price_field) in panel.columns:
            simple_panel[contract] = panel[(contract, price_field)]
    
    # Add meta columns
    if ('meta', 'front_contract') in panel.columns:
        simple_panel['front_contract'] = panel[('meta', 'front_contract')]
        simple_panel['next_contract'] = panel[('meta', 'next_contract')]
    if ('meta', 'bucket') in panel.columns:
        simple_panel['bucket'] = panel[('meta', 'bucket')]
    
    simple_csv = bucket_panels_dir / 'hg_bucket_panel_simple.csv'
    simple_panel.to_csv(simple_csv)
    print(f"  [OK] Simple panel saved to {simple_csv.name}")
    
    # Save signals
    spread_out = bucket_signals_dir / 'hg_bucket_spread.csv'
    spread.to_csv(spread_out, header=True)
    print(f"  [OK] Spreads saved to {spread_out.name}")
    
    widening_out = bucket_signals_dir / 'hg_bucket_widening.csv'
    widening.to_csv(widening_out, header=True)
    print(f"  [OK] Widening events saved to {widening_out.name}")
    
    liquidity_out = bucket_signals_dir / 'hg_bucket_liquidity.csv'
    liquidity_signal.to_csv(liquidity_out, header=True)
    print(f"  [OK] Liquidity signals saved to {liquidity_out.name}")
    
    # Save bucket summary
    bucket_summary = bucket_report['bucket_summary']
    summary_csv = bucket_analysis_dir / 'bucket_summary.csv'
    bucket_summary.to_csv(summary_csv, index=False)
    print(f"  [OK] Bucket summary saved to {summary_csv.name}")
    
    # Save preference scores if available
    if 'preference_scores' in bucket_report:
        pref_scores = bucket_report['preference_scores']
        pref_csv = bucket_analysis_dir / 'preference_scores.csv'
        pref_scores.to_csv(pref_csv, header=['preference_score'])
        print(f"  [OK] Preference scores saved to {pref_csv.name}")
    
    # Save comprehensive report as JSON
    report_json = bucket_analysis_dir / 'bucket_analysis_report.json'
    # Convert non-serializable objects for JSON
    json_report = {}
    for key, value in bucket_report.items():
        if isinstance(value, pd.DataFrame):
            json_report[key] = value.to_dict('records')
        elif isinstance(value, pd.Series):
            json_report[key] = value.to_dict()
        else:
            json_report[key] = value
    
    with open(report_json, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"  [OK] Analysis report saved to {report_json.name}")
    
    # 11) Generate summary report
    print("\n" + "=" * 60)
    print("BUCKET ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal Events: {bucket_report['total_events']:,}")
    print(f"Event Rate: {bucket_report['event_rate_pct']:.2f}%")
    
    print("\nEvent Distribution by Session:")
    us_vs_offpeak = bucket_report.get('us_vs_offpeak', {})
    print(f"  US Regular Hours: {us_vs_offpeak.get('us_events', 0):,} events ({us_vs_offpeak.get('us_share_pct', 0):.1f}%)")
    print(f"  Off-Peak Hours: {us_vs_offpeak.get('offpeak_events', 0):,} events ({us_vs_offpeak.get('offpeak_share_pct', 0):.1f}%)")
    
    print("\nTop 5 Most Active Buckets:")
    top_buckets = bucket_summary.nlargest(5, 'event_count')[['bucket', 'label', 'event_count', 'event_share_pct']]
    for _, row in top_buckets.iterrows():
        print(f"  Bucket {row['bucket']}: {row['label']} - {row['event_count']} events ({row['event_share_pct']:.1f}%)")
    
    print("\nPeak Activity Hours:")
    peak_buckets = bucket_report.get('peak_buckets', [])
    for bucket_id in peak_buckets[:3]:
        bucket_info = bucket_summary[bucket_summary['bucket'] == bucket_id].iloc[0]
        print(f"  {bucket_info['label']}: {bucket_info['event_count']} events")
    
    # 12) Optional: Validation against daily aggregation
    if args.validate:
        print("\n" + "=" * 60)
        print("VALIDATION: Bucket vs Daily Comparison")
        print("=" * 60)
        
        # Aggregate buckets to daily
        daily_from_buckets = aggregate_buckets_to_daily(panel)
        
        # Compare some key metrics
        print("\nValidation checks:")
        print(f"  Unique dates in bucket data: {len(daily_from_buckets)}")
        print(f"  Total bucket periods: {len(panel)}")
        print(f"  Average buckets per day: {len(panel) / len(daily_from_buckets):.1f}")
        
        # Volume conservation check
        if 'volume' in simple_panel.columns:
            daily_volume = daily_from_buckets.iloc[:, daily_from_buckets.columns.get_level_values(1) == 'volume'].sum(axis=1)
            print(f"  Total volume preserved: {daily_volume.sum():,.0f}")
    
    print("\n" + "=" * 60)
    print("Bucket analysis complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
