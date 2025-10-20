#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd

# Local imports
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from roll_analysis.ingest import find_hg_files, build_daily_by_contract
from roll_analysis.panel import assemble_panel
from roll_analysis.rolls import mark_front_next, liquidity_roll_signal, compute_fixed_roll_dates
from roll_analysis.spread import compute_spread
from roll_analysis.events import detect_widening


def load_settings(settings_path: Path) -> dict:
    import yaml
    with open(settings_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="HG Futures Roll Analysis")
    project_root = Path(__file__).resolve().parents[1]
    default_data_root = project_root.parent / "organized_data" / "copper"
    
    parser.add_argument("--root", default=str(default_data_root), help="Root directory to search for HG minute files")
    parser.add_argument("--settings", default=str(project_root / 'config' / 'settings.yaml'), help="Path to settings.yaml")
    parser.add_argument("--metadata", default=str(project_root / 'metadata' / 'contracts_metadata.csv'), help="Optional local metadata CSV with expiry dates")
    parser.add_argument("--output_dir", default=str(project_root / 'outputs'), help="Output directory")
    args = parser.parse_args()

    settings = load_settings(Path(args.settings))
    tz = settings.get('data', {}).get('timezone', 'US/Central')
    price_field = settings.get('data', {}).get('price_field', 'close')

    fixed_N = settings.get('roll_rules', {}).get('fixed_days_before_expiry', 5)
    alpha = settings.get('roll_rules', {}).get('liquidity_threshold', 0.8)
    confirm_days = settings.get('roll_rules', {}).get('confirm_days', 1)

    spread_cfg = settings.get('spread', {})
    spread_method = spread_cfg.get('method', 'zscore')
    spread_window = spread_cfg.get('window', 30)
    spread_z = spread_cfg.get('z_threshold', 1.5)
    spread_abs_min = spread_cfg.get('abs_min', None)
    spread_cool_down = spread_cfg.get('cool_down', 0)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'panels').mkdir(parents=True, exist_ok=True)
    (out_root / 'roll_signals').mkdir(parents=True, exist_ok=True)

    # 1) Discover HG files
    files = find_hg_files(args.root)
    if not files:
        print("No HG files found under", args.root)
        return 0

    # 2) Build daily per contract
    daily_by_contract = build_daily_by_contract(files, tz=tz)
    if not daily_by_contract:
        print("No daily data built.")
        return 0
    
    # 2.5) Apply data quality filtering if enabled
    data_quality_config = settings.get('data_quality', {})
    if data_quality_config.get('filter_enabled', False):
        print("\nApplying data quality filtering...")
        from roll_analysis.data_quality import DataQualityFilter
        
        filter = DataQualityFilter(data_quality_config)
        daily_by_contract, quality_metrics = filter.apply(daily_by_contract)
        
        # Save quality reports
        filter.save_exclusion_report(quality_metrics, out_root)
        
        if not daily_by_contract:
            print("ERROR: No contracts passed quality filtering!")
            return 1
        
        print(f"Continuing with {len(daily_by_contract)} quality-filtered contracts")

    # 3) Load explicit expiry metadata (required). We do NOT derive expiries from data.
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        print("ERROR: Explicit expiry metadata CSV not found at", meta_path)
        print("Please create it with columns: root, contract, expiry_date, source, source_url")
        print("and populate official expiry dates and sources (e.g., CME Copper Product Calendar).")
        return 1

    meta_df = pd.read_csv(meta_path)
    if 'contract' not in meta_df.columns or 'expiry_date' not in meta_df.columns:
        print("ERROR: Metadata CSV must include columns: root, contract, expiry_date, source, source_url")
        return 1

    # Filter to discovered contracts
    contracts = sorted(daily_by_contract.keys())
    meta_df['contract'] = meta_df['contract'].astype(str)
    expiry_df = meta_df.loc[meta_df['contract'].isin(contracts)].copy()

    # Validate all required contracts have explicit expiry
    missing = sorted(set(contracts) - set(expiry_df['contract'].unique()))
    if missing:
        print("ERROR: Missing explicit expiry for contracts:", ", ".join(missing))
        print("Please add rows to", meta_path, "with official expiry_date and source/source_url for each missing contract.")
        return 1

    # Coerce expiry_date to datetime
    expiry_df['expiry_date'] = pd.to_datetime(expiry_df['expiry_date'], errors='coerce').dt.normalize()
    if expiry_df['expiry_date'].isna().any():
        bad = expiry_df.loc[expiry_df['expiry_date'].isna(), 'contract'].tolist()
        print("ERROR: Some expiry_date values could not be parsed for:", ", ".join(bad))
        return 1

    # 4) Assemble panel and enrich with front/next
    panel = assemble_panel(daily_by_contract, expiry_df)
    panel = mark_front_next(panel)

    # 5) Compute roll signals
    liq_signal = liquidity_roll_signal(panel, alpha=alpha, confirm_days=confirm_days, price_field=price_field)

    # 6) Compute spread and widening
    spread = compute_spread(panel, price_field=price_field)
    widening = detect_widening(spread, method=spread_method, window=spread_window, 
                               z_threshold=spread_z, abs_min=spread_abs_min, 
                               cool_down=spread_cool_down)
    
    # Print summary statistics
    print(f"\nSpread analysis:")
    print(f"  Valid spreads: {spread.notna().sum():,} / {len(spread):,} ({spread.notna().sum()/len(spread)*100:.1f}%)")
    print(f"  Widening events detected: {widening.sum():,} ({widening.sum()/len(widening)*100:.2f}%)")
    print(f"  Parameters: method={spread_method}, window={spread_window}, z_threshold={spread_z}")
    
    # Validate data consistency before saving
    panel_dates = panel.index
    assert len(liq_signal) == len(panel_dates), f"Liquidity signal length mismatch: {len(liq_signal)} vs {len(panel_dates)}"
    assert len(spread) == len(panel_dates), f"Spread length mismatch: {len(spread)} vs {len(panel_dates)}"
    assert len(widening) == len(panel_dates), f"Widening length mismatch: {len(widening)} vs {len(panel_dates)}"
    print(f"\n[OK] Data validation passed: all outputs have {len(panel_dates):,} rows")

    # 7) Save outputs
    # Add suffix for filtered data
    suffix = '_filtered' if data_quality_config.get('filter_enabled', False) else ''
    panel_out = out_root / 'panels' / f'hg_panel{suffix}.parquet'
    try:
        panel.to_parquet(panel_out)
    except Exception as e:
        print("Warning: failed to write parquet (install pyarrow or fastparquet). Error:", e)

    # Create simplified CSV: one column per contract with close prices
    contracts = panel.columns.get_level_values(0).unique().tolist()
    contracts = [c for c in contracts if c != 'meta']
    
    simple_panel = pd.DataFrame(index=panel.index)
    expiry_row = {}
    
    for contract in contracts:
        # Get close price for each contract
        try:
            simple_panel[contract] = panel[(contract, price_field)]
            # Get expiry date
            exp = panel[(contract, 'expiry')].dropna()
            if not exp.empty:
                expiry_row[contract] = pd.to_datetime(exp.iloc[0]).strftime('%Y-%m-%d')
            else:
                expiry_row[contract] = ''
        except Exception:
            continue
    
    # Add meta columns
    simple_panel['front_contract'] = panel[('meta', 'front_contract')]
    simple_panel['next_contract'] = panel[('meta', 'next_contract')]
    
    # Create expiry row DataFrame
    expiry_df_row = pd.DataFrame([expiry_row], columns=simple_panel.columns)
    expiry_df_row.index = ['expiry_date']
    expiry_df_row['front_contract'] = ''
    expiry_df_row['next_contract'] = ''
    
    # Concatenate expiry row with data
    simple_csv = pd.concat([expiry_df_row, simple_panel])
    
    # Save simplified CSV
    simple_csv_out = out_root / 'panels' / f'hg_panel_simple{suffix}.csv'
    simple_csv.to_csv(simple_csv_out)
    
    # Create source documentation markdown
    source_md_out = out_root / 'panels' / 'hg_expiry_sources.md'
    with open(source_md_out, 'w') as f:
        f.write('# HG Copper Futures Expiry Date Sources\n\n')
        f.write('## Methodology\n\n')
        f.write('All expiry dates are calculated using the CME rule:\n')
        f.write('**"Third last business day of the contract month"**\n\n')
        f.write('## Primary Source\n\n')
        f.write('- **Source**: CME Group Copper Futures Calendar\n')
        f.write('- **URL**: https://www.cmegroup.com/markets/metals/base/copper.calendar.html\n')
        f.write('- **Contract Specifications**: https://www.cmegroup.com/markets/metals/base/copper.contractSpecs.html\n\n')
        f.write('## Contract Expiry Dates\n\n')
        f.write('| Contract | Expiry Date | Year | Month |\n')
        f.write('|----------|-------------|------|-------|\n')
        
        for _, row in expiry_df.iterrows():
            contract = row['contract']
            exp_date = pd.to_datetime(row['expiry_date']).strftime('%Y-%m-%d')
            year = contract[-4:]
            month_code = contract[-5]
            month_map = {'F':'Jan','G':'Feb','H':'Mar','J':'Apr','K':'May','M':'Jun',
                        'N':'Jul','Q':'Aug','U':'Sep','V':'Oct','X':'Nov','Z':'Dec'}
            month_name = month_map.get(month_code, month_code)
            f.write(f'| {contract} | {exp_date} | {year} | {month_name} |\n')
        
        f.write('\n## Notes\n\n')
        f.write('- Expiry dates do not account for exchange holidays beyond weekend exclusions\n')
        f.write('- For exact trading halt times, refer to CME official documentation\n')
        f.write('- Physical delivery occurs after the expiry date per contract specifications\n')

    # Also save the full panel as CSV with MultiIndex (for reference)
    panel_csv_out = out_root / 'panels' / f'hg_panel_full{suffix}.csv'
    panel.to_csv(panel_csv_out)

    expiry_out = out_root / 'panels' / f'hg_expiry_table{suffix}.csv'
    expiry_df.to_csv(expiry_out, index=False)

    signals_out = out_root / 'roll_signals' / f'hg_liquidity_roll{suffix}.csv'
    liq_signal.to_csv(signals_out, header=True)

    spread_out = out_root / 'roll_signals' / f'hg_spread{suffix}.csv'
    spread.to_csv(spread_out, header=True)

    widen_out = out_root / 'roll_signals' / f'hg_widening{suffix}.csv'
    widening.to_csv(widen_out, header=True)

    # 8) Generate summary report (optional)
    try:
        # Attempt local import first
        from generate_summary_report import generate_report
    except Exception:
        # Fallback: add scripts directory to path
        sys.path.append(str(Path(__file__).resolve().parent))
        try:
            from generate_summary_report import generate_report
        except Exception:
            generate_report = None

    if 'generate_report' in locals() and callable(generate_report):
        try:
            generate_report(out_root)
        except Exception as e:
            print("Warning: failed to generate summary report:", e)

    # Summary to stdout
    print("Contracts loaded:", len(daily_by_contract))
    print("Expiry sources saved:", source_md_out)
    print("Panel saved:", panel_out)
    print("Simple CSV saved:", simple_csv_out, "(one column per contract with expiry dates in row 2)")
    print("Full CSV saved:", panel_csv_out, "(complete data with all fields)")
    print("Signals saved:", signals_out, spread_out, widen_out)
    print("Note: All expiry dates sourced from CME. See hg_expiry_sources.md for details.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
