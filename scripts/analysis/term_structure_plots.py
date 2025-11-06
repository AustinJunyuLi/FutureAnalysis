#!/usr/bin/env python3
"""
Term Structure Visualization

Analyzes how the futures curve (F1-F12) evolves as the front contract approaches expiry.
Extracts representative snapshots at different phases of the expiry cycle and visualizes
how the term structure reshapes.

Key Questions:
- Does the curve flatten/steepen near expiry?
- Is contango/backwardation consistent across the cycle?
- Does the front spread (F2-F1) show systematic behavior 14 days before expiry?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm

from futures_roll_analysis.rolls import identify_front_to_f12, build_expiry_map


def load_and_prepare_data(panel_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load panel and extract F1-F12 contract chain."""
    print("Loading panel...")
    panel = pd.read_parquet(panel_path)
    
    # Build expiry map from panel
    contracts = [col[0] for col in panel.columns if col[0] != 'meta']
    expiry_map_data = []
    for contract in set(contracts):
        expiry_vals = panel[(contract, 'expiry')].dropna()
        if len(expiry_vals) > 0:
            expiry_map_data.append({
                'contract': contract,
                'expiry_date': expiry_vals.iloc[0]
            })
    
    expiry_df = pd.DataFrame(expiry_map_data)
    expiry_map = build_expiry_map(expiry_df)
    
    print(f"Panel shape: {panel.shape}")
    print(f"Date range: {panel.index[0]} to {panel.index[-1]}")
    print(f"Number of contracts: {len(set(contracts))}")
    
    return panel, expiry_map


def identify_contract_chain(panel: pd.DataFrame, expiry_map: pd.Series) -> pd.DataFrame:
    """Identify F1-F12 contract chain for each timestamp."""
    print("Identifying F1-F12 contract chain...")
    contract_chain = identify_front_to_f12(panel, expiry_map, max_contracts=12)
    print(f"Contract chain shape: {contract_chain.shape}")
    return contract_chain


def extract_term_structure_prices(
    panel: pd.DataFrame,
    contract_chain: pd.DataFrame,
    price_field: str = "close"
) -> pd.DataFrame:
    """
    Extract F1-F12 prices from panel using contract chain.
    
    Returns DataFrame with shape (timestamps, 12) where columns are F1, F2, ..., F12.
    """
    print("Extracting term structure prices...")
    
    # Get all contract prices
    contracts = [col[0] for col in panel.columns if col[0] != 'meta']
    contract_index = {contract: idx for idx, contract in enumerate(contracts)}
    
    # Extract close prices for all contracts
    close_cols = [(c, price_field) for c in contracts if (c, price_field) in panel.columns]
    close_df = panel.loc[:, close_cols].copy()
    close_df.columns = [c[0] for c in close_df.columns]
    close_matrix = close_df.values  # (timestamps, num_contracts)
    
    # For each timestamp, extract prices for F1-F12
    f_prices = np.full((len(contract_chain), 12), np.nan)
    
    for i in range(1, 13):
        f_col = f"F{i}"
        f_contracts = contract_chain[f_col].values
        
        for t, contract in enumerate(f_contracts):
            if contract is None or pd.isna(contract):
                continue
            if contract not in contract_index:
                continue
            c_idx = contract_index[contract]
            f_prices[t, i-1] = close_matrix[t, c_idx]
    
    result = pd.DataFrame(
        f_prices,
        index=contract_chain.index,
        columns=[f"F{i}" for i in range(1, 13)]
    )
    
    print(f"Term structure shape: {result.shape}")
    return result


def compute_days_to_expiry(panel: pd.DataFrame, contract_chain: pd.DataFrame) -> pd.Series:
    """Compute days to F1 expiry for each timestamp."""
    f1_contracts = contract_chain['F1'].values
    
    days_to_expiry = np.full(len(contract_chain), np.nan)
    
    for t, contract in enumerate(f1_contracts):
        if contract is None or pd.isna(contract):
            continue
        if (contract, 'expiry') not in panel.columns:
            continue
        
        expiry = panel.loc[contract_chain.index[t], (contract, 'expiry')]
        if pd.isna(expiry):
            continue
        
        days = (expiry - contract_chain.index[t]).days
        days_to_expiry[t] = days
    
    return pd.Series(days_to_expiry, index=contract_chain.index)


def select_representative_snapshots(
    term_structure: pd.DataFrame,
    days_to_expiry: pd.Series,
    target_days: list[int] = None
) -> dict[int, int]:
    """
    Select representative timestamps near target days-to-expiry.
    
    Returns dict mapping target_days -> index in term_structure.
    """
    if target_days is None:
        target_days = [60, 45, 30, 15, 7, 0]  # Days before expiry
    
    snapshots = {}
    
    for target in target_days:
        # Find closest timestamp with valid term structure
        valid_mask = (
            (days_to_expiry >= target - 2) &
            (days_to_expiry <= target + 2) &
            (term_structure.notna().sum(axis=1) >= 10)  # At least 10 contracts
        )
        
        if not valid_mask.any():
            print(f"  No valid snapshot near {target} days to expiry")
            continue
        
        # Get one near the target (prefer the one closest to target)
        valid_idx = np.where(valid_mask)[0]
        distances = np.abs(days_to_expiry.iloc[valid_idx].values - target)
        best_idx = valid_idx[np.argmin(distances)]
        
        snapshots[target] = best_idx
        actual_days = days_to_expiry.iloc[best_idx]
        actual_time = term_structure.index[best_idx]
        print(f"  {target} days: found at index {best_idx} ({actual_time}), actual={actual_days:.0f} days")
    
    return snapshots


def plot_term_structure_evolution(
    term_structure: pd.DataFrame,
    snapshots: dict[int, int],
    output_path: str
):
    """Create visualization of term structure evolution."""
    print(f"Creating term structure plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Define colors for each snapshot
    colors = cm.viridis(np.linspace(0, 1, len(snapshots)))
    
    # Sort by days to expiry for better color gradient
    sorted_snapshots = sorted(snapshots.items(), key=lambda x: x[0])
    
    # Plot 1: Term structure curves
    for (days, idx), color in zip(sorted_snapshots, colors):
        prices = term_structure.iloc[idx].values
        valid_mask = np.isfinite(prices)
        contracts = np.arange(1, 13)[valid_mask]
        valid_prices = prices[valid_mask]
        
        label = f"{days} days to F1 expiry"
        ax1.plot(contracts, valid_prices, marker='o', label=label, color=color, linewidth=2)
    
    ax1.set_xlabel('Contract Position (F1=1, F2=2, ..., F12=12)', fontsize=11)
    ax1.set_ylabel('Price (USD/lb)', fontsize=11)
    ax1.set_title('Copper Futures Term Structure Evolution Around F1 Expiry', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.arange(1, 13))
    
    # Plot 2: Spread analysis (F2-F1, F3-F2, etc.)
    for (days, idx), color in zip(sorted_snapshots, colors):
        prices = term_structure.iloc[idx].values
        valid_mask = np.isfinite(prices)
        
        # Compute consecutive spreads
        valid_prices = prices[valid_mask]
        spreads = np.diff(valid_prices)
        spread_positions = np.arange(1, len(spreads) + 1)
        
        label = f"{days} days to F1 expiry"
        ax2.plot(spread_positions, spreads, marker='s', label=label, color=color, linewidth=2)
    
    ax2.set_xlabel('Spread Position (S1=F2-F1, S2=F3-F2, ...)', fontsize=11)
    ax2.set_ylabel('Spread Value (USD/lb)', fontsize=11)
    ax2.set_title('Consecutive Spreads (F(i+1) - F(i)) Around Expiry', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_xticks(np.arange(1, 11))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


def compute_term_structure_metrics(
    term_structure: pd.DataFrame,
    snapshots: dict[int, int],
    days_to_expiry: pd.Series
) -> pd.DataFrame:
    """Compute metrics for each snapshot."""
    metrics = []
    
    for days, idx in sorted(snapshots.items()):
        prices = term_structure.iloc[idx].values
        valid_mask = np.isfinite(prices)
        valid_prices = prices[valid_mask]
        
        if len(valid_prices) < 2:
            continue
        
        # Slope: (F12 - F1) / 11
        slope = (prices[-1] - prices[0]) / 11 if np.isfinite(prices[0]) and np.isfinite(prices[-1]) else np.nan
        
        # Front spread: F2 - F1
        front_spread = prices[1] - prices[0] if np.isfinite(prices[0]) and np.isfinite(prices[1]) else np.nan
        
        # Convexity: Simple measure (F2-F1) - (F12-F11) or second difference average
        # Using simplified: variance of consecutive differences
        spreads = np.diff(valid_prices)
        convexity = np.std(spreads) if len(spreads) > 1 else np.nan
        
        # Average level
        avg_level = np.mean(valid_prices)
        
        metrics.append({
            'timestamp': term_structure.index[idx],
            'days_to_f1_expiry': days,
            'f1_price': prices[0],
            'f12_price': prices[-1],
            'avg_price': avg_level,
            'slope': slope,
            'front_spread_f2_f1': front_spread,
            'convexity_std': convexity,
            'num_contracts': np.sum(valid_mask)
        })
    
    return pd.DataFrame(metrics)


def analyze_near_expiry_behavior(
    term_structure: pd.DataFrame,
    contract_chain: pd.DataFrame,
    days_to_expiry: pd.Series,
    window_days: int = 14
):
    """
    Analyze behavior in the final window before expiry.
    Specifically look at whether F2-F1 spread shows patterns.
    """
    print(f"\n{'='*70}")
    print(f"Analysis: Last {window_days} days before F1 expiry")
    print(f"{'='*70}")
    
    # Filter for near-expiry period
    near_expiry_mask = (days_to_expiry >= 0) & (days_to_expiry <= window_days)
    
    if not near_expiry_mask.any():
        print("No data in near-expiry window")
        return
    
    near_expiry_data = term_structure[near_expiry_mask].copy()
    near_expiry_days = days_to_expiry[near_expiry_mask].copy()
    
    print(f"\nRecords in {window_days}-day window: {near_expiry_data.shape[0]}")
    print(f"Date range: {near_expiry_data.index[0]} to {near_expiry_data.index[-1]}")
    
    # Compute F2-F1 spread for this window
    f1_prices = near_expiry_data['F1'].values
    f2_prices = near_expiry_data['F2'].values
    spreads = f2_prices - f1_prices
    valid_spreads = spreads[np.isfinite(spreads)]
    
    print(f"\nF2-F1 Spread Statistics (last {window_days} days):")
    print(f"  Mean: {np.nanmean(spreads):.4f} USD/lb")
    print(f"  Median: {np.nanmedian(spreads):.4f} USD/lb")
    print(f"  Std Dev: {np.nanstd(spreads):.4f} USD/lb")
    print(f"  Min: {np.nanmin(spreads):.4f} USD/lb")
    print(f"  Max: {np.nanmax(spreads):.4f} USD/lb")
    
    # Check for trend: compare early vs late
    if len(valid_spreads) > 10:
        split_idx = len(valid_spreads) // 2
        early = valid_spreads[:split_idx]
        late = valid_spreads[split_idx:]
        
        print(f"\nF2-F1 Spread Comparison:")
        print(f"  Earlier half (mean): {np.mean(early):.4f} USD/lb")
        print(f"  Later half (mean):   {np.mean(late):.4f} USD/lb")
        print(f"  Change: {np.mean(late) - np.mean(early):+.4f} USD/lb")
    
    # Check term structure flattening
    print(f"\nTerm Structure (F1-F12) in near-expiry window:")
    f1_vals = near_expiry_data['F1'].values
    f12_vals = near_expiry_data['F12'].values
    slopes = (f12_vals - f1_vals) / 11
    valid_slopes = slopes[np.isfinite(slopes)]
    
    print(f"  Mean slope: {np.nanmean(slopes):.4f} USD/lb (F12-F1)/11")
    print(f"  Median slope: {np.nanmedian(slopes):.4f} USD/lb")
    print(f"  Contango (positive slope): {(valid_slopes > 0).mean()*100:.1f}%")
    print(f"  Backwardation (negative): {(valid_slopes < 0).mean()*100:.1f}%")


def main():
    """Main execution."""
    # Paths
    panel_path = 'outputs/panels/hourly_panel.parquet'
    output_dir = Path('outputs/exploratory')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / 'term_structure_evolution.png'
    metrics_path = output_dir / 'term_structure_metrics.csv'
    
    # Load and prepare
    panel, expiry_map = load_and_prepare_data(panel_path)
    contract_chain = identify_contract_chain(panel, expiry_map)
    term_structure = extract_term_structure_prices(panel, contract_chain)
    days_to_expiry = compute_days_to_expiry(panel, contract_chain)
    
    # Select snapshots
    print("\nSelecting representative snapshots...")
    snapshots = select_representative_snapshots(term_structure, days_to_expiry)
    
    if not snapshots:
        print("ERROR: Could not find any valid snapshots")
        return 1
    
    # Create visualization
    plot_term_structure_evolution(term_structure, snapshots, str(plot_path))
    
    # Compute metrics
    metrics_df = compute_term_structure_metrics(term_structure, snapshots, days_to_expiry)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")
    
    print("\nTerm Structure Metrics Summary:")
    print(metrics_df.to_string(index=False))
    
    # Analyze near-expiry behavior
    analyze_near_expiry_behavior(term_structure, contract_chain, days_to_expiry, window_days=14)
    
    print(f"\n{'='*70}")
    print("Outputs:")
    print(f"  - Visualization: {plot_path}")
    print(f"  - Metrics: {metrics_path}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
