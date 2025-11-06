#!/usr/bin/env python3
"""
Contract Month Reframing Analysis

Purpose
-------
Implements two timing lenses to test whether spread-widening events cluster at the
start of a "contract month":

1) First‑F1 appearance proxy (existing): days since the current contract first became F1
2) Supervisor definition (new): days since the previous F1's expiry (month starts the
   day after the old F1 expires)

Notes
-----
The supervisor's hypothesis refers to (2). We keep (1) for comparison because some
contracts become F1 before the prior expiry due to liquidity crossover.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def chi_square_test(observed, expected):
    """
    Compute chi-square test statistic and p-value.

    Uses SciPy if available; otherwise falls back to a coarse approximation.
    """
    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)

    # Chi-square statistic: sum((O - E)^2 / E)
    chi2_stat = np.sum((observed - expected) ** 2 / expected)

    try:
        from scipy.stats import chisquare  # type: ignore
        res = chisquare(f_obs=observed, f_exp=expected)
        p_value = float(res.pvalue)
        return chi2_stat, p_value
    except Exception:
        # Fallback: compare to critical values (approximate)
        df = len(observed) - 1
        critical_values = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070}
        critical_value = critical_values.get(df, 9.488)
        if chi2_stat < critical_value:
            p_value = 0.1  # Not significant (approx.)
        elif chi2_stat < critical_value * 2:
            p_value = 0.01
        else:
            p_value = 0.001
        return chi2_stat, p_value


def load_metadata(metadata_path: Path) -> pd.Series:
    """Load contract expiry metadata and return expiry_map."""
    metadata = pd.read_csv(metadata_path)
    metadata['expiry_date'] = pd.to_datetime(metadata['expiry_date'])
    expiry_map = metadata.set_index('contract')['expiry_date']
    return expiry_map


def get_contract_sequence(expiry_map: pd.Series) -> pd.DataFrame:
    """
    Build chronological sequence of contracts with previous contract for each.

    Returns DataFrame with columns: contract, expiry_date, prev_contract, prev_expiry
    """
    # Sort contracts by expiry date
    contract_df = pd.DataFrame({
        'contract': expiry_map.index,
        'expiry_date': expiry_map.values
    }).sort_values('expiry_date').reset_index(drop=True)

    # Identify previous contract (the one that expired before this one became F1)
    contract_df['prev_contract'] = contract_df['contract'].shift(1)
    contract_df['prev_expiry'] = contract_df['expiry_date'].shift(1)

    return contract_df


def build_f1_transition_chain(
    timestamps: pd.DatetimeIndex,
    f1_contracts: pd.Series,
) -> pd.DataFrame:
    """
    Construct the observed F1 transition chain from the panel.

    Returns a DataFrame with columns:
      - start_time: timestamp when new F1 segment begins
      - new_f1: contract that becomes F1 at start_time
      - prev_f1: contract that was F1 immediately before start_time
      - start_pos / end_pos: integer index bounds for the segment within timestamps
    """
    idx = pd.DatetimeIndex(timestamps)
    series = pd.Series(f1_contracts.values, index=idx)
    # Normalize to python objects to avoid categorical/nullable corner cases
    series = series.astype(object)

    changes = series.ne(series.shift(1)).fillna(True)
    change_points = series.index[changes]

    # Build segments
    starts = list(change_points)
    positions = np.searchsorted(idx.values, np.array(starts, dtype='datetime64[ns]'))
    segments = []
    for i, (t0, pos0) in enumerate(zip(starts, positions)):
        pos1 = positions[i + 1] if i + 1 < len(positions) else len(idx)
        new_f1 = series.loc[t0]
        prev_f1 = series.iloc[pos0 - 1] if pos0 > 0 else None
        segments.append(
            {
                'start_time': t0,
                'new_f1': new_f1,
                'prev_f1': prev_f1,
                'start_pos': int(pos0),
                'end_pos': int(pos1),
            }
        )
    return pd.DataFrame(segments)


def compute_days_since_prev_expiry_chain(
    timestamps: pd.DatetimeIndex,
    f1_contracts: pd.Series,
    expiry_map: pd.Series,
) -> pd.Series:
    """
    Compute days since the previous F1's expiry using the observed transition chain.

    This anchors each F1 segment to the expiry date of the F1 contract that
    actually preceded it in the panel (prev_f1 from the chain), avoiding errors
    from calendar-only ordering.
    """
    idx = pd.DatetimeIndex(timestamps)
    idx_dates = idx.normalize()
    chain = build_f1_transition_chain(idx, f1_contracts)

    prev_expiry_series = pd.Series(pd.NaT, index=idx)
    for _, row in chain.iterrows():
        prev_f1 = row['prev_f1']
        if prev_f1 is None or (isinstance(prev_f1, float) and pd.isna(prev_f1)):
            continue
        exp = expiry_map.get(str(prev_f1))
        if pd.isna(exp):
            continue
        exp = pd.to_datetime(exp).normalize()
        prev_expiry_series.iloc[row['start_pos']: row['end_pos']] = exp

    days_since_prev = (idx_dates - prev_expiry_series).dt.days
    return days_since_prev.rename('days_since_prev_expiry_chain')


def compute_days_since_became_f1(
    timestamps: pd.DatetimeIndex,
    f1_contracts: pd.Series,
    contract_sequence: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate "days since this contract became F1" for each timestamp.

    Key insight: A contract becomes F1 when it first appears as F1 in the dataset,
    which may be BEFORE the previous contract expires (due to liquidity crossover).

    Returns
    -------
    days_since_became_f1: Days since contract first became F1
    first_appearance_date: Date when contract first became F1
    """
    # Find first appearance date for each contract as F1
    first_appearance = f1_contracts.to_frame('contract')
    first_appearance = first_appearance[first_appearance['contract'].notna()]
    first_appearance = first_appearance.groupby('contract').first()
    first_appearance['first_f1_date'] = first_appearance.index.to_series().groupby(first_appearance.index).first()

    # Actually, we need to track when each contract FIRST appears as F1
    contract_first_f1 = {}
    for ts, contract in f1_contracts.items():
        if pd.notna(contract) and contract not in contract_first_f1:
            contract_first_f1[contract] = pd.Timestamp(ts).normalize()

    # Map each timestamp's F1 contract to its first F1 date
    first_f1_dates = f1_contracts.map(contract_first_f1)

    # Calculate days since became F1
    timestamp_dates = pd.to_datetime(timestamps).normalize()
    days_since_f1 = (timestamp_dates - first_f1_dates).dt.days

    return (
        pd.Series(days_since_f1.values, index=timestamps, name='days_since_became_f1'),
        pd.Series(first_f1_dates.values, index=timestamps, name='first_f1_date')
    )


def analyze_event_distribution(
    event_data: pd.DataFrame,
    output_dir: Path,
    *,
    days_col: str = 'days_since_became_f1',
    tag: str = 'became_f1',
) -> dict:
    """
    Analyze distribution of events by contract age.

    Returns dict with statistics and generates histogram plot + CSV.
    """
    # Filter to events only
    events_only = event_data[event_data['event_detected']].copy()

    print(f"\nTotal events: {len(events_only)}")
    print(f"Total periods: {len(event_data)}")
    print(f"Detection rate: {len(events_only) / len(event_data) * 100:.2f}%")

    # Remove rows with missing contract age
    if days_col not in events_only.columns:
        raise KeyError(f"Column '{days_col}' not found in event_data")
    valid_events = events_only[events_only[days_col].notna()].copy()
    print(f"Events with valid contract age: {len(valid_events)}")

    if len(valid_events) == 0:
        print("WARNING: No events with valid contract age data!")
        return {}

    # Distribution by week bins
    bins = [0, 7, 14, 21, 28, 60]
    labels = ['Days 1-7', 'Days 8-14', 'Days 15-21', 'Days 22-28', 'Days 28+']
    valid_events['age_bin'] = pd.cut(
        valid_events[days_col],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )

    bin_counts = valid_events['age_bin'].value_counts().sort_index()
    print("\n" + "="*60)
    print("EVENT DISTRIBUTION BY CONTRACT AGE")
    print("="*60)
    for bin_label, count in bin_counts.items():
        pct = count / len(valid_events) * 100
        print(f"{bin_label:15s}: {count:5d} events ({pct:5.1f}%)")

    # Key statistic: % in first 7 days
    early_events = valid_events[valid_events[days_col] <= 7]
    early_pct = len(early_events) / len(valid_events) * 100
    print(f"\n{'FIRST 7 DAYS':15s}: {len(early_events):5d} events ({early_pct:5.1f}%)")

    # Statistical test for uniformity
    # Expected: uniform distribution across bins (weighted by bin width)
    bin_widths = [7, 7, 7, 7, 32]  # Days in each bin
    total_days = sum(bin_widths)
    expected_probs = [w / total_days for w in bin_widths]
    expected_counts = [p * len(valid_events) for p in expected_probs]

    chi2_stat, p_value = chi_square_test(bin_counts.values, expected_counts)

    print("\n" + "="*60)
    print("CHI-SQUARE TEST FOR UNIFORM DISTRIBUTION")
    print("="*60)
    print(f"Chi-square statistic: {chi2_stat:.2f}")
    print(f"P-value: {p_value:.4e}")

    if p_value < 0.05:
        print(f"CONCLUSION: Non-uniform distribution (p < 0.05)")
        if early_pct > 100 / total_days * 7:  # More than expected by chance
            print(f"            Events cluster in FIRST 7 DAYS ({early_pct:.1f}% vs expected {100/total_days*7:.1f}%)")
        else:
            print(f"            Events do NOT cluster early")
    else:
        print(f"CONCLUSION: No significant clustering detected (p > 0.05)")

    # Generate histogram (best-effort; skip if headless font issues)
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual distribution
        bin_centers = [3.5, 10.5, 17.5, 24.5, 44]  # Midpoint of each bin
        ax.bar(bin_centers, bin_counts.values / len(valid_events) * 100,
               width=[7, 7, 7, 7, 32], alpha=0.7, label='Actual')

        # Plot expected uniform distribution
        expected_pcts = [p * 100 for p in expected_probs]
        ax.plot(bin_centers, expected_pcts, 'r--', linewidth=2, marker='o',
                markersize=8, label='Expected (uniform)')

        # Annotations
        for i, (center, count, pct) in enumerate(zip(bin_centers, bin_counts.values, bin_counts.values / len(valid_events) * 100)):
            ax.text(center, pct + 1, f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        x_label = 'Days Since Contract Became F1' if tag == 'became_f1' else 'Days Since Previous F1 Expiry'
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Percentage of Events (%)', fontsize=12)
        ax.set_title(
            f'Distribution of Spread Widening Events by Contract Age\n'
            f'Total Events: {len(valid_events)} | First 7 Days: {early_pct:.1f}% | χ²={chi2_stat:.2f}, p={p_value:.4e}',
            fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(labels, rotation=0)

        plt.tight_layout()
        plot_path = output_dir / f'contract_month_histogram_{tag}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"WARNING: Skipping plot generation due to rendering issue: {e}")

    # Summary statistics
    # Determine expected percentage for first 7 days under uniformity
    expected_first7_pct = 100 / total_days * 7

    # Conclusion text with over/under-representation
    if p_value < 0.05:
        if early_pct > expected_first7_pct:
            conclusion = 'Significant over-representation in first 7 days'
        else:
            conclusion = 'Significant under-representation in first 7 days'
    else:
        conclusion = 'No significant deviation from uniformity'

    stats_dict = {
        'total_events': len(valid_events),
        'first_7_days_count': len(early_events),
        'first_7_days_pct': early_pct,
        'median_age': valid_events[days_col].median(),
        'mean_age': valid_events[days_col].mean(),
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'expected_first7_pct': expected_first7_pct,
        'conclusion': conclusion,
        'days_basis': days_col,
    }

    return stats_dict


def main():
    """Main analysis workflow."""
    # Paths
    base_dir = Path(__file__).parent.parent
    metadata_path = base_dir / 'metadata' / 'contracts_metadata.csv'
    panel_path = base_dir / 'outputs' / 'panels' / 'hourly_panel.parquet'
    events_path = base_dir / 'outputs' / 'roll_signals' / 'hourly_widening.csv'
    output_dir = base_dir / 'outputs' / 'exploratory'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("CONTRACT MONTH REFRAMING ANALYSIS")
    print("="*60)
    print(f"\nLoading data from:")
    print(f"  Metadata:  {metadata_path}")
    print(f"  Panel:     {panel_path}")
    print(f"  Events:    {events_path}")

    # Load metadata
    print("\n[1/5] Loading contract metadata...")
    expiry_map = load_metadata(metadata_path)
    print(f"      Loaded {len(expiry_map)} contracts")

    # Build contract sequence
    print("\n[2/5] Building contract sequence...")
    contract_sequence = get_contract_sequence(expiry_map)
    print(f"      Built sequence of {len(contract_sequence)} contracts (calendar order)")

    # Load panel for F1 identification
    print("\n[3/5] Loading panel data...")
    panel = pd.read_parquet(panel_path)
    f1_contracts = panel[('meta', 'front_contract')]
    print(f"      Panel shape: {panel.shape}")
    print(f"      Date range: {panel.index[0]} to {panel.index[-1]}")

    # Load event data
    print("\n[4/5] Loading event data...")
    events = pd.read_csv(events_path, parse_dates=['bucket_start_ts'], index_col='bucket_start_ts')
    print(f"      Total periods: {len(events)}")
    print(f"      Events detected: {events['spread_widening'].sum()}")

    # Compute days since contract became F1 for all timestamps
    print("\n[5/5] Computing contract ages...")
    days_since_f1, first_f1_dates = compute_days_since_became_f1(
        panel.index,
        f1_contracts,
        contract_sequence
    )
    # Supervisor-defined days since previous F1 expiry (CHAIN-BASED)
    days_since_prev = compute_days_since_prev_expiry_chain(
        panel.index,
        f1_contracts,
        expiry_map,
    )

    # Build combined DataFrame for analysis
    event_data = pd.DataFrame({
        'timestamp': panel.index,
        'f1_contract': f1_contracts.values,
        'days_since_became_f1': days_since_f1.values,
        'days_since_prev_expiry_chain': days_since_prev.values,
        'first_f1_date': first_f1_dates.values,
        'event_detected': events['spread_widening'].reindex(panel.index, fill_value=False).values
    })

    # Also add traditional "days_to_expiry" for comparison
    f1_expiry_dates = f1_contracts.map(expiry_map)
    timestamp_dates = pd.to_datetime(panel.index).normalize()
    days_to_expiry = (f1_expiry_dates - timestamp_dates).dt.days
    event_data['days_to_expiry'] = days_to_expiry.values

    print(f"      Computed ages for {len(event_data)} periods")
    print(f"      Valid ages (non-null): {event_data['days_since_became_f1'].notna().sum()}")

    # Analyze distribution
    print("\n" + "="*60)
    print("ANALYZING EVENT DISTRIBUTION (first appearance as F1)")
    print("="*60)
    stats_first = analyze_event_distribution(
        event_data, output_dir, days_col='days_since_became_f1', tag='became_f1'
    )

    # Supervisor definition: days since previous F1 expiry (restrict to days >= 0)
    print("\n" + "="*60)
    print("ANALYZING EVENT DISTRIBUTION (days since previous F1 expiry — chain-based)")
    print("="*60)
    ed_prev = event_data.copy()
    ed_prev = ed_prev[ed_prev['days_since_prev_expiry_chain'].notna()]
    ed_prev = ed_prev[ed_prev['days_since_prev_expiry_chain'] >= 0]
    stats_prev = analyze_event_distribution(
        ed_prev, output_dir, days_col='days_since_prev_expiry_chain', tag='prev_expiry_chain'
    )

    # Save detailed event data
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)

    # Save all events with metadata
    csv_path = output_dir / 'events_with_contract_age.csv'
    event_data.to_csv(csv_path, index=False)
    print(f"Event data saved to: {csv_path}")

    # Save summary statistics for both bases
    if stats_first:
        summary_first_path = output_dir / 'contract_month_summary_became_f1.csv'
        pd.DataFrame([stats_first]).to_csv(summary_first_path, index=False)
        print(f"Summary stats saved to: {summary_first_path}")

    if stats_prev:
        summary_prev_path = output_dir / 'contract_month_summary_prev_expiry_chain.csv'
        pd.DataFrame([stats_prev]).to_csv(summary_prev_path, index=False)
        print(f"Summary stats saved to: {summary_prev_path}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nKey Findings:")
    if stats_first:
        print(f"  First‑F1 proxy: {stats_first['first_7_days_pct']:.1f}% in days 1–7 (p={stats_first['p_value']:.4e})")
        print(f"  Conclusion: {stats_first['conclusion']}")
    if stats_prev:
        print(f"  Supervisor metric: {stats_prev['first_7_days_pct']:.1f}% in days 1–7 (p={stats_prev['p_value']:.4e})")
        print(f"  Conclusion: {stats_prev['conclusion']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
