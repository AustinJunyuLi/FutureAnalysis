"""
Multi-spread comparative analysis for identifying institutional roll patterns.

This module provides functions to compare signal strength across S1, S2, ..., S11
to determine if front-spread (S1) events are driven by institutional rolling behavior
or general contract expiry mechanics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_spread_correlations(spreads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise correlations between all spread series.

    Parameters
    ----------
    spreads_df:
        DataFrame with spread columns (e.g., S1, S2, ..., S11).

    Returns
    -------
    DataFrame with correlation matrix (spread_i × spread_j).
    """
    return spreads_df.corr()


def compare_spread_signals(
    spreads_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare detection statistics across all spread levels.

    Parameters
    ----------
    spreads_df:
        DataFrame with spread columns (S1, S2, ..., S11).
    events_df:
        DataFrame with event columns (S1_events, S2_events, ..., S11_events).

    Returns
    -------
    DataFrame with columns:
        - spread: Spread identifier (S1, S2, ...)
        - event_count: Total events detected
        - detection_rate_pct: Percentage of periods with events
        - mean_spread: Average spread value
        - mean_event_spread: Average spread during events
        - std_event_spread: Std dev of spread during events
        - signal_to_noise: Ratio of event spread std to overall spread std
    """
    results = []

    for col in spreads_df.columns:
        spread_series = spreads_df[col]
        event_col = f"{col}_events"

        if event_col not in events_df.columns:
            continue

        events = events_df[event_col]

        event_count = int(events.sum())
        total_periods = len(events)
        detection_rate = (event_count / total_periods * 100) if total_periods > 0 else 0

        mean_spread = spread_series.mean() if len(spread_series) > 0 else np.nan
        std_spread = spread_series.std() if len(spread_series) > 0 else np.nan

        # Statistics during events
        event_spreads = spread_series[events]
        mean_event_spread = event_spreads.mean() if len(event_spreads) > 0 else np.nan
        std_event_spread = event_spreads.std() if len(event_spreads) > 0 else np.nan

        # Signal-to-noise ratio: std during events / overall std
        snr = (std_event_spread / std_spread) if (std_spread > 0 and not np.isnan(std_event_spread)) else np.nan

        results.append({
            'spread': col,
            'event_count': event_count,
            'detection_rate_pct': detection_rate,
            'mean_spread': mean_spread,
            'std_spread': std_spread,
            'mean_event_spread': mean_event_spread,
            'std_event_spread': std_event_spread,
            'signal_to_noise': snr,
        })

    return pd.DataFrame(results)


def analyze_spread_timing(
    spreads_df: pd.DataFrame,
    events_df: pd.DataFrame,
    contract_chain: pd.DataFrame,
    expiry_map: pd.Series,
) -> pd.DataFrame:
    """
    Analyze timing of events relative to F1 contract expiry for all spreads.

    FIXED: Now uses F1 expiry as common reference point for all spreads,
    allowing fair comparison across S1, S2, ..., S11.

    Parameters
    ----------
    spreads_df:
        DataFrame with spread columns (S1, S2, ..., S11).
    events_df:
        DataFrame with event columns (S1_events, S2_events, ...).
    contract_chain:
        DataFrame with contract identifiers (F1, F2, ..., F12).
    expiry_map:
        Series mapping contracts to expiry dates.

    Returns
    -------
    DataFrame with columns:
        - spread: Spread identifier
        - event_date: Timestamp of event
        - contract: Front contract identifier for this spread
        - expiry_date: Expiry date of spread's front contract
        - days_to_expiry: Calendar days until F1 expiry (common reference)
        - f1_contract: F1 contract identifier
        - f1_expiry: F1 expiry date
    """
    results = []

    for col in spreads_df.columns:
        event_col = f"{col}_events"
        if event_col not in events_df.columns:
            continue

        # Extract spread number (e.g., "S1" → 1)
        spread_num = int(col[1:])
        front_contract_col = f"F{spread_num}"

        if front_contract_col not in contract_chain.columns:
            continue

        events = events_df[event_col]
        event_timestamps = events[events].index

        for ts in event_timestamps:
            # Get both the spread's front contract AND F1 for common reference
            front_contract = contract_chain.loc[ts, front_contract_col]
            f1_contract = contract_chain.loc[ts, 'F1']  # Common reference point

            if pd.isna(front_contract) or front_contract is None:
                continue
            if pd.isna(f1_contract) or f1_contract is None:
                continue

            if front_contract not in expiry_map.index:
                continue
            if f1_contract not in expiry_map.index:
                continue

            # FIXED: Use F1 expiry as common reference for all spreads
            f1_expiry_date = expiry_map[f1_contract]
            days_to_f1_expiry = (f1_expiry_date - ts.normalize()).days

            # Also keep the spread's own front contract expiry for reference
            spread_expiry_date = expiry_map[front_contract]

            results.append({
                'spread': col,
                'event_date': ts,
                'contract': front_contract,
                'expiry_date': spread_expiry_date,
                'days_to_expiry': days_to_f1_expiry,  # Now relative to F1!
                'f1_contract': f1_contract,
                'f1_expiry': f1_expiry_date,
            })

    if not results:
        return pd.DataFrame(columns=['spread', 'event_date', 'contract', 'expiry_date', 'days_to_expiry', 'f1_contract', 'f1_expiry'])

    return pd.DataFrame(results)


def summarize_timing_by_spread(timing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize event timing statistics by spread level.

    Parameters
    ----------
    timing_df:
        Output from analyze_spread_timing().

    Returns
    -------
    DataFrame with columns:
        - spread: Spread identifier
        - event_count: Number of events
        - median_days_to_expiry: Median days to expiry
        - mean_days_to_expiry: Mean days to expiry
        - std_days_to_expiry: Std dev of days to expiry
        - q25_days_to_expiry: 25th percentile
        - q75_days_to_expiry: 75th percentile
    """
    if timing_df.empty:
        return pd.DataFrame(columns=[
            'spread', 'event_count', 'median_days_to_expiry',
            'mean_days_to_expiry', 'std_days_to_expiry',
            'q25_days_to_expiry', 'q75_days_to_expiry'
        ])

    summary = timing_df.groupby('spread')['days_to_expiry'].agg([
        ('event_count', 'count'),
        ('median_days_to_expiry', 'median'),
        ('mean_days_to_expiry', 'mean'),
        ('std_days_to_expiry', 'std'),
        ('q25_days_to_expiry', lambda x: x.quantile(0.25)),
        ('q75_days_to_expiry', lambda x: x.quantile(0.75)),
    ]).reset_index()

    return summary


def analyze_spread_changes(spreads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute change statistics for all spreads.

    Parameters
    ----------
    spreads_df:
        DataFrame with spread columns (S1, S2, ..., S11).

    Returns
    -------
    DataFrame with columns:
        - spread: Spread identifier
        - mean_change: Mean spread change
        - std_change: Std dev of spread change
        - abs_mean_change: Mean absolute change
        - max_increase: Maximum single-period increase
        - max_decrease: Maximum single-period decrease
    """
    results = []

    for col in spreads_df.columns:
        spread_series = spreads_df[col]
        changes = spread_series.diff()

        results.append({
            'spread': col,
            'mean_change': changes.mean(),
            'std_change': changes.std(),
            'abs_mean_change': changes.abs().mean(),
            'max_increase': changes.max(),
            'max_decrease': changes.min(),
        })

    return pd.DataFrame(results)


def compare_spread_magnitudes(spreads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare S1 magnitude vs S2-S11 at each timestamp.

    This answers the supervisor's question: On which days does S1 move
    significantly more than other spreads?

    Parameters
    ----------
    spreads_df:
        DataFrame with spread columns (S1, S2, ..., S11).

    Returns
    -------
    DataFrame with columns:
        - s1_change: Absolute change in S1
        - others_median: Median absolute change in S2-S11
        - others_mean: Mean absolute change in S2-S11
        - dominance_ratio: |ΔS1| / median(|ΔS2|,...,|ΔS11|)
        - s1_dominates: True when dominance_ratio > 2.0
        - rank_s1: Rank of S1 magnitude (1=largest, 11=smallest)
    """
    changes = spreads_df.diff().abs()

    # Separate S1 from others
    s1_change = changes['S1']
    other_cols = [col for col in changes.columns if col != 'S1']
    other_changes = changes[other_cols]

    # Compute statistics for comparison
    others_median = other_changes.median(axis=1)
    others_mean = other_changes.mean(axis=1)

    # Dominance ratio: how much larger is S1 compared to median of others
    # Add small epsilon to avoid division by zero
    dominance_ratio = s1_change / (others_median + 1e-10)

    # S1 dominates when it moves >2x the median of other spreads
    s1_dominates = dominance_ratio > 2.0

    # Rank S1 among all spreads for each period (1 = largest move)
    ranks = changes.rank(axis=1, ascending=False, method='average')
    rank_s1 = ranks['S1']

    return pd.DataFrame({
        's1_change': s1_change,
        'others_median': others_median,
        'others_mean': others_mean,
        'dominance_ratio': dominance_ratio,
        's1_dominates': s1_dominates,
        'rank_s1': rank_s1,
    }, index=spreads_df.index)


def analyze_s1_dominance_by_expiry_cycle(
    magnitude_comparison: pd.DataFrame,
    contract_chain: pd.DataFrame,
    expiry_map: pd.Series,
) -> pd.DataFrame:
    """
    Group S1 dominance events by position in monthly expiry cycle.

    This identifies if there are specific days before F1 expiry when S1
    consistently moves more than other spreads.

    Parameters
    ----------
    magnitude_comparison:
        Output from compare_spread_magnitudes().
    contract_chain:
        DataFrame with contract identifiers (F1, F2, ...).
    expiry_map:
        Series mapping contracts to expiry dates.

    Returns
    -------
    DataFrame with columns:
        - days_to_f1_expiry_bucket: Cycle position (e.g., '0-5', '6-10')
        - total_observations: Number of periods in this bucket
        - s1_dominance_count: Number of times S1 dominated
        - s1_dominance_rate: Percentage of times S1 dominated
        - avg_dominance_ratio: Average S1/others ratio when S1 dominates
        - avg_s1_magnitude: Average |ΔS1| in this bucket
        - avg_others_magnitude: Average median(|ΔS2|,...,|ΔS11|)
    """
    # Add F1 contract and expiry information
    magnitude_comparison = magnitude_comparison.copy()

    # Get F1 contract for each timestamp
    magnitude_comparison['f1_contract'] = contract_chain['F1']

    # Calculate days to F1 expiry for each row
    days_to_f1_expiry = []
    for idx, row in magnitude_comparison.iterrows():
        f1_contract = row['f1_contract']
        if pd.isna(f1_contract) or f1_contract not in expiry_map.index:
            days_to_f1_expiry.append(np.nan)
        else:
            f1_expiry = expiry_map[f1_contract]
            days = (f1_expiry - pd.Timestamp(idx).normalize()).days
            days_to_f1_expiry.append(days)

    magnitude_comparison['days_to_f1_expiry'] = days_to_f1_expiry

    # Filter to valid data
    valid_data = magnitude_comparison[magnitude_comparison['days_to_f1_expiry'].notna()].copy()

    # Create buckets for cycle position
    bins = [-np.inf, 5, 10, 15, 20, 25, 30, 40, 50, 60, np.inf]
    labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', '41-50', '51-60', '60+']
    valid_data['cycle_bucket'] = pd.cut(valid_data['days_to_f1_expiry'], bins=bins, labels=labels)

    # Group by bucket and compute statistics
    grouped = valid_data.groupby('cycle_bucket', observed=True)

    results = []
    for bucket, group in grouped:
        total_obs = len(group)
        s1_dom_count = group['s1_dominates'].sum()
        s1_dom_rate = (s1_dom_count / total_obs * 100) if total_obs > 0 else 0

        # Average dominance ratio when S1 dominates
        dominance_events = group[group['s1_dominates']]
        avg_dom_ratio = dominance_events['dominance_ratio'].mean() if len(dominance_events) > 0 else np.nan

        # Average magnitudes
        avg_s1_mag = group['s1_change'].mean()
        avg_others_mag = group['others_median'].mean()

        results.append({
            'days_to_f1_expiry_bucket': bucket,
            'total_observations': total_obs,
            's1_dominance_count': int(s1_dom_count),
            's1_dominance_rate': s1_dom_rate,
            'avg_dominance_ratio': avg_dom_ratio,
            'avg_s1_magnitude': avg_s1_mag,
            'avg_others_magnitude': avg_others_mag,
        })

    return pd.DataFrame(results)


def summarize_cross_spread_patterns(
    dominance_df: pd.DataFrame,
    significance_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Statistical summary of cross-spread patterns to identify if S1 is special.

    Parameters
    ----------
    dominance_df:
        Output from analyze_s1_dominance_by_expiry_cycle().
    significance_threshold:
        Minimum dominance rate (%) to consider significant.

    Returns
    -------
    DataFrame with summary statistics and interpretation.
    """
    # Find buckets where S1 significantly dominates
    significant = dominance_df[dominance_df['s1_dominance_rate'] > significance_threshold]

    if len(significant) == 0:
        interpretation = "No significant S1 dominance detected at any point in the cycle"
    else:
        peak_bucket = significant.loc[significant['s1_dominance_rate'].idxmax()]
        peak_days = peak_bucket['days_to_f1_expiry_bucket']
        peak_rate = peak_bucket['s1_dominance_rate']
        interpretation = f"S1 shows peak dominance {peak_days} days before F1 expiry ({peak_rate:.1f}% of periods)"

    summary = pd.DataFrame({
        'metric': [
            'Peak dominance bucket',
            'Peak dominance rate (%)',
            'Average dominance ratio at peak',
            'Interpretation',
            'Suggests institutional rolling',
        ],
        'value': [
            significant.loc[significant['s1_dominance_rate'].idxmax(), 'days_to_f1_expiry_bucket'] if len(significant) > 0 else 'None',
            f"{significant['s1_dominance_rate'].max():.1f}" if len(significant) > 0 else '0.0',
            f"{significant.loc[significant['s1_dominance_rate'].idxmax(), 'avg_dominance_ratio']:.2f}" if len(significant) > 0 else 'N/A',
            interpretation,
            'Yes' if len(significant) > 0 and significant['s1_dominance_rate'].max() > 15.0 else 'No',
        ]
    })

    return summary
