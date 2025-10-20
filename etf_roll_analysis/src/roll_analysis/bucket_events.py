"""
Bucket-level event detection for futures roll analysis.
Implements z-score based widening detection with variable-granularity buckets.

Enhancements:
- Default window_buckets set to 20 (≈2 trading days at 10 buckets/day)
- Time-based cool-down supported via cool_down_hours; falls back to bucket-based if not provided
"""

from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
from .bucket import get_bucket_info, BUCKET_DEFINITIONS


def detect_bucket_widening(
    spread: pd.Series,
    method: str = "zscore",
    window_buckets: int = 20,
    z_threshold: float = 1.5,
    abs_min: float = None,
    cool_down_buckets: int = 0,
    cool_down_hours: float = 3.0,
) -> pd.Series:
    """
    Detect spread widening events at bucket granularity.
    
    Parameters:
    -----------
    spread : pd.Series
        Calendar spread series at bucket level (next - front)
    method : str
        Detection method: "zscore", "abs", or "combined"
    window_buckets : int
        Rolling window for z-score calculation (in buckets, not days)
        Default: 480 = approximately 48 hours of trading
    z_threshold : float
        Z-score threshold for event detection
    abs_min : float
        Minimum absolute change threshold
    cool_down_buckets : int
        Minimum buckets between events (default: 30 = ~3 hours)
        
    Returns:
    --------
    pd.Series
        Boolean series indicating widening events per bucket
    """
    # Calculate bucket-level spread changes
    ds = spread.diff()
    
    if method == "zscore":
        # Z-score method with bucket-based window
        min_periods = max(10, window_buckets // 4)  # At least 10 buckets
        mu = ds.rolling(window_buckets, min_periods=min_periods).mean()
        sd = ds.rolling(window_buckets, min_periods=min_periods).std()
        
        # Avoid division by zero
        sd = sd.replace(0, np.nan)
        z = (ds - mu) / sd
        
        # Detect when z-score exceeds threshold
        trigger = z > z_threshold
        
        # Optional: require minimum absolute change
        if abs_min is not None:
            trigger &= (ds > abs_min)
            
    elif method == "abs":
        # Absolute threshold method
        if abs_min is None:
            raise ValueError("abs_min is required when method='abs'")
        trigger = ds > abs_min
        
    elif method == "combined":
        # Combined method: either z-score OR absolute threshold
        if abs_min is None:
            raise ValueError("abs_min is required when method='combined'")
        
        min_periods = max(10, window_buckets // 4)
        mu = ds.rolling(window_buckets, min_periods=min_periods).mean()
        sd = ds.rolling(window_buckets, min_periods=min_periods).std()
        sd = sd.replace(0, np.nan)
        z = (ds - mu) / sd
        
        trigger = (z > z_threshold) | (ds > abs_min)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply cool-down period
    if cool_down_hours is not None:
        # Time-based cool-down uses index timestamps
        trigger_filtered = trigger.copy()
        last_event_time = None
        idx = spread.index
        for i in range(len(trigger)):
            if not trigger.iloc[i]:
                continue
            t = pd.to_datetime(idx[i])
            if last_event_time is None or (t - last_event_time) >= pd.Timedelta(hours=cool_down_hours):
                last_event_time = t
            else:
                trigger_filtered.iloc[i] = False
        trigger = trigger_filtered
    elif cool_down_buckets and cool_down_buckets > 0:
        # Fallback: bucket-count cool-down
        trigger_filtered = trigger.copy()
        last_event_idx = -cool_down_buckets - 1
        for i in range(len(trigger)):
            if trigger.iloc[i]:
                if i - last_event_idx > cool_down_buckets:
                    last_event_idx = i
                else:
                    trigger_filtered.iloc[i] = False
        trigger = trigger_filtered
    
    # Fill NaN values with False
    trigger = trigger.fillna(False)
    
    return trigger.rename("bucket_widening_event")


def summarize_bucket_events(events: pd.Series, 
                           spread: pd.Series,
                           bucket_info: pd.Series = None) -> pd.DataFrame:
    """
    Create summary table of widening events by bucket.
    
    Parameters:
    -----------
    events : pd.Series
        Boolean series of detected events
    spread : pd.Series
        Spread values
    bucket_info : pd.Series
        Optional series with bucket IDs
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics per bucket
    """
    # Extract bucket information from index if MultiIndex
    if isinstance(events.index, pd.MultiIndex) and 'bucket' in events.index.names:
        bucket_ids = events.index.get_level_values('bucket')
    elif bucket_info is not None:
        bucket_ids = bucket_info
    else:
        # Try to extract hour and map to bucket
        hours = pd.to_datetime(events.index).hour
        from .bucket import assign_bucket
        bucket_ids = pd.Series(hours).apply(assign_bucket).values
    
    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({
        'event': events.values,
        'spread': spread.values,
        'bucket': bucket_ids
    }, index=events.index)
    
    # Calculate statistics per bucket
    summary = []
    for bucket_id in range(1, 11):
        bucket_data = analysis_df[analysis_df['bucket'] == bucket_id]
        bucket_config = get_bucket_info(bucket_id)
        
        if len(bucket_data) > 0:
            event_count = bucket_data['event'].sum()
            event_rate = bucket_data['event'].mean() * 100
            avg_spread = bucket_data['spread'].mean()
            
            # Calculate spread change statistics for events only
            event_data = bucket_data[bucket_data['event']]
            if len(event_data) > 0:
                avg_event_spread = event_data['spread'].mean()
                std_event_spread = event_data['spread'].std()
            else:
                avg_event_spread = np.nan
                std_event_spread = np.nan
        else:
            event_count = 0
            event_rate = 0
            avg_spread = np.nan
            avg_event_spread = np.nan
            std_event_spread = np.nan
        
        summary.append({
            'bucket': bucket_id,
            'label': bucket_config.label,
            'session': bucket_config.session,
            'duration_hours': bucket_config.duration_hours,
            'total_periods': len(bucket_data),
            'event_count': event_count,
            'event_rate_pct': round(event_rate, 2),
            'avg_spread': avg_spread,
            'avg_event_spread': avg_event_spread,
            'std_event_spread': std_event_spread
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Add percentage of total events
    total_events = summary_df['event_count'].sum()
    if total_events > 0:
        summary_df['event_share_pct'] = round(
            summary_df['event_count'] / total_events * 100, 2
        )
    else:
        summary_df['event_share_pct'] = 0
    
    return summary_df


def calculate_preference_scores(events_df: pd.DataFrame, 
                               volume_df: pd.DataFrame) -> pd.Series:
    """
    Calculate preference score adjusting for volume distribution.
    Higher score = more rolls than expected given volume patterns.
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        DataFrame with 'event' and 'bucket' columns
    volume_df : pd.DataFrame
        DataFrame with 'volume' and 'bucket' columns
        
    Returns:
    --------
    pd.Series
        Preference scores per bucket (normalized, mean=1.0)
    """
    # Events per bucket
    event_rate = events_df.groupby('bucket')['event'].mean()
    
    # Volume share per bucket
    volume_by_bucket = volume_df.groupby('bucket')['volume'].sum()
    volume_share = volume_by_bucket / volume_by_bucket.sum()
    
    # Preference score = actual rate / expected rate (based on volume)
    # Avoid division by zero
    volume_share = volume_share.replace(0, 0.0001)
    preference_score = event_rate / volume_share
    
    # Normalize so mean = 1.0
    if preference_score.mean() > 0:
        preference_score = preference_score / preference_score.mean()
    
    return preference_score


def analyze_consecutive_bucket_patterns(events: pd.Series) -> pd.DataFrame:
    """
    Identify patterns in consecutive bucket transitions for roll events.
    
    Parameters:
    -----------
    events : pd.Series
        Boolean series of events with bucket information
        
    Returns:
    --------
    pd.DataFrame
        Transition matrix showing probability of next bucket given current
    """
    # Extract bucket information
    if isinstance(events.index, pd.MultiIndex) and 'bucket' in events.index.names:
        buckets = events.index.get_level_values('bucket')
    else:
        # Try to extract from hour
        hours = pd.to_datetime(events.index).hour
        from .bucket import assign_bucket
        buckets = pd.Series(hours).apply(assign_bucket).values
    
    # Filter to events only
    event_indices = events[events].index
    event_buckets = []
    
    for idx in event_indices:
        if isinstance(events.index, pd.MultiIndex):
            bucket = idx[1] if 'bucket' in events.index.names else None
        else:
            hour = pd.to_datetime(idx).hour
            from .bucket import assign_bucket
            bucket = assign_bucket(hour)
        if bucket:
            event_buckets.append(bucket)
    
    # Create transition pairs
    transitions = []
    for i in range(len(event_buckets) - 1):
        # Check if events are within reasonable time window (same or next day)
        current_bucket = event_buckets[i]
        next_bucket = event_buckets[i + 1]
        transitions.append((current_bucket, next_bucket))
    
    # Create transition matrix
    if transitions:
        trans_df = pd.DataFrame(transitions, columns=['from_bucket', 'to_bucket'])
        transition_matrix = pd.crosstab(
            trans_df['from_bucket'],
            trans_df['to_bucket'],
            normalize='index'
        )
        
        # Ensure all buckets are represented
        for bucket in range(1, 11):
            if bucket not in transition_matrix.index:
                transition_matrix.loc[bucket] = 0
            if bucket not in transition_matrix.columns:
                transition_matrix[bucket] = 0
        
        transition_matrix = transition_matrix.sort_index().sort_index(axis=1)
    else:
        # Empty transition matrix if no consecutive events
        transition_matrix = pd.DataFrame(
            0, 
            index=range(1, 11), 
            columns=range(1, 11)
        )
    
    return transition_matrix


def identify_peak_buckets(events: pd.Series, 
                         threshold_percentile: float = 75) -> List[int]:
    """
    Identify peak rolling buckets based on event frequency.
    
    Parameters:
    -----------
    events : pd.Series
        Boolean series of events
    threshold_percentile : float
        Percentile threshold for identifying peaks (default: 75th percentile)
        
    Returns:
    --------
    list
        List of bucket IDs with peak activity
    """
    # Extract bucket information
    if isinstance(events.index, pd.MultiIndex) and 'bucket' in events.index.names:
        bucket_ids = events.index.get_level_values('bucket')
    else:
        hours = pd.to_datetime(events.index).hour
        from .bucket import assign_bucket
        bucket_ids = pd.Series(hours).apply(assign_bucket).values
    
    # Count events per bucket
    event_counts = pd.Series(index=range(1, 11), dtype=int)
    for bucket_id in range(1, 11):
        bucket_mask = bucket_ids == bucket_id
        event_counts[bucket_id] = events[bucket_mask].sum()
    
    # Identify peaks using percentile threshold
    threshold = event_counts.quantile(threshold_percentile / 100)
    peak_buckets = event_counts[event_counts >= threshold].index.tolist()
    
    return peak_buckets


def generate_bucket_report(events: pd.Series,
                          spread: pd.Series,
                          volume: pd.Series = None) -> Dict:
    """
    Generate comprehensive bucket analysis report.
    
    Parameters:
    -----------
    events : pd.Series
        Boolean series of detected events
    spread : pd.Series
        Spread values
    volume : pd.Series
        Optional volume data
        
    Returns:
    --------
    dict
        Comprehensive analysis results
    """
    report = {}
    
    # Basic statistics
    report['total_events'] = events.sum()
    report['event_rate_pct'] = events.mean() * 100
    
    # Bucket summary
    report['bucket_summary'] = summarize_bucket_events(events, spread)
    
    # Peak buckets
    report['peak_buckets'] = identify_peak_buckets(events)
    
    # Preference scores if volume available
    if volume is not None:
        # Create DataFrame for preference score calculation
        if isinstance(events.index, pd.MultiIndex) and 'bucket' in events.index.names:
            bucket_ids = events.index.get_level_values('bucket')
        else:
            hours = pd.to_datetime(events.index).hour
            from .bucket import assign_bucket
            bucket_ids = pd.Series(hours).apply(assign_bucket).values
        
        events_df = pd.DataFrame({'event': events.values, 'bucket': bucket_ids})
        volume_df = pd.DataFrame({'volume': volume.values, 'bucket': bucket_ids})
        
        report['preference_scores'] = calculate_preference_scores(events_df, volume_df)
    
    # Transition patterns
    report['transition_matrix'] = analyze_consecutive_bucket_patterns(events)
    
    # US vs Off-Peak comparison
    us_buckets = [1, 2, 3, 4, 5, 6, 7]  # US regular hours
    offpeak_buckets = [8, 9, 10]  # Off-peak sessions
    
    summary = report['bucket_summary']
    us_events = summary[summary['bucket'].isin(us_buckets)]['event_count'].sum()
    offpeak_events = summary[summary['bucket'].isin(offpeak_buckets)]['event_count'].sum()
    
    report['us_vs_offpeak'] = {
        'us_events': us_events,
        'us_share_pct': us_events / report['total_events'] * 100 if report['total_events'] > 0 else 0,
        'offpeak_events': offpeak_events,
        'offpeak_share_pct': offpeak_events / report['total_events'] * 100 if report['total_events'] > 0 else 0
    }
    
    return report
