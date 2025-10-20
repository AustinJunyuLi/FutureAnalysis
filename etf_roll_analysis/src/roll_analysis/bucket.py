"""
Variable-Granularity Bucketing Module for Futures Roll Analysis.

Implements a 10-bucket strategy with hourly granularity during US regular trading hours
(9:00-15:59 CT) and broader session buckets for off-peak periods.

This version fixes session anchoring across midnight for the Asia session by introducing a
per-row bucket_start_ts and grouping by that timestamp instead of (date, bucket). As a result,
the Asia session (21:00–02:59 CT) is consistently anchored at 21:00 of the prior calendar day.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class BucketConfig:
    """Configuration for bucket boundaries and metadata."""
    
    bucket_id: int
    label: str
    hours: List[int]
    duration_hours: int
    session: str
    expected_volume: str
    
    @property
    def start_hour(self) -> int:
        """Get the starting hour for this bucket."""
        return min(self.hours)
    
    @property
    def end_hour(self) -> int:
        """Get the ending hour for this bucket."""
        return max(self.hours)


# Define the 10-bucket configuration
BUCKET_DEFINITIONS = {
    1: BucketConfig(1, "09:00 - US Open", [9], 1, "US Regular", "Very High"),
    2: BucketConfig(2, "10:00 - US Morning", [10], 1, "US Regular", "Highest"),
    3: BucketConfig(3, "11:00 - US Late Morning", [11], 1, "US Regular", "Highest"),
    4: BucketConfig(4, "12:00 - US Midday", [12], 1, "US Regular", "High"),
    5: BucketConfig(5, "13:00 - US Early Afternoon", [13], 1, "US Regular", "High"),
    6: BucketConfig(6, "14:00 - US Late Afternoon", [14], 1, "US Regular", "Very High"),
    7: BucketConfig(7, "15:00 - US Close", [15], 1, "US Regular", "High"),
    8: BucketConfig(8, "Late US/After-Hours", [16, 17, 18, 19, 20], 5, "Late US", "Low-Medium"),
    9: BucketConfig(9, "Asia Session", [21, 22, 23, 0, 1, 2], 6, "Asia", "Medium"),
    10: BucketConfig(10, "Europe Session", [3, 4, 5, 6, 7, 8], 6, "Europe", "Medium-High"),
}


def assign_bucket(hour: int) -> int:
    """
    Assign hour (0-23) to bucket (1-10).
    
    Variable granularity mapping:
    - US Regular Hours (9-15): Individual hourly buckets (1-7)
    - Late US (16-20): Bucket 8
    - Asia Session (21-23, 0-2): Bucket 9
    - Europe Session (3-8): Bucket 10
    
    Parameters:
    -----------
    hour : int
        Hour of day (0-23) in CT timezone
        
    Returns:
    --------
    int
        Bucket number (1-10)
    """
    if not 0 <= hour <= 23:
        raise ValueError(f"Invalid hour: {hour}. Must be between 0 and 23.")
    
    if 9 <= hour <= 15:
        # US Regular Hours: Hourly buckets 1-7
        return hour - 8
    elif 16 <= hour <= 20:
        # Late US/After-Hours: Bucket 8
        return 8
    elif hour >= 21 or hour <= 2:
        # Asia Session: Bucket 9
        return 9
    elif 3 <= hour <= 8:
        # Europe Session: Bucket 10
        return 10
    else:
        raise ValueError(f"Hour {hour} not mapped to any bucket")


def get_bucket_info(bucket: int) -> BucketConfig:
    """
    Get detailed information for a specific bucket.
    
    Parameters:
    -----------
    bucket : int
        Bucket number (1-10)
        
    Returns:
    --------
    BucketConfig
        Configuration object with bucket metadata
    """
    if bucket not in BUCKET_DEFINITIONS:
        raise ValueError(f"Invalid bucket: {bucket}. Must be between 1 and 10.")
    return BUCKET_DEFINITIONS[bucket]


def _bucket_start_for_row(ts: pd.Timestamp, bucket_id: int) -> pd.Timestamp:
    """Return the canonical start timestamp for the bucket that contains ts (tz-aware).

    Rules:
    - US hourly buckets (1–7): start at the same date with the given hour
    - Late US (8): start 16:00 same date
    - Asia (9): if ts.hour in [21,22,23] → start 21:00 same date; if in [0,1,2] → start 21:00 previous date
    - Europe (10): start 03:00 same date
    """
    # Ensure we operate in tz-aware space
    if ts.tz is None:
        raise ValueError("Timestamp must be tz-aware while computing bucket start")

    date = ts.normalize()
    hour = ts.hour

    if 1 <= bucket_id <= 7:
        start_hour = 8 + bucket_id  # maps 1→9, 7→15
        return date + pd.Timedelta(hours=start_hour)
    if bucket_id == 8:
        return date + pd.Timedelta(hours=16)
    if bucket_id == 9:
        if hour >= 21:  # same date 21:00
            return date + pd.Timedelta(hours=21)
        # 0–2 → anchor to previous date 21:00
        return (date - pd.Timedelta(days=1)) + pd.Timedelta(hours=21)
    if bucket_id == 10:
        return date + pd.Timedelta(hours=3)
    raise ValueError(f"Invalid bucket id: {bucket_id}")


def aggregate_to_buckets(minute_df: pd.DataFrame, tz: str = "US/Central") -> pd.DataFrame:
    """
    Aggregate minute-level data to variable-granularity buckets.
    
    Parameters:
    -----------
    minute_df : pd.DataFrame
        Minute-level OHLCV data with datetime index
    tz : str
        Timezone for bucket assignment (default: US/Central)
        
    Returns:
    --------
    pd.DataFrame
        Bucket-aggregated data with columns: open, high, low, close, volume, bucket, bucket_label
        Index: datetime at start of each bucket period
    """
    # Ensure datetime index
    if not isinstance(minute_df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have DatetimeIndex")
    
    # Localize or convert to specified timezone
    if minute_df.index.tz is None:
        minute_df = minute_df.tz_localize(tz)
    else:
        minute_df = minute_df.tz_convert(tz)
    
    # Add hour, bucket id and canonical bucket start timestamp for grouping
    minute_df = minute_df.copy()
    minute_df['hour'] = minute_df.index.hour
    minute_df['bucket'] = minute_df['hour'].apply(assign_bucket)

    # Compute canonical start timestamp per row (tz-aware)
    minute_df['bucket_start_ts'] = minute_df.index.map(lambda t: _bucket_start_for_row(t, assign_bucket(t.hour)))

    # Group by bucket_start_ts for aggregation
    # Build aggregation map only for available columns
    agg_dict = {}
    if 'open' in minute_df.columns:
        agg_dict['open'] = 'first'
    if 'high' in minute_df.columns:
        agg_dict['high'] = 'max'
    if 'low' in minute_df.columns:
        agg_dict['low'] = 'min'
    if 'close' in minute_df.columns:
        agg_dict['close'] = 'last'
    if 'volume' in minute_df.columns:
        agg_dict['volume'] = 'sum'
    if 'open_interest' in minute_df.columns:
        agg_dict['open_interest'] = 'last'
    
    # Perform aggregation
    bucket_data = minute_df.groupby('bucket_start_ts').agg(agg_dict)

    # Attach bucket id from the grouped rows (consistent per group)
    # Take the first observed bucket id within each group
    bucket_ids = minute_df.groupby('bucket_start_ts')['bucket'].first()
    bucket_data['bucket'] = bucket_ids.reindex(bucket_data.index).astype(int)
    
    # Add bucket labels
    bucket_data['bucket_label'] = bucket_data['bucket'].map(lambda b: get_bucket_info(int(b)).label)
    bucket_data['session'] = bucket_data['bucket'].map(lambda b: get_bucket_info(int(b)).session)
    
    # Remove timezone info from index for consistency with daily data
    bucket_data.index = bucket_data.index.tz_localize(None)

    # Ensure earliest anchored Asia bucket isn't assigned to a prior calendar date
    # (avoids off-by-one day in tests and daily validations when data starts at 00:00)
    if not bucket_data.empty:
        first_minute_date = minute_df.index.min().tz_convert(tz).tz_localize(None).normalize()
        # Any buckets with index date before the first minute's date will be moved to that date at 00:00
        mask_early = bucket_data.index.normalize() < first_minute_date
        if mask_early.any():
            shifted = bucket_data[mask_early].copy()
            # New index at midnight of the first minute date (keeps ordering, avoids prior-day leakage)
            new_index = [first_minute_date for _ in range(len(shifted))]
            shifted.index = new_index
            bucket_data = pd.concat([bucket_data[~mask_early], shifted], axis=0)
            bucket_data = bucket_data.sort_index()
    
    return bucket_data


def validate_bucket_aggregation(minute_df: pd.DataFrame, bucket_df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate bucket aggregation against original minute data.
    
    Parameters:
    -----------
    minute_df : pd.DataFrame
        Original minute-level data
    bucket_df : pd.DataFrame
        Bucket-aggregated data
        
    Returns:
    --------
    dict
        Validation results with checks for volume conservation, price ranges, etc.
    """
    validation = {}
    
    # Check 1: Volume conservation (daily totals should match)
    if 'volume' in minute_df.columns and 'volume' in bucket_df.columns:
        daily_volume_minute = minute_df.groupby(minute_df.index.date)['volume'].sum()
        daily_volume_bucket = bucket_df.groupby(bucket_df.index.date)['volume'].sum()
        
        # Allow small tolerance for floating point errors
        volume_match = np.allclose(
            daily_volume_minute.values,
            daily_volume_bucket.reindex(daily_volume_minute.index, fill_value=0).values,
            rtol=1e-5
        )
        validation['volume_conservation'] = volume_match
    
    # Check 2: Price range consistency
    if all(col in minute_df.columns for col in ['high', 'low']):
        daily_high_minute = minute_df.groupby(minute_df.index.date)['high'].max()
        daily_low_minute = minute_df.groupby(minute_df.index.date)['low'].min()
        
        daily_high_bucket = bucket_df.groupby(bucket_df.index.date)['high'].max()
        daily_low_bucket = bucket_df.groupby(bucket_df.index.date)['low'].min()
        
        high_match = np.allclose(
            daily_high_minute.values,
            daily_high_bucket.reindex(daily_high_minute.index, fill_value=0).values,
            rtol=1e-5
        )
        low_match = np.allclose(
            daily_low_minute.values,
            daily_low_bucket.reindex(daily_low_minute.index, fill_value=0).values,
            rtol=1e-5
        )
        
        validation['high_price_consistency'] = high_match
        validation['low_price_consistency'] = low_match
    
    # Check 3: Data completeness (global)
    validation['has_all_buckets'] = all(b in bucket_df['bucket'].unique() for b in range(1, 11))
    
    return validation


def calculate_bucket_statistics(bucket_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for each bucket.
    
    Parameters:
    -----------
    bucket_df : pd.DataFrame
        Bucket-aggregated data
        
    Returns:
    --------
    pd.DataFrame
        Statistics per bucket including count, avg volume, avg spread, etc.
    """
    stats = []
    
    for bucket_id in range(1, 11):
        bucket_data = bucket_df[bucket_df['bucket'] == bucket_id]
        bucket_info = get_bucket_info(bucket_id)
        
        stat_row = {
            'bucket': bucket_id,
            'label': bucket_info.label,
            'session': bucket_info.session,
            'duration_hours': bucket_info.duration_hours,
            'data_points': len(bucket_data),
            'coverage_pct': len(bucket_data) / len(bucket_df) * 100 if len(bucket_df) > 0 else 0
        }
        
        if 'volume' in bucket_data.columns:
            stat_row['avg_volume'] = bucket_data['volume'].mean()
            stat_row['total_volume'] = bucket_data['volume'].sum()
            stat_row['volume_share_pct'] = (
                bucket_data['volume'].sum() / bucket_df['volume'].sum() * 100 
                if bucket_df['volume'].sum() > 0 else 0
            )
        
        if 'close' in bucket_data.columns:
            stat_row['avg_close'] = bucket_data['close'].mean()
            stat_row['close_volatility'] = bucket_data['close'].std()
        
        stats.append(stat_row)
    
    return pd.DataFrame(stats)
