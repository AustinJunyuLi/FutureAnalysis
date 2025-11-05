from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BucketConfig:
    bucket_id: int
    label: str
    hours: List[int]
    duration_hours: int
    session: str
    expected_volume: str

    @property
    def start_hour(self) -> int:
        return min(self.hours)

    @property
    def end_hour(self) -> int:
        return max(self.hours)


DEFAULT_BUCKETS: Dict[int, BucketConfig] = {
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
    """Return the bucket id for a Central Time hour (0-23)."""
    if not 0 <= hour <= 23:
        raise ValueError(f"Invalid hour {hour}; must be 0-23.")
    if 9 <= hour <= 15:
        return hour - 8
    if 16 <= hour <= 20:
        return 8
    if hour >= 21 or hour <= 2:
        return 9
    if 3 <= hour <= 8:
        return 10
    raise ValueError(f"Hour {hour} not mapped to any bucket")


def get_bucket_info(bucket_id: int) -> BucketConfig:
    try:
        return DEFAULT_BUCKETS[bucket_id]
    except KeyError as exc:  # pragma: no cover - defensive clause
        raise ValueError(f"Invalid bucket id {bucket_id}") from exc


def aggregate_to_buckets(minute_df: pd.DataFrame, tz: str = "US/Central") -> pd.DataFrame:
    """
    Aggregate minute-level OHLCV data to variable-granularity buckets.

    Returns
    -------
    DataFrame indexed by bucket start timestamps with OHLCV data and bucket metadata.
    """
    if not isinstance(minute_df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have DatetimeIndex")

    if minute_df.index.tz is None:
        minute_df = minute_df.tz_localize(tz)
    else:
        minute_df = minute_df.tz_convert(tz)

    if minute_df.empty:
        columns = {
            col: pd.Series(dtype=float) for col in ["open", "high", "low", "close", "volume"] if col in minute_df.columns
        }
        result = pd.DataFrame(columns)
        result["bucket"] = pd.Series(dtype=int)
        result["bucket_label"] = pd.Series(dtype=str)
        result["session"] = pd.Series(dtype=str)
        return result

    df = minute_df.copy()
    hours = df.index.hour
    bucket_ids = np.vectorize(assign_bucket)(hours)
    df["bucket"] = bucket_ids
    df["bucket_start_ts"] = _bucket_start_vectorised(df.index, bucket_ids)

    agg_spec = {}
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            agg_spec[col] = "first" if col == "open" else "last" if col == "close" else "max" if col == "high" else "min"
    if "volume" in df.columns:
        agg_spec["volume"] = "sum"
    if "open_interest" in df.columns:
        agg_spec["open_interest"] = "last"

    grouped = df.groupby("bucket_start_ts").agg(agg_spec)
    grouped["bucket"] = df.groupby("bucket_start_ts")["bucket"].first()
    grouped["bucket_label"] = grouped["bucket"].map(lambda b: get_bucket_info(int(b)).label)
    grouped["session"] = grouped["bucket"].map(lambda b: get_bucket_info(int(b)).session)
    grouped.index = grouped.index.tz_localize(None)

    return grouped


def calculate_bucket_statistics(bucket_df: pd.DataFrame) -> pd.DataFrame:
    stats: List[Dict[str, object]] = []
    for bucket_id in range(1, 11):
        subset = bucket_df[bucket_df["bucket"] == bucket_id]
        info = get_bucket_info(bucket_id)
        row: Dict[str, object] = {
            "bucket": bucket_id,
            "label": info.label,
            "session": info.session,
            "duration_hours": info.duration_hours,
            "data_points": len(subset),
            "coverage_pct": (len(subset) / len(bucket_df) * 100) if len(bucket_df) else 0.0,
        }
        if "volume" in subset.columns and bucket_df["volume"].sum() > 0:
            total_volume = bucket_df["volume"].sum()
            row["avg_volume"] = subset["volume"].mean()
            row["total_volume"] = subset["volume"].sum()
            row["volume_share_pct"] = subset["volume"].sum() / total_volume * 100
        if "close" in subset.columns:
            row["avg_close"] = subset["close"].mean()
            row["close_volatility"] = subset["close"].std()
        stats.append(row)
    return pd.DataFrame(stats)


def _bucket_start_vectorised(index: pd.DatetimeIndex, bucket_ids: Iterable[int]) -> pd.DatetimeIndex:
    dates = index.normalize()
    hours = index.hour
    bucket_ids = np.asarray(bucket_ids)
    start_hours = np.where(
        (bucket_ids >= 1) & (bucket_ids <= 7),
        8 + bucket_ids,
        np.where(bucket_ids == 8, 16, np.where(bucket_ids == 9, 21, 3)),
    )

    start_dates = dates.copy()
    # Asia session buckets (id 9) for hours 0-2 should anchor to previous day 21:00
    asia_mask = bucket_ids == 9
    prev_day_mask = asia_mask & np.isin(hours, [0, 1, 2])
    start_dates = start_dates.where(~prev_day_mask, start_dates - pd.Timedelta(days=1))

    start_times = start_dates + pd.to_timedelta(start_hours, unit="h")
    return start_times


def validate_bucket_aggregation(minute_df: pd.DataFrame, bucket_df: pd.DataFrame) -> Dict[str, bool]:
    """Validate bucket aggregation against original minute-level data."""
    validation: Dict[str, bool] = {}

    if "volume" in minute_df.columns and "volume" in bucket_df.columns:
        minute_trade_dates = _compute_trading_dates(minute_df.index)
        bucket_trade_dates = _compute_trading_dates(bucket_df.index, bucket_df["bucket"])
        daily_volume_minute = minute_df.groupby(minute_trade_dates)["volume"].sum()
        daily_volume_bucket = bucket_df.groupby(bucket_trade_dates)["volume"].sum()
        validation["volume_conservation"] = np.allclose(
            daily_volume_minute.values,
            daily_volume_bucket.reindex(daily_volume_minute.index, fill_value=0).values,
            rtol=1e-5,
        )

    if all(col in minute_df.columns for col in ["high", "low"]):
        minute_trade_dates = _compute_trading_dates(minute_df.index)
        bucket_trade_dates = _compute_trading_dates(bucket_df.index, bucket_df["bucket"])
        daily_high_minute = minute_df.groupby(minute_trade_dates)["high"].max()
        daily_low_minute = minute_df.groupby(minute_trade_dates)["low"].min()
        daily_high_bucket = bucket_df.groupby(bucket_trade_dates)["high"].max()
        daily_low_bucket = bucket_df.groupby(bucket_trade_dates)["low"].min()

        validation["high_price_consistency"] = np.allclose(
            daily_high_minute.values,
            daily_high_bucket.reindex(daily_high_minute.index, fill_value=0).values,
            rtol=1e-5,
        )
        validation["low_price_consistency"] = np.allclose(
            daily_low_minute.values,
            daily_low_bucket.reindex(daily_low_minute.index, fill_value=0).values,
            rtol=1e-5,
        )

    validation["has_all_buckets"] = all(b in bucket_df["bucket"].unique() for b in range(1, 11))
    return validation


def _compute_trading_dates(index: pd.DatetimeIndex, bucket_ids: Optional[pd.Series] = None) -> pd.Index:
    """
    Map timestamps (and optionally bucket IDs) to the corresponding trading date.

    Asia session buckets (id 9) start the next trading day at 21:00 prior day,
    so they are attributed to the following calendar date.
    """
    # Preserve wall-clock (local) dates/hours for mapping
    normalized = index.tz_localize(None) if index.tz is not None else index
    normalized = normalized.normalize()
    if bucket_ids is not None:
        shift = bucket_ids.eq(9).astype(int)
    else:
        hours = index.tz_localize(None).hour if index.tz is not None else index.hour
        shift = (hours >= 21).astype(int)
    return normalized + pd.to_timedelta(shift, unit="D")
