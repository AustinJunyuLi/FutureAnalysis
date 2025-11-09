from __future__ import annotations

from typing import Optional, Set

import numpy as np
import pandas as pd


def summarize_strip_dominance(
    magnitude_comparison: pd.DataFrame,
    contract_chain: pd.DataFrame,
    expiry_map: pd.Series,
    *,
    calendar: Optional[pd.DataFrame] = None,
    dominance_threshold: float = 2.0,
    expiry_window_bd: int = 18,
) -> pd.DataFrame:
    """
    Summarize S1 dominance vs other spreads for each front-contract day.

    Parameters
    ----------
    magnitude_comparison:
        Output from multi_spread_analysis.compare_spread_magnitudes().
    contract_chain:
        DataFrame with columns F1, F2, ..., aligning with magnitude_comparison index.
    expiry_map:
        Series mapping contracts to expiry dates.
    calendar:
        Optional trading calendar DataFrame with 'date' and 'is_trading_day' columns.
    dominance_threshold:
        Threshold for dominance_ratio to mark S1 dominance.
    expiry_window_bd:
        Number of business days before expiry to classify dominance as expiry-driven.
    """
    if magnitude_comparison.empty:
        return pd.DataFrame(
            columns=[
                "front_contract",
                "date",
                "s1_change",
                "other_median_change",
                "dominance_ratio",
                "business_days_to_expiry",
                "classification",
            ]
        )

    df = magnitude_comparison.copy()
    df["front_contract"] = contract_chain["F1"].reindex(df.index)

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    df["date"] = idx.normalize()

    expiry_series = df["front_contract"].map(expiry_map)
    max_expiry = expiry_series.dropna().max()
    start_date = df["date"].min()
    end_date = max(df["date"].max(), max_expiry) if pd.notna(max_expiry) else df["date"].max()

    if calendar is not None:
        open_days = _determine_open_days(calendar, start_date, end_date)
        df["business_days_to_expiry"] = [
            _business_days_to_expiry(date, expiry, open_days) for date, expiry in zip(df["date"], expiry_series)
        ]
    else:
        # Fall back to regular calendar days when no trading calendar is provided
        df["business_days_to_expiry"] = [
            (expiry - date).days if pd.notna(expiry) else None
            for date, expiry in zip(df["date"], expiry_series)
        ]

    grouped = df.groupby(["front_contract", "date"]).agg(
        s1_change=("s1_change", "median"),
        other_median_change=("others_median", "median"),
        dominance_ratio=("dominance_ratio", "max"),
        business_days_to_expiry=("business_days_to_expiry", "first"),
    )
    grouped = grouped.reset_index()

    grouped["dominance_ratio"] = grouped["dominance_ratio"].fillna(0.0)
    grouped["s1_dominates"] = grouped["dominance_ratio"] >= dominance_threshold
    grouped["expiry_window"] = grouped["business_days_to_expiry"].le(expiry_window_bd)

    grouped["classification"] = np.select(
        [
            grouped["s1_dominates"] & grouped["expiry_window"],
            grouped["s1_dominates"] & ~grouped["expiry_window"],
        ],
        ["expiry_dominance", "broad_roll"],
        default="normal",
    )

    return grouped[
        [
            "front_contract",
            "date",
            "s1_change",
            "other_median_change",
            "dominance_ratio",
            "business_days_to_expiry",
            "classification",
        ]
    ]


def filter_expiry_dominance_events(
    events: pd.Series,
    diagnostics: pd.DataFrame,
) -> tuple[pd.Series, int]:
    """
    Remove events that occur on days classified as expiry-dominant.

    Returns the filtered events and the number of removed points.
    """
    if diagnostics.empty or events.empty:
        return events, 0

    expiry_dates: Set[pd.Timestamp] = set(
        diagnostics.loc[diagnostics["classification"] == "expiry_dominance", "date"]
    )
    if not expiry_dates:
        return events, 0

    idx = pd.DatetimeIndex(events.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    mask = idx.normalize().isin(expiry_dates)
    removed = int((events & mask).sum())
    filtered = events & ~mask
    return filtered, removed


def _determine_open_days(calendar: Optional[pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Determine trading days from calendar.

    Strict mode: Requires a valid calendar. No fallback to weekdays.

    Raises
    ------
    ValueError
        If calendar is None or missing required columns.
    """
    if calendar is None:
        raise ValueError(
            "Trading calendar is required for strip analysis. "
            "Please provide business_days.calendar_paths in settings."
        )
    if "is_trading_day" not in calendar.columns:
        raise ValueError(
            "Trading calendar missing 'is_trading_day' column. "
            "Please check calendar file format."
        )

    days = pd.date_range(start, end, freq="D")
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    cal = cal.dropna(subset=["date"]).drop_duplicates("date").set_index("date")

    # Strict mode: only use calendar, no weekday fallback
    open_days = []
    for day in days:
        if day in cal.index:
            if bool(cal.at[day, "is_trading_day"]):
                open_days.append(day)
        # No fallback to weekdays - dates must be in calendar

    return pd.DatetimeIndex(open_days)


def _business_days_to_expiry(date: pd.Timestamp, expiry: Optional[pd.Timestamp], open_days: pd.DatetimeIndex) -> float:
    if pd.isna(expiry) or expiry < date:
        return np.nan
    mask = (open_days > date) & (open_days <= expiry)
    return int(mask.sum())
