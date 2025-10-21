from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def detect_spread_events(
    spread: pd.Series,
    *,
    method: str = "zscore",
    window: int = 30,
    z_threshold: float = 1.5,
    abs_min: Optional[float] = None,
    min_periods: Optional[int] = None,
    cool_down: Optional[object] = None,
) -> pd.Series:
    """
    Detect calendar-spread widening events using configurable heuristics.

    Parameters
    ----------
    spread:
        Calendar spread series (next - front).
    method:
        ``"zscore"``, ``"abs"``, or ``"combined"``.
    window:
        Rolling window size (in periods) for z-score calculation.
    abs_min:
        Minimum absolute spread change required for an event. Required for ``"abs"`` mode.
    cool_down:
        Optional cool-down constraint. Accepts an ``int`` (periods) or ``pd.Timedelta``.
    """
    change = spread.diff()
    method = method.lower()

    if method == "zscore":
        min_periods = min_periods or max(5, window // 2)
        mu = change.rolling(window, min_periods=min_periods).mean()
        sd = change.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
        z = (change - mu) / sd
        trigger = z > z_threshold
        if abs_min is not None:
            trigger &= change > abs_min
    elif method == "abs":
        if abs_min is None:
            raise ValueError("abs_min is required when method='abs'")
        trigger = change > abs_min
    elif method == "combined":
        if abs_min is None:
            raise ValueError("abs_min is required when method='combined'")
        min_periods = min_periods or max(5, window // 2)
        mu = change.rolling(window, min_periods=min_periods).mean()
        sd = change.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
        z = (change - mu) / sd
        trigger = (z > z_threshold) | (change > abs_min)
    else:
        raise ValueError(f"Unknown detection method: {method}")

    trigger = trigger.fillna(False)
    if cool_down:
        trigger = _apply_cool_down(trigger, cool_down)
    return trigger.rename("spread_widening")


def summarize_events(events: pd.Series, spread: pd.Series) -> pd.DataFrame:
    """Summarise widening events with spread deltas and spacing."""
    event_dates = events[events].index
    if len(event_dates) == 0:
        return pd.DataFrame(columns=["date", "spread_before", "spread_after", "change", "days_since_last"])

    rows: List[dict] = []
    prev_date = None
    for date in event_dates:
        loc = spread.index.get_loc(date)
        before = spread.iloc[loc - 1] if loc > 0 else np.nan
        after = spread.iloc[loc]
        change = after - before
        days_since = (date - prev_date).days if prev_date is not None else np.nan

        rows.append(
            {
                "date": date,
                "spread_before": before,
                "spread_after": after,
                "change": change,
                "days_since_last": days_since,
            }
        )
        prev_date = date

    return pd.DataFrame(rows)


def summarize_bucket_events(
    events: pd.Series,
    spread: pd.Series,
    bucket_ids: pd.Series,
    *,
    bucket_labels: Optional[pd.Series] = None,
    sessions: Optional[pd.Series] = None,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "event": events.values,
            "spread": spread.values,
            "bucket": bucket_ids.values,
        },
        index=events.index,
    )

    summaries: List[dict] = []
    unique_buckets = sorted(set(bucket_ids.dropna().astype(int)))

    for bucket_id in unique_buckets:
        subset = data[data["bucket"] == bucket_id]
        label = bucket_labels.loc[subset.index[0]] if bucket_labels is not None else bucket_id
        session = sessions.loc[subset.index[0]] if sessions is not None else np.nan
        total_periods = len(subset)
        event_count = int(subset["event"].sum())
        event_rate = subset["event"].mean() * 100 if total_periods else 0
        avg_spread = subset["spread"].mean()
        event_subset = subset[subset["event"]]
        avg_event_spread = event_subset["spread"].mean() if not event_subset.empty else np.nan
        std_event_spread = event_subset["spread"].std() if not event_subset.empty else np.nan

        summaries.append(
            {
                "bucket": bucket_id,
                "label": label,
                "session": session,
                "total_periods": total_periods,
                "event_count": event_count,
                "event_rate_pct": round(event_rate, 2),
                "avg_spread": avg_spread,
                "avg_event_spread": avg_event_spread,
                "std_event_spread": std_event_spread,
            }
        )
    result = pd.DataFrame(summaries)
    total_events = result["event_count"].sum()
    if total_events > 0:
        result["event_share_pct"] = (result["event_count"] / total_events * 100).round(2)
    else:
        result["event_share_pct"] = 0.0
    return result


def preference_scores(
    events: pd.Series,
    volumes: pd.Series,
    bucket_ids: pd.Series,
) -> pd.Series:
    events_df = pd.DataFrame({"event": events.values, "bucket": bucket_ids.values})
    volume_df = pd.DataFrame({"volume": volumes.values, "bucket": bucket_ids.values})

    event_rate = events_df.groupby("bucket")["event"].mean()
    volume_by_bucket = volume_df.groupby("bucket")["volume"].sum()
    volume_share = volume_by_bucket / volume_by_bucket.sum()
    volume_share = volume_share.replace(0, 0.0001)

    scores = event_rate / volume_share
    if scores.mean() > 0:
        scores = scores / scores.mean()
    return scores


def transition_matrix(events: pd.Series, bucket_ids: pd.Series) -> pd.DataFrame:
    indices = events[events].index
    relevant_buckets = bucket_ids.loc[indices].dropna()
    if len(relevant_buckets) < 2:
        return pd.DataFrame(0, index=range(1, 11), columns=range(1, 11))

    from_buckets = relevant_buckets.iloc[:-1].astype(int).to_numpy()
    to_buckets = relevant_buckets.iloc[1:].astype(int).to_numpy()

    matrix = pd.crosstab(from_buckets, to_buckets, normalize="index")
    for bucket in range(1, 11):
        if bucket not in matrix.index:
            matrix.loc[bucket] = 0
        if bucket not in matrix.columns:
            matrix[bucket] = 0
    return matrix.sort_index().sort_index(axis=1)


def _apply_cool_down(events: pd.Series, cool_down: object) -> pd.Series:
    events = events.copy()
    index = events.index

    if isinstance(cool_down, (int, float)):
        if cool_down <= 0:
            return events
        last_idx = -int(cool_down) - 1
        filtered = events.copy()
        for i, flag in enumerate(events):
            if not flag:
                continue
            if i - last_idx > cool_down:
                last_idx = i
            else:
                filtered.iat[i] = False
        return filtered

    if isinstance(cool_down, pd.Timedelta):
        last_time = None
        filtered = events.copy()
        for i, (ts, flag) in enumerate(events.items()):
            if not flag:
                continue
            if last_time is None or (ts - last_time) >= cool_down:
                last_time = ts
            else:
                filtered.iat[i] = False
        return filtered

    raise TypeError("cool_down must be int (periods) or pandas.Timedelta")
