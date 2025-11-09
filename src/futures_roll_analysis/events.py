from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .trading_days import business_day_gaps as _business_day_gaps


def preprocess_spread(
    spread: pd.Series,
    *,
    clip_quantile: Optional[float] = None,
    ema_span: Optional[int] = None,
) -> pd.Series:
    """Apply optional clipping/smoothing to reduce microstructure noise."""

    series = spread.copy()
    if clip_quantile is not None and 0 < clip_quantile < 0.5:
        lower = series.quantile(clip_quantile)
        upper = series.quantile(1 - clip_quantile)
        series = series.clip(lower=lower, upper=upper)

    if ema_span is not None and ema_span > 1:
        series = series.ewm(span=ema_span, adjust=False).mean()

    return series


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


def detect_multi_spread_events(
    spreads_df: pd.DataFrame,
    *,
    method: str = "zscore",
    window: int = 30,
    z_threshold: float = 1.5,
    abs_min: Optional[float] = None,
    min_periods: Optional[int] = None,
    cool_down: Optional[object] = None,
) -> pd.DataFrame:
    """
    Detect widening events across multiple spread series.

    Applies detect_spread_events() to each column in the spreads DataFrame.

    Parameters
    ----------
    spreads_df:
        DataFrame with spread columns (e.g., S1, S2, ..., S11).
    method:
        ``"zscore"``, ``"abs"``, or ``"combined"``.
    window:
        Rolling window size (in periods) for z-score calculation.
    z_threshold:
        Z-score threshold for detection.
    abs_min:
        Minimum absolute spread change required for an event.
    min_periods:
        Minimum periods for rolling window calculations.
    cool_down:
        Optional cool-down constraint. Accepts an ``int`` (periods) or ``pd.Timedelta``.

    Returns
    -------
    DataFrame with boolean event columns (e.g., S1_events, S2_events, ..., S11_events).
    """
    result_dict = {}

    for col in spreads_df.columns:
        spread_series = spreads_df[col]
        events = detect_spread_events(
            spread_series,
            method=method,
            window=window,
            z_threshold=z_threshold,
            abs_min=abs_min,
            min_periods=min_periods,
            cool_down=cool_down,
        )
        result_dict[f"{col}_events"] = events

    result = pd.DataFrame(result_dict, index=spreads_df.index)
    return result


def confirm_roll_events(
    primary: pd.Series,
    additional_signals: Optional[Dict[str, Optional[pd.Series]]] = None,
    *,
    min_signals: int = 1,
) -> pd.Series:
    """Require a minimum number of boolean signals (including primary) to confirm events."""

    signals = {"spread": primary.astype(bool)}
    if additional_signals:
        for name, series in additional_signals.items():
            if series is None:
                continue
            aligned = series.reindex(primary.index).fillna(False).astype(bool)
            signals[name] = aligned

    signals_df = pd.DataFrame(signals)
    effective_min = min(max(min_signals, 1), signals_df.shape[1])
    counts = signals_df.astype(int).sum(axis=1)
    mask = (counts >= effective_min) & signals_df["spread"]
    return signals_df["spread"] & mask


def align_events_to_business_days(
    events: pd.Series,
    business_days: Optional[pd.DatetimeIndex],
    policy: str = "none",
) -> pd.Series:
    """
    Optionally align event timestamps to business days.

    Policies
    --------
    - "none": return as-is.
    - "shift_next": if event falls on a non-business day, move it to the next business day.
    - "drop_closed": drop events on non-business days.
    """
    if not policy or policy == "none" or business_days is None or len(business_days) == 0:
        return events
    biz = pd.DatetimeIndex(business_days).normalize().drop_duplicates().sort_values()
    biz_set = set(biz)
    out = events.copy()
    if policy == "drop_closed":
        mask = out.index.normalize().isin(biz_set)
        return out[mask]
    if policy == "shift_next":
        new_index = []
        for ts in out.index:
            d = ts.normalize()
            if d in biz_set:
                new_index.append(ts)
                continue
            # find next business date >= d
            pos = biz.searchsorted(d)
            if pos < len(biz):
                new_index.append(biz[pos].replace(hour=ts.hour, minute=ts.minute, second=ts.second))
            else:
                # no next biz day; keep original
                new_index.append(ts)
        out.index = pd.DatetimeIndex(new_index)
        return out
    raise ValueError(f"Unknown alignment policy: {policy}")


def summarize_events(
    events: pd.Series,
    spread: pd.Series,
    business_days: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Summarise widening events with spread deltas and business-day spacing only.

    Parameters
    ----------
    events:
        Boolean series of events.
    spread:
        Calendar spread series.
    business_days:
        Optional DatetimeIndex of business days for computing business day gaps.

    Returns
    -------
    DataFrame with event summary including business-day gaps.
    """
    event_dates = events[events].index
    if len(event_dates) == 0:
        base_cols = ["date", "spread_before", "spread_after", "change", "business_days_since_last"]
        return pd.DataFrame(columns=base_cols)

    rows: List[dict] = []
    biz_gap_series = _business_day_gaps(events, business_days) if business_days is not None else None
    biz_iter = iter(biz_gap_series) if biz_gap_series is not None else None

    for date in event_dates:
        if spread.empty:
            before = np.nan
            after = np.nan
        else:
            try:
                loc = spread.index.get_loc(date)
                before = spread.iloc[loc - 1] if loc > 0 else np.nan
                after = spread.iloc[loc]
            except KeyError:
                pos = spread.index.searchsorted(date)
                before = spread.iloc[pos - 1] if pos - 1 >= 0 else np.nan
                if pos >= len(spread):
                    after = spread.iloc[-1]
                else:
                    after = spread.iloc[pos]
        change = after - before

        row = {
            "date": date,
            "spread_before": before,
            "spread_after": after,
            "change": change,
            "business_days_since_last": (next(biz_iter) if biz_iter is not None else np.nan),
        }
        rows.append(row)

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
    total_volume = volume_by_bucket.sum()
    if total_volume <= 0 or pd.isna(total_volume):
        return pd.Series(0.0, index=event_rate.index, name="preference")

    volume_share = volume_by_bucket / total_volume
    volume_share = volume_share.replace(0, 0.0001)

    scores = event_rate / volume_share
    if scores.mean() > 0:
        scores = scores / scores.mean()
    scores.name = "preference"
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
            if last_time is None or (ts - last_time) > cool_down:
                last_time = ts
            else:
                filtered.iat[i] = False
        return filtered

    raise TypeError("cool_down must be int (periods) or pandas.Timedelta")
