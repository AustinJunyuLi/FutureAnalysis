from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _ensure_paths(paths: Union[Path, str, Sequence[Union[Path, str]]]) -> List[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def load_calendar(
    paths: Union[Path, str, Sequence[Union[Path, str]]],
    hierarchy: str = "override",
) -> pd.DataFrame:
    """Load and merge trading calendars.

    Parameters
    ----------
    paths
        Path or list of paths to calendar CSV files.
    hierarchy
        Merge strategy when multiple calendars are supplied:
        ``override`` (first file wins), ``union`` (any closed day),
        ``intersection`` (only days open in every calendar).
    """

    calendar_paths = _ensure_paths(paths)
    if not calendar_paths:
        raise ValueError("At least one calendar path must be provided")

    calendars: List[pd.DataFrame] = []
    for path in calendar_paths:
        resolved = path.resolve()
        if not resolved.exists():
            LOGGER.warning("Calendar file not found: %s", resolved)
            continue

        # Skip comment lines when reading CSV
        df = pd.read_csv(resolved, comment='#')
        if "date" not in df.columns:
            raise ValueError(f"Calendar {resolved} missing 'date' column")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"]).reset_index(drop=True)

        for column in ("holiday_name", "session_note", "partial_hours"):
            if column not in df.columns:
                df[column] = ""
            df[column] = df[column].fillna("").astype(str)

        session_note = df["session_note"].str.strip().str.lower()
        # Normalize common aliases
        session_note = session_note.replace({"open": "regular"})
        holiday_name = df["holiday_name"].str.strip()
        # Validation of session_note values
        allowed = {"", "regular", "early close", "closed", "full close"}
        unknown_mask = ~session_note.isin(allowed)
        if unknown_mask.any():
            bad = sorted(session_note[unknown_mask].unique())
            raise ValueError(f"Unknown session_note values in {resolved.name}: {bad}")

        is_closed_note = session_note.isin({"closed", "full close"})

        # Treat days as trading days by default unless explicitly marked closed.
        # Early-close days are trading days (handled downstream via partials).
        df["is_trading_day"] = ~is_closed_note

        # If a holiday has no explicit session_note, assume open but log once.
        if ((holiday_name.str.len() > 0) & session_note.eq("")).any():
            LOGGER.warning(
                "Calendar rows with holiday_name but empty session_note found; assuming open: %s",
                resolved,
            )
        calendars.append(df[["date", "holiday_name", "session_note", "partial_hours", "is_trading_day"]])

    if not calendars:
        raise FileNotFoundError(f"No valid calendar files found in: {calendar_paths}")

    if hierarchy not in {"override", "union", "intersection"}:
        raise ValueError("Unknown hierarchy: %s" % hierarchy)

    if len(calendars) == 1:
        calendar = calendars[0]
    elif hierarchy == "override":
        calendar = calendars[0].copy()
        seen = set(calendar["date"])
        for df in calendars[1:]:
            mask = ~df["date"].isin(seen)
            calendar = pd.concat([calendar, df[mask]], ignore_index=True)
            seen.update(df.loc[mask, "date"].tolist())
    elif hierarchy == "union":
        combined = pd.concat(calendars, ignore_index=True)
        calendar = combined.groupby("date", sort=True).agg(
            holiday_name=("holiday_name", lambda x: x[x != ""].iloc[0] if (x != "").any() else ""),
            session_note=("session_note", lambda x: x[x != ""].iloc[0] if (x != "").any() else ""),
            partial_hours=("partial_hours", lambda x: x[x != ""].iloc[0] if (x != "").any() else ""),
            is_trading_day=("is_trading_day", "min"),
        ).reset_index()
    else:  # intersection
        base = calendars[0].set_index("date")
        for df in calendars[1:]:
            df_indexed = df.set_index("date")
            base = base.join(df_indexed, how="inner", rsuffix="_r")
            base["is_trading_day"] = base["is_trading_day"] & base["is_trading_day_r"]
            base = base[["holiday_name", "session_note", "partial_hours", "is_trading_day"]]
        calendar = base.reset_index()

    calendar = calendar.sort_values("date").reset_index(drop=True)
    return calendar


def load_calendar_hierarchy(
    commodity: str,
    exchange: str = "CME",
    calendar_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load calendars using commodity/exchange hierarchy."""

    if calendar_dir is None:
        calendar_dir = Path(__file__).resolve().parents[2] / "metadata" / "calendars"
    else:
        calendar_dir = Path(calendar_dir)

    candidates = [
        calendar_dir / f"{exchange.lower()}_{commodity.lower()}_holidays.csv",
        calendar_dir / f"{exchange.lower()}_{commodity.lower()}.csv",
        calendar_dir / f"{exchange.lower()}_globex_holidays.csv",
        calendar_dir / f"{exchange.lower()}_globex.csv",
        calendar_dir / "global_futures.csv",
    ]

    found = [path for path in candidates if path.exists()]
    if not found:
        raise FileNotFoundError(
            f"No calendar files found for {commodity} on {exchange}. "
            f"Searched: {[str(p) for p in candidates]}"
        )

    LOGGER.info("Loading calendar hierarchy for %s: %s", commodity, [p.name for p in found])
    return load_calendar(found, hierarchy="override")


def trading_day_index(
    ts_utc: pd.DatetimeIndex,
    anchor_local: str = "17:00",
    tz_exchange: str = "America/Chicago",
) -> pd.DatetimeIndex:
    """
    Map UTC timestamps to trading-day dates based on a local-time anchor.

    Parameters
    ----------
    ts_utc:
        UTC tz-aware DatetimeIndex.
    anchor_local:
        Local time (HH:MM) at/after which the trading day is considered the same
        calendar date, otherwise previous calendar date.
    tz_exchange:
        IANA timezone name for the exchange (e.g., America/Chicago).

    Returns
    -------
    Naive DatetimeIndex of dates representing trading days suitable for grouping.
    """
    idx = pd.DatetimeIndex(ts_utc)
    if idx.tz is None:
        raise ValueError("trading_day_index requires tz-aware UTC timestamps")
    local = idx.tz_convert(tz_exchange)
    try:
        h, m = map(int, anchor_local.split(":"))
    except Exception as e:
        raise ValueError(f"Invalid anchor_local '{anchor_local}': {e}")
    is_same = (local.hour > h) | ((local.hour == h) & (local.minute >= m))
    base = local.normalize()
    td = base.where(is_same, base - pd.Timedelta(days=1))
    return pd.DatetimeIndex(td.tz_localize(None))


def map_to_trading_date(
    index: pd.DatetimeIndex,
    bucket_ids: Optional[pd.Series] = None,
) -> pd.DatetimeIndex:
    """Map timestamps to trading dates (Asia session anchors at prior 21:00)."""

    from .buckets import _compute_trading_dates

    return _compute_trading_dates(index, bucket_ids)


def _normalize_index(index: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx.normalize()


def _extract_total_volume(panel: pd.DataFrame, front_next: pd.DataFrame) -> pd.Series:
    if not isinstance(panel.columns, pd.MultiIndex):
        return pd.Series(0.0, index=panel.index)

    volume_cols = [col for col in panel.columns if col[0] != "meta" and col[1] == "volume"]
    if not volume_cols:
        return pd.Series(0.0, index=panel.index)

    volume_df = panel.loc[:, volume_cols].copy()
    volume_df.columns = [col[0] for col in volume_cols]

    contracts = volume_df.columns.tolist()
    contract_to_pos = {contract: pos for pos, contract in enumerate(contracts)}
    values = volume_df.to_numpy(dtype=float, copy=False)
    row_indices = np.arange(len(volume_df))

    front_contracts = front_next["front_contract"].reindex(panel.index)
    next_contracts = front_next["next_contract"].reindex(panel.index)

    total = np.zeros(len(volume_df), dtype=float)

    front_idx = np.array([contract_to_pos.get(c, -1) for c in front_contracts], dtype=int)
    mask_front = front_idx >= 0
    if mask_front.any():
        total[mask_front] += values[mask_front, front_idx[mask_front]]

    next_idx = np.array([contract_to_pos.get(c, -1) for c in next_contracts], dtype=int)
    mask_next = next_idx >= 0
    if mask_next.any():
        total[mask_next] += values[mask_next, next_idx[mask_next]]

    return pd.Series(total, index=panel.index)


def _compute_hours_to_expiry(
    index: pd.DatetimeIndex,
    front_next: pd.DataFrame,
    expiry_map: pd.Series,
) -> pd.Series:
    """
    Compute hours remaining until expiry for each timestamp.

    Uses hour-based precision for better alignment with intraday expiry times.
    """
    if expiry_map is None:
        raise ValueError("expiry_map must be provided to compute business days")

    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)

    front_contracts = front_next["front_contract"].reindex(index)
    expiry_series = front_contracts.map(expiry_map)
    idx_series = pd.Series(idx, index=index)

    # Convert timedelta to hours using total_seconds() / 3600.0
    hours = (expiry_series - idx_series).dt.total_seconds() / 3600.0
    hours = hours.clip(lower=0)
    return hours.fillna(np.inf)


def _quantile_safe(series: pd.Series, q: float, default: float = 0.0) -> float:
    if series.empty:
        return default
    value = series.quantile(q)
    if pd.isna(value):
        return default
    return float(value)


def compute_dynamic_volume_threshold(
    panel: pd.DataFrame,
    front_next: pd.DataFrame,
    hours_to_expiry: pd.Series,
    config: Optional[Dict] = None,
) -> pd.Series:
    """
    Compute dynamic volume threshold based on hours to expiry.

    Config values (max_days) are converted to hours internally.
    """
    volume = _extract_total_volume(panel, front_next)
    if volume.empty:
        return pd.Series(0.0, index=panel.index)

    if config is None or config.get("method") == "fixed":
        percentile = config.get("fixed_percentile", 0.10) if config else 0.10
        threshold = _quantile_safe(volume, percentile, default=0.0)
        return pd.Series(threshold, index=panel.index, dtype=float)

    if config.get("method") != "dynamic":
        raise ValueError(f"Unknown volume threshold method: {config.get('method')}")

    ranges = sorted(
        config.get(
            "dynamic_ranges",
            [
                {"max_days": 5, "percentile": 0.30},
                {"max_days": 30, "percentile": 0.20},
                {"max_days": 60, "percentile": 0.10},
                {"max_days": 999, "percentile": 0.05},
            ],
        ),
        key=lambda r: r["max_days"],
    )

    thresholds = pd.Series(np.nan, index=panel.index, dtype=float)
    remaining = pd.Series(True, index=panel.index)
    hte = hours_to_expiry.fillna(np.inf)

    for range_cfg in ranges:
        max_days = range_cfg["max_days"]
        max_hours = max_days * 24  # Convert days to hours
        percentile = range_cfg["percentile"]
        mask = remaining & (hte <= max_hours)
        if mask.any():
            subset_volume = volume[mask]
            threshold_value = _quantile_safe(subset_volume, percentile, default=0.0)
            thresholds.loc[mask] = threshold_value
            remaining.loc[mask] = False

    if thresholds.isna().any():
        fallback_q = ranges[-1]["percentile"]
        fallback_value = _quantile_safe(volume, fallback_q, default=0.0)
        thresholds = thresholds.fillna(fallback_value)

    return thresholds


def compute_business_days(
    panel: pd.DataFrame,
    front_next: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    expiry_map: pd.Series,
    min_total_buckets: int = 6,
    min_us_buckets: int = 2,
    volume_threshold_config: Optional[Dict] = None,
    near_expiry_relax: int = 5,
    partial_day_min_buckets: int = 4,
    fallback_policy: str = "calendar_only",
    bucket_ids: Optional[pd.Series] = None,
    return_audit: bool = False,
) -> Union[pd.DatetimeIndex, tuple[pd.DatetimeIndex, pd.DataFrame]]:
    if panel.empty:
        LOGGER.info("Panel empty; no business days computed")
        return pd.DatetimeIndex([])

    if expiry_map is None:
        raise ValueError("expiry_map must be provided to compute business days")

    trading_dates = map_to_trading_date(panel.index, bucket_ids)
    trading_dates = _normalize_index(trading_dates)

    if bucket_ids is not None:
        bucket_aligned = bucket_ids.reindex(panel.index)
        total_bucket_counts = bucket_aligned.notna().astype(int)
        us_bucket_counts = bucket_aligned.between(1, 7, inclusive="both").fillna(False).astype(int)
        effective_min_total = min_total_buckets
        effective_min_us = min_us_buckets
        effective_partial_min = partial_day_min_buckets
    else:
        total_bucket_counts = pd.Series(1, index=panel.index, dtype=int)
        us_bucket_counts = pd.Series(0, index=panel.index, dtype=int)
        effective_min_total = max(1, min_total_buckets)
        effective_min_us = 0
        effective_partial_min = max(1, partial_day_min_buckets)

    coverage_df = pd.DataFrame(
        {
            "date": trading_dates,
            "total_bucket": total_bucket_counts,
            "us_bucket": us_bucket_counts,
        },
        index=panel.index,
    )

    coverage_by_date = coverage_df.groupby("date").agg(
        total_buckets=("total_bucket", "sum"),
        us_buckets=("us_bucket", "sum"),
    )

    volume_series = _extract_total_volume(panel, front_next)
    volume_by_date = volume_series.groupby(trading_dates).sum()

    hte_series = _compute_hours_to_expiry(panel.index, front_next, expiry_map)
    min_hte_by_date = hte_series.groupby(trading_dates).min()

    threshold_series = compute_dynamic_volume_threshold(panel, front_next, hte_series, volume_threshold_config)
    threshold_by_date = threshold_series.groupby(trading_dates).first()

    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce").dt.normalize()
    calendar = calendar.dropna(subset=["date"]).drop_duplicates("date").set_index("date")

    if "is_trading_day" not in calendar.columns:
        calendar["is_trading_day"] = True

    # Determine closed dates from calendar; dates not listed are assumed open.
    closed_dates = set(calendar.index[~calendar["is_trading_day"]])
    # Candidate calendar-open dates restricted to observed data dates
    observed_dates = pd.Index(sorted(set(coverage_by_date.index)))
    # Only include weekdays (Mon=0 to Fri=4) that are not holidays
    calendar_trading = pd.DatetimeIndex([
        d for d in observed_dates 
        if d.weekday() < 5 and d not in closed_dates
    ])

    partial_hours = calendar.get("partial_hours", pd.Series("", index=calendar.index)).fillna(" ")
    partial_dates = set(calendar.loc[calendar["is_trading_day"] & partial_hours.str.strip().ne(""), :].index)

    all_dates = observed_dates
    summary = pd.DataFrame(index=all_dates)
    summary["total_buckets"] = coverage_by_date["total_buckets"].reindex(all_dates, fill_value=0)
    summary["us_buckets"] = coverage_by_date["us_buckets"].reindex(all_dates, fill_value=0)
    summary["volume"] = volume_by_date.reindex(all_dates, fill_value=0.0)
    summary["min_hte"] = min_hte_by_date.reindex(all_dates, fill_value=np.inf)
    summary["threshold"] = threshold_by_date.reindex(all_dates, fill_value=0.0)
    summary["has_data"] = summary["total_buckets"] > 0
    summary["calendar_closed"] = summary.index.map(lambda d: d in closed_dates)
    summary["calendar_open"] = ~summary["calendar_closed"]
    summary["is_partial_day"] = summary.index.map(lambda d: d in partial_dates)

    partial_mask = summary.index.map(lambda d: d in partial_dates)
    summary["required_buckets"] = np.where(partial_mask, effective_partial_min, effective_min_total)

    coverage_ok = summary["total_buckets"] >= summary["required_buckets"]
    if effective_min_us > 0:
        coverage_ok |= summary["us_buckets"] >= effective_min_us

    if near_expiry_relax is not None:
        # Convert days to hours for comparison
        near_expiry_hours = near_expiry_relax * 24
        summary["near_expiry_ok"] = summary["min_hte"] <= near_expiry_hours
        coverage_ok |= summary["near_expiry_ok"]
    else:
        summary["near_expiry_ok"] = False

    threshold = summary["threshold"].fillna(0.0)
    volume_ok = (threshold <= 0) | (summary["volume"] >= threshold)

    data_approved_mask = summary["has_data"] & coverage_ok & volume_ok
    data_approved = summary.index[data_approved_mask]

    calendar_set = set(calendar_trading)
    data_set = set(data_approved)

    if fallback_policy == "calendar_only":
        final_set = calendar_set
        summary["final_included"] = summary.index.map(lambda d: d in calendar_set)
    elif fallback_policy == "union_with_data":
        final_set = calendar_set | data_set
        summary["final_included"] = summary.index.map(lambda d: (d in calendar_set) or (d in data_set))
    elif fallback_policy == "intersection_strict":
        final_set = calendar_set & data_set
        summary["final_included"] = summary.index.map(lambda d: (d in calendar_set) and (d in data_set))
    else:
        raise ValueError(f"Unknown fallback_policy: {fallback_policy}")

    LOGGER.info(
        "Business days computed: calendar=%d data_approved=%d final=%d",
        len(calendar_set),
        len(data_set),
        len(final_set),
    )

    # Attach helpful audit columns
    summary["coverage_ok"] = coverage_ok.values
    summary["volume_ok_flag"] = volume_ok.values
    summary["data_approved"] = data_approved_mask.values
    # Compute simple exclusion reason
    def _reason(row):
        if row["calendar_closed"]:
            return "calendar_closed"
        if not row["has_data"]:
            return "no_data"
        if not row["coverage_ok"] and not row["volume_ok_flag"]:
            return "coverage_and_volume_fail"
        if not row["coverage_ok"]:
            return "coverage_fail"
        if not row["volume_ok_flag"]:
            return "volume_fail"
        return "included"
    summary["reason"] = summary.apply(_reason, axis=1)

    business_days = pd.DatetimeIndex(sorted(final_set)).tz_localize(None)

    if return_audit:
        audit = summary.copy()
        audit.index.name = "date"
        audit.reset_index(inplace=True)
        audit["policy"] = fallback_policy
        return business_days, audit

    return business_days


def business_day_gaps(events: pd.Series, business_days: pd.DatetimeIndex) -> pd.Series:
    events = events[events.astype(bool)]
    if business_days is None or len(business_days) == 0 or events.empty:
        return pd.Series(dtype=float, name="business_days_since_last")

    biz_index = pd.DatetimeIndex(business_days)
    if biz_index.tz is not None:
        biz_index = biz_index.tz_convert(None)
    biz_index = biz_index.normalize().drop_duplicates().sort_values()

    pos_map = {date: pos for pos, date in enumerate(biz_index)}

    event_index = pd.DatetimeIndex(events.index)
    if event_index.tz is not None:
        event_index = event_index.tz_convert(None)
    event_norm = event_index.normalize()

    gaps = []
    prev_pos: Optional[int] = None
    missing_dates: set[pd.Timestamp] = set()

    for date in event_norm:
        pos = pos_map.get(date)
        if pos is None:
            gaps.append(np.nan)
            missing_dates.add(date)
            continue
        if prev_pos is None:
            gaps.append(np.nan)
        else:
            gaps.append(pos - prev_pos)
        prev_pos = pos

    if missing_dates:
        LOGGER.warning(
            "Events found on dates outside business day index: %s",
            ", ".join(str(d.date()) for d in sorted(missing_dates)),
        )

    return pd.Series(gaps, index=events.index, name="business_days_since_last")
