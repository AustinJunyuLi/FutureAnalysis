from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import config as cfg
from . import trading_days
from . import spreads as strip_tools
from .events import (
    preprocess_spread,
    detect_spread_events,
    detect_multi_spread_events,
    align_events_to_business_days,
    confirm_roll_events,
    preference_scores,
    summarize_bucket_events,
    summarize_events,
    transition_matrix,
)
from .ingest import build_contract_frames, find_contract_files
from .panel import assemble_panel
from .quality import DataQualityFilter
from .rolls import (
    build_expiry_map,
    compute_liquidity_signal,
    compute_open_interest_signal,
    compute_spread,
    compute_multi_spreads,
    identify_front_next,
    identify_front_to_f12,
)
from . import multi_spread_analysis
from .results import BucketAnalysisResult, DailyAnalysisResult

LOGGER = logging.getLogger(__name__)


def run_bucket_analysis(
    settings: cfg.Settings,
    *,
    max_files: Optional[int] = None,
    metadata_path: Optional[Path] = None,
) -> BucketAnalysisResult:
    root_symbol = settings.products[0]
    data_root = Path(settings.data["minute_root"])
    files = find_contract_files(data_root, root_symbol=root_symbol)
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No {root_symbol} files found under {data_root}")

    tz = settings.data.get("timezone", "US/Central")
    ts_format = settings.data.get("timestamp_format")
    buckets = build_contract_frames(
        files,
        root_symbol=root_symbol,
        tz=tz,
        aggregate="bucket",
        timestamp_format=ts_format,
    )
    if not buckets:
        raise RuntimeError("Bucket aggregation returned no data")

    price_field = settings.data.get("price_field", "close")
    _ensure_required_fields(buckets, [price_field])

    metadata = _load_metadata(metadata_path or settings.metadata_path, buckets.keys())
    panel = assemble_panel(buckets, metadata, include_bucket_meta=True)

    expiry_map = build_expiry_map(metadata)

    # Use deterministic expiry-based labeling
    tz_ex = (settings.time or {}).get("tz_exchange", "America/Chicago")
    strip_cfg = settings.strip_analysis or {}
    strip_length = int(strip_cfg.get("strip_length", 12))
    chain = identify_front_to_f12(panel, expiry_map, tz_exchange=tz_ex, max_contracts=strip_length)
    # Construct front/next from chain
    front_next = chain.loc[:, ["F1", "F2"]].rename(columns={"F1": "front_contract", "F2": "next_contract"})

    panel[("meta", "front_contract")] = front_next["front_contract"].values
    panel[("meta", "next_contract")] = front_next["next_contract"].values

    spread = compute_spread(panel, front_next, price_field=price_field)
    spread_cfg = settings.spread
    spread_signal = preprocess_spread(
        spread,
        clip_quantile=spread_cfg.get("clip_quantile"),
        ema_span=spread_cfg.get("ema_span"),
    )

    cool_down = None
    if spread_cfg.get("cool_down_hours") is not None:
        cool_down = pd.Timedelta(hours=spread_cfg["cool_down_hours"])
    elif spread_cfg.get("cool_down_buckets"):
        cool_down = int(spread_cfg["cool_down_buckets"])

    widening = detect_spread_events(
        spread_signal,
        method=spread_cfg.get("method", "zscore"),
        window=spread_cfg.get("window_buckets", spread_cfg.get("window", 30)),
        z_threshold=spread_cfg.get("z_threshold", 1.5),
        abs_min=spread_cfg.get("abs_min"),
        cool_down=cool_down,
    )
    widening = widening.astype(bool)

    roll_rules = settings.roll_rules or {}
    liquidity_signal = compute_liquidity_signal(
        panel,
        front_next,
        volume_field=settings.data.get("volume_field", "volume"),
        alpha=roll_rules.get("liquidity_threshold", 0.8),
        confirm=roll_rules.get("confirm_days", 1),
    )
    oi_signal = compute_open_interest_signal(
        panel,
        front_next,
        ratio=roll_rules.get("oi_ratio", 0.75),
        confirm=roll_rules.get("oi_confirm_days", 1),
    )
    confirmation_min = int(roll_rules.get("confirmation_min_signals", 1))
    additional_signals = {"liquidity": liquidity_signal, "open_interest": oi_signal}
    widening = confirm_roll_events(widening, additional_signals, min_signals=confirmation_min)

    strip_cfg = settings.strip_analysis or {}

    volume_series = _front_volume_series(panel, front_next)
    bucket_ids = pd.to_numeric(panel[("meta", "bucket")], errors="coerce").astype("Int64")
    labels = panel[("meta", "bucket_label")]
    sessions = panel[("meta", "session")]

    # Optional calendar paths: business day logic only runs if calendars provided
    calendar_paths = settings.business_days.get("calendar_paths", [])
    calendar = None
    if calendar_paths:
        try:
            calendar = trading_days.load_calendar(
                calendar_paths,
                hierarchy=settings.business_days.get("calendar_hierarchy", "override"),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load trading calendar from {calendar_paths}: {exc}\n"
                "Please ensure the calendar file exists and is properly formatted."
            ) from exc
    else:
        LOGGER.info("No calendar paths provided; business day alignment will be skipped")

    # Multi-spread analysis for supervisor's test
    LOGGER.info("Computing multi-spread analysis (F1-F12, S1-S11)...")
    contract_chain = chain
    LOGGER.info(f"Contract chain identified: {len(contract_chain)} periods × {len(contract_chain.columns)} contracts")

    # Always compute business days (business-day-only pipeline)
    business_days_index = None
    business_days_audit = None
    if calendar is not None:
        try:
            result = trading_days.compute_business_days(
                panel,
                front_next,
                calendar,
                expiry_map=expiry_map,
                min_total_buckets=settings.business_days.get("min_total_buckets", 6),
                min_us_buckets=settings.business_days.get("min_us_buckets", 2),
                volume_threshold_config=settings.business_days.get("volume_threshold"),
                near_expiry_relax=settings.business_days.get("near_expiry_relax", 5),
                partial_day_min_buckets=settings.business_days.get("partial_day_min_buckets", 4),
                fallback_policy=settings.business_days.get("fallback_policy", "calendar_only"),
                bucket_ids=bucket_ids,
                return_audit=True,
            )
            business_days_index, business_days_audit = result
            LOGGER.info(f"Business days computed: {len(business_days_index)} days")
        except (ValueError, KeyError) as e:
            # Configuration or data structure errors - these are acceptable to catch
            LOGGER.warning(f"Business day computation failed due to configuration issue: {e}")
        except Exception as e:
            # Unexpected errors - log with full context and re-raise
            LOGGER.error(f"Unexpected error in business day computation: {e}", exc_info=True)
            raise RuntimeError(f"Business day computation encountered unexpected error: {e}") from e

    # Optional alignment of events to business days for summary/reporting
    align_policy = settings.business_days.get("align_events", "none")
    if align_policy and business_days_index is not None:
        widening = align_events_to_business_days(widening, business_days_index, policy=align_policy)

    approved_mask = None
    if business_days_audit is not None:
        approved_dates = pd.to_datetime(
            business_days_audit.loc[business_days_audit["data_approved"].fillna(False), "date"]
        ).dt.normalize()
        approved_set = set(approved_dates)
        event_dates = pd.to_datetime(widening.index).normalize()
        approved_mask = pd.Series(event_dates.isin(approved_set), index=widening.index)
        if approved_mask.notna().any():
            invalid = int((widening.astype(bool) & ~approved_mask).sum())
            if invalid:
                LOGGER.info("Dropping %s widening events outside approved business days", invalid)
            widening = widening & approved_mask.fillna(False)
        else:
            LOGGER.warning("No approved business days detected; suppressing all widening events")
            widening = widening & False

    LOGGER.info("Computing multi-spread analysis (F1-F12, S1-S11)...")
    multi_spreads = compute_multi_spreads(
        panel,
        chain,
        price_field=settings.data.get("price_field", "close"),
    )
    LOGGER.info(f"Multi-spreads computed: {multi_spreads.shape}")

    multi_events = detect_multi_spread_events(
        multi_spreads,
        method=spread_cfg.get("method", "zscore"),
        window=spread_cfg.get("window_buckets", spread_cfg.get("window", 30)),
        z_threshold=spread_cfg.get("z_threshold", 1.5),
        abs_min=spread_cfg.get("abs_min"),
        cool_down=cool_down,
    )
    if approved_mask is not None:
        mask = approved_mask.reindex(multi_events.index, fill_value=False)
        mask_df = pd.DataFrame(
            np.broadcast_to(mask.to_numpy()[:, None], multi_events.shape),
            index=multi_events.index,
            columns=multi_events.columns,
        )
        multi_events = multi_events.where(mask_df, False)
    LOGGER.info(f"Multi-spread events detected: {multi_events.sum().sum()} total events across all spreads")

    event_counts = multi_events.sum()
    zero_spreads = [col.replace("_events", "") for col, val in event_counts.items() if val == 0]
    if zero_spreads:
        LOGGER.warning(
            "No detections for spreads: %s. Consider reducing strip_analysis.strip_length or reviewing data coverage.",
            ", ".join(zero_spreads),
        )

    spread_correlations = multi_spread_analysis.compute_spread_correlations(multi_spreads)
    spread_comparison = multi_spread_analysis.compare_spread_signals(multi_spreads, multi_events)
    spread_timing = multi_spread_analysis.analyze_spread_timing(
        multi_spreads,
        multi_events,
        chain,
        expiry_map,
        tz_exchange=tz_ex,
    )
    timing_summary = multi_spread_analysis.summarize_timing_by_spread(spread_timing)
    spread_changes = multi_spread_analysis.analyze_spread_changes(multi_spreads)

    LOGGER.info("Computing cross-spread magnitude comparison (S1 vs S2-S11)...")
    magnitude_comp = multi_spread_analysis.compare_spread_magnitudes(multi_spreads)
    s1_dominance_by_cycle = multi_spread_analysis.analyze_s1_dominance_by_expiry_cycle(
        magnitude_comp, chain, expiry_map
    )
    cross_spread_summary = multi_spread_analysis.summarize_cross_spread_patterns(s1_dominance_by_cycle)

    LOGGER.info(
        "S1 events: %s, S2 events: %s, S3 events: %s",
        multi_events["S1_events"].sum(),
        multi_events["S2_events"].sum(),
        multi_events["S3_events"].sum(),
    )

    strip_diagnostics = pd.DataFrame()
    if strip_cfg.get("enabled", True):
        strip_diagnostics = strip_tools.summarize_strip_dominance(
            magnitude_comp,
            chain,
            expiry_map,
            calendar=calendar,
            dominance_threshold=strip_cfg.get("dominance_ratio_threshold", 2.0),
            expiry_window_bd=strip_cfg.get("expiry_window_business_days", 18),
        )
        if strip_cfg.get("filter_expiry_dominance", True):
            widened_before = int(widening.sum())
            widening, removed = strip_tools.filter_expiry_dominance_events(widening, strip_diagnostics)
            if removed:
                LOGGER.info("Filtered %s widening events classified as expiry dominance (from %s)", removed, widened_before)

    support_index = widening.index
    tol = pd.Timedelta(hours=1)
    spread_for_events = _reindex_like(spread, support_index, tolerance=tol)
    volume_series = _reindex_like(volume_series, support_index, tolerance=tol)
    bucket_ids = _reindex_like(bucket_ids, support_index, tolerance=tol)
    labels = _reindex_like(labels, support_index, tolerance=tol)
    sessions = _reindex_like(sessions, support_index, tolerance=tol)
    bucket_ids = bucket_ids.astype("Int64")

    bucket_summary = summarize_bucket_events(
        widening,
        spread_for_events,
        bucket_ids,
        bucket_labels=labels,
        sessions=sessions,
    )
    session_summary = _session_event_summary(bucket_summary)
    preference = preference_scores(widening, volume_series, bucket_ids)
    transitions = transition_matrix(widening, bucket_ids)
    event_summary = summarize_events(widening, spread_for_events, business_days=business_days_index)

    LOGGER.info(
        "Bucket analysis complete: rows=%s events=%s",
        len(panel),
        int(widening.sum()),
    )

    return BucketAnalysisResult(
        panel=panel,
        front_next=front_next,
        spread=spread,
        widening=widening,
        bucket_summary=bucket_summary,
        session_summary=session_summary,
        preference_scores=preference,
        transition_matrix=transitions,
        event_summary=event_summary,
        business_days_index=business_days_index,
        business_days_audit=business_days_audit,
        multi_spreads=multi_spreads,
        multi_events=multi_events,
        strip_diagnostics=strip_diagnostics,
        metadata=metadata,
    )


def run_daily_analysis(
    settings: cfg.Settings,
    *,
    metadata_path: Optional[Path] = None,
) -> DailyAnalysisResult:
    root_symbol = settings.products[0]
    data_root = Path(settings.data["minute_root"])
    files = find_contract_files(data_root, root_symbol=root_symbol)
    if not files:
        raise FileNotFoundError(f"No {root_symbol} files found under {data_root}")

    tz = settings.data.get("timezone", "US/Central")
    ts_format = settings.data.get("timestamp_format")
    daily = build_contract_frames(
        files,
        root_symbol=root_symbol,
        tz=tz,
        aggregate="daily",
        timestamp_format=ts_format,
    )

    metadata = _load_metadata(metadata_path or settings.metadata_path, daily.keys())

    dq = DataQualityFilter(settings.data_quality)
    filtered, metrics = dq.apply(daily)
    quality_metrics = pd.DataFrame(metrics)
    if not filtered:
        raise RuntimeError("No contracts passed data quality filtering")

    panel = assemble_panel(filtered, metadata, include_bucket_meta=False)

    expiry_map = build_expiry_map(metadata)

    # Use deterministic expiry-based labeling
    tz_ex = (settings.time or {}).get("tz_exchange", "America/Chicago")
    chain = identify_front_to_f12(panel, expiry_map, tz_exchange=tz_ex, max_contracts=2)
    front_next = chain.rename(columns={"F1": "front_contract", "F2": "next_contract"})

    panel[("meta", "front_contract")] = front_next["front_contract"].values
    panel[("meta", "next_contract")] = front_next["next_contract"].values

    price_field = settings.data.get("price_field", "close")
    _ensure_required_fields(filtered, [price_field])
    spread = compute_spread(panel, front_next, price_field=price_field)

    spread_cfg = settings.spread
    spread_signal = preprocess_spread(
        spread,
        clip_quantile=spread_cfg.get("clip_quantile"),
        ema_span=spread_cfg.get("ema_span"),
    )
    cool_down = spread_cfg.get("cool_down")
    widening = detect_spread_events(
        spread_signal,
        method=spread_cfg.get("method", "zscore"),
        window=spread_cfg.get("window", spread_cfg.get("window_buckets", 30)),
        z_threshold=spread_cfg.get("z_threshold", 1.5),
        abs_min=spread_cfg.get("abs_min"),
        cool_down=int(cool_down) if cool_down else None,
    )
    widening = widening.astype(bool)

    liquidity = compute_liquidity_signal(
        panel,
        front_next,
        volume_field=settings.data.get("volume_field", "volume"),
        alpha=settings.roll_rules.get("liquidity_threshold", 0.8),
        confirm=settings.roll_rules.get("confirm_days", 1),
    )
    oi_signal = compute_open_interest_signal(
        panel,
        front_next,
        ratio=settings.roll_rules.get("oi_ratio", 0.75),
        confirm=settings.roll_rules.get("oi_confirm_days", 1),
    )
    confirmation_min = int(settings.roll_rules.get("confirmation_min_signals", 1))
    additional_signals = {"liquidity": liquidity, "open_interest": oi_signal}
    widening = confirm_roll_events(widening, additional_signals, min_signals=confirmation_min)
    spread_for_events = _reindex_like(spread, widening.index, tolerance=pd.Timedelta(days=1))

    # Optional calendar paths: business day logic only runs if calendars provided
    calendar_paths = settings.business_days.get("calendar_paths", [])
    calendar = None
    if calendar_paths:
        try:
            calendar = trading_days.load_calendar(
                calendar_paths,
                hierarchy=settings.business_days.get("calendar_hierarchy", "override"),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load trading calendar from {calendar_paths}: {exc}\n"
                "Please ensure the calendar file exists and is properly formatted."
            ) from exc
    else:
        LOGGER.info("No calendar paths provided; business day alignment will be skipped")

    business_days_index = None
    business_days_audit = None
    if calendar is not None:  # This will always be True now, but keeping for clarity
        try:
            result = trading_days.compute_business_days(
                panel,
                front_next,
                calendar,
                expiry_map=expiry_map,
                min_total_buckets=settings.business_days.get("min_total_buckets", 6),
                min_us_buckets=settings.business_days.get("min_us_buckets", 2),
                volume_threshold_config=settings.business_days.get("volume_threshold"),
                near_expiry_relax=settings.business_days.get("near_expiry_relax", 5),
                partial_day_min_buckets=settings.business_days.get("partial_day_min_buckets", 4),
                fallback_policy=settings.business_days.get("fallback_policy", "calendar_only"),
                bucket_ids=None,
                return_audit=True,
            )
            business_days_index, business_days_audit = result
            LOGGER.info(f"Business days computed: {len(business_days_index)} days")
        except (ValueError, KeyError) as e:
            # Configuration or data structure errors - these are acceptable to catch
            LOGGER.warning(f"Business day computation failed due to configuration issue: {e}")
        except Exception as e:
            # Unexpected errors - log with full context and re-raise
            LOGGER.error(f"Unexpected error in business day computation: {e}", exc_info=True)
            raise RuntimeError(f"Business day computation encountered unexpected error: {e}") from e

    align_policy = settings.business_days.get("align_events", "none")
    if align_policy and business_days_index is not None:
        widening = align_events_to_business_days(widening, business_days_index, policy=align_policy)

    if business_days_audit is not None:
        approved_dates = pd.to_datetime(
            business_days_audit.loc[business_days_audit["data_approved"].fillna(False), "date"]
        ).dt.normalize()
        approved_set = set(approved_dates)
        event_dates = pd.to_datetime(widening.index).normalize()
        approved_mask = pd.Series(event_dates.isin(approved_set), index=widening.index)
        invalid = int((widening & ~approved_mask).sum())
        if invalid:
            LOGGER.info("Dropping %s daily widening events outside approved business days", invalid)
        widening = widening & approved_mask.fillna(False)

    summary = summarize_events(widening, spread_for_events, business_days=business_days_index)

    LOGGER.info(
        "Daily analysis complete: rows=%s events=%s",
        len(panel),
        int(widening.sum()),
    )

    return DailyAnalysisResult(
        panel=panel,
        front_next=front_next,
        spread=spread,
        widening=widening,
        liquidity=liquidity,
        summary=summary,
        quality_metrics=quality_metrics,
        business_days_index=business_days_index,
        business_days_audit=business_days_audit,
        metadata=metadata,
    )


def _load_metadata(path: Path, contracts: Iterable[str]) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    metadata = pd.read_csv(path)
    metadata["contract"] = metadata["contract"].astype(str)
    metadata["expiry_date"] = pd.to_datetime(metadata["expiry_date"]).dt.normalize()
    subset = metadata[metadata["contract"].isin(contracts)]
    contract_set = set(contracts)
    missing = sorted(contract_set - set(subset["contract"]))
    if missing:
        raise ValueError(f"Missing explicit expiry for contracts: {', '.join(missing[:10])}")
    return subset


def _ensure_required_fields(frames_by_contract: dict[str, pd.DataFrame], fields: Iterable[str]) -> None:
    missing = {}
    for contract, frame in frames_by_contract.items():
        absent = [field for field in fields if field not in frame.columns]
        if absent:
            missing[contract] = absent
    if missing:
        preview = ", ".join(f"{contract}: {absent}" for contract, absent in list(missing.items())[:5])
        raise ValueError(
            "Missing required columns in aggregated data. "
            f"Requested fields {list(fields)} but absent for contracts: {preview}"
        )


def _front_volume_series(panel: pd.DataFrame, front_next: pd.DataFrame) -> pd.Series:
    contracts = [
        contract
        for contract in panel.columns.get_level_values(0).unique()
        if contract != "meta"
    ]
    tuples = [(contract, "volume") for contract in contracts if (contract, "volume") in panel.columns]
    if not tuples:
        return pd.Series(index=panel.index, dtype=float)
    volumes = panel.loc[:, tuples]
    volumes.columns = [contract for contract, _ in volumes.columns]
    values = volumes.to_numpy(dtype=float)
    contract_index = {contract: idx for idx, contract in enumerate(volumes.columns)}
    front_idx = np.array([contract_index.get(c, -1) for c in front_next["front_contract"]], dtype=int)
    result = np.full(len(panel), np.nan)
    mask = front_idx >= 0
    if mask.any():
        row_idx = np.nonzero(mask)[0]
        result[mask] = values[row_idx, front_idx[mask]]
    return pd.Series(result, index=panel.index, name="front_volume")


def _session_event_summary(bucket_summary: pd.DataFrame) -> pd.DataFrame:
    if bucket_summary.empty or "session" not in bucket_summary.columns:
        return pd.DataFrame(columns=["session", "event_count", "event_share_pct"])

    total_events = float(bucket_summary["event_count"].sum())
    grouped = (
        bucket_summary.groupby("session", as_index=False)["event_count"].sum()
        .sort_values("event_count", ascending=False)
        .reset_index(drop=True)
    )
    if total_events > 0:
        grouped["event_share_pct"] = grouped["event_count"] / total_events * 100.0
    else:
        grouped["event_share_pct"] = 0.0
    return grouped


def _reindex_like(series: pd.Series, target: pd.Index, *, tolerance: Optional[pd.Timedelta] = None) -> pd.Series:
    aligned = series.reindex(target)
    if tolerance is not None and aligned.isna().any():
        missing_index = aligned[aligned.isna()].index
        if len(missing_index) > 0:
            fallback = series.reindex(missing_index, method="nearest", tolerance=tolerance)
            aligned.loc[missing_index] = fallback
    return aligned
