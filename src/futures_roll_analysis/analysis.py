from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from . import config as cfg
from . import trading_days
from . import spreads as strip_tools
from .events import (
    detect_spread_events,
    detect_multi_spread_events,
    align_events_to_business_days,
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
    compute_spread,
    compute_multi_spreads,
    identify_front_next,
    identify_front_to_f12,
)
from . import multi_spread_analysis

LOGGER = logging.getLogger(__name__)


def run_bucket_analysis(
    settings: cfg.Settings,
    *,
    max_files: Optional[int] = None,
    metadata_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    root_symbol = settings.products[0]
    data_root = Path(settings.data["minute_root"])
    files = find_contract_files(data_root, root_symbol=root_symbol)
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No {root_symbol} files found under {data_root}")

    tz = settings.data.get("timezone", "US/Central")
    buckets = build_contract_frames(
        files,
        root_symbol=root_symbol,
        tz=tz,
        aggregate="bucket",
    )
    if not buckets:
        raise RuntimeError("Bucket aggregation returned no data")

    metadata = _load_metadata(metadata_path or settings.metadata_path, buckets.keys())
    panel = assemble_panel(buckets, metadata, include_bucket_meta=True)

    expiry_map = build_expiry_map(metadata)

    # Use deterministic expiry-based labeling
    tz_ex = (settings.time or {}).get("tz_exchange", "America/Chicago")
    chain = identify_front_to_f12(panel, expiry_map, tz_exchange=tz_ex, max_contracts=12)
    # Construct front/next from chain
    front_next = chain.loc[:, ["F1", "F2"]].rename(columns={"F1": "front_contract", "F2": "next_contract"})

    panel[("meta", "front_contract")] = front_next["front_contract"].values
    panel[("meta", "next_contract")] = front_next["next_contract"].values

    spread = compute_spread(panel, front_next, price_field=settings.data.get("price_field", "close"))

    spread_cfg = settings.spread
    cool_down = None
    if spread_cfg.get("cool_down_hours") is not None:
        cool_down = pd.Timedelta(hours=spread_cfg["cool_down_hours"])
    elif spread_cfg.get("cool_down_buckets"):
        cool_down = int(spread_cfg["cool_down_buckets"])

    widening = detect_spread_events(
        spread,
        method=spread_cfg.get("method", "zscore"),
        window=spread_cfg.get("window_buckets", spread_cfg.get("window", 30)),
        z_threshold=spread_cfg.get("z_threshold", 1.5),
        abs_min=spread_cfg.get("abs_min"),
        cool_down=cool_down,
    )

    strip_cfg = settings.strip_analysis or {}

    volume_series = _front_volume_series(panel, front_next)
    bucket_ids = pd.to_numeric(panel[("meta", "bucket")], errors="coerce").astype("Int64")
    labels = panel[("meta", "bucket_label")]
    sessions = panel[("meta", "session")]

    # Strict calendar requirement: fail fast if calendar loading fails
    calendar_paths = settings.business_days.get("calendar_paths", [])
    if not calendar_paths:
        raise ValueError(
            "business_days.calendar_paths is required. "
            "Please provide a trading calendar in settings."
        )

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

    # Multi-spread analysis for supervisor's test
    LOGGER.info("Computing multi-spread analysis (F1-F12, S1-S11)...")
    contract_chain = chain
    LOGGER.info(f"Contract chain identified: {len(contract_chain)} periods × {len(contract_chain.columns)} contracts")

    multi_spreads = compute_multi_spreads(
        panel,
        contract_chain,
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
    LOGGER.info(f"Multi-spread events detected: {multi_events.sum().sum()} total events across all spreads")

    # Comparative analysis
    spread_correlations = multi_spread_analysis.compute_spread_correlations(multi_spreads)
    spread_comparison = multi_spread_analysis.compare_spread_signals(multi_spreads, multi_events)
    spread_timing = multi_spread_analysis.analyze_spread_timing(
        multi_spreads,
        multi_events,
        contract_chain,
        expiry_map,
        tz_exchange=tz_ex,
    )
    timing_summary = multi_spread_analysis.summarize_timing_by_spread(spread_timing)
    spread_changes = multi_spread_analysis.analyze_spread_changes(multi_spreads)

    # Cross-spread magnitude comparison (supervisor's specific request)
    LOGGER.info("Computing cross-spread magnitude comparison (S1 vs S2-S11)...")
    magnitude_comp = multi_spread_analysis.compare_spread_magnitudes(multi_spreads)
    s1_dominance_by_cycle = multi_spread_analysis.analyze_s1_dominance_by_expiry_cycle(
        magnitude_comp, contract_chain, expiry_map
    )
    cross_spread_summary = multi_spread_analysis.summarize_cross_spread_patterns(s1_dominance_by_cycle)

    LOGGER.info(f"S1 dominance analysis complete: {magnitude_comp['s1_dominates'].sum()} periods where S1 > 2x median(S2-S11)")
    LOGGER.info(f"S1 events: {multi_events['S1_events'].sum()}, "
                f"S2 events: {multi_events['S2_events'].sum()}, "
                f"S3 events: {multi_events['S3_events'].sum()}")

    strip_diagnostics = pd.DataFrame()
    if strip_cfg.get("enabled", True):
        strip_diagnostics = strip_tools.summarize_strip_dominance(
            magnitude_comp,
            contract_chain,
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
        except Exception as e:
            LOGGER.warning(f"Business day computation failed: {e}.")

    # Optional alignment of events to business days for summary/reporting
    align_policy = settings.business_days.get("align_events", "none")
    if align_policy and business_days_index is not None:
        widening = align_events_to_business_days(widening, business_days_index, policy=align_policy)

    bucket_summary = summarize_bucket_events(
        widening,
        spread,
        bucket_ids.astype("Int64"),
        bucket_labels=labels,
        sessions=sessions,
    )
    preference = preference_scores(widening, volume_series, bucket_ids)
    transitions = transition_matrix(widening, bucket_ids)
    event_summary = summarize_events(widening, spread, business_days=business_days_index)

    out_dir = (output_dir or settings.output_dir).resolve()
    panels_dir = out_dir / "panels"
    signals_dir = out_dir / "roll_signals"
    analysis_dir = out_dir / "analysis"
    for path in (panels_dir, signals_dir, analysis_dir):
        path.mkdir(parents=True, exist_ok=True)

    panel_path = panels_dir / "hourly_panel.parquet"
    _write_parquet(panel, panel_path)
    _write_csv(spread, signals_dir / "hourly_spread.csv", header=True)
    _write_csv(widening, signals_dir / "hourly_widening.csv", header=True)

    _write_csv(bucket_summary, analysis_dir / "bucket_summary.csv", index=False)
    _write_csv(preference, analysis_dir / "preference_scores.csv", header=True)
    _write_csv(transitions, analysis_dir / "transition_matrix.csv")
    if business_days_audit is not None:
        _write_csv(business_days_audit, analysis_dir / "business_days_audit_hourly.csv", index=False)
    _write_csv(event_summary, analysis_dir / "hourly_widening_summary.csv", index=False)

    # Write multi-spread analysis outputs
    _write_csv(multi_spreads, signals_dir / "multi_spreads.csv", header=True)
    _write_csv(multi_events, signals_dir / "multi_spread_events.csv", header=True)
    _write_csv(spread_correlations, analysis_dir / "spread_correlations.csv")
    _write_csv(spread_comparison, analysis_dir / "spread_signal_comparison.csv", index=False)
    _write_csv(spread_timing, analysis_dir / "spread_event_timing.csv", index=False)
    _write_csv(timing_summary, analysis_dir / "spread_timing_summary.csv", index=False)
    _write_csv(spread_changes, analysis_dir / "spread_change_statistics.csv", index=False)

    # Write switch log for F1 transitions
    try:
        _write_switch_log(contract_chain["F1"], expiry_map, tz_ex, analysis_dir / "roll_switches.csv")
    except Exception as e:
        LOGGER.warning("Failed to write roll_switches.csv: %s", e)

    # Write cross-spread magnitude comparison outputs
    _write_csv(magnitude_comp, analysis_dir / "s1_vs_others_magnitude.csv", header=True)
    _write_csv(s1_dominance_by_cycle, analysis_dir / "s1_dominance_by_cycle.csv", index=False)
    _write_csv(cross_spread_summary, analysis_dir / "cross_spread_summary.csv", index=False)
    if not strip_diagnostics.empty:
        _write_csv(strip_diagnostics, analysis_dir / "strip_spread_diagnostics.csv", index=False)

    LOGGER.info("Multi-spread analysis outputs written successfully")

    _write_run_settings(settings, analysis_dir / "run_settings.json")
    LOGGER.info(
        "Bucket analysis complete: rows=%s events=%s",
        len(panel),
        int(widening.sum()),
    )

    return {
        "panel": panel_path,
        "spread": signals_dir / "hourly_spread.csv",
        "events": signals_dir / "hourly_widening.csv",
        "bucket_summary": analysis_dir / "bucket_summary.csv",
        "preference_scores": analysis_dir / "preference_scores.csv",
        "transition_matrix": analysis_dir / "transition_matrix.csv",
        "event_summary": analysis_dir / "hourly_widening_summary.csv",
    }


def run_daily_analysis(
    settings: cfg.Settings,
    *,
    metadata_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    root_symbol = settings.products[0]
    data_root = Path(settings.data["minute_root"])
    files = find_contract_files(data_root, root_symbol=root_symbol)
    if not files:
        raise FileNotFoundError(f"No {root_symbol} files found under {data_root}")

    tz = settings.data.get("timezone", "US/Central")
    daily = build_contract_frames(
        files,
        root_symbol=root_symbol,
        tz=tz,
        aggregate="daily",
    )

    metadata = _load_metadata(metadata_path or settings.metadata_path, daily.keys())

    dq = DataQualityFilter(settings.data_quality)
    filtered, metrics = dq.apply(daily)
    dq.save_reports(metrics, (output_dir or settings.output_dir) / "data_quality", root_symbol)
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

    spread = compute_spread(panel, front_next, price_field=settings.data.get("price_field", "close"))

    spread_cfg = settings.spread
    cool_down = spread_cfg.get("cool_down")
    widening = detect_spread_events(
        spread,
        method=spread_cfg.get("method", "zscore"),
        window=spread_cfg.get("window", spread_cfg.get("window_buckets", 30)),
        z_threshold=spread_cfg.get("z_threshold", 1.5),
        abs_min=spread_cfg.get("abs_min"),
        cool_down=int(cool_down) if cool_down else None,
    )

    liquidity = compute_liquidity_signal(
        panel,
        front_next,
        volume_field=settings.data.get("volume_field", "volume"),
        alpha=settings.roll_rules.get("liquidity_threshold", 0.8),
        confirm=settings.roll_rules.get("confirm_days", 1),
    )

    # Strict calendar requirement: fail fast if calendar loading fails
    calendar_paths = settings.business_days.get("calendar_paths", [])
    if not calendar_paths:
        raise ValueError(
            "business_days.calendar_paths is required. "
            "Please provide a trading calendar in settings."
        )

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
        except Exception as e:
            LOGGER.warning(f"Business day computation failed: {e}.")

    align_policy = settings.business_days.get("align_events", "none")
    if align_policy and business_days_index is not None:
        widening = align_events_to_business_days(widening, business_days_index, policy=align_policy)

    summary = summarize_events(widening, spread, business_days=business_days_index)

    out_dir = (output_dir or settings.output_dir).resolve()
    panels_dir = out_dir / "panels"
    signals_dir = out_dir / "roll_signals"
    analysis_dir = out_dir / "analysis"
    for path in (panels_dir, signals_dir, analysis_dir):
        path.mkdir(parents=True, exist_ok=True)

    panel_path = panels_dir / "hg_panel_filtered.parquet"
    _write_parquet(panel, panel_path)
    _write_csv(panel, panels_dir / "hg_panel_full_filtered.csv")

    _write_csv(summary, analysis_dir / "daily_widening_summary.csv", index=False)

    _write_csv(spread, signals_dir / "hg_spread_filtered.csv")
    _write_csv(widening, signals_dir / "hg_widening_filtered.csv")
    _write_csv(liquidity, signals_dir / "hg_liquidity_roll_filtered.csv")
    if business_days_audit is not None:
        _write_csv(business_days_audit, analysis_dir / "business_days_audit_daily.csv", index=False)
    _write_run_settings(settings, analysis_dir / "run_settings.json")

    LOGGER.info(
        "Daily analysis complete: rows=%s events=%s",
        len(panel),
        int(widening.sum()),
    )

    return {
        "panel": panel_path,
        "spread": signals_dir / "hg_spread_filtered.csv",
        "events": signals_dir / "hg_widening_filtered.csv",
    }


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


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.unlink(missing_ok=True)
    df.to_parquet(path)


def _write_csv(obj: pd.DataFrame | pd.Series, path: Path, **kwargs) -> None:
    path.unlink(missing_ok=True)
    obj.to_csv(path, **kwargs)


def _write_run_settings(settings: cfg.Settings, path: Path) -> None:
    import json
    payload = {
        "products": list(settings.products),
        "data": settings.data,
        "spread": settings.spread,
        "selection": getattr(settings, "selection", {}),
        "time": getattr(settings, "time", {}),
        "business_days": {k: v for k, v in settings.business_days.items() if k != "calendar_paths"},
        "output_dir": str(settings.output_dir),
        "metadata_path": str(settings.metadata_path),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))


def _write_switch_log(f1_series: pd.Series, expiry_map: pd.Series, tz_exchange: str, out_path: Path) -> None:
    idx = pd.DatetimeIndex(f1_series.index)
    if idx.tz is None:
        idx_local = idx.tz_localize(tz_exchange, nonexistent="shift_forward", ambiguous="infer")
    else:
        idx_local = idx.tz_convert(tz_exchange)
    changes = f1_series.astype(object).ne(f1_series.shift(1)).fillna(True)
    change_points = f1_series.index[changes]
    rows = []
    for cp in change_points[1:]:  # skip first segment with no previous F1
        pos = f1_series.index.get_loc(cp)
        prev = f1_series.iloc[pos - 1]
        newf = f1_series.iloc[pos]
        prev_exp = pd.to_datetime(expiry_map.get(str(prev), pd.NaT))
        if pd.notna(prev_exp) and prev_exp.tz is None:
            # For individual timestamps, use True (treat as DST) for ambiguous times
            prev_exp = prev_exp.tz_localize(tz_exchange, ambiguous=True, nonexistent="shift_forward")
        rows.append(
            {
                "prev_f1": prev,
                "new_f1": newf,
                "transition_time_local": idx_local[pos],
                "prev_f1_expiry_local": prev_exp,
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
