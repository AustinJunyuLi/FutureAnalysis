from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from . import config as cfg
from .events import (
    detect_spread_events,
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
    identify_front_next,
)

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
    front_next = identify_front_next(panel, expiry_map, price_field=settings.data.get("price_field", "close"))
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

    volume_series = _front_volume_series(panel, front_next)
    bucket_ids = pd.to_numeric(panel[("meta", "bucket")], errors="coerce").astype("Int64")
    labels = panel[("meta", "bucket_label")]
    sessions = panel[("meta", "session")]

    bucket_summary = summarize_bucket_events(
        widening,
        spread,
        bucket_ids.astype("Int64"),
        bucket_labels=labels,
        sessions=sessions,
    )
    preference = preference_scores(widening, volume_series, bucket_ids)
    transitions = transition_matrix(widening, bucket_ids)

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
    front_next = identify_front_next(panel, expiry_map, price_field=settings.data.get("price_field", "close"))
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

    summary = summarize_events(widening, spread)

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
