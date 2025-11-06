from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Dict, Iterable, Optional

import yaml


@dataclass(frozen=True)
class Settings:
    """Validated settings loaded from the project YAML configuration."""

    products: Iterable[str]
    bucket_config: Dict[str, Any]
    data: Dict[str, Any]
    data_quality: Dict[str, Any]
    roll_rules: Dict[str, Any]
    selection: Dict[str, Any]
    time: Dict[str, Any]
    expiries: Dict[str, Any]
    strip_analysis: Dict[str, Any]
    spread: Dict[str, Any]
    business_days: Dict[str, Any]
    output_dir: Path
    metadata_path: Path


def _resolve_path(value: Optional[str], base_path: Path) -> Optional[Path]:
    if value is None:
        return None
    # Expand environment variables for portability
    value = os.path.expandvars(value)
    path = Path(value)
    if not path.is_absolute():
        path = (base_path / path).resolve()
    return path


def load_settings(
    settings_path: Path,
    overrides: Optional[Dict[str, Any]] = None,
) -> Settings:
    """
    Load project settings from a YAML file, applying optional overrides.

    Parameters
    ----------
    settings_path:
        Path to settings YAML file.
    overrides:
        Optional dictionary of values that should override the YAML contents.
    """
    settings_path = settings_path.resolve()
    base_dir = settings_path.parent

    with settings_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if overrides:
        raw = _merge_dicts(raw, overrides)

    products = list(raw.get("products", [])) or ["HG"]
    bucket_config = raw.get("bucket_config", {})
    data_cfg = raw.get("data", {})
    data_cfg.setdefault("timezone", "US/Central")
    data_cfg.setdefault("price_field", "close")

    spread_cfg = raw.get("spread", {})
    roll_rules = raw.get("roll_rules", {})
    selection_cfg = raw.get("selection", {}) or {}
    selection_cfg.setdefault("mode", "expiry_v1")  # 'expiry_v1' | 'expiry_v2'
    time_cfg = raw.get("time", {}) or {}
    time_cfg.setdefault("tz_exchange", "America/Chicago")
    expiries_cfg = raw.get("expiries", {}) or {}
    dq_cfg = raw.get("data_quality", {})

    strip_analysis_cfg = raw.get("strip_analysis", {}) or {}
    strip_analysis_cfg.setdefault("enabled", True)
    strip_analysis_cfg.setdefault("strip_length", 12)
    strip_analysis_cfg.setdefault("dominance_ratio_threshold", 2.0)
    strip_analysis_cfg.setdefault("expiry_window_business_days", 18)
    strip_analysis_cfg.setdefault("filter_expiry_dominance", True)

    business_days_cfg = raw.get("business_days", {})
    # Always enabled; ignore any 'enabled' field if present
    if "enabled" in business_days_cfg:
        business_days_cfg.pop("enabled", None)
    business_days_cfg.setdefault("align_events", "none")
    if "calendar_paths" in business_days_cfg:
        resolved_paths = [_resolve_path(p, base_dir) for p in business_days_cfg["calendar_paths"]]
        business_days_cfg["calendar_paths"] = [p for p in resolved_paths if p is not None]

    metadata_path = _resolve_path(raw.get("metadata", {}).get("contracts"), base_dir)
    if metadata_path is None:
        metadata_path = _resolve_path(
            raw.get("metadata_path", "../metadata/contracts_metadata.csv"), base_dir
        )

    output_dir = _resolve_path(raw.get("output_dir", "../outputs"), base_dir)
    if output_dir is None:
        output_dir = base_dir.parent / "outputs"

    data_root = _resolve_path(data_cfg.get("minute_root"), base_dir)
    if data_root is not None:
        data_cfg["minute_root"] = data_root

    _validate_settings(products, data_cfg, spread_cfg, business_days_cfg)

    return Settings(
        products=products,
        bucket_config=bucket_config,
        data=data_cfg,
        data_quality=dq_cfg,
        roll_rules=roll_rules,
        selection=selection_cfg,
        time=time_cfg,
        expiries=expiries_cfg,
        strip_analysis=strip_analysis_cfg,
        spread=spread_cfg,
        business_days=business_days_cfg,
        output_dir=output_dir,
        metadata_path=metadata_path,
    )


def _merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _validate_settings(products: Iterable[str], data_cfg: Dict[str, Any], spread_cfg: Dict[str, Any], business_days_cfg: Dict[str, Any]) -> None:
    # Basic validations with helpful messages
    tz = data_cfg.get("timezone")
    if not isinstance(tz, str):
        raise ValueError("data.timezone must be a string timezone name")

    method = spread_cfg.get("method", "zscore").lower()
    if method not in {"zscore", "abs", "combined"}:
        raise ValueError("spread.method must be one of: zscore, abs, combined")
    # No noise-reduction features (winsorization) in this iteration

    align = business_days_cfg.get("align_events", "none")
    if align not in {"none", "shift_next", "drop_closed"}:
        raise ValueError("business_days.align_events must be one of: none, shift_next, drop_closed")

    vt = business_days_cfg.get("volume_threshold", {}) or {}
    if vt:
        method_vt = vt.get("method", "dynamic")
        if method_vt not in {"dynamic", "fixed"}:
            raise ValueError("business_days.volume_threshold.method must be 'dynamic' or 'fixed'")
        ranges = vt.get("dynamic_ranges")
        if method_vt == "dynamic":
            if not isinstance(ranges, list) or not ranges:
                raise ValueError("business_days.volume_threshold.dynamic_ranges must be a non-empty list")
            last = -1
            for r in ranges:
                md = int(r.get("max_days", -1))
                if md <= last:
                    raise ValueError("dynamic_ranges must be strictly increasing by max_days")
                last = md
