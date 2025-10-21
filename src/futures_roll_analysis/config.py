from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    spread: Dict[str, Any]
    output_dir: Path
    metadata_path: Path


def _resolve_path(value: Optional[str], base_path: Path) -> Optional[Path]:
    if value is None:
        return None
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
    dq_cfg = raw.get("data_quality", {})

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

    return Settings(
        products=products,
        bucket_config=bucket_config,
        data=data_cfg,
        data_quality=dq_cfg,
        roll_rules=roll_rules,
        spread=spread_cfg,
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
