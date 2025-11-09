from __future__ import annotations

from pathlib import Path

from futures_roll_analysis.config import load_settings


def test_load_settings_accepts_empty_calendar_paths(tmp_path):
    metadata_path = tmp_path / "contracts.csv"
    metadata_path.write_text("contract,expiry_date\n")

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        f"""
products: ["HG"]
data:
  minute_root: "{tmp_path}"
  timezone: "US/Central"
  price_field: "close"
metadata:
  contracts: "{metadata_path}"
business_days:
  calendar_paths: []
"""
    )

    settings = load_settings(Path(settings_path))

    assert settings.business_days.get("calendar_paths") == []
