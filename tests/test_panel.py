"""Tests for panel assembly."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from futures_roll_analysis.panel import assemble_panel


def test_bucket_metadata_union():
    idx = pd.date_range("2024-01-01", periods=2, freq="H")

    frames = {
        "HGZ2024": pd.DataFrame(
            {
                "open": [4.1, 4.2],
                "close": [4.15, 4.25],
                "bucket": [np.nan, 2],
                "bucket_label": [np.nan, "US Morning"],
                "session": [np.nan, "US Regular"],
            },
            index=idx,
        ),
        "HGF2025": pd.DataFrame(
            {
                "open": [4.3, 4.4],
                "close": [4.35, 4.45],
                "bucket": [1, 2],
                "bucket_label": ["US Open", "US Morning"],
                "session": ["US Regular", "US Regular"],
            },
            index=idx,
        ),
    }

    metadata = pd.DataFrame(
        {
            "contract": ["HGZ2024", "HGF2025"],
            "expiry_date": ["2024-12-15", "2025-01-15"],
        }
    )

    panel = assemble_panel(frames, metadata)

    assert ("meta", "bucket") in panel.columns
    assert panel["meta", "bucket"].iloc[0] == 1
    assert panel["meta", "bucket"].iloc[1] == 2
