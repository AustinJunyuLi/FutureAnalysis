from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from futures_roll_analysis.ingest import load_minutes


def test_load_minutes_honors_timestamp_format(tmp_path: Path) -> None:
    csv = "\n".join(
        [
            "timestamp,open,high,low,close,volume",
            "2024-01-02 09:00:00,4.0,4.5,3.9,4.2,10",
            "2024-01-02 09:01:00,4.1,4.6,4.0,4.3,12",
        ]
    )
    csv_path = tmp_path / "HGZ2024.csv"
    csv_path.write_text(csv)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        minutes = load_minutes(csv_path, timestamp_format="%Y-%m-%d %H:%M:%S")

    assert not record, f"Unexpected warnings: {[w.message for w in record]}"
    assert isinstance(minutes.index, pd.DatetimeIndex)
    assert minutes.index[0] == pd.Timestamp("2024-01-02 09:00:00")
