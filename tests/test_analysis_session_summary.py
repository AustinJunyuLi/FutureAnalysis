from __future__ import annotations

import pandas as pd

import pytest

from futures_roll_analysis.analysis import _session_event_summary


def test_session_event_summary_grouping():
    df = pd.DataFrame(
        {
            "bucket": [1, 2, 8, 9],
            "session": ["US Regular", "US Regular", "Late US", "Asia"],
            "event_count": [10, 5, 3, 2],
        }
    )
    summary = _session_event_summary(df)
    assert list(summary["session"]) == ["US Regular", "Late US", "Asia"]
    assert summary.loc[summary["session"] == "US Regular", "event_count"].iloc[0] == 15
    assert summary["event_share_pct"].sum() == pytest.approx(100.0)
