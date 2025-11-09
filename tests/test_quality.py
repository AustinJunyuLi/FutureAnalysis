from __future__ import annotations

import pandas as pd

from futures_roll_analysis.quality import DataQualityFilter


def _make_frame(start: str = "2024-01-01", periods: int = 10) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame({"close": range(periods)}, index=index)


def test_quality_filter_evaluates_all_contracts_by_default():
    frames = {
        "HGA2024": _make_frame(),
        "CLA2024": _make_frame(),
    }
    dq = DataQualityFilter({"min_data_points": 1})

    filtered, metrics = dq.apply(frames)

    assert set(filtered.keys()) == set(frames.keys())
    assert {m["contract"] for m in metrics} == set(frames.keys())


def test_quality_filter_respects_commodity_guard():
    frames = {
        "HGA2024": _make_frame(),
        "CLA2024": _make_frame(),
    }
    dq = DataQualityFilter({"commodity": "HG", "min_data_points": 1})

    filtered, metrics = dq.apply(frames)

    assert set(filtered.keys()) == set(frames.keys())
    assert {m["contract"] for m in metrics} == {"HGA2024"}
