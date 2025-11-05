from __future__ import annotations

import pandas as pd

from futures_roll_analysis import spreads


def test_summarize_strip_dominance_classifies_expiry_and_roll():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    magnitude = pd.DataFrame(
        {
            "s1_change": [0.1, 0.1, 1.0, 0.2, 0.2, 1.2],
            "others_median": [0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
            "dominance_ratio": [1.0, 1.0, 5.0, 1.1, 1.2, 4.0],
        },
        index=idx,
    )
    contract_chain = pd.DataFrame(
        {
            "F1": ["HGF2024"] * 3 + ["HGG2024"] * 3,
        },
        index=idx,
    )
    expiry_map = pd.Series(
        {
            "HGF2024": pd.Timestamp("2024-01-05"),
            "HGG2024": pd.Timestamp("2024-02-20"),
        }
    )
    calendar = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", "2024-02-28", freq="D"),
            "is_trading_day": [
                day.weekday() < 5 for day in pd.date_range("2024-01-01", "2024-02-28", freq="D")
            ],
        }
    )

    diag = spreads.summarize_strip_dominance(
        magnitude,
        contract_chain,
        expiry_map,
        calendar=calendar,
        dominance_threshold=2.0,
        expiry_window_bd=3,
    )

    expiry_days = diag[diag["classification"] == "expiry_dominance"]
    broad_days = diag[diag["classification"] == "broad_roll"]

    assert not expiry_days.empty
    assert not broad_days.empty
    assert (expiry_days["front_contract"] == "HGF2024").all()
    assert (broad_days["front_contract"] == "HGG2024").all()


def test_filter_expiry_dominance_events_masks_matching_days():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    events = pd.Series([False, True, True, False], index=idx)
    diagnostics = pd.DataFrame(
        {
            "front_contract": ["HGF2024"] * 2,
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "s1_change": [0.5, 0.6],
            "other_median_change": [0.1, 0.1],
            "dominance_ratio": [4.0, 3.0],
            "business_days_to_expiry": [2, 1],
            "classification": ["expiry_dominance", "normal"],
        }
    )

    filtered, removed = spreads.filter_expiry_dominance_events(events, diagnostics)

    assert removed == 1
    assert bool(filtered.loc["2024-01-02"]) is False
    assert bool(filtered.loc["2024-01-03"]) is True
