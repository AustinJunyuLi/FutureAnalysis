from __future__ import annotations

import pandas as pd

from futures_roll_analysis.labeler import compute_strip_labels


def test_strip_switches_at_expiry_instant():
    # Two contracts: C1 expires earlier than C2
    c1, c2 = "HGX2025", "HGF2026"
    expiries = {
        c1: pd.Timestamp("2025-01-15T12:00:00Z"),
        c2: pd.Timestamp("2025-03-15T12:00:00Z"),
    }
    # Build UTC index around the expiry instant of C1
    idx = pd.date_range("2025-01-15T11:58:00Z", periods=5, freq="min", tz="UTC")
    strip = compute_strip_labels(idx, [c1, c2], expiries, depth=2)
    # Before expiry, F1 is c1; at expiry and after, F1 is c2
    assert strip.loc["2025-01-15T11:58:00Z", "F1"] == c1
    assert strip.loc["2025-01-15T11:59:00Z", "F1"] == c1
    assert strip.loc["2025-01-15T12:00:00Z", "F1"] == c2
    assert strip.loc["2025-01-15T12:01:00Z", "F1"] == c2


def test_f2_becomes_f1_exactly_at_expiry():
    c1, c2, c3 = "C1", "C2", "C3"
    expiries = {
        c1: pd.Timestamp("2025-01-01T00:00:00Z"),
        c2: pd.Timestamp("2025-02-01T00:00:00Z"),
        c3: pd.Timestamp("2025-03-01T00:00:00Z"),
    }
    idx = pd.date_range("2024-12-31T23:58:00Z", periods=5, freq="min", tz="UTC")
    strip = compute_strip_labels(idx, [c1, c2, c3], expiries, depth=2)
    # Just before expiry: F1=c1, F2=c2; at expiry: F1=c2
    assert strip.loc["2024-12-31T23:59:00Z", "F1"] == c1
    assert strip.loc["2024-12-31T23:59:00Z", "F2"] == c2
    assert strip.loc["2025-01-01T00:00:00Z", "F1"] == c2


def test_front_next_equal_f1_f2():
    c1, c2 = "A", "B"
    expiries = {
        c1: pd.Timestamp("2025-01-10T00:00:00Z"),
        c2: pd.Timestamp("2025-02-10T00:00:00Z"),
    }
    idx = pd.date_range("2025-01-09T23:58:00Z", periods=3, freq="min", tz="UTC")
    strip = compute_strip_labels(idx, [c1, c2], expiries, depth=2)
    # F1/F2 are consistent across the 2-depth strip
    assert (strip.columns[:2] == ["F1", "F2"]).all()
    assert strip.shape[1] >= 2
    # Simple sanity: the sorted expiries determine order
    assert strip.iloc[0]["F1"] == c1
    assert strip.iloc[-1]["F1"] == c2

