"""
Comprehensive tests for contract labeling (F1/F2/F3...F12 strip identification).

Tests cover:
- Exact expiry instant boundary conditions
- DST fall-back and spring-forward handling
- Multi-contract strip generation
- Edge cases (no valid contracts, single contract)
- Expiry presence validation
"""

from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from futures_roll_analysis.rolls import (
    identify_front_next,
    identify_front_to_f12,
    build_expiry_map,
    compute_open_interest_signal,
)


def test_front_next_switches_exactly_at_expiry():
    """
    Regression test: F2 should become F1 exactly at expiry instant.

    This tests the supervisor's requirement: "F1 should appear at exactly
    the same time as the previous one expires."
    """
    # Create panel with three timestamps around expiry
    expiry_instant = pd.Timestamp("2025-03-27 17:00:00")
    index = pd.DatetimeIndex([
        expiry_instant - pd.Timedelta(minutes=1),  # Before expiry
        expiry_instant,                             # Exact expiry
        expiry_instant + pd.Timedelta(minutes=1),  # After expiry
    ])

    # Two contracts: HGH2025 expires at 17:00, HGK2025 expires later
    panel = pd.DataFrame(
        {
            ("HGH2025", "close"): [400.0, 400.5, 401.0],
            ("HGK2025", "close"): [405.0, 405.5, 406.0],
        },
        index=index,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)

    # Expiry map
    expiry_map = pd.Series({
        "HGH2025": expiry_instant,
        "HGK2025": pd.Timestamp("2025-05-29 17:00:00"),
    })

    # Call the labeler
    result = identify_front_next(panel, expiry_map, tz_exchange="America/Chicago")

    # Assertions
    # Before expiry: F1 = HGH2025 (old front)
    assert result.loc[expiry_instant - pd.Timedelta(minutes=1), "front_contract"] == "HGH2025"

    # At exact expiry instant: F1 = HGK2025 (F2 becomes F1)
    assert result.loc[expiry_instant, "front_contract"] == "HGK2025", \
        "F2 should become F1 exactly at the expiry instant of old F1"

    # After expiry: F1 = HGK2025 (new front continues)
    assert result.loc[expiry_instant + pd.Timedelta(minutes=1), "front_contract"] == "HGK2025"


def test_dst_fallback_both_occurrences():
    """
    Test that data around DST fall-back is handled without creating NaT values.

    We test with trading hours (9 AM - 5 PM) which avoid the ambiguous 1-2 AM window.
    Real-world futures trading data typically uses UTC or avoids overnight DST transitions.
    """
    # 2024-11-03 is DST fall-back day
    # Use trading hours timestamps (9 AM, 12 PM, 3 PM) to avoid 1-2 AM ambiguous window
    index = pd.DatetimeIndex([
        pd.Timestamp("2024-11-02 15:00:00"),  # Day before DST
        pd.Timestamp("2024-11-03 09:00:00"),  # Morning of DST day (unambiguous)
        pd.Timestamp("2024-11-03 12:00:00"),  # Midday of DST day (unambiguous)
        pd.Timestamp("2024-11-03 15:00:00"),  # Afternoon of DST day (unambiguous)
        pd.Timestamp("2024-11-04 09:00:00"),  # Day after DST
    ])

    panel = pd.DataFrame(
        {
            ("HGZ2024", "close"): [420.0, 420.5, 421.0, 421.5, 422.0],
            ("HGH2025", "close"): [425.0, 425.5, 426.0, 426.5, 427.0],
        },
        index=index,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)

    expiry_map = pd.Series({
        "HGZ2024": pd.Timestamp("2024-12-27 17:00:00"),
        "HGH2025": pd.Timestamp("2025-03-27 17:00:00"),
    })

    # This should NOT raise an error and should NOT produce NaT
    result = identify_front_next(panel, expiry_map, tz_exchange="America/Chicago")

    # Assert no NaT values in result
    assert not result["front_contract"].isna().any(), "Data around DST should not create NaT"
    assert not result["next_contract"].isna().any(), "Data around DST should not create NaT"

    # All timestamps should have valid contracts
    assert len(result) == len(index), "Should have result for all timestamps"


def test_dst_spring_forward_shifted():
    """
    DST spring-forward creates non-existent times (clock jumps forward).
    Using nonexistent='shift_forward' should handle this correctly.
    """
    # 2024-03-10 02:30:00 America/Chicago doesn't exist (DST spring-forward)
    # Clock jumps from 02:00 → 03:00

    nonexistent_time = pd.Timestamp("2024-03-10 02:30:00")
    index = pd.DatetimeIndex([
        pd.Timestamp("2024-03-10 01:30:00"),  # Before jump
        nonexistent_time,                      # Non-existent time
        pd.Timestamp("2024-03-10 03:30:00"),  # After jump
    ])

    panel = pd.DataFrame(
        {
            ("HGH2024", "close"): [400.0, 400.5, 401.0],
            ("HGK2024", "close"): [405.0, 405.5, 406.0],
        },
        index=index,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)

    expiry_map = pd.Series({
        "HGH2024": pd.Timestamp("2024-03-27 17:00:00"),
        "HGK2024": pd.Timestamp("2024-05-29 17:00:00"),
    })

    # This should NOT raise an error and should shift the time forward
    result = identify_front_next(panel, expiry_map, tz_exchange="America/Chicago")

    # Assert no NaT values
    assert not result["front_contract"].isna().any(), "DST non-existent times should be shifted, not NaT"
    assert not result["next_contract"].isna().any(), "DST non-existent times should be shifted, not NaT"


def test_utc_aware_index_handling():
    """
    Test that panel with already UTC-aware index is handled correctly (no double conversion).
    """
    # Create UTC-aware timestamps
    expiry_instant = pd.Timestamp("2025-03-27 22:00:00", tz="UTC")  # 17:00 Chicago = 22:00 UTC
    index = pd.DatetimeIndex([
        expiry_instant - pd.Timedelta(minutes=1),
        expiry_instant,
        expiry_instant + pd.Timedelta(minutes=1),
    ], tz="UTC")

    panel = pd.DataFrame(
        {
            ("HGH2025", "close"): [400.0, 400.5, 401.0],
            ("HGK2025", "close"): [405.0, 405.5, 406.0],
        },
        index=index,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)

    expiry_map = pd.Series({
        "HGH2025": pd.Timestamp("2025-03-27 17:00:00"),  # Naive, will be localized
        "HGK2025": pd.Timestamp("2025-05-29 17:00:00"),
    })

    # Should handle UTC-aware index without error
    result = identify_front_next(panel, expiry_map, tz_exchange="America/Chicago")

    # Should switch at the correct instant
    assert result.loc[expiry_instant - pd.Timedelta(minutes=1), "front_contract"] == "HGH2025"
    assert result.loc[expiry_instant, "front_contract"] == "HGK2025"


def test_full_strip_f1_to_f12():
    """Test that full 12-contract strip is populated correctly."""
    # Create 13 contracts with monthly expiries
    base_date = pd.Timestamp("2025-01-15 17:00:00")

    # Use realistic contract codes (HG uses H, K, N, U, Z months)
    month_codes = ['H', 'K', 'N', 'U', 'Z']
    contracts = []
    for i in range(13):
        year = 2025 + (i // 5)
        month_idx = i % 5
        contracts.append(f"HG{month_codes[month_idx]}{year}")

    expiry_map = pd.Series({
        contract: base_date + pd.DateOffset(months=i)
        for i, contract in enumerate(contracts)
    })

    # Create panel at a time when all contracts are active
    query_time = base_date - pd.Timedelta(days=30)
    panel = pd.DataFrame(
        {(contract, "close"): [400.0 + i] for i, contract in enumerate(contracts)},
        index=pd.DatetimeIndex([query_time]),
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)

    # Get 12-contract strip
    result = identify_front_to_f12(
        panel, expiry_map, tz_exchange="America/Chicago", max_contracts=12
    )

    # Assert all 12 positions populated
    for i in range(1, 13):
        col = f"F{i}"
        assert col in result.columns, f"Missing column {col}"
        assert result[col].notna().all(), f"Column {col} has NaT values"

    # Assert ordering is correct (F1 expires before F2, etc.)
    for i in range(1, 12):
        f_i = result.iloc[0][f"F{i}"]
        f_i_plus_1 = result.iloc[0][f"F{i+1}"]

        exp_i = expiry_map[f_i]
        exp_i_plus_1 = expiry_map[f_i_plus_1]

        assert exp_i < exp_i_plus_1, f"F{i} should expire before F{i+1}"


def test_no_valid_contracts_all_expired():
    """When all contracts have expired, should return None for all positions."""
    # All contracts expired yesterday
    expiry_yesterday = pd.Timestamp("2025-01-14 17:00:00")
    query_today = pd.Timestamp("2025-01-15 10:00:00")

    panel = pd.DataFrame(
        {
            ("HGH2025", "close"): [400.0],
            ("HGK2025", "close"): [405.0],
        },
        index=pd.DatetimeIndex([query_today]),
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)

    expiry_map = pd.Series({
        "HGH2025": expiry_yesterday - pd.Timedelta(days=1),
        "HGK2025": expiry_yesterday,
    })

    result = identify_front_next(panel, expiry_map, tz_exchange="America/Chicago")

    # When no valid contracts, should return None
    assert result.loc[query_today, "front_contract"] is None or pd.isna(result.loc[query_today, "front_contract"])
    assert result.loc[query_today, "next_contract"] is None or pd.isna(result.loc[query_today, "next_contract"])


def test_single_contract_only():
    """With only one valid contract, F1 should be populated but F2 should be None."""
    now = pd.Timestamp("2025-01-15 10:00:00")

    panel = pd.DataFrame(
        {
            ("HGH2025", "close"): [400.0],
            ("HGK2025", "close"): [405.0],
        },
        index=pd.DatetimeIndex([now]),
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)

    expiry_map = pd.Series({
        "HGH2025": now - pd.Timedelta(days=1),  # Expired
        "HGK2025": now + pd.Timedelta(days=30),  # Valid
    })

    result = identify_front_next(panel, expiry_map, tz_exchange="America/Chicago")

    # Only one valid contract
    assert result.loc[now, "front_contract"] == "HGK2025"
    assert result.loc[now, "next_contract"] is None or pd.isna(result.loc[now, "next_contract"])


def test_non_monotonic_index_handling():
    """
    Test that non-monotonic index is automatically sorted before tz operations.
    This is required for ambiguous='infer' to work.
    """
    # Create deliberately non-monotonic index
    expiry_instant = pd.Timestamp("2025-03-27 17:00:00")
    index = pd.DatetimeIndex([
        expiry_instant + pd.Timedelta(minutes=1),  # Out of order
        expiry_instant - pd.Timedelta(minutes=1),
        expiry_instant,
    ])

    panel = pd.DataFrame(
        {
            ("HGH2025", "close"): [401.0, 400.0, 400.5],
            ("HGK2025", "close"): [406.0, 405.0, 405.5],
        },
        index=index,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)

    expiry_map = pd.Series({
        "HGH2025": expiry_instant,
        "HGK2025": pd.Timestamp("2025-05-29 17:00:00"),
    })

    # Should handle non-monotonic index without error
    result = identify_front_next(panel, expiry_map, tz_exchange="America/Chicago")

    # Result should have entries for all timestamps
    assert len(result) == len(index)
    # No NaT values should be present
    assert not result["front_contract"].isna().any()


def test_compute_open_interest_signal_basic():
    index = pd.date_range("2025-03-01", periods=3, freq="D")
    panel = pd.DataFrame(
        {
            ("HGH2025", "open_interest"): [1000, 800, 600],
            ("HGK2025", "open_interest"): [500, 700, 900],
        },
        index=index,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)

    front_next = pd.DataFrame(
        {
            "front_contract": ["HGH2025"] * 3,
            "next_contract": ["HGK2025"] * 3,
        },
        index=index,
    )

    signals = compute_open_interest_signal(panel, front_next, ratio=0.7, confirm=1)
    assert signals.tolist() == [False, True, True]

    confirmed = compute_open_interest_signal(panel, front_next, ratio=0.7, confirm=2)
    assert confirmed.tolist() == [False, False, True]


def test_compute_open_interest_signal_missing_field_returns_none():
    index = pd.date_range("2025-03-01", periods=2, freq="D")
    panel = pd.DataFrame({("HGH2025", "close"): [4.0, 4.1]}, index=index)
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    front_next = pd.DataFrame(
        {"front_contract": ["HGH2025", "HGH2025"], "next_contract": ["HGK2025", "HGK2025"]},
        index=index,
    )

    assert compute_open_interest_signal(panel, front_next) is None
