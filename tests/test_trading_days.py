from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from futures_roll_analysis import trading_days


EXPIRY_MAP = pd.Series(
    {
        "HGH24": pd.Timestamp("2024-03-27"),
        "HGK24": pd.Timestamp("2024-05-29"),
    }
)


class TestCalendarLoading:
    """Test calendar loading and hierarchy resolution."""

    def test_load_single_calendar(self, tmp_path):
        csv_path = tmp_path / "test_calendar.csv"
        csv_path.write_text(
            "date,holiday_name,session_note,partial_hours,exchanges_affected\n"
            "2024-01-01,New Year,Closed,,CME\n"
            "2024-12-25,Christmas,Closed,,CME\n"
            "2024-11-29,Day After Thanksgiving,Early close,09:00-13:00,CME\n"
        )

        calendar = trading_days.load_calendar(csv_path)

        assert len(calendar) == 3
        assert "date" in calendar.columns
        assert "is_trading_day" in calendar.columns
        assert calendar["is_trading_day"].sum() == 1
        assert calendar[calendar["date"] == pd.Timestamp("2024-11-29")]["is_trading_day"].iloc[0]

    def test_regular_holiday_is_trading_day(self, tmp_path):
        """A holiday with session_note='Regular' must be treated as open."""
        csv_path = tmp_path / "test_calendar_regular.csv"
        csv_path.write_text(
            "date,holiday_name,session_note,partial_hours,exchanges_affected\n"
            "2025-01-20,Martin Luther King Jr Day,Regular,,CME\n"
        )
        calendar = trading_days.load_calendar(csv_path)
        assert len(calendar) == 1
        assert bool(calendar.iloc[0]["is_trading_day"]) is True

    def test_calendar_override_hierarchy(self, tmp_path):
        cal1 = tmp_path / "cal1.csv"
        cal1.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-01-01,New Year,Closed,\n"
            "2024-07-04,Independence Day,Closed,\n"
        )

        cal2 = tmp_path / "cal2.csv"
        cal2.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-01-01,Custom Holiday,Open,\n"
            "2024-12-25,Christmas,Closed,\n"
        )

        calendar = trading_days.load_calendar([cal1, cal2], hierarchy="override")

        assert len(calendar) == 3
        jan1 = calendar[calendar["date"] == pd.Timestamp("2024-01-01")].iloc[0]
        assert jan1["holiday_name"] == "New Year"

    def test_calendar_union_hierarchy(self, tmp_path):
        cal1 = tmp_path / "cal1.csv"
        cal1.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-01-01,New Year,Closed,\n"
        )

        cal2 = tmp_path / "cal2.csv"
        cal2.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-12-25,Christmas,Closed,\n"
        )

        calendar = trading_days.load_calendar([cal1, cal2], hierarchy="union")

        assert len(calendar) == 2
        assert not calendar["is_trading_day"].any()

    def test_calendar_missing_file(self, tmp_path):
        nonexistent = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            trading_days.load_calendar([nonexistent])


class TestDynamicVolumeThreshold:
    """Test dynamic volume threshold computation."""

    def test_fixed_threshold(self):
        index = pd.date_range("2024-01-01", periods=100, freq="h")
        panel = pd.DataFrame({
            ("HGH24", "volume"): np.random.randint(100, 1000, 100),
            ("meta", "bucket"): 1,
        }, index=index)
        front_next = pd.DataFrame({
            "front_contract": "HGH24",
            "next_contract": "HGK24",
        }, index=index)
        days_to_expiry = pd.Series(60, index=index)

        config = {"method": "fixed", "fixed_percentile": 0.10}
        threshold = trading_days.compute_dynamic_volume_threshold(
            panel, front_next, days_to_expiry, config
        )

        assert len(threshold) == len(panel)
        assert threshold.iloc[0] == threshold.iloc[-1]

    def test_dynamic_threshold_near_expiry(self):
        index = pd.date_range("2024-01-01", periods=100, freq="h")
        panel = pd.DataFrame({
            ("HGH24", "volume"): np.random.randint(100, 1000, 100),
            ("meta", "bucket"): 1,
        }, index=index)
        front_next = pd.DataFrame({
            "front_contract": "HGH24",
            "next_contract": "HGK24",
        }, index=index)
        days_to_expiry = pd.Series(3, index=index)

        config = {
            "method": "dynamic",
            "dynamic_ranges": [
                {"max_days": 5, "percentile": 0.30},
                {"max_days": 999, "percentile": 0.05},
            ]
        }
        threshold = trading_days.compute_dynamic_volume_threshold(
            panel, front_next, days_to_expiry, config
        )

        assert len(threshold) == len(panel)
        assert threshold.iloc[0] > 0


class TestBusinessDayComputation:
    """Test business day computation with data guards."""

    def test_calendar_only_policy(self, tmp_path):
        csv_path = tmp_path / "test_calendar.csv"
        csv_path.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-01-01,New Year,Closed,\n"
            "2024-01-02,,Regular,\n"
            "2024-01-03,,Regular,\n"
        )
        calendar = trading_days.load_calendar(csv_path)

        index = pd.date_range("2024-01-02 09:00", periods=20, freq="h")
        panel = pd.DataFrame({
            ("HGH24", "close"): 4.0,
            ("HGH24", "volume"): 1000,
            ("HGK24", "close"): 4.01,
            ("HGK24", "volume"): 500,
            ("meta", "bucket"): 1,
        }, index=index)
        front_next = pd.DataFrame({
            "front_contract": "HGH24",
            "next_contract": "HGK24",
        }, index=index)

        biz_days = trading_days.compute_business_days(
            panel,
            front_next,
            calendar,
            expiry_map=EXPIRY_MAP,
            fallback_policy="calendar_only"
        )

        assert len(biz_days) == 2
        assert pd.Timestamp("2024-01-02") in biz_days
        assert pd.Timestamp("2024-01-03") in biz_days

    def test_data_coverage_guard(self, tmp_path):
        csv_path = tmp_path / "test_calendar.csv"
        csv_path.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-01-02,,Regular,\n"
            "2024-01-03,,Regular,\n"
        )
        calendar = trading_days.load_calendar(csv_path)

        index = pd.date_range("2024-01-02 09:00", periods=2, freq="h")
        panel = pd.DataFrame({
            ("HGH24", "close"): 4.0,
            ("HGH24", "volume"): 1000,
            ("HGK24", "close"): 4.01,
            ("HGK24", "volume"): 500,
            ("meta", "bucket"): 1,
        }, index=index)
        front_next = pd.DataFrame({
            "front_contract": "HGH24",
            "next_contract": "HGK24",
        }, index=index)

        biz_days = trading_days.compute_business_days(
            panel,
            front_next,
            calendar,
            expiry_map=EXPIRY_MAP,
            min_total_buckets=10,
            fallback_policy="intersection_strict"
        )

        assert len(biz_days) == 0

    def test_partial_day_handling(self, tmp_path):
        csv_path = tmp_path / "test_calendar.csv"
        csv_path.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-01-02,,Early close,09:00-13:00\n"
        )
        calendar = trading_days.load_calendar(csv_path)

        index = pd.date_range("2024-01-02 09:00", periods=5, freq="h")
        panel = pd.DataFrame({
            ("HGH24", "close"): 4.0,
            ("HGH24", "volume"): 1000,
            ("HGK24", "close"): 4.01,
            ("HGK24", "volume"): 500,
            ("meta", "bucket"): list(range(1, 6)),
        }, index=index)
        front_next = pd.DataFrame({
            "front_contract": "HGH24",
            "next_contract": "HGK24",
        }, index=index)

        biz_days = trading_days.compute_business_days(
            panel,
            front_next,
            calendar,
            expiry_map=EXPIRY_MAP,
            min_total_buckets=10,
            partial_day_min_buckets=4,
            fallback_policy="union_with_data"
        )

        assert len(biz_days) >= 1

    def test_fallback_policies(self, tmp_path):
        csv_path = tmp_path / "cal.csv"
        csv_path.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-01-01,New Year,Closed,\n"
            "2024-01-02,,Regular,\n"
            "2024-01-03,,Regular,\n"
        )
        calendar = trading_days.load_calendar(csv_path)

        # sparse data only on Jan 01 (closed) and Jan 03
        index = pd.to_datetime([
            "2024-01-01 10:00", "2024-01-03 10:00"
        ])
        panel = pd.DataFrame({
            ("HGH24", "close"): [4.0, 4.1],
            ("HGH24", "volume"): [100, 200],
            ("HGK24", "close"): [4.02, 4.11],
            ("HGK24", "volume"): [90, 180],
            ("meta", "bucket"): [1, 1],
        }, index=index)
        front_next = pd.DataFrame({
            "front_contract": ["HGH24", "HGH24"],
            "next_contract": ["HGK24", "HGK24"],
        }, index=index)

        # calendar_only should include only calendar-open dates that are observed in data
        biz_cal_only = trading_days.compute_business_days(
            panel, front_next, calendar, expiry_map=EXPIRY_MAP,
            fallback_policy="calendar_only",
            min_total_buckets=1,
            min_us_buckets=0,
            partial_day_min_buckets=1,
        )
        assert pd.Timestamp("2024-01-02") in calendar[calendar["is_trading_day"]]["date"].values
        assert pd.Timestamp("2024-01-02") not in biz_cal_only

        # intersection_strict should likely be only Jan 03 (has data and open)
        biz_intersection = trading_days.compute_business_days(
            panel, front_next, calendar, expiry_map=EXPIRY_MAP,
            fallback_policy="intersection_strict",
            min_total_buckets=1,
            min_us_buckets=0,
            partial_day_min_buckets=1,
        )
        assert pd.Timestamp("2024-01-03") in biz_intersection

        # union_with_data should include calendar-open and data-approved union (here Jan 03 only)
        biz_union = trading_days.compute_business_days(
            panel, front_next, calendar, expiry_map=EXPIRY_MAP,
            fallback_policy="union_with_data",
            min_total_buckets=1,
            min_us_buckets=0,
            partial_day_min_buckets=1,
        )
        assert pd.Timestamp("2024-01-03") in biz_union and pd.Timestamp("2024-01-02") not in biz_union


class TestBusinessDayGaps:
    """Test business day gap computation."""

    def test_empty_events(self):
        events = pd.Series([False] * 10, index=pd.date_range("2024-01-01", periods=10))
        biz_days = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=10))

        gaps = trading_days.business_day_gaps(events, biz_days)

        assert len(gaps) == 0

    def test_single_event(self):
        index = pd.date_range("2024-01-01", periods=10)
        events = pd.Series([False] * 10, index=index)
        events.iloc[5] = True
        biz_days = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=10))

        gaps = trading_days.business_day_gaps(events, biz_days)

        assert len(gaps) == 1
        assert pd.isna(gaps.iloc[0])

    def test_multiple_events(self):
        index = pd.date_range("2024-01-01", periods=10)
        events = pd.Series([False] * 10, index=index)
        events.iloc[2] = True
        events.iloc[6] = True
        biz_days = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=10))

        gaps = trading_days.business_day_gaps(events, biz_days)

        assert len(gaps) == 2
        assert pd.isna(gaps.iloc[0])
        assert gaps.iloc[1] == 4

    def test_weekend_gap(self):
        index = pd.date_range("2024-01-01", periods=10)
        events = pd.Series([False] * 10, index=index)
        events.iloc[4] = True
        events.iloc[7] = True

        biz_days_list = [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-04"),
            pd.Timestamp("2024-01-05"),
            pd.Timestamp("2024-01-08"),
            pd.Timestamp("2024-01-09"),
            pd.Timestamp("2024-01-10"),
        ]
        biz_days = pd.DatetimeIndex(biz_days_list)

        gaps = trading_days.business_day_gaps(events, biz_days)

        assert len(gaps) == 2
        assert gaps.iloc[1] >= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_calendar(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("date,holiday_name,session_note,partial_hours\n")

        calendar = trading_days.load_calendar(csv_path)
        assert len(calendar) == 0

    def test_invalid_hierarchy(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-01-01,New Year,Closed,\n"
        )

        with pytest.raises(ValueError, match="Unknown hierarchy"):
            trading_days.load_calendar([csv_path], hierarchy="invalid")

    def test_compute_business_days_requires_expiry_map(self, tmp_path):
        csv_path = tmp_path / "cal.csv"
        csv_path.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-01-02,,Regular,\n"
        )
        calendar = trading_days.load_calendar(csv_path)

        index = pd.date_range("2024-01-02 09:00", periods=5, freq="h")
        panel = pd.DataFrame({
            ("HGH24", "close"): 4.0,
            ("HGH24", "volume"): 100,
            ("HGK24", "close"): 4.01,
            ("HGK24", "volume"): 90,
        }, index=index)
        front_next = pd.DataFrame({
            "front_contract": "HGH24",
            "next_contract": "HGK24",
        }, index=index)

        with pytest.raises(ValueError):
            trading_days.compute_business_days(panel, front_next, calendar, expiry_map=None)


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_christmas_holiday(self, tmp_path):
        csv_path = tmp_path / "holidays.csv"
        csv_path.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-12-24,,Early close,09:00-13:00\n"
            "2024-12-25,Christmas,Closed,\n"
            "2024-12-26,,Regular,\n"
        )
        calendar = trading_days.load_calendar(csv_path)

        assert len(calendar[calendar["is_trading_day"]]) == 2

    def test_leap_day(self, tmp_path):
        csv_path = tmp_path / "leap.csv"
        csv_path.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-02-29,,Regular,\n"
        )
        calendar = trading_days.load_calendar(csv_path)

        assert len(calendar) == 1
        assert calendar.iloc[0]["is_trading_day"]

    def test_year_boundary(self, tmp_path):
        csv_path = tmp_path / "year_end.csv"
        csv_path.write_text(
            "date,holiday_name,session_note,partial_hours\n"
            "2024-12-31,,Early close,09:00-13:00\n"
            "2025-01-01,New Year,Closed,\n"
            "2025-01-02,,Regular,\n"
        )
        calendar = trading_days.load_calendar(csv_path)

        assert len(calendar[calendar["is_trading_day"]]) == 2
