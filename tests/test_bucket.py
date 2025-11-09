"""
Tests for bucket aggregation module
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure src directory is on the path for local test execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import from the project
from futures_roll_analysis.buckets import (
    assign_bucket,
    get_bucket_info,
    aggregate_to_buckets,
    validate_bucket_aggregation,
    calculate_bucket_statistics,
)


class TestBucketAssignment:
    """Test suite for bucket assignment logic"""
    
    def test_us_regular_hours_assignment(self):
        """Test that US regular hours (9-15) map to buckets 1-7"""
        assert assign_bucket(9) == 1   # US Open
        assert assign_bucket(10) == 2  # US Morning
        assert assign_bucket(11) == 3  # US Late Morning
        assert assign_bucket(12) == 4  # US Midday
        assert assign_bucket(13) == 5  # US Early Afternoon
        assert assign_bucket(14) == 6  # US Late Afternoon
        assert assign_bucket(15) == 7  # US Close
    
    def test_off_peak_assignment(self):
        """Test that off-peak hours map to correct session buckets"""
        # Late US (16-20) -> Bucket 8
        for hour in [16, 17, 18, 19, 20]:
            assert assign_bucket(hour) == 8, f"Hour {hour} should map to bucket 8"
        
        # Asia Session (21-23, 0-2) -> Bucket 9
        for hour in [21, 22, 23, 0, 1, 2]:
            assert assign_bucket(hour) == 9, f"Hour {hour} should map to bucket 9"
        
        # Europe Session (3-8) -> Bucket 10
        for hour in [3, 4, 5, 6, 7, 8]:
            assert assign_bucket(hour) == 10, f"Hour {hour} should map to bucket 10"
    
    def test_all_hours_covered(self):
        """Test that all 24 hours map to a bucket"""
        for hour in range(24):
            bucket = assign_bucket(hour)
            assert 1 <= bucket <= 10, f"Hour {hour} maps to invalid bucket {bucket}"
    
    def test_invalid_hour_raises_error(self):
        """Test that invalid hours raise ValueError"""
        with pytest.raises(ValueError):
            assign_bucket(-1)
        with pytest.raises(ValueError):
            assign_bucket(24)
        with pytest.raises(ValueError):
            assign_bucket(100)


class TestBucketInfo:
    """Test suite for bucket information retrieval"""
    
    def test_all_buckets_defined(self):
        """Test that all 10 buckets have definitions"""
        for bucket_id in range(1, 11):
            info = get_bucket_info(bucket_id)
            assert info is not None, f"Bucket {bucket_id} has no definition"
            assert info.bucket_id == bucket_id
            assert info.label is not None
            assert info.session is not None
    
    def test_bucket_duration(self):
        """Test that bucket durations are correct"""
        # US hours should be 1 hour each
        for bucket_id in range(1, 8):
            info = get_bucket_info(bucket_id)
            assert info.duration_hours == 1, f"US bucket {bucket_id} should be 1 hour"
        
        # Late US should be 5 hours
        assert get_bucket_info(8).duration_hours == 5
        
        # Asia and Europe should be 6 hours each
        assert get_bucket_info(9).duration_hours == 6
        assert get_bucket_info(10).duration_hours == 6
    
    def test_invalid_bucket_raises_error(self):
        """Test that invalid bucket IDs raise ValueError"""
        with pytest.raises(ValueError):
            get_bucket_info(0)
        with pytest.raises(ValueError):
            get_bucket_info(11)
        with pytest.raises(ValueError):
            get_bucket_info(-1)


class TestAggregation:
    """Test suite for bucket aggregation"""
    
    def create_sample_minute_data(self, start_date='2024-01-15', hours=24):
        """Create sample minute-level OHLCV data"""
        start = pd.Timestamp(f'{start_date} 00:00:00')
        periods = hours * 60  # minutes
        
        dates = pd.date_range(start=start, periods=periods, freq='1min')
        
        # Generate synthetic price data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(periods) * 0.01)
        
        df = pd.DataFrame({
            'open': prices + np.random.randn(periods) * 0.001,
            'high': prices + abs(np.random.randn(periods) * 0.002),
            'low': prices - abs(np.random.randn(periods) * 0.002),
            'close': prices,
            'volume': np.random.randint(10, 100, periods)
        }, index=dates)
        
        return df
    
    def test_aggregate_to_buckets_basic(self):
        """Test basic bucket aggregation"""
        minute_df = self.create_sample_minute_data()
        bucket_df = aggregate_to_buckets(minute_df)
        
        # Should have buckets in result
        assert 'bucket' in bucket_df.columns
        assert 'bucket_label' in bucket_df.columns
        assert 'session' in bucket_df.columns
        
        # Check bucket values are valid
        assert bucket_df['bucket'].min() >= 1
        assert bucket_df['bucket'].max() <= 10
    
    def test_volume_conservation(self):
        """Test that total volume is preserved in aggregation"""
        minute_df = self.create_sample_minute_data()
        bucket_df = aggregate_to_buckets(minute_df)
        
        # Total volume should match (with small tolerance for float precision)
        minute_volume = minute_df['volume'].sum()
        bucket_volume = bucket_df['volume'].sum()
        
        assert abs(minute_volume - bucket_volume) < 1e-6, \
            f"Volume not conserved: minute={minute_volume}, bucket={bucket_volume}"
    
    def test_price_ranges(self):
        """Test that high/low price ranges are preserved"""
        minute_df = self.create_sample_minute_data()
        bucket_df = aggregate_to_buckets(minute_df)
        
        # Daily high should be max of bucket highs
        daily_high_minute = minute_df['high'].max()
        daily_high_bucket = bucket_df['high'].max()
        assert abs(daily_high_minute - daily_high_bucket) < 1e-6
        
        # Daily low should be min of bucket lows
        daily_low_minute = minute_df['low'].min()
        daily_low_bucket = bucket_df['low'].min()
        assert abs(daily_low_minute - daily_low_bucket) < 1e-6
    
    def test_bucket_count_per_day(self):
        """Test that we get expected number of buckets per day"""
        # Create 3 days of data
        minute_df = self.create_sample_minute_data(hours=72)
        bucket_df = aggregate_to_buckets(minute_df)

        trading_dates = bucket_df.index.normalize()
        trading_dates = trading_dates + pd.to_timedelta((bucket_df["bucket"] == 9).astype(int), unit="D")
        buckets_per_day = bucket_df.groupby(trading_dates)["bucket"].nunique()

        assert buckets_per_day.max() <= 10, "Should have at most 10 buckets per day"
        full_days = buckets_per_day[buckets_per_day >= 8]
        assert not full_days.empty, "Expected at least one fully populated trading day"
        assert full_days.min() >= 8, "Fully populated trading days should have at least eight buckets"
    
    def test_validation_function(self):
        """Test the validation function"""
        minute_df = self.create_sample_minute_data()
        bucket_df = aggregate_to_buckets(minute_df)
        
        validation = validate_bucket_aggregation(minute_df, bucket_df)
        
        # Check validation results
        assert 'volume_conservation' in validation
        assert 'high_price_consistency' in validation
        assert 'low_price_consistency' in validation
        
        # All validations should pass
        assert validation['volume_conservation'] == True
        assert validation['high_price_consistency'] == True
        assert validation['low_price_consistency'] == True

    def _create_dst_dataset(self, start: str, minutes: int) -> pd.DataFrame:
        idx = pd.date_range(start=start, periods=minutes, freq="min", tz="US/Central")
        data = np.linspace(100, 101, len(idx))
        return pd.DataFrame(
            {
                "open": data,
                "high": data + 0.1,
                "low": data - 0.1,
                "close": data,
                "volume": np.full(len(idx), 50),
            },
            index=idx,
        )

    def test_aggregate_handles_dst_spring_forward(self):
        minute_df = self._create_dst_dataset("2024-03-09 18:00", minutes=60 * 30)
        bucket_df = aggregate_to_buckets(minute_df)
        assert bucket_df.index.is_monotonic_increasing
        assert not bucket_df.index.duplicated().any()

    def test_aggregate_handles_dst_fall_back(self):
        minute_df = self._create_dst_dataset("2024-11-02 18:00", minutes=60 * 30)
        bucket_df = aggregate_to_buckets(minute_df)
        assert bucket_df.index.is_monotonic_increasing
        assert not bucket_df.index.duplicated().any()

    def test_cross_midnight_anchor_without_duplicates(self):
        """Cross-midnight buckets should anchor to the prior day without duplicate timestamps."""
        idx = pd.date_range("2024-01-03 20:30", periods=360, freq="1min")
        np.random.seed(0)
        minute_df = pd.DataFrame(
            {
                "open": np.linspace(100, 105, len(idx)),
                "high": np.linspace(100.5, 105.5, len(idx)),
                "low": np.linspace(99.5, 104.5, len(idx)),
                "close": np.linspace(100.1, 105.1, len(idx)),
                "volume": np.random.randint(10, 50, len(idx)),
            },
            index=idx,
        )

        bucket_df = aggregate_to_buckets(minute_df)

        assert bucket_df.index.is_unique, "Bucket aggregation should not create duplicate timestamps"
        asia = bucket_df[bucket_df["bucket"] == 9]
        assert not asia.empty
        earliest_asia = asia.index.min()
        assert earliest_asia.hour == 21


class TestBucketStatistics:
    """Test suite for bucket statistics calculation"""
    
    def test_calculate_statistics(self):
        """Test calculation of bucket statistics"""
        # Create sample bucket data
        minute_df = TestAggregation().create_sample_minute_data(hours=48)
        bucket_df = aggregate_to_buckets(minute_df)
        
        # Calculate statistics
        stats = calculate_bucket_statistics(bucket_df)
        
        # Should have 10 rows (one per bucket)
        assert len(stats) == 10
        
        # Check required columns
        required_cols = ['bucket', 'label', 'session', 'duration_hours', 'data_points']
        for col in required_cols:
            assert col in stats.columns, f"Missing column: {col}"
        
        # Buckets should be 1-10
        assert list(stats['bucket']) == list(range(1, 11))
        
        # Check that statistics make sense
        assert stats['data_points'].sum() == len(bucket_df)
        assert (stats['coverage_pct'].sum() - 100) < 0.1  # Should sum to ~100%


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        empty_df.index = pd.DatetimeIndex([])
        
        result = aggregate_to_buckets(empty_df)
        assert len(result) == 0
    
    def test_single_minute(self):
        """Test handling of single minute of data"""
        single_minute = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [50]
        }, index=pd.date_range('2024-01-15 10:30:00', periods=1, freq='1min'))
        
        result = aggregate_to_buckets(single_minute)
        assert len(result) == 1
        assert result['bucket'].iloc[0] == 2  # 10:30 is in bucket 2 (10:00-10:59)
    
    def test_missing_columns(self):
        """Test handling of missing OHLCV columns"""
        # Create data with only close and volume
        dates = pd.date_range('2024-01-15 09:00:00', periods=60, freq='1min')
        partial_df = pd.DataFrame({
            'close': np.random.randn(60) + 100,
            'volume': np.random.randint(10, 100, 60)
        }, index=dates)
        
        # Should handle missing columns gracefully
        result = aggregate_to_buckets(partial_df)
        assert 'close' in result.columns
        assert 'volume' in result.columns


if __name__ == "__main__":
    # Run tests
    test_classes = [
        TestBucketAssignment(),
        TestBucketInfo(),
        TestAggregation(),
        TestBucketStatistics(),
        TestEdgeCases()
    ]
    
    print("Running bucket module tests...")
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"  [PASS] {method_name}")
            except Exception as e:
                print(f"  [FAIL] {method_name}: {str(e)}")
    
    print("\nAll bucket tests completed!")
