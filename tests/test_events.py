"""
Tests for event detection module
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the project
from etf_roll_analysis.src.roll_analysis.events import detect_widening


class TestEventDetection:
    """Test suite for roll event detection"""
    
    def test_detect_widening_basic(self):
        """Test basic widening detection with synthetic data"""
        # Create sample spread data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        spread = pd.Series(np.random.randn(100) * 0.01, index=dates)
        
        # Add artificial widening event at position 50
        spread.iloc[50] = spread.iloc[49] + 0.1  # Large positive change
        
        # Detect events
        events = detect_widening(spread, method='zscore', window=20, z_threshold=1.5)
        
        # Should detect at least one event
        assert events.sum() >= 1, "Should detect at least one widening event"
        
        # The artificial event should be detected (around position 50-51)
        assert events.iloc[49:52].any(), "Should detect the artificial widening around position 50"
    
    def test_detect_widening_no_events(self):
        """Test that no events are detected in flat data"""
        # Create flat spread data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        spread = pd.Series(0.05, index=dates)  # Constant spread
        
        # Detect events
        events = detect_widening(spread, method='zscore', window=20, z_threshold=1.5)
        
        # Should detect no events in flat data
        assert events.sum() == 0, "Should detect no events in flat spread data"
    
    def test_detect_widening_with_nan(self):
        """Test that detection handles NaN values properly"""
        # Create spread with NaN values
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        spread = pd.Series(np.random.randn(100) * 0.01, index=dates)
        spread.iloc[10:15] = np.nan  # Add some NaN values
        
        # Detect events - should not crash
        events = detect_widening(spread, method='zscore', window=20, z_threshold=1.5)
        
        # Should return a series of same length
        assert len(events) == len(spread), "Output should have same length as input"
        
        # NaN positions should be False
        assert not events.iloc[10:15].any(), "NaN positions should not be events"
    
    def test_cool_down_period(self):
        """Test that cool-down period prevents duplicate signals"""
        # Create spread with two consecutive large changes
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        spread = pd.Series(np.random.randn(100) * 0.01, index=dates)
        spread.iloc[50] = spread.iloc[49] + 0.1  # Large change
        spread.iloc[51] = spread.iloc[50] + 0.1  # Another large change
        
        # Detect events with cool-down
        events = detect_widening(spread, method='zscore', window=20, 
                                z_threshold=1.5, cool_down=3)
        
        # Should not detect both consecutive events due to cool-down
        consecutive_events = events.iloc[50] and events.iloc[51]
        assert not consecutive_events, "Cool-down should prevent consecutive events"
    
    def test_different_thresholds(self):
        """Test that lower threshold detects more events"""
        # Create sample spread data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        spread = pd.Series(np.random.randn(200) * 0.02, index=dates)
        
        # Detect with different thresholds
        events_high = detect_widening(spread, method='zscore', window=20, z_threshold=2.0)
        events_low = detect_widening(spread, method='zscore', window=20, z_threshold=1.0)
        
        # Lower threshold should detect more events
        assert events_low.sum() >= events_high.sum(), "Lower threshold should detect more events"
        
        # High threshold events should be subset of low threshold events
        high_event_dates = events_high[events_high].index
        for date in high_event_dates:
            assert events_low[date], "High threshold events should be subset of low threshold"


if __name__ == "__main__":
    # Run tests
    test_class = TestEventDetection()
    
    print("Running event detection tests...")
    test_class.test_detect_widening_basic()
    print("✓ Basic detection test passed")
    
    test_class.test_detect_widening_no_events()
    print("✓ No events test passed")
    
    test_class.test_detect_widening_with_nan()
    print("✓ NaN handling test passed")
    
    test_class.test_cool_down_period()
    print("✓ Cool-down test passed")
    
    test_class.test_different_thresholds()
    print("✓ Threshold comparison test passed")
    
    print("\nAll tests passed!")
