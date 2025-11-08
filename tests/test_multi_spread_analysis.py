"""
Tests for multi-spread comparative analysis module
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure src directory is importable for local execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import from the project
from futures_roll_analysis.multi_spread_analysis import (
    compute_spread_correlations,
    compare_spread_signals,
    analyze_spread_timing,
    summarize_timing_by_spread,
    analyze_spread_changes,
    compare_spread_magnitudes,
    analyze_s1_dominance_by_expiry_cycle,
    summarize_cross_spread_patterns,
)


class TestMultiSpreadAnalysis:
    """Test suite for multi-spread comparative analysis"""

    @pytest.fixture
    def sample_spreads(self):
        """Create sample spread data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        spreads = pd.DataFrame({
            'S1': np.random.randn(100) * 0.01 + 0.05,
            'S2': np.random.randn(100) * 0.008 + 0.04,
            'S3': np.random.randn(100) * 0.006 + 0.03,
        }, index=dates)
        return spreads

    @pytest.fixture
    def sample_events(self):
        """Create sample event data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        np.random.seed(42)
        events = pd.DataFrame({
            'S1_events': np.random.random(100) > 0.95,  # ~5% detection rate
            'S2_events': np.random.random(100) > 0.97,  # ~3% detection rate
            'S3_events': np.random.random(100) > 0.98,  # ~2% detection rate
        }, index=dates)
        return events

    @pytest.fixture
    def sample_contract_chain(self):
        """Create sample contract chain for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        chain = pd.DataFrame({
            'F1': ['HGH2024'] * 50 + ['HGK2024'] * 50,
            'F2': ['HGK2024'] * 50 + ['HGM2024'] * 50,
            'F3': ['HGM2024'] * 50 + ['HGN2024'] * 50,
        }, index=dates)
        return chain

    @pytest.fixture
    def sample_expiry_map(self):
        """Create sample expiry map for testing"""
        return pd.Series({
            'HGH2024': pd.Timestamp('2024-01-25'),
            'HGK2024': pd.Timestamp('2024-02-25'),
            'HGM2024': pd.Timestamp('2024-03-25'),
            'HGN2024': pd.Timestamp('2024-04-25'),
        })

    def test_compute_spread_correlations_basic(self, sample_spreads):
        """Test basic correlation computation"""
        corr = compute_spread_correlations(sample_spreads)

        # Should be a square matrix
        assert corr.shape == (3, 3), "Correlation matrix should be 3x3"

        # Diagonal should be 1.0
        assert np.allclose(np.diag(corr), 1.0), "Diagonal correlations should be 1.0"

        # Should be symmetric
        assert np.allclose(corr, corr.T), "Correlation matrix should be symmetric"

    def test_compute_spread_correlations_perfect(self):
        """Test correlations with perfectly correlated data"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        spreads = pd.DataFrame({
            'S1': np.linspace(0, 1, 50),
            'S2': np.linspace(0, 1, 50),  # Identical to S1
        }, index=dates)

        corr = compute_spread_correlations(spreads)

        # S1 and S2 should be perfectly correlated
        assert np.abs(corr.loc['S1', 'S2'] - 1.0) < 0.001, "Identical series should have corr ≈ 1.0"

    def test_compare_spread_signals_basic(self, sample_spreads, sample_events):
        """Test spread signal comparison"""
        comparison = compare_spread_signals(sample_spreads, sample_events)

        # Should have one row per spread
        assert len(comparison) == 3, "Should have 3 rows for S1, S2, S3"

        # Should have expected columns
        expected_cols = {'spread', 'event_count', 'detection_rate_pct', 'mean_spread',
                        'std_spread', 'mean_event_spread', 'std_event_spread', 'signal_to_noise'}
        assert set(comparison.columns) == expected_cols, "Should have all expected columns"

        # Event counts should match
        assert comparison.loc[comparison['spread'] == 'S1', 'event_count'].iloc[0] == sample_events['S1_events'].sum()

    def test_compare_spread_signals_empty_events(self, sample_spreads):
        """Test with no events detected"""
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        empty_events = pd.DataFrame({
            'S1_events': [False] * 100,
            'S2_events': [False] * 100,
            'S3_events': [False] * 100,
        }, index=dates)

        comparison = compare_spread_signals(sample_spreads, empty_events)

        # Should still produce results
        assert len(comparison) == 3

        # All event counts should be 0
        assert (comparison['event_count'] == 0).all()

        # Detection rates should be 0
        assert (comparison['detection_rate_pct'] == 0).all()

    def test_analyze_spread_timing_basic(self, sample_spreads, sample_events,
                                         sample_contract_chain, sample_expiry_map):
        """Test basic timing analysis"""
        timing = analyze_spread_timing(
            sample_spreads,
            sample_events,
            sample_contract_chain,
            sample_expiry_map,
            tz_exchange="America/Chicago"
        )

        # Should have rows for detected events
        total_events = sample_events['S1_events'].sum() + sample_events['S2_events'].sum() + sample_events['S3_events'].sum()
        # Some events might be filtered if contracts/expiries missing
        assert len(timing) <= total_events, "Should have rows for events"

        # Should have expected columns
        expected_cols = {'spread', 'event_date', 'contract', 'expiry_date',
                        'days_to_expiry', 'hours_to_expiry', 'f1_contract', 'f1_expiry'}
        assert set(timing.columns) == expected_cols, "Should have all expected columns"

    def test_analyze_spread_timing_empty(self, sample_spreads, sample_contract_chain, sample_expiry_map):
        """Test timing analysis with no events"""
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        empty_events = pd.DataFrame({
            'S1_events': [False] * 100,
            'S2_events': [False] * 100,
        }, index=dates)

        timing = analyze_spread_timing(
            sample_spreads[['S1', 'S2']],
            empty_events,
            sample_contract_chain,
            sample_expiry_map
        )

        # Should return empty DataFrame with correct columns
        assert len(timing) == 0
        assert 'spread' in timing.columns
        assert 'days_to_expiry' in timing.columns

    def test_summarize_timing_by_spread_basic(self, sample_spreads, sample_events,
                                              sample_contract_chain, sample_expiry_map):
        """Test timing summary computation"""
        timing = analyze_spread_timing(
            sample_spreads,
            sample_events,
            sample_contract_chain,
            sample_expiry_map
        )

        summary = summarize_timing_by_spread(timing)

        # Should have summary statistics columns
        expected_cols = {'spread', 'event_count', 'median_days_to_expiry',
                        'mean_days_to_expiry', 'std_days_to_expiry',
                        'q25_days_to_expiry', 'q75_days_to_expiry'}
        assert set(summary.columns) == expected_cols, "Should have all summary columns"

        # Event counts should match timing data
        if len(timing) > 0:
            for _, row in summary.iterrows():
                spread = row['spread']
                timing_count = len(timing[timing['spread'] == spread])
                assert row['event_count'] == timing_count

    def test_summarize_timing_by_spread_empty(self):
        """Test timing summary with empty input"""
        empty_timing = pd.DataFrame(columns=['spread', 'event_date', 'contract',
                                             'expiry_date', 'days_to_expiry',
                                             'f1_contract', 'f1_expiry'])

        summary = summarize_timing_by_spread(empty_timing)

        # Should return empty DataFrame with correct structure
        assert len(summary) == 0
        assert 'median_days_to_expiry' in summary.columns

    def test_analyze_spread_changes_basic(self, sample_spreads):
        """Test spread change analysis"""
        changes = analyze_spread_changes(sample_spreads)

        # Should have one row per spread
        assert len(changes) == 3

        # Should have expected columns
        expected_cols = {'spread', 'mean_change', 'std_change', 'abs_mean_change',
                        'max_increase', 'max_decrease'}
        assert set(changes.columns) == expected_cols

        # Standard deviation should be positive
        assert (changes['std_change'] > 0).all()

    def test_compare_spread_magnitudes_basic(self, sample_spreads):
        """Test spread magnitude comparison"""
        magnitudes = compare_spread_magnitudes(sample_spreads)

        # Should have one row per timestamp
        assert len(magnitudes) == len(sample_spreads)

        # Should have expected columns
        expected_cols = {'s1_change', 'others_median', 'others_mean',
                        'dominance_ratio', 's1_dominates', 'rank_s1'}
        assert set(magnitudes.columns) == expected_cols

        # Dominance ratio should be positive (ignoring NaN from first diff)
        assert (magnitudes['dominance_ratio'].dropna() >= 0).all()

        # Ranks should be between 1 and number of spreads (ignoring NaN from first diff)
        assert (magnitudes['rank_s1'].dropna() >= 1).all()
        assert (magnitudes['rank_s1'].dropna() <= 3).all()

    def test_compare_spread_magnitudes_s1_always_dominant(self):
        """Test with S1 always having largest changes"""
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        spreads = pd.DataFrame({
            'S1': np.arange(50) * 0.1,  # Linear increase, large changes
            'S2': np.arange(50) * 0.01,  # Smaller changes
            'S3': np.arange(50) * 0.01,  # Smaller changes
        }, index=dates)

        magnitudes = compare_spread_magnitudes(spreads)

        # S1 should dominate most periods (after first NaN)
        dominance_rate = magnitudes['s1_dominates'].sum() / (len(magnitudes) - 1)  # Exclude first NaN
        assert dominance_rate > 0.8, "S1 should dominate >80% of periods"

    def test_analyze_s1_dominance_by_expiry_cycle(self, sample_spreads,
                                                   sample_contract_chain, sample_expiry_map):
        """Test S1 dominance analysis by expiry cycle"""
        magnitudes = compare_spread_magnitudes(sample_spreads)

        dominance = analyze_s1_dominance_by_expiry_cycle(
            magnitudes,
            sample_contract_chain,
            sample_expiry_map
        )

        # Should have buckets
        assert len(dominance) > 0, "Should produce cycle buckets"

        # Should have expected columns
        expected_cols = {'days_to_f1_expiry_bucket', 'total_observations',
                        's1_dominance_count', 's1_dominance_rate',
                        'avg_dominance_ratio', 'avg_s1_magnitude', 'avg_others_magnitude'}
        assert set(dominance.columns) == expected_cols

        # Rates should be between 0 and 100
        assert (dominance['s1_dominance_rate'] >= 0).all()
        assert (dominance['s1_dominance_rate'] <= 100).all()

    def test_summarize_cross_spread_patterns_no_dominance(self, sample_spreads,
                                                          sample_contract_chain, sample_expiry_map):
        """Test pattern summary when S1 doesn't dominate"""
        # Create data where S1 never dominates
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        weak_spreads = pd.DataFrame({
            'S1': np.random.randn(50) * 0.001,  # Very small changes
            'S2': np.random.randn(50) * 0.01,   # Larger changes
            'S3': np.random.randn(50) * 0.01,   # Larger changes
        }, index=dates)

        magnitudes = compare_spread_magnitudes(weak_spreads)
        dominance = analyze_s1_dominance_by_expiry_cycle(
            magnitudes,
            sample_contract_chain.iloc[:50],
            sample_expiry_map
        )

        summary = summarize_cross_spread_patterns(dominance, significance_threshold=10.0)

        # Should indicate no dominance
        assert len(summary) == 5, "Should have 5 summary rows"
        interpretation = summary.loc[summary['metric'] == 'Suggests institutional rolling', 'value'].iloc[0]
        assert interpretation == 'No', "Should not suggest institutional rolling"

    def test_summarize_cross_spread_patterns_with_dominance(self):
        """Test pattern summary with strong S1 dominance"""
        # Create mock dominance data with strong S1 signal
        dominance = pd.DataFrame({
            'days_to_f1_expiry_bucket': ['0-5', '6-10', '11-15', '16-20'],
            'total_observations': [100, 100, 100, 100],
            's1_dominance_count': [20, 5, 3, 2],
            's1_dominance_rate': [20.0, 5.0, 3.0, 2.0],
            'avg_dominance_ratio': [3.5, 2.1, 1.8, 1.5],
            'avg_s1_magnitude': [0.05, 0.02, 0.01, 0.01],
            'avg_others_magnitude': [0.01, 0.01, 0.01, 0.01],
        })

        summary = summarize_cross_spread_patterns(dominance, significance_threshold=10.0)

        # Should identify peak at 0-5 days
        peak_bucket = summary.loc[summary['metric'] == 'Peak dominance bucket', 'value'].iloc[0]
        assert peak_bucket == '0-5', "Should identify 0-5 days as peak"

        # Should suggest institutional rolling
        suggests = summary.loc[summary['metric'] == 'Suggests institutional rolling', 'value'].iloc[0]
        assert suggests == 'Yes', "Should suggest institutional rolling with strong signal"


if __name__ == "__main__":
    # Run tests
    test_class = TestMultiSpreadAnalysis()

    print("Running multi-spread analysis tests...")

    # Create fixtures
    spreads = test_class.sample_spreads(None)
    events = test_class.sample_events(None)
    chain = test_class.sample_contract_chain(None)
    expiry = test_class.sample_expiry_map(None)

    print("\n✓ Testing compute_spread_correlations...")
    test_class.test_compute_spread_correlations_basic(spreads)
    test_class.test_compute_spread_correlations_perfect()

    print("✓ Testing compare_spread_signals...")
    test_class.test_compare_spread_signals_basic(spreads, events)
    test_class.test_compare_spread_signals_empty_events(spreads)

    print("✓ Testing analyze_spread_timing...")
    test_class.test_analyze_spread_timing_basic(spreads, events, chain, expiry)
    test_class.test_analyze_spread_timing_empty(spreads, chain, expiry)

    print("✓ Testing summarize_timing_by_spread...")
    test_class.test_summarize_timing_by_spread_empty()

    print("✓ Testing analyze_spread_changes...")
    test_class.test_analyze_spread_changes_basic(spreads)

    print("✓ Testing compare_spread_magnitudes...")
    test_class.test_compare_spread_magnitudes_basic(spreads)
    test_class.test_compare_spread_magnitudes_s1_always_dominant()

    print("✓ Testing analyze_s1_dominance_by_expiry_cycle...")
    test_class.test_analyze_s1_dominance_by_expiry_cycle(spreads, chain, expiry)

    print("✓ Testing summarize_cross_spread_patterns...")
    test_class.test_summarize_cross_spread_patterns_with_dominance()

    print("\nAll multi-spread analysis tests passed!")
