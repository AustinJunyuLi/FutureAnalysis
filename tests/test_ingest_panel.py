from __future__ import annotations

from pathlib import Path
import pandas as pd

from futures_roll_analysis.ingest import normalize_contract_code, minutes_to_daily
from futures_roll_analysis.buckets import aggregate_to_buckets, validate_bucket_aggregation


def test_normalize_contract_code_variants():
    assert normalize_contract_code('HGZ25.csv', 'HG') == 'HGZ2025'
    assert normalize_contract_code('hg_z_2025.txt', 'HG') == 'HGZ2025'
    assert normalize_contract_code('foo_HGZ9.parquet', 'HG') == 'HGZ2009'
    assert normalize_contract_code('HGK2024-data.csv', 'HG') == 'HGK2024'


def test_aggregation_conservation():
    # Build 1 day of minutes with deterministic values
    idx = pd.date_range('2024-03-01 09:00', periods=60*7, freq='min', tz='US/Central')
    # Create a small minute frame with monotonic prices and constant volume
    minutes = pd.DataFrame({
        'open': 4.0,
        'high': 4.5,
        'low': 3.5,
        'close': 4.2,
        'volume': 10,
    }, index=idx)

    bucket = aggregate_to_buckets(minutes)
    daily = minutes_to_daily(minutes)

    v = validate_bucket_aggregation(minutes, bucket)
    assert v['volume_conservation']
    assert v['high_price_consistency']
    assert v['low_price_consistency']

    # Daily totals match summed minutes
    assert daily['volume'].iloc[0] == 10 * len(minutes)
