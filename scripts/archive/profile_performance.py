#!/usr/bin/env python
"""Performance profiling script for Futures Roll Analysis."""

import time
import tracemalloc
from pathlib import Path
import pandas as pd
import numpy as np
from futures_roll_analysis.ingest import find_contract_files, load_minutes
from futures_roll_analysis.labeler import compute_strip_labels
from futures_roll_analysis.buckets import aggregate_to_buckets
from futures_roll_analysis.events import detect_spread_events

def profile_operation(func, *args, **kwargs):
    """Profile a single operation for time and memory."""
    tracemalloc.start()
    start_time = time.perf_counter()

    result = func(*args, **kwargs)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, elapsed, peak / 1024 / 1024  # Convert to MB

def main():
    """Run performance profiling on key operations."""
    print("Performance Profiling Results")
    print("=" * 50)

    # Test data setup
    data_root = Path("../organized_data/copper")

    # 1. File discovery
    files = find_contract_files(data_root, "HG")[:5]  # Limit for testing
    print(f"\n1. File Discovery: {len(files)} files found")

    # 2. Profile data loading
    if files:
        file_path = files[0]
        df, elapsed, memory = profile_operation(load_minutes, file_path)
        print(f"\n2. Data Loading:")
        print(f"   - Time: {elapsed:.3f} seconds")
        print(f"   - Memory: {memory:.2f} MB")
        print(f"   - Rows: {len(df)}")

        # 3. Profile bucket aggregation
        if len(df) > 0:
            bucket_df, elapsed, memory = profile_operation(
                aggregate_to_buckets, df, bucket_config=None
            )
            print(f"\n3. Bucket Aggregation:")
            print(f"   - Time: {elapsed:.3f} seconds")
            print(f"   - Memory: {memory:.2f} MB")
            print(f"   - Throughput: {len(df)/elapsed:.0f} rows/second")

    # 4. Profile labeling
    # Create synthetic expiry data
    contract_order = [f'HG{chr(70+i)}2024' for i in range(12)]  # HGF2024 to HGQ2024
    expiries_utc = {
        contract: pd.Timestamp(f'2024-{i+1:02d}-28 17:00:00', tz='UTC')
        for i, contract in enumerate(contract_order)
    }

    timestamps = pd.date_range('2024-01-01', '2024-12-31', freq='h', tz='UTC')
    labels, elapsed, memory = profile_operation(
        compute_strip_labels, timestamps, contract_order, expiries_utc
    )
    print(f"\n4. Strip Labeling:")
    print(f"   - Time: {elapsed:.3f} seconds")
    print(f"   - Memory: {memory:.2f} MB")
    print(f"   - Timestamps: {len(timestamps)}")

    # 5. Profile event detection
    spread_data = pd.Series(
        np.random.randn(1000) * 0.01,
        index=pd.date_range('2024-01-01', periods=1000, freq='h')
    )
    events, elapsed, memory = profile_operation(
        detect_spread_events, spread_data, method='zscore', window=20, z_threshold=2.0
    )
    print(f"\n5. Event Detection:")
    print(f"   - Time: {elapsed:.3f} seconds")
    print(f"   - Memory: {memory:.2f} MB")
    print(f"   - Events detected: {events.sum()}")

    print("\n" + "=" * 50)
    print("Profiling complete.")

if __name__ == "__main__":
    main()