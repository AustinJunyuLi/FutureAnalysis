# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python framework for analyzing calendar spread dynamics in futures markets using minute-level CME data (2008-2024). The codebase processes ~450M data points to detect spread widening events and characterize contract expiry mechanics through multi-spread comparative analysis.

**Key Discovery**: Multi-spread analysis (S1-S11) revealed that detected events reflect systematic contract maturity effects (occurring 28-30 days before any contract expires) rather than discretionary institutional roll timing decisions.

## Environment Setup

```bash
# Create conda environment (Python 3.11+)
conda create -n futures-roll python=3.11
conda activate futures-roll

# Install package in editable mode
pip install -e .[dev,viz]

# Reinstall after code changes
pip install -e . --no-deps
```

## Essential Commands

### Running Analysis

```bash
# Hourly (bucket) analysis - full dataset
futures-roll analyze --mode hourly --settings config/settings.yaml \
  --root organized_data/copper \
  --metadata metadata/contracts_metadata.csv \
  --output-dir outputs

# Quick test with limited files
futures-roll analyze --mode hourly --max-files 10 --output-dir outputs/test

# Daily analysis with quality filtering
futures-roll analyze --mode daily --settings config/settings.yaml \
  --root organized_data/copper \
  --output-dir outputs

# Organize raw data by commodity
futures-roll organize --source raw_data --destination organized_data
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_bucket.py -v

# With coverage report
pytest tests/ --cov=futures_roll_analysis --cov-report=html

# Run single test
pytest tests/test_bucket.py::TestBucketAssignment::test_us_regular_hours_assignment -v
```

### Report Generation

```bash
# Compile LaTeX report to PDF (must be in presentation_docs/)
cd presentation_docs
pdflatex -interaction=nonstopmode analysis_report.tex
pdflatex -interaction=nonstopmode analysis_report.tex  # Run twice for TOC
```

## Architecture Overview

### Core Data Pipeline

1. **Ingest** (`ingest.py`) → Load minute-level futures data from CSV/Parquet
2. **Aggregate** (`buckets.py`) → Convert to 10 intraday periods (US/Asia/Europe sessions) OR daily bars
3. **Panel Assembly** (`panel.py`) → Construct wide-format DataFrame with all contracts
4. **Contract Chain** (`rolls.py`) → Identify F1-F12 contracts at each timestamp using vectorized expiry-based logic
5. **Multi-Spread Computation** (`rolls.py`) → Calculate S1=F2-F1, S2=F3-F2, ..., S11=F12-F11 simultaneously
6. **Event Detection** (`events.py`) → Apply z-score methodology to detect spread widening across all spreads
7. **Comparative Analysis** (`multi_spread_analysis.py`) → Compare detection rates, timing, and correlations across S1-S11
8. **Summarization** (`events.py`) → Generate bucket statistics, preference scores, transition matrices

### Key Design Patterns

**Vectorization Throughout**: All operations use NumPy array operations to process 44K periods simultaneously. Never use iterative loops for per-period calculations.

**MultiIndex Columns**: Panel DataFrames use `(contract, field)` tuple columns:
```python
panel[("HGF2009", "close")]  # Price for specific contract
panel[("meta", "bucket")]     # Metadata column
```

**Metadata Namespace**: All non-price columns stored under `("meta", field_name)` to separate from contract data.

**Contract Identification Logic**:
- Compute days-to-expiry matrix: `(num_periods × num_contracts)`
- Mask expired/unavailable contracts → set delta to `np.inf`
- Use `argmin` to find nearest expiry → F1
- Set F1 to `np.inf`, repeat for F2, F3, ...
- This replaces O(n²) iteration with O(n×m) vectorized operations

**Event Detection Cool-down**: Time-based (3 hours) prevents cascade detections from single large moves. Enforced via timestamp comparison, not period count.

### Critical Module Interactions

**`analysis.py`** orchestrates everything:
- Calls `ingest.find_contract_files()` + `build_contract_frames()`
- Builds `expiry_map` from metadata
- Runs `identify_front_to_f12()` to get contract chain
- Computes `compute_multi_spreads()` for all S1-S11
- Detects events via `detect_multi_spread_events()`
- Performs comparative analysis via `multi_spread_analysis.*`
- Writes outputs to `outputs/{panels,roll_signals,analysis}/`

**`trading_days.py`** (business day computation):
- Loads CME/Globex holiday calendar
- Maps intraday timestamps to trading dates (Asia session 21:00 anchor)
- Applies data guards: coverage (min buckets), volume thresholds (lifecycle-aware)
- Returns business day index for gap calculations

**`multi_spread_analysis.py`** (supervisor's test):
- `compare_spread_signals()`: Detection rates, event counts per spread
- `analyze_spread_timing()`: Days-to-expiry for each spread's events
- `summarize_timing_by_spread()`: Median/IQR timing statistics
- `compute_spread_correlations()`: 11×11 correlation matrix

### Configuration System

**`config/settings.yaml`** drives all analysis parameters:

```yaml
products: [HG]  # Commodity code

bucket_config:  # Hourly aggregation periods
  us_regular_hours: {start: 9, end: 15}
  off_peak_sessions: {late_us, asia, europe}

data:
  minute_root: "../organized_data/copper"
  timezone: "US/Central"

data_quality:  # Daily analysis filtering
  cutoff_year: 2015
  min_data_points: 500
  min_coverage_percent: 25

spread:  # Event detection
  method: "zscore"
  window_buckets: 20
  z_threshold: 1.5
  cool_down_hours: 3.0

business_days:
  calendar_paths: ["../metadata/calendars/cme_globex_holidays.csv"]
  min_total_buckets: 6
  volume_threshold: {method: "dynamic", ...}
```

**Override system**: CLI flags → `overrides` dict → merged into settings

## Output Files Reference

### `outputs/panels/`
- `hourly_panel.parquet`: (44K periods × 1015 cols) Full dataset with all contracts + metadata
- `hg_panel_filtered.parquet`: Daily analysis with quality filtering applied

### `outputs/roll_signals/`
- `hourly_spread.csv`: Original S1 spread series
- `hourly_widening.csv`: Boolean event flags for S1
- `multi_spreads.csv`: **All 11 spreads (S1-S11)** side-by-side
- `multi_spread_events.csv`: Boolean event flags for all spreads

### `outputs/analysis/`
- `bucket_summary.csv`: Event counts/rates by intraday period (1-10)
- `preference_scores.csv`: Volume-normalized event concentration (score > 1.0 = elevated activity)
- `transition_matrix.csv`: Probability of next event in each bucket given current bucket
- `spread_correlations.csv`: **11×11 correlation matrix between spreads**
- `spread_signal_comparison.csv`: **Detection rates, event counts for S1-S11**
- `spread_timing_summary.csv`: **Median days-to-expiry per spread**
- `business_days_audit_hourly.csv`: Trading day validation report

## Common Development Tasks

### Adding a New Spread Calculation

1. Extend `rolls.compute_multi_spreads()` with vectorized logic
2. Add corresponding event detection in `events.detect_multi_spread_events()`
3. Update analysis outputs in `analysis.run_bucket_analysis()`
4. Add unit tests in `tests/test_bucket.py` or `tests/test_events.py`

### Modifying Event Detection

All detection happens in `events.detect_spread_events()`:
- Supports `"zscore"`, `"abs"`, or `"combined"` methods
- Rolling window calculations via pandas `.rolling()`
- Cool-down via `_apply_cool_down()` helper
- Returns boolean Series with same index as input spread

### Adding Bucket Periods

Bucket definitions in `buckets.py`:
- `assign_hour_to_bucket()`: Map hour (0-23) → bucket ID
- `get_bucket_info()`: Return (label, session, duration) for bucket ID
- Update `config/settings.yaml` with new period ranges
- Ensure 24-hour coverage (all hours must map to exactly one bucket)

## Critical Implementation Notes

**Contract Code Normalization**:
- Input: `HGZ25`, `HG_Z25`, `HGZ2025`
- Normalized: `HGZ2025` (always 4-digit year)
- Done in `ingest._normalize_contract_code()`

**Asia Session Cross-Midnight**:
- 21:00-02:59 CT maps to bucket 9
- Timestamps 21:00-23:59 belong to previous calendar day's trading date
- Handled via `_compute_trading_dates()` in `trading_days.py`

**Spread Convention**: Always `next - front` (contango positive, backwardation negative)

**Business Day Counting**: Exclusive of event date, inclusive of expiry date. Gap between events uses `pd.DatetimeIndex.get_indexer()` for rank difference.

## Test Suite Structure

- `test_bucket.py`: Bucket assignment, OHLCV aggregation, cross-midnight handling, statistics
- `test_events.py`: Z-score detection, cool-down, different thresholds
- `test_trading_days.py`: Calendar loading, business day computation, dynamic thresholds, gap calculation
- `test_panel.py`: Metadata merging, bucket metadata union
- `test_ingest_panel.py`: Contract normalization, aggregation conservation

**Coverage**: 46 tests, 96% code coverage

## LaTeX Report Structure

`presentation_docs/analysis_report.tex`:

1. **Introduction**: Project overview
2. **Data Architecture**: Raw data specs, organization, business days, metadata
3. **Methodology**:
   - Intraday aggregation (Section 3.1)
   - Contract identification (Section 3.2)
   - Calendar spreads (Section 3.3)
   - **Multi-spread analysis** (Section 3.4) - Added for supervisor's test
   - Event detection (Section 3.5)
   - Quality filtering (Section 3.6)
4. **Implementation**: Package structure, CLI
5. **Empirical Results**:
   - Hourly analysis (Section 5.1)
   - Daily analysis (Section 5.2)
   - **Multi-spread comparative results** (Section 6) - Added with findings
6. **Performance**: Efficiency metrics, scalability, vectorization benefits
7. **Output Specifications**: File formats
8. **Testing**: Coverage, validation, edge cases
9. **Conclusions**: Revised to reflect expiry mechanics interpretation

**Key Sections Modified**: Abstract, Section 3.4, Section 6, Section 9 (conclusions) updated to present corrected interpretation based on multi-spread analysis.

## Research Context

**Original Hypothesis**: Events at ~19 days before expiry indicate institutional roll timing decisions.

**Supervisor's Concern**: Pattern might reflect contract expiry mechanics, not institutional strategy.

**Test Implemented**: Compute S1-S11 and check if pattern repeats across all spreads at equivalent times-to-expiry.

**Results**:
- S1: 2,737 events (6.16%), median 28 days to F1 expiry
- S2: 2,582 events (5.81%), median 59 days to F2 expiry
- S3: 2,270 events (5.11%), median 89 days to F3 expiry

**Conclusion**: Pattern "ripples through" contract chain. Each spread shows events ~28-30 days before its own front contract expires. This is **systematic contract maturity effect**, not discretionary institutional behavior.

**Implication**: Framework characterizes contract lifecycle dynamics, not strategic roll timing. Use findings to model expiry-driven effects for continuous futures series construction and basis trading strategies.
