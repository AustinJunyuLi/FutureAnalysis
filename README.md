# Futures Roll Analysis

A Python framework for analyzing institutional roll patterns in futures markets through minute-level data aggregation and calendar-spread analytics.

## Overview

This repository provides the `futures_roll_analysis` package for:
- Loading minute-level futures data from CSV/Parquet files
- Aggregating to daily bars or custom hourly buckets
- Detecting front/next contracts and computing calendar spreads
- Identifying institutional roll activity through spread widening events
- Generating comprehensive roll signals and analysis outputs

## Installation

### Prerequisites
- Miniconda or Anaconda installed
- Python 3.11+ (tested with 3.11.14)

### Setup

1. Create and activate the conda environment:
```bash
conda create -n futures-roll python=3.11
conda activate futures-roll
```

2. Install the package in editable mode:
```bash
cd /path/to/futures_individual_contracts_1min
pip install -e .[dev,viz]
```

3. Set up your shell environment (add to `~/.bashrc`):
```bash
export FUTURES_PROJECT="/path/to/futures_individual_contracts_1min"
conda activate futures-roll 2>/dev/null || true
```

## Quick Start

### Using the Unified CLI

The project provides a single command-line interface with subcommands:

```bash
# Run hourly (bucket) analysis
futures-roll analyze --mode hourly --settings config/settings.yaml

# Run daily analysis with data quality filtering
futures-roll analyze --mode daily --settings config/settings.yaml

# Organize raw data files by commodity
futures-roll organize --source raw_data --destination organized_data

# Quick test with limited files
futures-roll analyze --mode hourly --max-files 5 --output-dir outputs/test
```

### Command Options

#### Analyze Command
- `--mode`: Choose `hourly` (bucket aggregation) or `daily`
- `--settings`: Path to YAML configuration file
- `--root`: Override data root directory
- `--metadata`: Override metadata CSV path
- `--output-dir`: Custom output directory
- `--max-files`: Limit files for testing (hourly mode only)
- `--log-level`: Set logging verbosity (DEBUG, INFO, WARNING, ERROR)

#### Organize Command
- `--source`: Directory containing raw futures files
- `--destination`: Output directory for organized data
- `--inventory`: Path for inventory CSV
- `--dry-run`: Preview without moving files

### Contract Labeling (F1/F2)

The package uses **deterministic expiry-based labeling** to identify front-month (F1) and next-month (F2) contracts:

- **Exact expiry switching**: F2 becomes F1 at the precise expiry instant (e.g., 17:00 CT)
- **Timezone-aware**: Handles DST transitions correctly (no data gaps)
- **Hour-based precision**: All timing calculations use hours, not days
- **UTC internal representation**: Ensures consistent cross-timezone behavior

**Key behavior**: Contracts switch based solely on expiry timestamps, independent of data availability. The switch occurs exactly when the previous F1 expires.

### Required Trading Calendar

A valid trading calendar is **required** for analysis. The package will fail with a clear error if the calendar is missing:

```yaml
business_days:
  calendar_paths:
    - "../metadata/calendars/cme_globex_holidays.csv"  # REQUIRED
```

To use a custom calendar via CLI:
```bash
futures-roll analyze --mode hourly \
  --settings config/settings.yaml \
  --calendar metadata/calendars/cme_globex_holidays.csv
```

**Calendar Configuration (`config/settings.yaml`):**
```yaml
business_days:
  calendar_paths:  # REQUIRED - list of trading calendar files
    - "../metadata/calendars/cme_globex_holidays.csv"
  calendar_hierarchy: "override"  # override | union | intersection
  min_total_buckets: 6            # Min buckets per day for data validation
  min_us_buckets: 2              # Min US session buckets
  volume_threshold:
    method: "dynamic"
    dynamic_ranges:              # Days converted to hours internally
      - {max_days: 5, percentile: 0.30}    # Delivery month
      - {max_days: 30, percentile: 0.20}   # Near expiry
      - {max_days: 60, percentile: 0.10}   # Active roll
      - {max_days: 999, percentile: 0.05}  # Far contracts
  near_expiry_relax: 5           # Days (converted to hours)
  fallback_policy: "calendar_only"
  align_events: "none"           # none | shift_next | drop_closed
```

**Key Features:**
- **Trading Calendar**: Uses official CME/Globex holiday schedules (2015-2025) - **REQUIRED**
- **Data Guards**: Validates trading activity with coverage and volume thresholds
- **Dynamic Thresholds**: Volume requirements adapt to contract lifecycle (config in days, computed in hours)
- **Partial Days**: Handles early close sessions (Thanksgiving, Christmas Eve)
- **Strict Mode**: Fails fast with clear errors if calendar is missing or invalid

### Calendar Linting

Validate one or more calendar CSVs and emit a JSON report:

```bash
futures-roll-cal-lint metadata/calendars/cme_globex_holidays.csv --json outputs/calendar_lint.json
```

### Event Detection Options

- `business_days.align_events`: Shift or drop events that land on closed days when auditing.

**Output:**  
Event summaries include business-day gaps only (`business_days_since_last`).

## Project Structure

```
futures_individual_contracts_1min/
├── src/futures_roll_analysis/
│   ├── analysis.py         # High-level analysis runners
│   ├── buckets.py          # Bucket definitions and aggregation
│   ├── unified_cli.py      # Unified CLI entry point
│   ├── cli/                # Legacy subcommands (hourly, daily, organize)
│   ├── config.py           # Settings loader
│   ├── events.py           # Spread event detection
│   ├── ingest.py           # Data ingestion and normalization
│   ├── panel.py            # Panel assembly from contracts
│   ├── quality.py          # Data quality filtering
│   ├── trading_days.py     # Business day calendar utilities
│   └── rolls.py            # Roll detection utilities
├── tests/                   # Pytest test suites
├── config/
│   └── settings.yaml       # Main configuration
├── metadata/               # Contract metadata CSVs
├── organized_data/         # Minute data by commodity
├── outputs/                # Analysis outputs
│   ├── panels/            # Multi-contract panels
│   ├── roll_signals/      # Spread and roll signals
│   ├── analysis/          # Summaries and matrices
│   └── data_quality/      # Quality metrics
└── scripts/
    ├── analysis/           # Exploratory and one-off analysis scripts
    │   ├── contract_month_analysis.py
    │   ├── term_structure_*.py
    │   └── volume_crossover.py
    ├── setup/              # Environment setup and maintenance
    │   ├── setup_env.sh
    │   └── cleanup.sh
    └── validation/         # Data validation utilities
        └── verify_cme_holidays.py
```

## Key Features

### Data Processing
- **Robust CSV Loading**: Handles both headed and headerless minute data files
- **Multi-format Support**: CSV, TXT, and Parquet input formats
- **Contract Normalization**: Standardizes contract codes (e.g., HGZ25 → HGZ2025)
- **Quality Filtering**: Configurable filters for data completeness and date ranges

### Analysis Capabilities
- **Bucket Aggregation**: Variable-granularity time buckets for intraday analysis
  - US Regular Hours: 7 hourly buckets (9:00-15:00 CT)
  - Off-peak Sessions: Asia, Europe, and Late US sessions
- **Roll Detection**: Vectorized front/next contract identification
- **Spread Analysis**: Calendar spread computation with event detection
- **Event Detection**: Z-score and absolute threshold methods with cool-down periods
- **Strip Diagnostics**: Full F1–F12 strip with S1–S11 spreads, dominance metrics, and expiry-window filtering to distinguish genuine rolling from expiry mechanics

### Output Files
- **Panels**: Wide-format DataFrames with all contracts (Parquet/CSV)
- **Roll Signals**: Time series of spreads, widening events, liquidity rolls
- **Analysis Summaries**: Bucket statistics, preference scores, transition matrices,
  and hourly/daily widening summaries (calendar and business day gaps when enabled)
- **Quality Reports**: Filtering metrics and contract status

## Testing

Run the test suite:
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_bucket.py -v

# With coverage
pytest tests/ --cov=futures_roll_analysis --cov-report=html
```

## Configuration

The main configuration file is `config/settings.yaml`:

```yaml
products: [HG]  # Commodity symbols
bucket_config:  # Hourly bucket settings
data:           # Data paths and timezones
data_quality:   # Filtering parameters
roll_rules:     # Roll detection rules
spread:         # Event detection settings
output_dir:     # Output directory
```

## Code Quality

### Recent Improvements
- Fixed cross-midnight bucket handling for Asia session
- Improved CSV loading for headerless files
- Enhanced panel metadata merging
- Added comprehensive test coverage for bucket aggregation

### Architecture Highlights
- **Modular Design**: Clear separation of concerns
- **Vectorized Operations**: NumPy-based computations for performance
- **Type Hints**: Improved IDE support and documentation
- **Comprehensive Testing**: 23 test cases covering core functionality

## Development

### Environment Setup
```bash
# Use the setup script for each session
source scripts/setup/setup_env.sh

# Clean cache files
./scripts/setup/cleanup.sh
```

### Making Changes
1. Always use the conda environment
2. Run tests before committing
3. Update documentation for API changes
4. Follow existing code style

## Technical Details

For detailed implementation notes and analysis methodology, see:
- [`AGENTS.md`](AGENTS.md): Repository guidelines, architecture, and development standards
- [`presentation_docs/analysis_report.pdf`](presentation_docs/analysis_report.pdf): Complete methodology documentation

## License

Internal use only.

## Author

Austin Li
