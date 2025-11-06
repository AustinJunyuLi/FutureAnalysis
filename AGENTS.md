# Futures Roll Analysis – Repository Guidelines

> **Note**: This document consolidates repository guidelines, architecture overview, and development standards for the futures roll analysis framework (v2.0.0+).

---

## Project Structure & Module Organization

```
config/                    # Runtime configuration (YAML)
metadata/                  # Contract metadata and trading calendars (CSV)
  ├── calendars/           # CME/Globex trading calendars (REQUIRED)
  └── contracts_metadata.csv
organized_data/            # Raw minute data by commodity (external, ~26GB)
outputs/                   # Generated analysis outputs (gitignored)
  ├── panels/              # Multi-contract panels (Parquet/CSV)
  ├── roll_signals/        # Spread and liquidity series
  ├── analysis/            # Bucket summaries, preference scores
  └── data_quality/        # Quality metrics and filtering reports
presentation_docs/         # LaTeX report sources
  ├── analysis_report.tex  # Main report
  ├── figures/             # Generated plots
  └── Makefile             # Build system
scripts/                   # Analysis and utility scripts
  ├── analysis/            # One-off analysis scripts
  ├── validation/          # Data validation utilities
  └── setup/               # Environment setup scripts
src/futures_roll_analysis/ # Core package
  ├── analysis.py          # Pipeline orchestration (hourly/daily)
  ├── buckets.py           # Intraday period aggregation
  ├── labeler.py           # Deterministic expiry-based labeling (F1-F12)
  ├── expiries.py          # Expiry metadata loading with DST fail-fast
  ├── rolls.py             # Front/next identification pipeline
  ├── events.py            # Spread event detection
  ├── spreads.py           # Calendar spread computation
  ├── trading_days.py      # Business day computation with calendar
  ├── quality.py           # Data quality filtering
  ├── ingest.py            # Minute data loading
  ├── panel.py             # Panel assembly
  ├── config.py            # Settings loader and validation
  ├── calendar_tools.py    # Calendar linting and validation
  ├── unified_cli.py       # Main CLI entry point
  └── cli/                 # Command implementations
tests/                     # Pytest suite (59 tests, 31% coverage)
  ├── test_bucket.py       # 17 tests - bucket assignment
  ├── test_rolls.py        # 8 tests - front/next identification, DST
  ├── test_labeler.py      # 3 tests - strip labeling
  ├── test_trading_days.py # 21 tests - calendar and business days
  └── ...
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+ (3.11 recommended for development)
- Miniconda or Anaconda

### Environment Setup
```bash
# Create conda environment
conda create -n futures-roll python=3.11
conda activate futures-roll

# Install package in editable mode
cd /path/to/futures_individual_contracts_1min
pip install -e .[dev,viz]

# Verify installation
futures-roll --help
pytest -q  # Run test suite
```

### Shell Configuration (Optional)
Add to `~/.bashrc` for convenience:
```bash
export FUTURES_PROJECT="/path/to/futures_individual_contracts_1min"
conda activate futures-roll 2>/dev/null || true
```

---

## Development Workflow

### Running Analysis

**Hourly (bucket) pipeline:**
```bash
futures-roll analyze --mode hourly --settings config/settings.yaml
```

**Daily pipeline with data quality filtering:**
```bash
futures-roll analyze --mode daily --settings config/settings.yaml
```

**Test run with limited files:**
```bash
futures-roll analyze --mode hourly --max-files 5 --output-dir outputs/test
```

**Organize raw data:**
```bash
futures-roll organize --source raw_data --destination organized_data --dry-run
```

### Testing

**Run full test suite:**
```bash
pytest
```

**With coverage:**
```bash
pytest --cov=src/futures_roll_analysis --cov-report=term-missing
```

**Run specific test file:**
```bash
pytest tests/test_rolls.py -v
```

### Building Documentation

**From `presentation_docs/` directory:**
```bash
make analysis       # Compile LaTeX report
make view-analysis  # Open PDF
make clean          # Remove auxiliary files
```

---

## Coding Standards

### Python Style
- **Version**: Python 3.9+ required, 3.11 recommended for development
- **Style**: PEP 8, 4-space indentation
- **Type Hints**: Add for all new/modified functions
- **Naming**:
  - Modules/functions: `snake_case`
  - Classes: `CapWords`
  - Constants: `UPPER_SNAKE_CASE`

### Best Practices
- Use `pathlib.Path` for all file operations
- Use `logging.getLogger(__name__)` instead of `print()` in library code
- Configuration from `config/settings.yaml`, not hard-coded paths
- Avoid `sys.path` manipulation - use proper package imports
- Document complex algorithms with docstrings
- Keep functions focused and single-purpose

### Testing Guidelines
- Place tests in `tests/test_<module>.py`
- Follow Arrange–Act–Assert pattern
- Cover edge cases:
  - Calendar holidays and partial trading days
  - Sparse or near-expiry data
  - DST transitions
  - Error conditions
- Target ≥80% coverage for changed code
- Maintain existing coverage when modifying files

---

## Version Control & Commits

### Commit Messages
Use Conventional Commits format:
```
<type>: <short imperative summary>

[optional detailed explanation]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types:** `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `perf`

**Examples:**
- `feat: add multi-spread analysis (S1-S11)`
- `fix(rolls): fix DST handling and add tz-aware index guards`
- `docs: consolidate guidelines and archive session logs`
- `chore: remove duplicate files and build artifacts`

### Pull Requests
- Include purpose and rationale
- Provide sample CLI invocation showing the change
- Attach before/after snippets or plots when relevant
- Link to related issues
- Ensure tests pass and coverage maintained

---

## Critical v2.0.0+ Requirements

### Contract Labeling (Deterministic Expiry-Based)

**Functions:**
- Use `identify_front_next()` and `identify_front_to_f12()` from `rolls.py`
- Legacy v1 functions **removed** in v2.0.0
- `selection.mode` config option **removed** - expiry-based labeling is always used

**Behavior:**
- Contracts switch at **exact expiry instant** (e.g., 17:00 CT)
- **Independent of data availability** - uses only expiry timestamps
- F2 becomes F1 precisely when previous F1 expires (supervisor requirement)
- Pass `tz_exchange` parameter for proper DST handling (default: "America/Chicago")

**Implementation:**
- Core logic in `labeler.py`: `compute_strip_labels()`
- Uses `np.searchsorted(side='right')` for exact switching
- UTC nanosecond representation enables vectorized comparison
- Requires monotonic index (automatically sorted if needed)

### Trading Calendar (Mandatory)

**Requirements:**
- Calendar is **REQUIRED** - analysis fails without it
- No weekday fallback exists in v2.0.0
- Specify in `business_days.calendar_paths` in settings.yaml
- Config validation fails fast if calendar missing/invalid

**Calendar File:**
- Location: `metadata/calendars/cme_globex_holidays.csv`
- Format: CSV with columns: `date`, `holiday_name`, `session_note`, `partial_hours`, `exchanges_affected`
- Validation: Use `futures-roll-cal-lint` to verify format

**Fallback Policy:**
- Defaults to `"calendar_only"` (strict)
- Alternative modes exist (`union_with_data`, `intersection_strict`) but **not recommended** for production

### Timing Precision (Hour-Based)

**Critical Rules:**
- Use **hours** for all time-to-expiry calculations: `(expiry - ts).total_seconds() / 3600.0`
- **NEVER use `.dt.days`** - it loses intraday precision
- Config accepts days for readability, but internally converts to hours
- Example: `near_expiry_relax: 5` (days in config) → `5 * 24` hours internally
- Results reported in days for readability: `days = hours / 24`

**Why:** Contracts expire at specific times (17:00 CT), not at midnight. Day-based arithmetic causes systematic timing errors when switches occur intraday.

### DST Handling (Comprehensive)

**Three-Strategy Approach:**

1. **For DatetimeIndex** (panel data):
   ```python
   if idx.tz is None:
       idx_local = idx.tz_localize("America/Chicago",
                                    ambiguous="infer",
                                    nonexistent="shift_forward")
   idx_utc = idx_local.tz_convert("UTC")
   ```
   - `ambiguous="infer"`: Auto-detect based on surrounding timestamps
   - **Requires monotonic index** - check `idx.is_monotonic_increasing` first
   - Automatically sort if needed: `idx = idx.sort_values()`

2. **For Individual Timestamps** (contract expiries):
   ```python
   if ts.tz is None:
       ts_local = ts.tz_localize("America/Chicago",
                                  ambiguous=True,
                                  nonexistent="shift_forward")
   ts_utc = ts_local.tz_convert("UTC")
   ```
   - `ambiguous=True`: Treat as DST (safe since expiries at 17:00 CT)
   - Expiries outside 1:00-2:00 AM ambiguous window

3. **For Expiry Loading** (fail-fast):
   ```python
   # In expiries.py
   local = pd.to_datetime(df['expiry_local_iso'], utc=False).dt.tz_localize(
       tz_exchange, ambiguous="raise", nonexistent="raise"
   )
   ```
   - `ambiguous="raise"`: Fail loudly if expiry data contains ambiguous times
   - Prevents silent corruption

**Always add tz-aware guards:**
```python
if idx.tz is None:
    # Apply localization
```

**Fixed Locations** (commits 34c828a, 9eaf075):
- `rolls.py:72, 94`
- `analysis.py:493, 506`
- `expiries.py:56`

---

## Configuration

### Settings File Structure

Main config: `config/settings.yaml`

**Key sections:**
- `products`: Commodities to analyze (e.g., `["HG"]`)
- `bucket_config`: Intraday period definitions (10 buckets)
- `data`: Input paths and timezone
- `data_quality`: Filtering criteria (v2.0.0: more lenient defaults)
- `spread`: Event detection parameters (z-score, cool-down)
- `business_days`: **Calendar paths (REQUIRED)**, coverage thresholds, fallback policy
- `strip_analysis`: Multi-spread analysis settings
- `output_dir`: Where to write results

**Critical config fields:**
```yaml
business_days:
  calendar_paths:  # REQUIRED - analysis fails if missing
    - "../metadata/calendars/cme_globex_holidays.csv"
  fallback_policy: "calendar_only"  # DEFAULT: strict calendar-only

time:
  tz_exchange: America/Chicago  # Exchange timezone for DST handling
```

### CLI Overrides

All config values can be overridden via CLI:
```bash
futures-roll analyze --mode hourly \
  --settings config/settings.yaml \
  --root organized_data/copper \
  --metadata metadata/contracts_metadata.csv \
  --output-dir outputs \
  --max-files 10
```

---

## Security & Best Practices

### Data Management
- Keep large data in `organized_data/` and `outputs/` (gitignored)
- Verify `.gitignore` before committing
- Never commit credentials or API keys
- Use environment variables for sensitive config

### Path Handling
- Use relative paths in config (works across machines)
- Use `pathlib.Path` for cross-platform compatibility
- Resolve paths in `config.py`, not in business logic

---

## Breaking Changes in v2.0.0

**Removed:**
- Legacy v1 labeler (`identify_front_next` with data-availability logic)
- `selection.mode` config option
- Weekday fallback for business days
- Day-based timing calculations

**Renamed:**
- `identify_front_next_v2` → `identify_front_next` (now canonical)
- `identify_front_to_f12_v2` → `identify_front_to_f12` (now canonical)
- `roll_switches_v2.csv` → `roll_switches.csv`

**Added:**
- `labeler.py` module with deterministic expiry-based labeling
- `expiries.py` module with UTC conversion and DST fail-fast
- Comprehensive DST handling (5 fixed locations)
- Hour-precision timing throughout
- Mandatory calendar validation

**See:** `CHANGELOG.md` for complete v2.0.0 release notes.

---

## Common Tasks

### Add a New Analysis Script
1. Create in `scripts/analysis/your_script.py`
2. Use logging instead of print
3. Load config via `config.load_settings()`
4. Write outputs to `outputs/exploratory/`
5. Add test if complex logic

### Modify Configuration Schema
1. Update `config/settings.yaml` with new field
2. Add validation in `config.py::load_settings()`
3. Update README.md and this file
4. Add test in `tests/test_config.py` (if exists)

### Fix a Bug
1. Write failing test first (`tests/test_<module>.py`)
2. Fix the bug
3. Verify test passes
4. Check coverage didn't decrease
5. Commit with `fix(<module>): description`

### Add Feature
1. Discuss design (create issue if large)
2. Write tests for new functionality
3. Implement feature
4. Update documentation
5. Commit with `feat(<module>): description`

---

## Troubleshooting

### "calendar_paths is required"
- Add calendar file path to `business_days.calendar_paths` in settings.yaml
- Verify calendar file exists at specified path
- Run `futures-roll-cal-lint` to validate calendar format

### "ambiguous times cannot be inferred"
- Check if index is monotonic: `idx.is_monotonic_increasing`
- Sort if needed: `idx = idx.sort_values()`
- Verify timezone-aware: `idx.tz is not None`

### Tests failing after changes
- Run `pytest -v` to see detailed failures
- Check if coverage decreased: `pytest --cov`
- Verify config/settings.yaml matches expected schema
- Check git status for untracked test artifacts

### Build artifacts showing in git
- Run `git status --ignored` to see what's ignored
- Clean: `make clean` (in presentation_docs/)
- Remove caches: `find . -name __pycache__ -exec rm -rf {} +`

---

## Additional Resources

- **README.md**: User-facing documentation and quick start
- **CHANGELOG.md**: Version history and release notes
- **CLAUDE.md**: AI agent instructions (for Claude Code development)
- **presentation_docs/analysis_report.pdf**: Complete methodology documentation
- **GitHub Issues**: Bug reports and feature requests

---

*Last updated: v2.0.0 (November 2025)*
*This document consolidates root AGENTS.md and docs/AGENTS.md*
