# Futures Roll Analysis – Agent Anchor

## Overview
This repository provides a Python package, `futures_roll_analysis`, for loading
minute-level futures data, aggregating it to daily or session buckets, and
detecting institutional roll activity through calendar-spread analytics. The
codebase was restructured into a conventional `src/` layout with shared service
layers for ingest, panel assembly, roll identification, event detection, and
CLI entry points.

## Layout
```
config/                 # Project configuration (YAML)
metadata/               # Official contract metadata (CSV)
organized_data/         # Raw minute data organised by commodity (external)
outputs/                # Generated panels, signals, summaries
presentation_docs/      # LaTeX sources and published report
src/futures_roll_analysis/
  ├── analysis.py       # High-level bucket & daily analysis runners
  ├── buckets.py        # Bucket definitions and aggregation routines
  ├── cli/              # Command-line entry points (hourly, daily, organise)
  ├── config.py         # Settings loader with override support
  ├── events.py         # Spread event detection & summaries
  ├── ingest.py         # Minute data ingestion and contract aggregation
  ├── panel.py          # Panel assembly from per-contract frames
  ├── quality.py        # Data-quality filtering & reporting helpers
  └── rolls.py          # Vectorised front/next, spread & liquidity utilities
src/tests/              # Pytest suites for buckets/events
```

## Installation
```bash
python -m pip install --upgrade pip
python -m pip install -e .[dev,viz]
```
The editable install exposes console entry points registered in
`pyproject.toml`.

## Preparing Data
Raw text files can be organised into `organized_data/` using:
```bash
python -m futures_roll_analysis.cli.organize --source <raw_dir> --destination organized_data
```
An inventory CSV is written alongside the organised folders.

## Running Analyses
Hourly (bucket) pipeline:
```bash
python -m futures_roll_analysis.cli.hourly \
  --settings config/settings.yaml \
  --root organized_data/copper \
  --output-dir outputs
```
Daily pipeline with data-quality filtering:
```bash
python -m futures_roll_analysis.cli.daily \
  --settings config/settings.yaml \
  --root organized_data/copper \
  --output-dir outputs
```
Both commands honour optional `--metadata`, `--max-files`, and `--log-level`
overrides.

## Outputs
- `outputs/panels/` – Parquet and CSV panels with multi-contract fields.
- `outputs/roll_signals/` – Spread/widening/liquidity series.
- `outputs/analysis/` – Bucket summaries, preference scores, transition matrices,
  and daily widening summaries.
- `outputs/data_quality/` – CSV/JSON quality metrics when daily filtering runs.

## Testing & Quality
Run the automated suites with:
```bash
pytest
```
The refactor introduces vectorised front/next contract detection, shared
spread-event logic, and centralised configuration loading to remove script
duplication and `sys.path` hacks while maintaining previous analytical results.

## Business Days – Hybrid Calendar + Data Guard (Plan)

This section specifies an additive, opt‑in feature to compute business‑day gaps (instead of calendar‑day gaps) consistently across hourly and daily analyses using an authoritative CME/Globex calendar combined with a sophisticated data‑activity guard. Scope is limited to calendar→business filtering; not broader detection refinements.

Objectives
- Count event gaps in business days for both hourly and daily pipelines with market-aware precision.
- Use a hierarchical futures trading calendar system with dynamic activity validation to handle global complexity.
- Keep changes modular, configurable, backward‑compatible, and performant through intelligent caching.

Design Principles
- Single‑responsibility module for trading‑day logic; no CLI/IO coupling beyond calendar load.
- Reuse existing session→trading‑date mapping (Asia session anchor 21:00 prior day) to keep hourly/daily aligned.
- Compute once per pipeline run and pass the business‑day index into summarizers.
- Business days are always enabled; remove enable/disable paths.
- Support hierarchical calendar resolution for multi-exchange products.

New Module
- File: `src/futures_roll_analysis/trading_days.py`
- Core API:
  - `load_calendar(paths: Union[Path, List[Path]], hierarchy: str = "override") -> pd.DataFrame`
    - Load normalized trading calendar from CSV(s) with holiday/partial day info.
    - Support hierarchy: "override" (first wins), "union" (combine all), "intersection" (common only).
    - Return DataFrame with columns: date, holiday_name, session_note, partial_hours, is_trading_day.
  - `load_calendar_hierarchy(commodity: str, exchange: str = "CME") -> pd.DataFrame`
    - Auto-resolve calendar hierarchy: commodity-specific → exchange → global fallback.
    - Search order: `{exchange}_{commodity}.csv` → `{exchange}_globex.csv` → `global_futures.csv`.
  - `map_to_trading_date(index: pd.DatetimeIndex, bucket_ids: Optional[pd.Series]=None) -> pd.DatetimeIndex`
    - Wraps `_compute_trading_dates` (Asia cross‑midnight) so both pipelines agree on trading date.
  - `compute_dynamic_volume_threshold(panel: pd.DataFrame, front_next: pd.DataFrame, days_to_expiry: pd.Series, config: Optional[Dict]) -> pd.Series`
    - Returns lifecycle-aware volume thresholds (per timestamp) with fixed or dynamic percentile ranges.
  - `compute_business_days(panel: pd.DataFrame, front_next: pd.DataFrame, calendar: pd.DataFrame, *, expiry_map: pd.Series, min_total_buckets=6, min_us_buckets=2, volume_threshold_config=None, near_expiry_relax=5, partial_day_min_buckets=4, fallback_policy="calendar_only") -> pd.DatetimeIndex`
    - Intersect authoritative calendar with sophisticated data guard:
      - Coverage guard: total buckets ≥ `min_total_buckets` OR US buckets ≥ `min_us_buckets`.
      - Partial day handling: if calendar marks partial, require only `partial_day_min_buckets`.
      - Dynamic volume guard: per‑date front+next volume ≥ lifecycle-aware threshold.
      - Near expiry: if days_to_expiry(front) ≤ `near_expiry_relax`, allow reduced coverage.
      - Fallback policies: `calendar_only` (strict), `union_with_data` (inclusive), `intersection_strict` (conservative).
  - `business_day_gaps(events: pd.Series, business_days: pd.DatetimeIndex) -> pd.Series`
    - Compute per‑event gaps in business days (rank difference over the business‑day index).

Metadata
- Primary calendar: `metadata/calendars/cme_globex_holidays.csv`
  - Columns: `date` (YYYY‑MM‑DD), `holiday_name`, `session_note`, `partial_hours` (HH:MM-HH:MM or empty), `exchanges_affected` (comma-separated).
  - Example rows:
    ```csv
    date,holiday_name,session_note,partial_hours,exchanges_affected
    2024-12-24,Christmas Eve,Early close,09:00-13:00,CME,ICE,EUREX
    2024-12-25,Christmas,Closed,,CME,ICE,EUREX,LME
    2024-12-31,New Year's Eve,Early close,09:00-13:00,CME,ICE
    2025-01-01,New Year's Day,Closed,,ALL
    ```
- Commodity overrides: `metadata/calendars/cme_comex_hg.csv` (same schema, takes precedence).
- Global fallback: `metadata/calendars/global_futures.csv` (union of major exchanges).

Configuration
- Extend `config/settings.yaml` with comprehensive `business_days` section:
  ```yaml
  business_days:
    enabled: false
    calendar_paths:  # List for hierarchy
      - "../metadata/calendars/cme_comex_hg.csv"
      - "../metadata/calendars/cme_globex_holidays.csv"
    calendar_hierarchy: "override"  # "override" | "union" | "intersection"
    min_total_buckets: 6
    min_us_buckets: 2
    partial_day_min_buckets: 4
    volume_threshold:
      method: "dynamic"  # "dynamic" | "fixed"
      fixed_percentile: 0.10
      dynamic_ranges:  # Used when method="dynamic"
        - {max_days: 5, percentile: 0.30}
        - {max_days: 30, percentile: 0.20}
        - {max_days: 60, percentile: 0.10}
        - {max_days: 999, percentile: 0.05}
    near_expiry_relax: 5
    fallback_policy: "calendar_only"  # "calendar_only" | "union_with_data" | "intersection_strict"
  ```
- Update `config.load_settings` to parse nested volume_threshold config; maintain backward compatibility.

Pipeline Integration
- Hourly pipeline (`analysis.run_bucket_analysis`):
  - After `panel`, `front_next`, and `spread` are computed:
    - If `business_days.enabled`:
      - `cal = trading_days.load_calendar(settings.business_days.calendar_path)`
      - `biz = trading_days.compute_business_days(panel, front_next, cal, thresholds...)`
      - Pass `biz` to event summarization.
- Daily pipeline (`analysis.run_daily_analysis`):
  - Same pattern; coverage guard at daily granularity (presence of data + day volume); reuse `map_to_trading_date` for consistency if needed.

Events Summarization
- Extend `events.summarize_events` to emit `business_days_since_last` only (business-day spacing).
- Preserve existing behavior when `business_days` is omitted.

CLI Options
- Unified CLI (`unified_cli.py`):
  - Business days are always on; no `--business-days` flag.
  - `--calendar` to override `calendar_path`.
  - Feed as overrides into `load_settings` (consistent with other flags).

Outputs
- Hourly analysis emits `analysis/hourly_widening_summary.csv` with both calendar and business-day gaps.
- Daily and hourly widening summaries include `business_days_since_last`.

Performance & Robustness
- Vectorized operations for coverage/volume calculations.
- Graceful degradation: if calendar unavailable, fall back to data-only detection with warning.
- Handle timezone complications: all calculations in exchange timezone (US/Central for CME).
- Weekend handling: Saturday never business day; Sunday night (>18:00) counts toward Monday.

Testing
- Unit tests (`tests/test_trading_days.py`):
  - Calendar parsing with partial days and multiple exchanges.
  - Hierarchical calendar resolution.
  - Dynamic volume threshold calculation across lifecycle.
  - Trading date mapping for Asia cross-midnight.
  - All three fallback policies.
- Edge case tests:
  - Contract expiring on holiday (Dec 25).
  - Partial trading days (Thanksgiving, Christmas Eve).
  - Sunday night futures opening.
  - Month/year boundary transitions.
  - Leap day handling.
  - DST transitions.
- Integration tests:
  - Synthetic panels with known business days.
  - Comparison with exchange-published trading calendars.
- Non‑regression: with `enabled=false`, outputs byte-identical.

Validation
- Phase 2.5 (Parallel Validation):
  - Run both calendar and business day calculations for 2 weeks.
  - Compare outputs; verify business day counts match exchange records.
  - Audit log discrepancies for investigation.
  - Success criteria: >99% agreement with CME official trading days.

Documentation
- Document the calendar CSV schema and governance; README shows business-day usage and custom calendar override.

Migration Plan (4-Week Implementation)
- Phase 1: Add `trading_days.py`, calendar CSV, config parsing, and unit tests.
- Phase 2: Extend `events.summarize_events` to support business‑day gaps.
- Phase 3: Wire into hourly/daily pipelines behind the config flag; add CLI flags; write summary CSV.
- Phase 4: Update docs; optionally emit `business_days_index.csv`; log counts for candidate vs accepted business days.

Acceptance Criteria (Production Readiness)
- Feature off: outputs unchanged.
- Feature on: summaries include `business_days_since_last`; business‑day index is monotonic and reproducible; guard thresholds exclude spurious days but retain valid near‑expiry shortened sessions.
