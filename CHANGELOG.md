## 2025-11-07 – Streamlined Reporting

- **Single deliverable:** Removed all CSV/Parquet manifest outputs. `run_bucket_analysis()` and `run_daily_analysis()` now return structured results instead of writing to `outputs/`.
- **Automated LaTeX report:** Added `futures_roll_analysis.reporting.generate_report()` which renders a consolidated LaTeX report to a configurable path (`--report-path`, default: `presentation_docs/technical_implementation_report.tex`).
- **Unified CLI update:** `--mode all` executes hourly and daily pipelines sequentially. `--output-dir` has been retired; the only persistent artifact is the refreshed LaTeX report.
- **Repository cleanup:** Removed `outputs/` artifacts, intermediate audit scripts, and stale email attachments. The `presentation_docs/Makefile` now invokes the consolidated analysis before compiling the PDF.
- **Data quality integration:** Daily analysis retains contract-quality diagnostics in-memory and surfaces them in the report instead of writing JSON/CSV summaries.

## 2025-11-06 – Deterministic Expiry-Based Labeling

**BREAKING CHANGES:**

- **Contract Labeling**: Legacy availability-driven labeler removed. The framework now uses deterministic expiry-based switching exclusively
  - F2 becomes F1 **exactly at expiry instant** (supervisor requirement)
  - `selection.mode` config option removed – expiry-based labeling is always used
  - Deterministic helpers formerly exported as `_v2` now live at `identify_front_next` / `identify_front_to_f12`

- **Required Calendar**: Trading calendar is **mandatory**
  - Config validation fails fast if `business_days.calendar_paths` is empty or files missing
  - Defaults to strict calendar-only mode (no weekday fallback)
  - Advanced `fallback_policy` options still available but not recommended for production
  - Clear error messages guide users to provide valid calendar files

- **Hour-Based Timing**: All timing metrics now use hour precision instead of days
  - `_compute_days_to_expiry()` → `_compute_hours_to_expiry()`
  - All `.dt.days` replaced with `.dt.total_seconds() / 3600.0`
  - Config values (max_days) automatically converted to hours internally
  - Better alignment with intraday expiry times (e.g., 17:00 CT)

**Improvements:**

- **DST Handling**: Fixed 4 locations using `ambiguous="NaT"` causing silent data loss
  - Now uses `ambiguous="infer"` for DatetimeIndex (requires monotonic sorting)
  - Handles spring-forward and fall-back transitions correctly
  - Added tz-aware index guards to prevent crashes

- **Test Coverage**: Comprehensive test suite in `tests/test_rolls.py`
  - 8 tests verify exact expiry switching, DST handling, edge cases
  - Tests confirm supervisor requirement: "F1 appears exactly when previous expires"

- **Output Files**:
  - Renamed `roll_switches_v2.csv` → `roll_switches.csv`
  - Switch log now written unconditionally (not conditional on mode)

**Migration Guide:**

1. **Remove `selection` section** from `config/settings.yaml`
2. **Ensure calendar exists** at path in `business_days.calendar_paths`
3. **Update imports**: Change `identify_front_next_v2` → `identify_front_next`
4. **Expect hour-based metrics**: Timing outputs now in hours, not days

**Files Changed:**
- `src/futures_roll_analysis/rolls.py`: Removed 158 lines of legacy code
- `src/futures_roll_analysis/config.py`: Added calendar validation
- `src/futures_roll_analysis/spreads.py`: Removed weekday fallback
- `src/futures_roll_analysis/trading_days.py`: Hour-based timing
- `config/settings.yaml`: Removed `selection` and `business_days.enabled`

---

## 1.1.0 (2025-10-30)

- Calendar: strict validation of `session_note` values; accept alias `Open` → `Regular`.
- Business days: richer audit (`coverage_ok`, `volume_ok_flag`, `data_approved`, `reason`).
- Events: business-day summaries only; calendar-day gaps removed. Added optional `align_events` policy (`none`, `shift_next`, `drop_closed`).
- Config: validation of YAML with helpful errors; environment variable expansion in paths.
- Analysis: write `run_settings.json` per run for reproducibility.
- Buckets: fix trading date mapping to preserve local (wall-clock) dates, not UTC conversion.
- Ingest: improve filename regex to prefer 4-digit years and handle 1-digit years; logging cleanup.
- Scripts: add `futures-roll-cal-lint` calendar linter; fix `setup_env.sh` quoting.
- Tests: add fallback policy checks, ingest/aggregation tests; remove deprecation warning.
- Docs: README updates for business days, calendar lint, and CLI; calendar-day comparisons removed.
