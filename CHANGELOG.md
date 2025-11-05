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
